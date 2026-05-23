import torch
import torch.nn as nn
import numpy as np
import os.path as osp
import networkx as nx
import torch.nn.functional as func
from tqdm import tqdm
from torch import optim
from datetime import datetime
from torch_geometric.utils import to_dense_batch

from src.model.ewc import EWC
from torch_geometric.loader import DataLoader
from dataer.SpatioTemporalDataset import SpatioTemporalDataset
from utils.metric import cal_metric, masked_mae_np
from utils.common_tools import mkdirs, load_best_model

# ---------------------------------------------------------------------------
# Ported from ST-TTC (https://github.com/Onedean/ST-TTC, arxiv 2506.00635)
#   continual_learning_setting/src/trainer/default_trainer.py :: FRPlusModule
# Spectral-domain phase/amplitude calibrator. Consumes per-year predictions
# [B,1,N,T] and emits bias-corrected predictions of the same shape. Trained
# online from a streaming queue inside test_model_with_ttc.
# ---------------------------------------------------------------------------
class FRPlusModule(nn.Module):
    def __init__(self, num_nodes, freq_bins, groups=4):
        super().__init__()
        self.groups = groups
        self.group_size = max(1, freq_bins // groups)
        self.lambda_amp = nn.Parameter(torch.zeros(groups, num_nodes, 1))
        self.lambda_phi = nn.Parameter(torch.zeros(groups, num_nodes, 1))

    def forward(self, y_pred):
        B, C, N, T = y_pred.shape
        y = y_pred[:, 0]
        Yf = torch.fft.rfft(y, dim=-1)
        A = torch.abs(Yf)
        P = torch.angle(Yf)

        Yf_corr = torch.zeros_like(Yf)
        for g in range(self.groups):
            start = g * self.group_size
            end = T // 2 + 1 if g == self.groups - 1 else (g + 1) * self.group_size
            lam_a = self.lambda_amp[g].unsqueeze(0)
            lam_p = self.lambda_phi[g].unsqueeze(0)
            A_g = A[:, :, start:end] * (1 + lam_a)
            P_g = P[:, :, start:end] + lam_p
            Yf_corr[:, :, start:end] = A_g * torch.exp(1j * P_g)

        y_time = torch.fft.irfft(Yf_corr, n=T, dim=-1)
        return y_time.unsqueeze(1)


def train(inputs, args):
    path = osp.join(args.path, str(args.year))  # Define the current year model save path
    mkdirs(path)
    
    # Setting the loss function
    if args.loss == "mse":
        lossfunc = func.mse_loss
    elif args.loss == "huber":
        lossfunc = func.smooth_l1_loss
    
    # Dataset definition
    if args.strategy == 'incremental' and args.year > args.begin_year:
        # Incremental Policy Data Loader
        train_loader = DataLoader(SpatioTemporalDataset("", "", x=inputs["train_x"][:, :, args.subgraph.numpy()], y=inputs["train_y"][:, :, args.subgraph.numpy()], \
            edge_index="", mode="subgraph"), batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=32)
        val_loader = DataLoader(SpatioTemporalDataset("", "", x=inputs["val_x"][:, :, args.subgraph.numpy()], y=inputs["val_y"][:, :, args.subgraph.numpy()], \
            edge_index="", mode="subgraph"), batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=32)
        # Construct the adjacency matrix of the subgraph
        graph = nx.Graph()
        graph.add_nodes_from(range(args.subgraph.size(0)))
        graph.add_edges_from(args.subgraph_edge_index.numpy().T)
        adj = nx.to_numpy_array(graph)  # Convert to adjacency matrix
        adj = adj / (np.sum(adj, 1, keepdims=True) + 1e-6)  # Normalized adjacency matrix
        vars(args)["sub_adj"] = torch.from_numpy(adj).to(torch.float).to(args.device)
    else:
        # Common Data Loader
        train_loader = DataLoader(SpatioTemporalDataset(inputs, "train"), batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=32)
        val_loader = DataLoader(SpatioTemporalDataset(inputs, "val"), batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=32)
        vars(args)["sub_adj"] = vars(args)["adj"]  # Use the adjacency matrix of the entire graph
    
    # Test Data Loader
    test_loader = DataLoader(SpatioTemporalDataset(inputs, "test"), batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=32)
    
    args.logger.info("[*] Year " + str(args.year) + " Dataset load!")

    # Model definition
    if args.init == True and args.year > args.begin_year:
        gnn_model, _ = load_best_model(args)  # If it is not the first year, load the optimal model
        if args.ewc:  # If you use the ewc strategy, use the ewc model
            args.logger.info("[*] EWC! lambda {:.6f}".format(args.ewc_lambda))  # Record EWC related parameters
            model = EWC(gnn_model, args.adj, args.ewc_lambda, args.ewc_strategy)  # Initialize the EWC model
            ewc_loader = DataLoader(SpatioTemporalDataset(inputs, "train"), batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=32)
            model.register_ewc_params(ewc_loader, lossfunc, args.device)  # Register EWC parameters
        else:
            model = gnn_model  # Otherwise, use the best model loaded
        
        if args.method == 'EAC':
            for name, param in model.named_parameters():
                if "gcn1" in name or "tcn1" in name or "gcn2" in name or "fc" in name:
                    param.requires_grad = False
        
        if args.method == 'EAC':
            model.expand_adaptive_params(args.graph_size)
        
        if args.method == 'Universal' and args.use_eac == True:
            for name, param in model.named_parameters():
                if "gcn1" in name or "tcn1" in name or "gcn2" in name or "fc" in name:
                    param.requires_grad = False
        
        if args.method == 'Universal' and args.use_eac == True:
            model.expand_adaptive_params(args.graph_size)
        
    else:
        gnn_model = args.methods[args.method](args).to(args.device)  # If it is the first year, use the base model
        model = gnn_model
        if args.method == 'EAC':
            model.expand_adaptive_params(args.graph_size)
        
        if args.method == 'Universal' and args.use_eac == True:
            model.expand_adaptive_params(args.graph_size)
    
    if hasattr(model, 'count_parameters'):
        model.count_parameters()
    # for name, param in model.named_parameters():
    #     print(f"Parameter: {name} | Requires Grad: {param.requires_grad}")
    
    
    # Model Optimizer
    # optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    

    args.logger.info("[*] Year " + str(args.year) + " Training start")
    lowest_validation_loss = 1e7
    counter = 0
    patience = 5
    model.train()
    use_time = []
    
    for epoch in range(args.epoch):
        
        start_time = datetime.now()
        
        # Training the model
        cn = 0
        training_loss = 0.0
        for batch_idx, data in enumerate(train_loader):
            if epoch == 0 and batch_idx == 0:
                args.logger.info("node number {}".format(data.x.shape))
            data = data.to(args.device, non_blocking=True)
            optimizer.zero_grad()
            pred = model(data, args.sub_adj)
            
            if args.strategy == "incremental" and args.year > args.begin_year:
                pred, _ = to_dense_batch(pred, batch=data.batch)  # to_dense_batch is used to convert a batch of sparse adjacency matrices into a batch of dense adjacency matrices
                data.y, _ = to_dense_batch(data.y, batch=data.batch)
                pred = pred[:, args.mapping, :]  # Slice according to the mapping to obtain the prediction and true value of the change node
                data.y = data.y[:, args.mapping, :]
            
            loss = lossfunc(data.y, pred, reduction="mean")
            
            if args.ewc and args.year > args.begin_year:
                loss += model.compute_consolidation_loss()  # Calculate and add ewc loss if necessary
            
            # NaN / Inf guard: some years in xxltrafficdata have missing sensor values.
            # If upstream preprocessing hasn't fully cleaned them, a single bad batch
            # would otherwise poison the optimizer and make every subsequent year NaN.
            if not torch.isfinite(loss):
                args.logger.warning(
                    f"[NaN-guard] year={args.year} epoch={epoch} batch={batch_idx}: "
                    f"non-finite loss={loss.item()}, "
                    f"x_nan={torch.isnan(data.x).any().item()} "
                    f"y_nan={torch.isnan(data.y).any().item()} "
                    f"pred_nan={torch.isnan(pred).any().item()} — skipping batch"
                )
                optimizer.zero_grad(set_to_none=True)
                continue
            
            training_loss += float(loss)
            cn += 1
            
            loss.backward()
            optimizer.step()
        
        
        if epoch == 0:
            total_time = (datetime.now() - start_time).total_seconds()
        else:
            total_time += (datetime.now() - start_time).total_seconds()
        use_time.append((datetime.now() - start_time).total_seconds())
        # Guard against the degenerate case where every batch in this epoch was skipped
        # by the NaN-guard above (shouldn't happen once upstream data is clean).
        training_loss = training_loss / cn if cn > 0 else float("nan")
        
        # Validate the model
        validation_loss = 0.0
        cn = 0
        with torch.no_grad():
            for batch_idx, data in enumerate(val_loader):
                data = data.to(args.device, non_blocking=True)
                pred = model(data, args.sub_adj)
                if args.strategy == "incremental" and args.year > args.begin_year:
                    pred, _ = to_dense_batch(pred, batch=data.batch)
                    data.y, _ = to_dense_batch(data.y, batch=data.batch)
                    pred = pred[:, args.mapping, :]
                    data.y = data.y[:, args.mapping, :]
                
                loss = masked_mae_np(data.y.cpu().data.numpy(), pred.cpu().data.numpy(), 0)
                if not np.isfinite(loss):
                    continue
                validation_loss += float(loss)
                cn += 1
        validation_loss = float(validation_loss / cn) if cn > 0 else float("inf")
        

        args.logger.info(f"epoch:{epoch}, training loss:{training_loss:.4f} validation loss:{validation_loss:.4f}")
        
        # Early Stopping Strategy
        if validation_loss <= lowest_validation_loss:
            counter = 0
            lowest_validation_loss = round(validation_loss, 4)
            if args.ewc:
                torch.save({'model_state_dict': gnn_model.state_dict()}, osp.join(path, str(round(validation_loss,4))+".pkl"))
            else:
                torch.save({'model_state_dict': model.state_dict()}, osp.join(path, str(round(validation_loss,4))+".pkl"))
        else:
            counter += 1
            if counter > patience:
                break
        
    best_model_path = osp.join(path, str(lowest_validation_loss)+".pkl")
        
    if args.ewc:
        best_model = args.methods[args.method](args)
    else:
        best_model = model
    
    best_model.load_state_dict(torch.load(best_model_path, args.device)["model_state_dict"])
    best_model = best_model.to(args.device)
    
    # Test the Model — dispatch ST-TTC overlay if configured
    if getattr(args, "use_ttc", 0):
        test_model_with_ttc(best_model, args, test_loader, True)
    else:
        test_model(best_model, args, test_loader, True)
    args.result[args.year] = {"total_time": total_time, "average_time": sum(use_time)/len(use_time), "epoch_num": epoch+1}
    args.logger.info("Finished optimization, total time:{:.2f} s, best model:{}".format(total_time, best_model_path))


def test_model(model, args, testset, pin_memory):
    
    model.eval()
    pred_ = []
    truth_ = []
    loss = 0.0
    with torch.no_grad():
        cn = 0
        for data in testset:
            data = data.to(args.device, non_blocking=pin_memory)
            pred = model(data, args.adj)
            
            loss += func.mse_loss(data.y, pred, reduction="mean")
            pred, _ = to_dense_batch(pred, batch=data.batch)
            data.y, _ = to_dense_batch(data.y, batch=data.batch)
                        
            pred_.append(pred.cpu().data.numpy())
            truth_.append(data.y.cpu().data.numpy())
            cn += 1
        loss = loss / cn
        args.logger.info("[*] loss:{:.4f}".format(loss))
        pred_ = np.concatenate(pred_, 0)
        truth_ = np.concatenate(truth_, 0)
        cal_metric(truth_, pred_, args)



def masked_mae(prediction: torch.Tensor, target: torch.Tensor, null_val: float = np.nan) -> torch.Tensor:
    if np.isnan(null_val):
        mask = ~torch.isnan(target)
    else:
        eps = 5e-5
        mask = ~torch.isclose(target, torch.tensor(null_val).expand_as(target).to(target.device), atol=eps, rtol=0.0)

    mask = mask.float()
    mask /= torch.mean(mask)  # Normalize mask to avoid bias in the loss due to the number of valid entries
    mask = torch.nan_to_num(mask)  # Replace any NaNs in the mask with zero

    loss = torch.abs(prediction - target)
    loss = loss * mask  # Apply the mask to the loss
    loss = torch.nan_to_num(loss)  # Replace any NaNs in the loss with zero

    return torch.mean(loss)


# ---------------------------------------------------------------------------
# Ported from ST-TTC (https://github.com/Onedean/ST-TTC, arxiv 2506.00635)
#   continual_learning_setting/src/trainer/default_trainer.py :: test_model_with_ttc
# Differences vs upstream:
#   * Queue keeps each sparse Batch's own .batch index, so the online-update
#     step uses x_o.batch — upstream incorrectly reuses the current batch's
#     .batch and silently mis-shapes yb_o whenever batch sizes differ at the
#     tail of the year.
#   * NaN guard skips bad batches instead of poisoning the calibrator.
#   * ttc_lr / ttc_groups read from args so configs can tune per dataset.
# ---------------------------------------------------------------------------
def test_model_with_ttc(model, args, testset, pin_memory):
    import queue as _queue

    model.eval()

    T = args.y_len
    M = T // 2 + 1
    groups = int(getattr(args, "ttc_groups", 4))
    ttc_lr = float(getattr(args, "ttc_lr", 1e-4))

    FRP = FRPlusModule(args.graph_size, M, groups).to(args.device)
    ttc_optim = optim.Adam(FRP.parameters(), lr=ttc_lr)
    q = _queue.Queue(maxsize=T)

    pred_ = []
    truth_ = []

    for data in tqdm(testset):
        data = data.to(args.device, non_blocking=pin_memory)

        with torch.no_grad():
            pred = model(data, args.adj)
        pred, _ = to_dense_batch(pred, batch=data.batch)
        y_dense, _ = to_dense_batch(data.y, batch=data.batch)

        FRP.eval()
        with torch.no_grad():
            pred_corr = FRP(pred.unsqueeze(1)).squeeze(1)

        pred_.append(pred_corr.cpu().data.numpy())
        truth_.append(y_dense.cpu().data.numpy())

        q.put((data, y_dense))
        if q.full():
            x_o, y_o = q.get()
            with torch.no_grad():
                yb_o_sparse = model(x_o, args.adj)
            yb_o, _ = to_dense_batch(yb_o_sparse, batch=x_o.batch)

            FRP.train()
            yc_o = FRP(yb_o.unsqueeze(1)).squeeze(1)
            loss = func.mse_loss(yc_o, y_o)
            if not torch.isfinite(loss):
                args.logger.warning(
                    f"[TTC NaN-guard] year={args.year}: non-finite calibrator loss, skipping update"
                )
                ttc_optim.zero_grad(set_to_none=True)
                FRP.eval()
                continue
            ttc_optim.zero_grad()
            loss.backward()
            ttc_optim.step()
            FRP.eval()

    pred_ = np.concatenate(pred_, 0)
    truth_ = np.concatenate(truth_, 0)
    cal_metric(truth_, pred_, args)