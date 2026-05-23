"""SE-LeWM: Sensor-Evolving Latent World Model entry point.

Mirrors `a2tta_main.py`:
  * loads a frozen per-year backbone checkpoint (Online-AN by default;
    falls back to retrain ckpts)
  * builds a `SELeWMAdapter` once and lets it persist (and grow) across years
  * (optionally) warms up the adapter on the year's train+val split
  * runs causal online evaluation on the test split with delayed-label
    adaptation that updates ONLY the SE-LeWM adapter (backbone stays frozen)
  * appends per-year × method × seed × horizon rows to a CSV

Usage:
  python se_lewm_main.py --conf conf/PEMS05/se_lewm_pems05.json \
      --method se_lewm --seed 51 --gpuid 1
"""
from __future__ import annotations

import argparse
import os
import os.path as osp
import sys

sys.path.append("src/")

import numpy as np
import networkx as nx
import torch

from utils.data_convert import generate_samples
from utils.initialize import init, seed_anything, init_log
from utils.common_tools import mkdirs

from src.model.model import (
    TrafficStream_Model, STKEC_Model, EAC_Model, Universal_Model,
    STGNN_Model, DCRNN_Model, ASTGNN_Model, TGCN_Model,
    PECPM_Model, RAP_Model, STTTC_Model,
)
from src.model.se_lewm import SELeWMAdapter
from src.trainer.se_lewm_trainer import (
    warmup_adapter, online_se_lewm_eval, append_results_csv,
)
from dataer.SpatioTemporalDataset import SpatioTemporalDataset
from torch_geometric.loader import DataLoader


METHOD_REGISTRY = {
    'TrafficStream': TrafficStream_Model,
    'STKEC': STKEC_Model,
    'EAC': EAC_Model,
    'Universal': Universal_Model,
    'STGNN': STGNN_Model,
    'DCRNN': DCRNN_Model,
    'ASTGNN': ASTGNN_Model,
    'TGCN': TGCN_Model,
    'PECPM': PECPM_Model,
    'RAP': RAP_Model,
    'STTTC': STTTC_Model,
}


SE_LEWM_METHODS = (
    "backbone",            # frozen base, no adapter at all (sanity)
    "residual_only",       # residual decoder, no latent / sigreg losses
    "latent_only",         # latent prediction loss, no sigreg
    "sigreg_only",         # sigreg only, no latent prediction loss
    "se_lewm",             # full: residual + latent + sigreg
    "se_lewm_no_residual", # latent + sigreg, but emit y_base (no decode-to-y)
    "se_lewm_no_online",   # warmup-only; no online adaptation (delayed-label disabled)
)


# ---------------------------------------------------------------------------
# Backbone loading (identical resolution to a2tta_main)
# ---------------------------------------------------------------------------

def _load_backbone_for_year(args, year: int):
    backbone_method = getattr(args, "backbone_method", "TrafficStream")
    logname_primary = args.backbone_ckpt_logname
    logname_fallback = getattr(args, "backbone_ckpt_logname_fallback", "retrain_st_pems05")

    saved_method = getattr(args, "method", None)
    saved_year = getattr(args, "year", None)
    vars(args)["method"] = backbone_method
    vars(args)["year"] = year

    def _try(logname):
        ckpt_dir = osp.join(args.model_path, f"{logname}-{args.seed}", str(year))
        if not osp.isdir(ckpt_dir):
            return None
        files = [f for f in os.listdir(ckpt_dir) if f.endswith(".pkl")]
        if not files:
            return None
        files = sorted(files, key=lambda f: float(f[:-4]))
        return osp.join(ckpt_dir, files[0])

    ckpt = _try(logname_primary) or _try(logname_fallback)
    if ckpt is None:
        raise FileNotFoundError(
            f"No backbone .pkl for year={year} under "
            f"{logname_primary}-{args.seed}/ or {logname_fallback}-{args.seed}/"
        )

    args.logger.info(f"  [backbone] year={year} loading {ckpt}")
    state = torch.load(ckpt, map_location=args.device)["model_state_dict"]
    model = METHOD_REGISTRY[backbone_method](args).to(args.device)
    model.load_state_dict(state, strict=False)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    vars(args)["method"] = saved_method
    vars(args)["year"] = saved_year
    return model


# ---------------------------------------------------------------------------
# Per-year orchestration
# ---------------------------------------------------------------------------

def _run_year(args, year: int, adapter: SELeWMAdapter):
    graph = nx.from_numpy_array(np.load(osp.join(args.graph_path, f"{year}_adj.npz"))["x"])
    if year == args.begin_year:
        vars(args)["base_node_size"] = graph.number_of_nodes()
        vars(args)["init_graph_size"] = graph.number_of_nodes()
    vars(args)["graph_size"] = graph.number_of_nodes()
    vars(args)["year"] = year

    adj = np.load(osp.join(args.graph_path, f"{year}_adj.npz"))["x"]
    adj = adj / (np.sum(adj, 1, keepdims=True) + 1e-6)
    vars(args)["adj"] = torch.from_numpy(adj).to(torch.float).to(args.device)

    inputs = (
        generate_samples(31, osp.join(args.save_data_path, str(year)),
                         np.load(osp.join(args.raw_data_path, f"{year}.npz"))["x"],
                         graph, val_test_mix=False)
        if args.data_process
        else np.load(osp.join(args.save_data_path, f"{year}.npz"), allow_pickle=True)
    )
    args.logger.info(f"[*] Year {year} loaded")

    backbone = _load_backbone_for_year(args, year)

    adapter.expand_nodes(graph.number_of_nodes())
    adapter = adapter.to(args.device)

    cfg = args.se_lewm

    # Optional warmup (skipped for `backbone`).
    if (
        cfg.get("warmup_epochs", 0) > 0
        and cfg["method"] != "backbone"
    ):
        train_loader = DataLoader(
            SpatioTemporalDataset(inputs, "train"),
            batch_size=args.batch_size, shuffle=True,
            pin_memory=True, num_workers=0,
        )
        val_loader = DataLoader(
            SpatioTemporalDataset(inputs, "val"),
            batch_size=args.batch_size, shuffle=False,
            pin_memory=True, num_workers=0,
        )
        warmup_adapter(
            backbone=backbone, adapter=adapter,
            train_loader=train_loader, val_loader=val_loader,
            args=args, epochs=cfg["warmup_epochs"], lr=cfg.get("warmup_lr", 1e-3),
        )

    # Online evaluation. `se_lewm_no_online` is handled inside the trainer
    # (uses the warmed adapter for prediction but skips online updates).
    online_se_lewm_eval(backbone, adapter, inputs, args)

    del backbone
    torch.cuda.empty_cache()


def main(args):
    args.logger.info("params : %s", vars(args))
    args.result = {
        "3":   {" MAE": {}, "MAPE": {}, "RMSE": {}},
        "6":   {" MAE": {}, "MAPE": {}, "RMSE": {}},
        "12":  {" MAE": {}, "MAPE": {}, "RMSE": {}},
        "Avg": {" MAE": {}, "MAPE": {}, "RMSE": {}},
    }
    mkdirs(args.save_data_path)

    cfg = args.se_lewm
    first_graph = nx.from_numpy_array(np.load(osp.join(args.graph_path, f"{args.begin_year}_adj.npz"))["x"])
    adapter = SELeWMAdapter(
        num_nodes_max=first_graph.number_of_nodes(),
        x_len=args.x_len, y_len=args.y_len,
        z_dim=cfg.get("z_dim", 64),
        hidden_dim=cfg.get("latent_hidden_dim", 128),
        node_emb_dim=cfg.get("node_emb_dim", 16),
        residual_gate_init=cfg.get("residual_gate_init", -3.0),
        target_detach=cfg.get("target_detach", False),
        use_residual=(cfg["method"] != "se_lewm_no_residual"),
        dropout=cfg.get("dropout_adapter", 0.1),
    ).to(args.device)

    vars(args)["graph_size_list"] = []

    for year in range(args.begin_year, args.end_year + 1):
        try:
            _run_year(args, year, adapter)
        except FileNotFoundError as e:
            args.logger.warning(f"[skip] year {year}: {e}")
            continue
        args.graph_size_list.append(args.graph_size)
        if cfg.get("fast_dev_run", False):
            args.logger.info("[fast_dev_run] stopping after first year")
            break

    # Pretty-print final table (mirrors main.py / a2tta_main.py).
    args.logger.info("\n\n")
    for i in ["3", "6", "12", "Avg"]:
        for j in [" MAE", "RMSE", "MAPE"]:
            info, vals = "", []
            for year in range(args.begin_year, args.end_year + 1):
                if year in args.result[i][j]:
                    info += "{:>10.2f}\t".format(args.result[i][j][year])
                    vals.append(args.result[i][j][year])
            if vals:
                args.logger.info("{:<4}\t{}\t".format(i, j) + info + "\t{:>8.2f}".format(np.mean(vals)))

    # CSV.
    csv_path = cfg.get("csv_path", "run_logs/se_lewm_pems05_results.csv")
    csv_path = osp.join(osp.dirname(__file__), csv_path) if not osp.isabs(csv_path) else csv_path
    append_results_csv(args, csv_path)
    args.logger.info(f"[se_lewm] CSV → {csv_path}")


def _build_parser():
    p = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--conf", type=str, default="conf/PEMS05/se_lewm_pems05.json")
    p.add_argument("--seed", type=int, default=51)
    p.add_argument("--gpuid", type=int, default=1)
    p.add_argument("--logname", type=str, default="se_lewm_pems05")
    p.add_argument("--method", type=str, default="se_lewm", choices=list(SE_LEWM_METHODS))
    p.add_argument("--dataset", type=str, default="PEMS05")
    p.add_argument("--checkpoint", type=str, default=None)

    # Backbone resolution
    p.add_argument("--backbone_ckpt_logname", type=str, default="oneline_st_an_pems05")
    p.add_argument("--backbone_ckpt_logname_fallback", type=str, default="retrain_st_pems05")
    p.add_argument("--backbone_method", type=str, default="TrafficStream",
                   choices=list(METHOD_REGISTRY.keys()))
    p.add_argument("--freeze_backbone", type=int, default=1)

    # Adapter dims
    p.add_argument("--z_dim", type=int, default=64)
    p.add_argument("--latent_hidden_dim", type=int, default=128)
    p.add_argument("--node_emb_dim", type=int, default=16)
    p.add_argument("--residual_gate_init", type=float, default=-3.0)
    p.add_argument("--dropout_adapter", type=float, default=0.1)
    p.add_argument("--target_detach", type=int, default=0)

    # Losses
    p.add_argument("--alpha_latent", type=float, default=0.05)
    p.add_argument("--lambda_sigreg", type=float, default=0.01)
    p.add_argument("--lambda_delta", type=float, default=1e-4)
    p.add_argument("--sigreg_warmup_epochs", type=int, default=3)
    p.add_argument("--sigreg_on_context", type=int, default=1)
    p.add_argument("--sigreg_on_target", type=int, default=1)
    p.add_argument("--sigreg_on_pred", type=int, default=0)

    # SIGReg shape
    p.add_argument("--num_projections", type=int, default=128)
    p.add_argument("--num_knots", type=int, default=16)
    p.add_argument("--max_sigreg_samples", type=int, default=4096)

    # Warmup / online adapt
    p.add_argument("--warmup_epochs", type=int, default=3)
    p.add_argument("--warmup_lr", type=float, default=1e-3)
    p.add_argument("--adapt_lr", type=float, default=3e-4)
    p.add_argument("--adapt_steps", type=int, default=1)
    p.add_argument("--adapt_every_batches", type=int, default=1)
    p.add_argument("--candidate_pool_size", type=int, default=512)

    # Eval / output
    p.add_argument("--eval_batch_size", type=int, default=64)
    p.add_argument("--csv_path", type=str, default="run_logs/se_lewm_pems05_results.csv")
    p.add_argument("--fast_dev_run", type=int, default=0)
    return p


def _stash_se_lewm_cfg(args):
    args.se_lewm = {
        "dataset": args.dataset,
        "method": args.method,
        "z_dim": args.z_dim,
        "latent_hidden_dim": args.latent_hidden_dim,
        "node_emb_dim": args.node_emb_dim,
        "residual_gate_init": args.residual_gate_init,
        "dropout_adapter": args.dropout_adapter,
        "target_detach": bool(args.target_detach),

        "alpha_latent": args.alpha_latent,
        "lambda_sigreg": args.lambda_sigreg,
        "lambda_delta": args.lambda_delta,
        "sigreg_warmup_epochs": args.sigreg_warmup_epochs,
        "sigreg_on_context": bool(args.sigreg_on_context),
        "sigreg_on_target": bool(args.sigreg_on_target),
        "sigreg_on_pred": bool(args.sigreg_on_pred),

        "num_projections": args.num_projections,
        "num_knots": args.num_knots,
        "max_sigreg_samples": args.max_sigreg_samples,

        "warmup_epochs": args.warmup_epochs,
        "warmup_lr": args.warmup_lr,
        "adapt_lr": args.adapt_lr,
        "adapt_steps": args.adapt_steps,
        "adapt_every_batches": args.adapt_every_batches,
        "candidate_pool_size": args.candidate_pool_size,

        "eval_batch_size": args.eval_batch_size,
        "csv_path": args.csv_path,
        "fast_dev_run": bool(args.fast_dev_run),
    }


if __name__ == "__main__":
    parser = _build_parser()
    args = parser.parse_args()
    vars(args)["device"] = (
        torch.device(f"cuda:{args.gpuid}")
        if torch.cuda.is_available() and args.gpuid != -1
        else torch.device("cpu")
    )
    vars(args)["methods"] = METHOD_REGISTRY
    init(args)
    seed_anything(args.seed)
    init_log(args)
    _stash_se_lewm_cfg(args)
    main(args)
