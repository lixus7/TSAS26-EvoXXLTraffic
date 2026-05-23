"""SE-LeWM trainer & online evaluator.

Mirrors the A2TTA-Lite trainer's pipeline:
  1. Frozen backbone per year (loaded outside this module).
  2. Optional warmup of the SE-LeWM adapter on train+val split.
  3. Causal online evaluation on the year's test split. Delayed-label
     adaptation: a window is only used for adaptation after its target
     horizon has elapsed.
  4. Per-horizon metrics via `cal_metric` (identical to other baselines).
"""
from __future__ import annotations

import os
import csv
import math
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch_geometric.loader import DataLoader
from torch_geometric.utils import to_dense_batch

from dataer.SpatioTemporalDataset import SpatioTemporalDataset
from utils.metric import cal_metric, masked_mae_np
from utils.sigreg import sigreg_loss

from src.model.se_lewm import SELeWMAdapter


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _to_dense(pred_flat, y_flat, batch):
    pred, _ = to_dense_batch(pred_flat, batch=batch)
    y, _ = to_dense_batch(y_flat, batch=batch)
    return pred, y


def _chronological_test_arrays(inputs):
    test_x = inputs["test_x"][::-1].copy()
    test_y = inputs["test_y"][::-1].copy()
    return test_x, test_y


def _masked_mae(pred: torch.Tensor, target: torch.Tensor, null_val: float = 0.0) -> torch.Tensor:
    """Differentiable masked MAE consistent with `masked_mae_np` (null_val=0)."""
    if math.isnan(null_val):
        mask = (~torch.isnan(target)).float()
    else:
        mask = (target != null_val).float()
    mask = torch.nan_to_num(mask, 0.0)
    denom = mask.mean().clamp_min(1e-6)
    mask = mask / denom
    err = (pred - target).abs()
    err = torch.nan_to_num(err * mask, 0.0)
    return err.mean()


# ---------------------------------------------------------------------------
# Loss assembly
# ---------------------------------------------------------------------------

def _compute_se_lewm_losses(out, y_target_flat, cfg, method_mode, N, H, sigreg_active=True):
    """Aggregate forecast / latent / sigreg / delta-reg into a single scalar."""
    pred = out["prediction"]              # [B*N, H]
    delta = out["delta"]                  # [B*N, H]
    latent_loss = out["loss_components"]["latent"]

    # Forecast loss.
    forecast_loss = _masked_mae(pred, y_target_flat, null_val=0.0)

    # SIGReg: on z_ctx and/or z_tgt_seq (and optionally z_pred_seq).
    sigreg_l = pred.new_zeros(())
    if sigreg_active and method_mode != "residual_only":
        terms = []
        if cfg.get("sigreg_on_context", True) and out.get("z_ctx") is not None:
            terms.append(sigreg_loss(
                out["z_ctx"],
                num_projections=cfg.get("num_projections", 128),
                num_knots=cfg.get("num_knots", 16),
                max_samples=cfg.get("max_sigreg_samples", 4096),
            ))
        if cfg.get("sigreg_on_target", True) and out.get("z_tgt_seq") is not None:
            terms.append(sigreg_loss(
                out["z_tgt_seq"],
                num_projections=cfg.get("num_projections", 128),
                num_knots=cfg.get("num_knots", 16),
                max_samples=cfg.get("max_sigreg_samples", 4096),
            ))
        if cfg.get("sigreg_on_pred", False) and out.get("z_pred_seq") is not None:
            terms.append(sigreg_loss(
                out["z_pred_seq"],
                num_projections=cfg.get("num_projections", 128),
                num_knots=cfg.get("num_knots", 16),
                max_samples=cfg.get("max_sigreg_samples", 4096),
            ))
        if terms:
            sigreg_l = sum(terms)

    # Residual reg.
    if cfg.get("lambda_delta", 0.0) > 0:
        delta_reg = (delta * delta).mean()
    else:
        delta_reg = pred.new_zeros(())

    # Method-specific gating of loss terms.
    if method_mode == "backbone":
        # Backbone-only: no adapter learning at all. Return None to signal
        # the caller to skip the optimizer step entirely.
        return None
    if method_mode == "residual_only":
        total = forecast_loss + cfg["lambda_delta"] * delta_reg
        latent_for_log = pred.new_zeros(())
        sigreg_for_log = pred.new_zeros(())
    elif method_mode == "latent_only":
        total = (
            forecast_loss
            + cfg["alpha_latent"] * latent_loss
            + cfg["lambda_delta"] * delta_reg
        )
        latent_for_log = latent_loss
        sigreg_for_log = pred.new_zeros(())
    elif method_mode == "sigreg_only":
        total = (
            forecast_loss
            + cfg["lambda_sigreg"] * sigreg_l
            + cfg["lambda_delta"] * delta_reg
        )
        latent_for_log = pred.new_zeros(())
        sigreg_for_log = sigreg_l
    elif method_mode == "se_lewm_no_residual":
        total = (
            cfg["alpha_latent"] * latent_loss
            + cfg["lambda_sigreg"] * sigreg_l
        )
        latent_for_log = latent_loss
        sigreg_for_log = sigreg_l
    else:  # "se_lewm" full, "se_lewm_no_online" (same losses, just no online adapt)
        total = (
            forecast_loss
            + cfg["alpha_latent"] * latent_loss
            + cfg["lambda_sigreg"] * sigreg_l
            + cfg["lambda_delta"] * delta_reg
        )
        latent_for_log = latent_loss
        sigreg_for_log = sigreg_l

    return {
        "total": total,
        "forecast": forecast_loss.detach(),
        "latent": latent_for_log.detach() if torch.is_tensor(latent_for_log) else latent_for_log,
        "sigreg": sigreg_for_log.detach() if torch.is_tensor(sigreg_for_log) else sigreg_for_log,
        "delta_reg": delta_reg.detach(),
    }


# ---------------------------------------------------------------------------
# Warmup
# ---------------------------------------------------------------------------

def warmup_adapter(
    backbone: nn.Module,
    adapter: SELeWMAdapter,
    train_loader: DataLoader,
    val_loader: DataLoader,
    args,
    epochs: int = 3,
    lr: float = 1e-3,
):
    """Train SE-LeWM adapter on train split, backbone frozen."""
    cfg = args.se_lewm
    method_mode = cfg["method"]
    if method_mode == "backbone":
        return float("inf")  # nothing to warm up

    backbone.eval()
    for p in backbone.parameters():
        p.requires_grad = False

    opt = optim.AdamW(
        filter(lambda p: p.requires_grad, adapter.parameters()), lr=lr
    )
    N = args.adj.shape[0]
    H = args.y_len
    sigreg_warmup = int(cfg.get("sigreg_warmup_epochs", 3))

    best_val = float("inf")
    for ep in range(epochs):
        sigreg_active = (ep >= sigreg_warmup - 1) or sigreg_warmup <= 0
        adapter.train()
        tot, fc, lt, sg, cnt = 0.0, 0.0, 0.0, 0.0, 0
        for data in train_loader:
            data = data.to(args.device, non_blocking=True)
            with torch.no_grad():
                y_base = backbone(data, args.adj)               # [B*N, H]
            out = adapter(
                x=data.x, y_base=y_base, adj=args.adj,
                y=data.y, mode="train", N=N,
            )
            losses = _compute_se_lewm_losses(out, data.y, cfg, method_mode, N, H, sigreg_active)
            if losses is None:
                continue
            loss = losses["total"]
            if not torch.isfinite(loss):
                continue
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(adapter.parameters(), 5.0)
            opt.step()
            tot += float(loss); fc += float(losses["forecast"])
            lt += float(losses["latent"]); sg += float(losses["sigreg"])
            cnt += 1

        # Validation: forecast-only MAE on val split (raw scale, like A2TTA).
        adapter.eval()
        v_total, v_cnt = 0.0, 0
        with torch.no_grad():
            for data in val_loader:
                data = data.to(args.device, non_blocking=True)
                y_base = backbone(data, args.adj)
                out = adapter(
                    x=data.x, y_base=y_base, adj=args.adj,
                    y=None, mode="eval", N=N,
                )
                vloss = masked_mae_np(data.y.cpu().numpy(), out["prediction"].cpu().numpy(), 0)
                if not np.isfinite(vloss):
                    continue
                v_total += float(vloss); v_cnt += 1
        v_mae = v_total / max(v_cnt, 1)
        if hasattr(args, "logger"):
            args.logger.info(
                f"  [warmup] year={args.year} ep={ep} "
                f"loss={tot/max(cnt,1):.4f} fc={fc/max(cnt,1):.4f} "
                f"lat={lt/max(cnt,1):.4f} sig={sg/max(cnt,1):.4f} "
                f"val_mae={v_mae:.4f} gate={float(torch.sigmoid(adapter.gate)):.4f}"
            )
        if v_mae < best_val:
            best_val = v_mae
    return best_val


# ---------------------------------------------------------------------------
# Online causal evaluation
# ---------------------------------------------------------------------------

class _Sample:
    __slots__ = ("idx", "x_flat", "y_flat", "release_idx")

    def __init__(self, idx, x_flat, y_flat, release_idx):
        self.idx = idx
        self.x_flat = x_flat
        self.y_flat = y_flat
        self.release_idx = release_idx


def online_se_lewm_eval(
    backbone: nn.Module,
    adapter: SELeWMAdapter,
    inputs,
    args,
):
    """Causal online eval on the year's test split."""
    backbone.eval()
    for p in backbone.parameters():
        p.requires_grad = False

    N = args.adj.shape[0]
    H = args.y_len
    device = args.device

    cfg = args.se_lewm
    method_mode: str = cfg["method"]
    adapt_lr: float = cfg["adapt_lr"]
    adapt_steps: int = cfg["adapt_steps"]
    adapt_every: int = cfg["adapt_every_batches"]
    fast_dev_run: bool = bool(cfg.get("fast_dev_run", False))
    bs: int = int(cfg.get("eval_batch_size", 64))

    do_adapt = method_mode in (
        "residual_only", "latent_only", "sigreg_only",
        "se_lewm", "se_lewm_no_residual",
    )
    use_adapter_pred = method_mode in (
        "residual_only", "latent_only", "sigreg_only",
        "se_lewm", "se_lewm_no_online",
    )
    # "se_lewm_no_residual" learns latent losses but emits y_base.
    # "se_lewm_no_online" uses the warmed adapter but skips online updates.

    opt = optim.AdamW(adapter.parameters(), lr=adapt_lr) if do_adapt else None

    test_x, test_y = _chronological_test_arrays(inputs)
    chrono_inputs = {"test_x": test_x, "test_y": test_y}
    test_loader = DataLoader(
        SpatioTemporalDataset(chrono_inputs, "test"),
        batch_size=bs, shuffle=False, pin_memory=True, num_workers=0,
    )

    pending = deque()
    pool: deque = deque(maxlen=int(cfg.get("candidate_pool_size", 512)))

    pred_chrono, truth_chrono = [], []
    next_idx = 0
    batch_idx = 0
    n_adapt_steps_total = 0
    last_latent = float("nan")
    last_sigreg = float("nan")

    for batch in test_loader:
        if fast_dev_run and batch_idx >= 4:
            break
        batch = batch.to(device, non_blocking=True)
        B = batch.x.shape[0] // N
        x_flat = batch.x
        y_flat = batch.y

        # Release pending samples whose horizon has elapsed.
        while pending and pending[0].release_idx <= next_idx:
            pool.append(pending.popleft())

        # Adapt on pool (delayed-label, post-release only).
        if (
            do_adapt and opt is not None
            and len(pool) >= max(4, int(0.1 * pool.maxlen))
            and (batch_idx % adapt_every == 0)
        ):
            ps = list(pool)
            xs = torch.cat([s.x_flat for s in ps], dim=0)
            ys = torch.cat([s.y_flat for s in ps], dim=0)
            # Stack of per-sample chunks; each is exactly N rows.
            P = len(ps)
            for _ in range(adapt_steps):
                adapter.train()
                with torch.no_grad():
                    # Re-run backbone on pool windows (cheap, frozen).
                    # Use Data-less call: backbone expects `data.x`. Build a
                    # transient namespace object compatible with backbone fwd.
                    yb = _backbone_forward_on_flat(backbone, xs, args)
                out = adapter(
                    x=xs, y_base=yb, adj=args.adj,
                    y=ys, mode="train", N=N,
                )
                losses = _compute_se_lewm_losses(out, ys, cfg, method_mode, N, H, sigreg_active=True)
                if losses is None:
                    break
                loss = losses["total"]
                if not torch.isfinite(loss):
                    continue
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(adapter.parameters(), 5.0)
                opt.step()
                n_adapt_steps_total += 1
                last_latent = float(losses["latent"]) if torch.is_tensor(losses["latent"]) else last_latent
                last_sigreg = float(losses["sigreg"]) if torch.is_tensor(losses["sigreg"]) else last_sigreg

        # Predict current batch.
        adapter.eval()
        with torch.no_grad():
            y_base = backbone(batch, args.adj)
            if use_adapter_pred:
                out = adapter(
                    x=x_flat, y_base=y_base, adj=args.adj,
                    y=None, mode="eval", N=N,
                )
                y_pred_flat = out["prediction"]
            else:
                y_pred_flat = y_base

        pred_dense, y_dense = _to_dense(y_pred_flat, y_flat, batch.batch)
        pred_chrono.append(pred_dense.detach().cpu().numpy())
        truth_chrono.append(y_dense.detach().cpu().numpy())

        # Enqueue this batch's samples for future delayed adaptation.
        for j in range(B):
            i_global = next_idx + j
            x_j = x_flat[j * N : (j + 1) * N].detach()
            y_j = y_flat[j * N : (j + 1) * N].detach()
            pending.append(_Sample(
                idx=i_global, x_flat=x_j, y_flat=y_j,
                release_idx=i_global + H,
            ))
        next_idx += B
        batch_idx += 1

    pred_arr = np.concatenate(pred_chrono, axis=0)
    truth_arr = np.concatenate(truth_chrono, axis=0)
    cal_metric(truth_arr, pred_arr, args)

    if hasattr(args, "logger"):
        gate_v = float(torch.sigmoid(adapter.gate))
        args.logger.info(
            f"  [se_lewm] year={args.year} method={method_mode} "
            f"adapt_steps={n_adapt_steps_total} gate={gate_v:.4f} "
            f"latent={last_latent:.4f} sigreg={last_sigreg:.4f} "
            f"pool_size_final={len(pool)} pending_final={len(pending)}"
        )
    # Record summary stats on args for CSV writer.
    args.se_lewm_runtime = {
        "gate": float(torch.sigmoid(adapter.gate)),
        "latent_loss": last_latent,
        "sigreg_loss": last_sigreg,
        "adapt_steps": n_adapt_steps_total,
    }
    return n_adapt_steps_total


def _backbone_forward_on_flat(backbone: nn.Module, x_flat: torch.Tensor, args) -> torch.Tensor:
    """Call backbone(data, adj) when we only have a flat [B*N, T] tensor.

    Backbone forwards consume `data.x` only (see TrafficStream_Model.forward),
    so we wrap with a minimal namespace.
    """
    class _D:  # tiny stand-in for `Data` — backbone only reads `.x`.
        pass
    d = _D()
    d.x = x_flat
    return backbone(d, args.adj)


# ---------------------------------------------------------------------------
# CSV result writer
# ---------------------------------------------------------------------------

CSV_COLUMNS = [
    "dataset", "method", "seed", "year", "horizon",
    "MAE", "RMSE", "MAPE",
    "gate", "latent_loss", "sigreg_loss",
    "alpha_latent", "lambda_sigreg", "lambda_delta",
    "z_dim", "latent_hidden_dim",
]


def append_results_csv(args, csv_path: str):
    os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
    write_header = not os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        if write_header:
            w.writeheader()
        cfg = args.se_lewm
        rt = getattr(args, "se_lewm_runtime", {}) or {}
        for h in ["3", "6", "12", "Avg"]:
            res = args.result.get(h, {})
            mae_d = res.get(" MAE", {})
            rmse_d = res.get("RMSE", {})
            mape_d = res.get("MAPE", {})
            for year in sorted(set(mae_d.keys()) | set(rmse_d.keys()) | set(mape_d.keys())):
                w.writerow({
                    "dataset": cfg.get("dataset", "PEMS05"),
                    "method": cfg.get("method", "?"),
                    "seed": args.seed,
                    "year": year,
                    "horizon": h,
                    "MAE": mae_d.get(year, ""),
                    "RMSE": rmse_d.get(year, ""),
                    "MAPE": mape_d.get(year, ""),
                    "gate": rt.get("gate", ""),
                    "latent_loss": rt.get("latent_loss", ""),
                    "sigreg_loss": rt.get("sigreg_loss", ""),
                    "alpha_latent": cfg.get("alpha_latent", ""),
                    "lambda_sigreg": cfg.get("lambda_sigreg", ""),
                    "lambda_delta": cfg.get("lambda_delta", ""),
                    "z_dim": cfg.get("z_dim", ""),
                    "latent_hidden_dim": cfg.get("latent_hidden_dim", ""),
                })
