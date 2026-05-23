#!/usr/bin/env python3
"""
evaluate_new_sensors.py

Re-run inference on existing checkpoints and report per-year MAE / RMSE / MAPE
restricted to the **newly added sensors** (node indices [prev_year_N, cur_year_N)).
Years where the graph did not grow are skipped.

The full-graph metrics are also reported alongside, so you can sanity-check
that the recomputed all-sensor numbers match the original training-time log.

Usage (run from the eac/ directory):
  python scripts/evaluate_new_sensors.py \
      --conf conf/PEMS05/eac.json --seed 51 --gpuid 0
"""
import argparse
import csv
import os
import os.path as osp
import sys

HERE = osp.dirname(osp.abspath(__file__))
ROOT = osp.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, osp.join(ROOT, "src"))

import numpy as np
import torch
from torch_geometric.loader import DataLoader
from torch_geometric.utils import to_dense_batch

from utils.common_tools import load_json_file, mkdirs
from utils.metric import masked_mae_np, masked_mse_np, masked_mape_np
from src.model.model import (
    TrafficStream_Model, STKEC_Model, EAC_Model, Universal_Model,
    STGNN_Model, DCRNN_Model, ASTGNN_Model, TGCN_Model,
    PECPM_Model, RAP_Model, STTTC_Model,
)
from dataer.SpatioTemporalDataset import SpatioTemporalDataset


METHODS = {
    "TrafficStream": TrafficStream_Model,
    "STKEC": STKEC_Model,
    "EAC": EAC_Model,
    "Universal": Universal_Model,
    "STGNN": STGNN_Model,
    "DCRNN": DCRNN_Model,
    "ASTGNN": ASTGNN_Model,
    "TGCN": TGCN_Model,
    "PECPM": PECPM_Model,
    "RAP": RAP_Model,
    "STTTC": STTTC_Model,
}


class _Args:
    """Mutable bag-of-attributes; mirrors argparse.Namespace + JSON merge."""
    pass


def _to_args(conf, seed, gpuid):
    a = _Args()
    for k, v in conf.items():
        setattr(a, k, v)
    a.seed = seed
    a.gpuid = gpuid
    a.device = (
        torch.device("cuda:{}".format(gpuid))
        if torch.cuda.is_available() and gpuid != -1
        else torch.device("cpu")
    )
    # main.py argparse defaults that some models read via getattr
    if not hasattr(a, "method"):
        # stkec.json / trafficstream.json don't carry a "method" field — they
        # are dispatched via stkec_main.py / main.py argparse defaults at run
        # time. Reconstruct the right model class from logname here.
        if getattr(a, "logname", "") == "stkec":
            a.method = "STKEC"
        else:
            a.method = "TrafficStream"
    if not hasattr(a, "backbone_type"):
        a.backbone_type = "stgnn"
    return a


def _instantiate(args):
    cls = METHODS.get(args.method, TrafficStream_Model)
    return cls(args)


def _slice_metrics(truth, pred, sl):
    """Mirror utils.metric.cal_metric: MAE/RMSE/MAPE for horizons 3, 6, 12, Avg."""
    g = truth[:, sl, :]
    p = pred[:, sl, :]
    if g.size == 0 or g.shape[1] == 0:
        return None
    mae_list, rmse_list, mape_list = [], [], []
    for i in range(1, 13):
        mae_list.append(masked_mae_np(g[:, :, :i], p[:, :, :i], 0))
        rmse_list.append(masked_mse_np(g[:, :, :i], p[:, :, :i], 0) ** 0.5)
        mape_list.append(masked_mape_np(g[:, :, :i], p[:, :, :i], 0))
    return {
        "3":   (float(mae_list[2]),  float(rmse_list[2]),  float(mape_list[2])),
        "6":   (float(mae_list[5]),  float(rmse_list[5]),  float(mape_list[5])),
        "12":  (float(mae_list[11]), float(rmse_list[11]), float(mape_list[11])),
        "Avg": (float(np.mean(mae_list)), float(np.mean(rmse_list)), float(np.mean(mape_list))),
    }


def _build_model(args, year, graph_size_list):
    """Recreate the per-year model the same way utils.common_tools.load_test_best_model does:
    instantiate, run the EAC/Universal expansions in the right order, then load the
    alphabetically-first .pkl from log/<DS>/<logname>-<seed>/<year>/.
    """
    seed_dir = osp.join(args.model_path, "{}-{}".format(args.logname, args.seed))
    year_dir = osp.join(seed_dir, str(year))
    if not osp.isdir(year_dir):
        return None, None
    pkls = sorted([f for f in os.listdir(year_dir) if f.endswith(".pkl")])
    if not pkls:
        return None, None
    load_path = osp.join(year_dir, pkls[0])

    state = torch.load(load_path, map_location=args.device)["model_state_dict"]

    # STKEC's `memory` may have been capped if cluster > #nodes-of-first-year;
    # honor whatever shape was actually saved so load_state_dict succeeds.
    if args.method == "STKEC" and "memory" in state:
        args.cluster = state["memory"].shape[0]

    model = _instantiate(args)

    if args.method == "EAC":
        if year == args.begin_year:
            model.expand_adaptive_params(args.base_node_size)
        else:
            for idx in range(year - args.begin_year):
                model.expand_adaptive_params(graph_size_list[idx + 1])
    if args.method == "Universal" and getattr(args, "use_eac", False):
        if year == args.begin_year:
            model.expand_adaptive_params(args.base_node_size)
        else:
            for idx in range(year - args.begin_year):
                model.expand_adaptive_params(graph_size_list[idx])

    model.load_state_dict(state)
    model = model.to(args.device).eval()
    return model, load_path


def evaluate(conf_path, seed, gpuid, out_log=None, out_csv=None,
             num_workers=4, batch_size_override=None):
    conf = load_json_file(conf_path)
    args = _to_args(conf, seed, gpuid)

    seed_dir = osp.join(args.model_path, "{}-{}".format(args.logname, args.seed))
    if not osp.isdir(seed_dir):
        print("[error] seed dir not found: {}".format(seed_dir), file=sys.stderr)
        sys.exit(1)

    if out_log is None:
        out_log = osp.join(seed_dir, "eval_new_sensors.log")
    if out_csv is None:
        out_csv = osp.join(seed_dir, "eval_new_sensors.csv")
    mkdirs(seed_dir)
    log_f = open(out_log, "w")

    def log(msg):
        print(msg)
        log_f.write(msg + "\n")
        log_f.flush()

    log("[*] conf={} seed={} gpu={} dest={}".format(conf_path, seed, gpuid, seed_dir))
    log("[*] method={} logname={} years={}-{}".format(
        args.method, args.logname, args.begin_year, args.end_year))

    bs = batch_size_override or getattr(args, "batch_size", 64)

    rows = []
    new_metrics = {}  # year -> dict
    all_metrics = {}  # year -> dict
    new_counts = {}   # year -> int

    graph_size_list = []
    prev_size = None

    for year in range(args.begin_year, args.end_year + 1):
        adj_path = osp.join(args.graph_path, "{}_adj.npz".format(year))
        if not osp.exists(adj_path):
            log("  [year {}] missing {} — skip".format(year, adj_path))
            continue
        adj = np.load(adj_path)["x"]
        n = adj.shape[0]
        graph_size_list.append(n)

        if year == args.begin_year:
            args.base_node_size = n
            args.init_graph_size = n
            prev_size = n
            log("  [year {}] begin-year, N={} (no prior year — skipping)".format(year, n))
            continue

        new_count = n - prev_size
        if new_count <= 0:
            log("  [year {}] N={} (no newly added sensors — skipping)".format(year, n))
            prev_size = n
            continue

        data_path = osp.join(args.save_data_path, "{}.npz".format(year))
        if not osp.exists(data_path):
            log("  [year {}] missing test data {} — skip".format(year, data_path))
            prev_size = n
            continue

        # Per-year scalar args read by some models / load_best_model paths
        norm_adj = adj / (np.sum(adj, 1, keepdims=True) + 1e-6)
        args.adj = torch.from_numpy(norm_adj).to(torch.float).to(args.device)
        args.graph_size = n
        args.year = year

        try:
            model, ckpt = _build_model(args, year, graph_size_list)
        except Exception as e:
            log("  [year {}] model build/load FAILED: {}".format(year, e))
            prev_size = n
            continue
        if model is None:
            log("  [year {}] no checkpoint, skipping".format(year))
            prev_size = n
            continue

        inputs = np.load(data_path, allow_pickle=True)
        loader = DataLoader(
            SpatioTemporalDataset(inputs, "test"),
            batch_size=bs, shuffle=False, pin_memory=True, num_workers=num_workers,
        )

        pred_chunks, truth_chunks = [], []
        with torch.no_grad():
            for data in loader:
                data = data.to(args.device, non_blocking=True)
                out = model(data, args.adj)
                if isinstance(out, tuple):  # STKEC returns (pred, scores)
                    out = out[0]
                pred_dense, _ = to_dense_batch(out, batch=data.batch)
                y_dense, _ = to_dense_batch(data.y, batch=data.batch)
                pred_chunks.append(pred_dense.cpu().numpy())
                truth_chunks.append(y_dense.cpu().numpy())

        truth = np.concatenate(truth_chunks, 0)
        pred = np.concatenate(pred_chunks, 0)
        # truth/pred shape: [B, N_year, T]
        if truth.shape[1] != n:
            log("  [year {}] WARN: dense N={} != adj N={}; clamping prev_size".format(
                year, truth.shape[1], n))
            n_eff = truth.shape[1]
            new_slice = slice(min(prev_size, n_eff), n_eff)
            new_count = max(0, n_eff - min(prev_size, n_eff))
            all_slice = slice(0, n_eff)
        else:
            new_slice = slice(prev_size, n)
            all_slice = slice(0, n)

        if new_count <= 0:
            log("  [year {}] dense slice empty after clamp — skip".format(year))
            prev_size = n
            del model
            continue

        nm = _slice_metrics(truth, pred, new_slice)
        am = _slice_metrics(truth, pred, all_slice)
        new_metrics[year] = nm
        all_metrics[year] = am
        new_counts[year] = new_count

        log("  [year {}] N={} new={} (idx {}..{})  ckpt={}".format(
            year, n, new_count, new_slice.start, new_slice.stop - 1, osp.basename(ckpt)))
        for k in ("3", "6", "12", "Avg"):
            log("    h={:>3} new   MAE={:>7.4f}  RMSE={:>7.4f}  MAPE={:>7.4f}".format(k, *nm[k]))
            log("    h={:>3} all   MAE={:>7.4f}  RMSE={:>7.4f}  MAPE={:>7.4f}".format(k, *am[k]))

        for k in ("3", "6", "12", "Avg"):
            rows.append([year, n, new_count, "new", k, *nm[k]])
            rows.append([year, n, new_count, "all", k, *am[k]])

        prev_size = n
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Aggregate (mean over years that actually had new sensors)
    log("\n========== Newly added sensors — aggregate (mean over years with growth) ==========")
    log("years evaluated: {}".format(sorted(new_metrics.keys())))
    if new_metrics:
        log("h     MAE       RMSE      MAPE")
        for k in ("3", "6", "12", "Avg"):
            ms = [new_metrics[y][k][0] for y in new_metrics]
            rs = [new_metrics[y][k][1] for y in new_metrics]
            ps = [new_metrics[y][k][2] for y in new_metrics]
            log("{:>3}   {:>7.4f}  {:>7.4f}  {:>7.4f}".format(
                k, float(np.mean(ms)), float(np.mean(rs)), float(np.mean(ps))))
    else:
        log("  (no year had newly added sensors with available data)")

    log("\n========== All sensors — aggregate (recomputed for sanity) ==========")
    if all_metrics:
        log("h     MAE       RMSE      MAPE")
        for k in ("3", "6", "12", "Avg"):
            ms = [all_metrics[y][k][0] for y in all_metrics]
            rs = [all_metrics[y][k][1] for y in all_metrics]
            ps = [all_metrics[y][k][2] for y in all_metrics]
            log("{:>3}   {:>7.4f}  {:>7.4f}  {:>7.4f}".format(
                k, float(np.mean(ms)), float(np.mean(rs)), float(np.mean(ps))))

    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["year", "n_total", "n_new", "scope", "horizon", "MAE", "RMSE", "MAPE"])
        w.writerows(rows)
    log("[*] wrote {}".format(out_csv))
    log_f.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--conf", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--gpuid", type=int, default=0)
    parser.add_argument("--out_log", default=None)
    parser.add_argument("--out_csv", default=None)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=None,
                        help="override conf batch_size for inference")
    a = parser.parse_args()
    evaluate(a.conf, a.seed, a.gpuid, a.out_log, a.out_csv,
             num_workers=a.num_workers, batch_size_override=a.batch_size)
