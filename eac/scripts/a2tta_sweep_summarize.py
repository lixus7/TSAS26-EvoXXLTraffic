"""
Read the sweep results CSV and print best hyperparameter configurations per
method, ranked by a chosen horizon's MAE.

Usage:
  python scripts/a2tta_sweep_summarize.py run_logs/sweep_a2tta_pems05_<ts>/results.csv
  python scripts/a2tta_sweep_summarize.py results.csv --horizon Avg --top 15

Groups by (method, adapt_lr, adapt_steps, budget_frac, candidate_pool_size,
lambda_cons, hidden_dim). Within each group, averages MAE across years per
seed first, then reports mean ± std across seeds. Configurations with fewer
than --min_seeds are listed as ``partial''.
"""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict


KEY_FIELDS = (
    "method",
    "adapt_lr",
    "adapt_steps",
    "budget_frac",
    "candidate_pool_size",
    "lambda_cons",
    "hidden_dim",
)


def _stats(per_seed_means):
    if not per_seed_means:
        return float("nan"), float("nan")
    n = len(per_seed_means)
    mean = sum(per_seed_means) / n
    if n > 1:
        var = sum((x - mean) ** 2 for x in per_seed_means) / (n - 1)
        return mean, var ** 0.5
    return mean, 0.0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("csv", help="Path to merged sweep results CSV.")
    p.add_argument("--horizon", default="Avg",
                   help="Which horizon column to rank by: 3 / 6 / 12 / Avg.")
    p.add_argument("--top", type=int, default=15,
                   help="Top-K configurations to print per method.")
    p.add_argument("--min_seeds", type=int, default=2,
                   help="Configs with fewer seeds are flagged as partial.")
    p.add_argument("--ref_avg_mae", type=float, default=11.10,
                   help="Online-AN PEMS05 reference for delta column.")
    args = p.parse_args()

    # bucket[cfg_tuple][seed] = list of per-year MAE for the chosen horizon
    bucket = defaultdict(lambda: defaultdict(list))
    with open(args.csv) as f:
        reader = csv.DictReader(f)
        for r in reader:
            if r.get("horizon") != args.horizon:
                continue
            try:
                mae = float(r["MAE"])
            except (ValueError, KeyError, TypeError):
                continue
            cfg = tuple(r.get(k, "") for k in KEY_FIELDS)
            bucket[cfg][r["seed"]].append(mae)

    if not bucket:
        print(f"[!] no rows for horizon={args.horizon!r} in {args.csv}")
        return

    # Reduce to (method, mean, std, n_seeds, cfg)
    records = []
    for cfg, seeds in bucket.items():
        seed_means = [sum(v) / len(v) for v in seeds.values() if v]
        mean, std = _stats(seed_means)
        records.append((cfg[0], mean, std, len(seeds), cfg))

    # Group + sort per method
    by_method = defaultdict(list)
    for method, mean, std, n, cfg in records:
        by_method[method].append((mean, std, n, cfg))

    print(f"\n=== A2TTA-Lite sweep — top configs by horizon={args.horizon} MAE ===")
    print(f"    (mean ± std across seeds; ranked ascending; ref Online-AN={args.ref_avg_mae:.2f})\n")

    # Ordering of methods in the output
    method_order = ["a2tta_lite", "tta_random", "tta_recent", "tta_error",
                    "tta_all", "calibrator", "backbone"]
    methods_present = [m for m in method_order if m in by_method] + \
                      sorted(m for m in by_method if m not in method_order)

    fmt_header = (
        f"{'method':<12} {'lr':>6} {'st':>3} {'bud':>6} {'pool':>5} "
        f"{'lcon':>5} {'hid':>4} {'#sd':>3} {'mean':>8} {'std':>6} {'Δ':>7}"
    )

    # Best per method (single-line summary)
    print(">>> Best single config per method:")
    print(fmt_header)
    print("-" * len(fmt_header))
    overall_best = None
    for m in methods_present:
        items = sorted(by_method[m], key=lambda x: (x[0] if x[2] >= args.min_seeds else 1e9))
        if not items:
            continue
        mean, std, n, cfg = items[0]
        _, lr, st, bud, pool, lcons, hid = cfg
        delta = mean - args.ref_avg_mae
        line = (f"{m:<12} {lr:>6} {st:>3} {bud:>6} {pool:>5} "
                f"{lcons:>5} {hid:>4} {n:>3} {mean:>8.3f} {std:>6.3f} {delta:>+7.3f}")
        print(line)
        if overall_best is None or mean < overall_best[1]:
            overall_best = (m, mean, std, n, cfg)
    print()

    # Top-K full table per method
    print(f">>> Top-{args.top} configs per method (full):")
    for m in methods_present:
        items = sorted(by_method[m], key=lambda x: (x[0] if x[2] >= args.min_seeds else 1e9))
        if not items:
            continue
        print(f"\n--- {m} ---")
        print(fmt_header)
        print("-" * len(fmt_header))
        for mean, std, n, cfg in items[: args.top]:
            _, lr, st, bud, pool, lcons, hid = cfg
            delta = mean - args.ref_avg_mae
            partial = "" if n >= args.min_seeds else "  *partial"
            line = (f"{m:<12} {lr:>6} {st:>3} {bud:>6} {pool:>5} "
                    f"{lcons:>5} {hid:>4} {n:>3} {mean:>8.3f} {std:>6.3f} {delta:>+7.3f}{partial}")
            print(line)

    # Compare a2tta_lite vs tta_random at every shared (lr, steps, bud, pool, lcons, hid)
    if "a2tta_lite" in by_method and "tta_random" in by_method:
        print("\n>>> a2tta_lite − tta_random delta at matching configs:")
        a2 = {cfg[1:]: (mean, std, n) for mean, std, n, cfg in by_method["a2tta_lite"]}
        tr = {cfg[1:]: (mean, std, n) for mean, std, n, cfg in by_method["tta_random"]}
        shared = sorted(set(a2) & set(tr), key=lambda c: a2[c][0] - tr[c][0])
        if not shared:
            print("    (no shared configs)")
        else:
            print(f"{'lr':>6} {'st':>3} {'bud':>6} {'pool':>5} {'lcon':>5} {'hid':>4}"
                  f"   {'a2tta':>8} {'random':>8} {'Δ':>8}")
            for cfg_tail in shared:
                lr, st, bud, pool, lcons, hid = cfg_tail
                am, _, _ = a2[cfg_tail]
                rm, _, _ = tr[cfg_tail]
                d = am - rm
                marker = " ← a2tta wins" if d < -0.005 else (" ← random wins" if d > 0.005 else "")
                print(f"{lr:>6} {st:>3} {bud:>6} {pool:>5} {lcons:>5} {hid:>4}"
                      f"   {am:>8.3f} {rm:>8.3f} {d:>+8.3f}{marker}")

    if overall_best:
        m, mean, std, n, cfg = overall_best
        _, lr, st, bud, pool, lcons, hid = cfg
        print(f"\n>>> Global best: {m}  Avg MAE={mean:.3f}±{std:.3f} (n={n})")
        print(f"    --method {m} --adapt_lr {lr} --adapt_steps {st} --budget_frac {bud} "
              f"--candidate_pool_size {pool} --lambda_cons {lcons} --adapter_hidden_dim {hid}")


if __name__ == "__main__":
    main()
