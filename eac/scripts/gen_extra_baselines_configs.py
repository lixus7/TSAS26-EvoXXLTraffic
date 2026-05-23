#!/usr/bin/env python3
"""Generate retrain configs for the STBP-paper conventional baselines + DLinear.

For each (dataset, method) pair we write
    conf/<DATASET>/retrain_<method>_<dataset>.json

where method ∈ {gwn, stid, itransformer, dlinear}. The first-year and paths are
copied from each dataset's existing retrain_st_<dataset>.json so begin_year /
end_year / raw_data_path / save_data_path / graph_path / model_path stay in
sync. Only the `method` and `logname` fields differ across the four backbones.

Run from `eac/`:
    python scripts/gen_extra_baselines_configs.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATASETS = ["PEMS03", "PEMS04", "PEMS05", "PEMS06", "PEMS07", "PEMS08", "PEMS10", "PEMS11", "PEMS12"]
METHOD_KEY = {
    # CLI/json `method` value -> filename slug
    "GWN": "gwn",
    "STID": "stid",
    "ITRANSFORMER": "itransformer",
    "DLINEAR": "dlinear",
}


def main() -> None:
    if not (ROOT / "conf").is_dir():
        sys.exit(f"Expected to find {ROOT / 'conf'}; run this from the repo root or eac/.")
    written = 0
    for ds in DATASETS:
        low = ds.lower()
        base_path = ROOT / "conf" / ds / f"retrain_st_{low}.json"
        if not base_path.is_file():
            print(f"  [skip] {ds}: missing template {base_path}")
            continue
        with open(base_path) as f:
            base = json.load(f)
        for method_name, slug in METHOD_KEY.items():
            cfg = dict(base)
            cfg["logname"] = f"retrain_{slug}_{low}"
            cfg["method"] = method_name
            # Each backbone uses the same hidden_channel / x_len / y_len as the
            # existing STGNN retrain template; no per-method tuning here.
            out_path = ROOT / "conf" / ds / f"retrain_{slug}_{low}.json"
            with open(out_path, "w") as f:
                json.dump(cfg, f, indent=4)
            written += 1
            print(f"  [ok] {out_path.relative_to(ROOT)}")
    print(f"Wrote {written} config files")


if __name__ == "__main__":
    main()
