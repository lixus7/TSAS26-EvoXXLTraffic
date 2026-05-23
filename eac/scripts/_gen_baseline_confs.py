#!/usr/bin/env python3
"""
Generate baseline JSON configs for PEMS03..PEMS12 by cloning each dataset's
existing `retrain_st_pems<XX>.json` and only overriding the method-specific
fields. Run once from eac/:

    python scripts/_gen_baseline_confs.py

Produces per-dataset configs:
    retrain_stgnn_<ds>.json   (method=STGNN)
    retrain_dcrnn_<ds>.json   (method=DCRNN)
    retrain_astgnn_<ds>.json  (method=ASTGNN)
    retrain_tgcn_<ds>.json    (method=TGCN)
    pecpm_<ds>.json           (method=PECPM, backbone_type=stgnn)
    strap_<ds>.json           (method=RAP,  backbone_type=stgnn)

STKEC already has its own per-dataset config (stkec.json); untouched.
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]                  # eac/
CONF = ROOT / "conf"
DATASETS = ["PEMS03", "PEMS04", "PEMS05", "PEMS06", "PEMS07",
            "PEMS08", "PEMS10", "PEMS11", "PEMS12"]


def _load_base(ds: str) -> dict:
    low = ds.lower()
    src = CONF / ds / f"retrain_st_{low}.json"
    return json.loads(src.read_text())


def _write(ds: str, name: str, data: dict) -> None:
    out = CONF / ds / f"{name}.json"
    out.write_text(json.dumps(data, indent=4))
    print(f"  [gen] {out.relative_to(ROOT)}")


def _mk_retrain(base: dict, ds: str, short: str, method: str) -> dict:
    low = ds.lower()
    d = dict(base)
    d["method"] = method
    d["logname"] = f"retrain_{short}_{low}"
    # retrain configs keep strategy=retrain, train=1, no init
    d["init"] = False
    d["train"] = 1
    d["auto_test"] = 0
    d["strategy"] = "retrain"
    d["detect"] = False
    d["ewc"] = False
    d["replay"] = False
    return d


def _mk_pecpm(base: dict, ds: str) -> dict:
    low = ds.lower()
    d = dict(base)
    d["method"] = "PECPM"
    d["backbone_type"] = "stgnn"
    d["logname"] = f"pecpm_{low}"
    # PECPM in STRAP repo uses strategy=retrain (its baseline mode) — keep simple
    d["init"] = True
    d["train"] = 1
    d["auto_test"] = 0
    d["strategy"] = "retrain"
    d["detect"] = False
    d["ewc"] = False
    d["replay"] = False
    # pattern-match top-k (STRAP repo default when attention_weight is absent)
    d["attention_weight"] = 5
    return d


def _mk_strap(base: dict, ds: str) -> dict:
    low = ds.lower()
    d = dict(base)
    d["method"] = "RAP"
    d["backbone_type"] = "stgnn"
    d["logname"] = f"strap_{low}"
    d["init"] = True
    d["train"] = 1
    d["auto_test"] = 0
    d["strategy"] = "retrain"
    d["detect"] = False
    d["ewc"] = False
    d["replay"] = False
    # STRAP hyperparams (STRAP-main defaults)
    d["use_strap"] = True
    d["k_neighbors"] = 16
    d["max_patterns"] = 2048
    d["fusion_weight"] = 0.7
    return d


def main() -> None:
    for ds in DATASETS:
        base = _load_base(ds)
        _write(ds, f"retrain_stgnn_{ds.lower()}",  _mk_retrain(base, ds, "stgnn",  "STGNN"))
        _write(ds, f"retrain_dcrnn_{ds.lower()}",  _mk_retrain(base, ds, "dcrnn",  "DCRNN"))
        _write(ds, f"retrain_astgnn_{ds.lower()}", _mk_retrain(base, ds, "astgnn", "ASTGNN"))
        _write(ds, f"retrain_tgcn_{ds.lower()}",   _mk_retrain(base, ds, "tgcn",   "TGCN"))
        _write(ds, f"pecpm_{ds.lower()}", _mk_pecpm(base, ds))
        _write(ds, f"strap_{ds.lower()}", _mk_strap(base, ds))


if __name__ == "__main__":
    main()
