#!/bin/bash
# ===========================================================================
# Extra baselines from the STBP paper (ICLR'26) + DLinear, on all 9 PEMS
# datasets (5 seeds each).
# ---------------------------------------------------------------------------
# Methods covered (4 lightweight re-impls, see src/model/model.py docstrings):
#   GWN          (Graph WaveNet, Wu et al. 2019)
#   STID         (Shao et al. 2022, node-MLP variant)
#   ITRANSFORMER (Liu et al. 2024, N-as-tokens)
#   DLINEAR      (Zeng et al. 2023, trend + seasonal linear)
#
# Configs:   conf/<DATASET>/retrain_<method>_<dataset>.json
# Per-year metric logs: log/<DATASET>/retrain_<method>_<dataset>-<seed>/...
# Aggregate stdout log: run_logs/extra_baselines_<timestamp>.log  (NOHUP=1)
# ---------------------------------------------------------------------------
# Usage:
#   cd eac/
#   bash scripts/extra_baselines_run.sh                              # all 9 × 4 × 5
#   NOHUP=1 bash scripts/extra_baselines_run.sh                      # background + log
#   GPU=0 bash scripts/extra_baselines_run.sh                        # pin GPU
#   DATASETS="PEMS03 PEMS04" bash scripts/extra_baselines_run.sh     # subset of datasets
#   METHODS="gwn stid" bash scripts/extra_baselines_run.sh           # subset of methods
#   SEEDS="42 43 44 45 46" bash scripts/extra_baselines_run.sh       # override seeds
#
# Seed pool defaults to 42-46 to align with the SEEDS_MAIN of each dataset's
# pemsXX_run.sh (so this column compares 1-to-1 with the existing Retrain
# column in tables/main_table_full.tex).
# ===========================================================================

set -euo pipefail
cd "$(dirname "$0")/.."

# ---------------------------------------------------------------------------
# Background mode: NOHUP=1 -> tee output to run_logs/extra_baselines_<tag>.log
# When launching multiple instances in parallel (one per GPU), pass RUN_TAG to
# disambiguate log files. If RUN_TAG is unset we auto-derive it from
# `gpu${GPU}` so concurrent launches on different GPUs don't clobber each
# other's log.
# ---------------------------------------------------------------------------
if [[ "${NOHUP:-0}" == "1" && -z "${EAC_BG:-}" ]]; then
    mkdir -p run_logs
    TAG="${RUN_TAG:-gpu${GPU:-0}}"
    LOG_FILE="run_logs/extra_baselines_${TAG}_$(date +%Y%m%d_%H%M%S).log"
    echo "[nohup] backgrounding to $LOG_FILE"
    EAC_BG=1 nohup bash "$0" "$@" > "$LOG_FILE" 2>&1 &
    BG_PID=$!
    echo "[nohup] PID=$BG_PID"
    echo "[nohup] tail -f $LOG_FILE"
    exit 0
fi

DATASETS=${DATASETS:-"PEMS03 PEMS04 PEMS05 PEMS06 PEMS07 PEMS08 PEMS10 PEMS11 PEMS12"}
SEEDS=${SEEDS:-"42 43 44 45 46"}
METHODS=${METHODS:-"gwn stid itransformer dlinear"}
GPU=${GPU:-0}

echo "[run] DATASETS=$DATASETS"
echo "[run] METHODS=$METHODS"
echo "[run] SEEDS=$SEEDS"
echo "[run] GPU=$GPU"

for ds in $DATASETS; do
    low=$(echo "$ds" | tr 'A-Z' 'a-z')
    echo ""
    echo "############################################################"
    echo "### Dataset = $ds"
    echo "############################################################"

    for m in $METHODS; do
        conf="conf/${ds}/retrain_${m}_${low}.json"
        if [[ ! -f "$conf" ]]; then
            echo "  [skip] missing config: $conf"
            continue
        fi
        echo "---------- [$ds] retrain backbone=$m ----------"
        for seed in $SEEDS; do
            python main.py --conf "$conf" --gpuid "$GPU" --seed "$seed"
        done
    done
done

echo ""
echo "==================== EXTRA BASELINES ALL DONE ===================="
