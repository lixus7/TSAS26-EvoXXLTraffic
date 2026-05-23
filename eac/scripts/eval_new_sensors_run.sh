#!/bin/bash
# ===========================================================================
# Re-evaluate trained checkpoints on **newly added sensors only** across all
# baselines used in tables/tsas_main_table_part{1,2}.tex. Runs in parallel
# across multiple GPUs.
#
# Per-(dataset, method, seed) this writes:
#   log/<DS>/<logname>-<seed>/eval_new_sensors.log
#   log/<DS>/<logname>-<seed>/eval_new_sensors.csv
# Plus an aggregate summary CSV at:
#   run_logs/eval_new_sensors_<timestamp>/summary.csv
#
# ---------------------------------------------------------------------------
# Seed pools (mirrors pemsXX_run.sh / baselines_pems_run.sh / sttc_run.sh):
#   retrain_st / pretrain_st / oneline_st_nn / trafficstream / sttc : 42 43 44 45 46
#   oneline_st_an / eac                                              : 51 52 53 54 55
#   stkec                                                            : 47 48 49 50 51
#   retrain_{stgnn,astgnn,dcrnn,tgcn} / pecpm / strap                : 47 48 49 50 51
#
# ---------------------------------------------------------------------------
# Usage:
#   cd eac/
#   bash scripts/eval_new_sensors_run.sh                              # 8-GPU parallel
#   NOHUP=1 bash scripts/eval_new_sensors_run.sh                      # background
#
# Overrides (env vars):
#   DATASETS="PEMS03 PEMS04 PEMS05 PEMS06 PEMS07 PEMS08 PEMS10 PEMS11 PEMS12"
#                                       default; subset to scope down
#   METHODS="eac stkec"                 run only these (default: all)
#   GPU_IDS="0 1 2 3 4 5 6 7"           GPUs to use (default: 0..7)
#   GPU=2                               (legacy) single-GPU; equivalent to GPU_IDS="2"
#   NOHUP=1                             run in background, log to run_logs/
# ===========================================================================

set -euo pipefail
cd "$(dirname "$0")/.."

# ---------------------------------------------------------------------------
# Background mode
# ---------------------------------------------------------------------------
if [[ "${NOHUP:-0}" == "1" && -z "${EAC_BG:-}" ]]; then
    mkdir -p run_logs
    LOG_FILE="run_logs/eval_new_sensors_$(date +%Y%m%d_%H%M%S).log"
    echo "[nohup] backgrounding to $LOG_FILE"
    EAC_BG=1 nohup bash "$0" "$@" > "$LOG_FILE" 2>&1 &
    BG_PID=$!
    echo "[nohup] PID=$BG_PID"
    echo "[nohup] tail -f $LOG_FILE"
    exit 0
fi

DATASETS=${DATASETS:-"PEMS03 PEMS04 PEMS05 PEMS06 PEMS07 PEMS08 PEMS10 PEMS11 PEMS12"}

# GPU pool. Honor single-GPU legacy `GPU=` env if user set it; else default to 8 GPUs.
if [[ -n "${GPU:-}" && -z "${GPU_IDS:-}" ]]; then
    GPU_IDS="$GPU"
fi
GPU_IDS=${GPU_IDS:-"0 1 2 3 4 5 6 7"}
read -r -a GPU_ARR <<< "$GPU_IDS"
NGPU=${#GPU_ARR[@]}

# Methods recognized below (lognames match the per-dataset configs)
ALL_METHODS="retrain_st pretrain_st oneline_st_nn oneline_st_an trafficstream stkec eac \
retrain_stgnn retrain_astgnn retrain_dcrnn retrain_tgcn pecpm strap sttc"
METHODS=${METHODS:-"$ALL_METHODS"}

SUMMARY_DIR="run_logs/eval_new_sensors_$(date +%Y%m%d_%H%M%S)"
FRAG_DIR="$SUMMARY_DIR/fragments"
mkdir -p "$FRAG_DIR"
SUMMARY_CSV="$SUMMARY_DIR/summary.csv"

echo "[run] DATASETS=$DATASETS"
echo "[run] METHODS=$METHODS"
echo "[run] GPU_IDS=${GPU_ARR[*]}  (NGPU=$NGPU)"
echo "[run] SUMMARY=$SUMMARY_CSV"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
seeds_for_method() {
    case "$1" in
        retrain_st|pretrain_st|oneline_st_nn|trafficstream)  echo "42 43 44 45 46" ;;
        sttc)                                                echo "42 43 44 45 46" ;;  # sttc_run.sh uses SEEDS_MAIN
        oneline_st_an|eac)                                   echo "51 52 53 54 55" ;;
        stkec)                                               echo "47 48 49 50 51" ;;
        retrain_stgnn|retrain_astgnn|retrain_dcrnn|retrain_tgcn|pecpm|strap)
                                                             echo "47 48 49 50 51" ;;
        *) echo "" ;;
    esac
}

conf_for() {
    local ds="$1" m="$2"
    local low; low=$(echo "$ds" | tr 'A-Z' 'a-z')
    local conf
    case "$m" in
        eac|stkec|trafficstream) conf="conf/${ds}/${m}.json" ;;
        retrain_st|pretrain_st|oneline_st_nn|oneline_st_an|retrain_stgnn|retrain_astgnn|retrain_dcrnn|retrain_tgcn|pecpm|sttc)
            conf="conf/${ds}/${m}_${low}.json" ;;
        strap)
            conf="conf/${ds}/strap_${low}.json" ;;
        *) conf="" ;;
    esac
    [[ -f "$conf" ]] && echo "$conf" || echo ""
}

logname_for() {
    local ds="$1" m="$2"
    local low; low=$(echo "$ds" | tr 'A-Z' 'a-z')
    case "$m" in
        eac|stkec|trafficstream) echo "$m" ;;
        retrain_st|pretrain_st|oneline_st_nn|oneline_st_an|retrain_stgnn|retrain_astgnn|retrain_dcrnn|retrain_tgcn|pecpm|sttc)
            echo "${m}_${low}" ;;
        strap) echo "strap_${low}" ;;
        *) echo "" ;;
    esac
}

# ---------------------------------------------------------------------------
# Build job list: only jobs where checkpoint dir exists
# ---------------------------------------------------------------------------
JOBS=()
for ds in $DATASETS; do
    for m in $METHODS; do
        conf=$(conf_for "$ds" "$m")
        if [[ -z "$conf" ]]; then continue; fi
        logname=$(logname_for "$ds" "$m")
        seeds=$(seeds_for_method "$m")
        if [[ -z "$seeds" ]]; then continue; fi
        for seed in $seeds; do
            seed_dir="log/${ds}/${logname}-${seed}"
            if [[ ! -d "$seed_dir" ]]; then
                echo "  [no-ckpt] $ds/$m seed=$seed: $seed_dir missing"
                continue
            fi
            JOBS+=("${ds}|${m}|${seed}|${conf}|${logname}")
        done
    done
done

NJOBS=${#JOBS[@]}
echo "[run] $NJOBS jobs queued; dispatching to $NGPU GPUs (round-robin by job index)"
if (( NJOBS == 0 )); then
    echo "[run] nothing to do, exiting."
    exit 0
fi

# ---------------------------------------------------------------------------
# Worker: each worker is pinned to one GPU and processes (i % NGPU == widx).
# Each job's verbose Python output goes to fragments/<ds>_<m>_<seed>.stdout
# and its per-seed CSV is converted to a per-job fragment CSV (no header).
# After all workers finish, we concat all fragments into summary.csv.
# ---------------------------------------------------------------------------
run_worker() {
    local widx="$1"
    local gpu="$2"
    local done_count=0
    for i in "${!JOBS[@]}"; do
        if (( i % NGPU != widx )); then continue; fi
        IFS='|' read -r ds m seed conf logname <<< "${JOBS[$i]}"
        local seed_dir="log/${ds}/${logname}-${seed}"
        local tag="${ds}_${m}_${seed}"

        done_count=$((done_count + 1))
        echo "[gpu $gpu | w$widx | $done_count] start $ds/$m seed=$seed"

        python scripts/evaluate_new_sensors.py \
            --conf "$conf" --seed "$seed" --gpuid "$gpu" \
            > "$FRAG_DIR/${tag}.stdout" 2>&1 \
            || echo "[gpu $gpu | w$widx] WARN python failed for $tag (see $FRAG_DIR/${tag}.stdout)"

        local csv_file="$seed_dir/eval_new_sensors.csv"
        if [[ -f "$csv_file" ]]; then
            python - "$csv_file" "$FRAG_DIR/${tag}.csv" "$ds" "$m" "$seed" <<'PY'
import csv, sys
from collections import defaultdict
import statistics as st
csv_path, frag_path, ds, m, seed = sys.argv[1:6]
rows_in = list(csv.DictReader(open(csv_path)))
agg = defaultdict(list)
for r in rows_in:
    agg[(r["scope"], r["horizon"])].append(
        (float(r["MAE"]), float(r["RMSE"]), float(r["MAPE"])))
with open(frag_path, "w", newline="") as f:
    w = csv.writer(f)
    for (scope, horizon), vs in agg.items():
        if not vs:
            continue
        w.writerow((ds, m, seed, scope, horizon, len(vs),
                    st.mean(v[0] for v in vs),
                    st.mean(v[1] for v in vs),
                    st.mean(v[2] for v in vs)))
PY
        fi
        echo "[gpu $gpu | w$widx | $done_count] done  $ds/$m seed=$seed"
    done
    echo "[gpu $gpu | w$widx] worker DONE ($done_count jobs)"
}

# ---------------------------------------------------------------------------
# Spawn one worker per GPU
# ---------------------------------------------------------------------------
PIDS=()
for widx in $(seq 0 $((NGPU - 1))); do
    gpu=${GPU_ARR[$widx]}
    run_worker "$widx" "$gpu" &
    PIDS+=($!)
done

# Wait for all workers; trap to surface partial failures
trap 'echo "[run] interrupt; killing workers"; kill "${PIDS[@]}" 2>/dev/null; exit 130' INT TERM
for pid in "${PIDS[@]}"; do
    wait "$pid" || echo "[run] worker pid=$pid exited non-zero"
done

# ---------------------------------------------------------------------------
# Assemble summary.csv
# ---------------------------------------------------------------------------
echo "[run] assembling $SUMMARY_CSV"
{
    echo "dataset,method,seed,scope,horizon,year_count,MAE,RMSE,MAPE"
    # cat may exit non-zero if no fragments exist; guard with || true
    cat "$FRAG_DIR"/*.csv 2>/dev/null || true
} > "$SUMMARY_CSV"

n_rows=$(($(wc -l < "$SUMMARY_CSV") - 1))
echo ""
echo "==================== ALL EVAL DONE ===================="
echo "Jobs queued / completed fragments: $NJOBS / $(ls "$FRAG_DIR"/*.csv 2>/dev/null | wc -l)"
echo "Summary rows:  $n_rows"
echo "Per-run logs:  log/<DS>/<logname>-<seed>/eval_new_sensors.{log,csv}"
echo "Worker stdout: $FRAG_DIR/<ds>_<method>_<seed>.stdout"
echo "Summary:       $SUMMARY_CSV"
