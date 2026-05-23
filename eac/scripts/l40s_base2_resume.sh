#!/bin/bash

#$ -P CRUISE
#$ -N evo-base2-resume
#$ -j y
#$ -m ea
#$ -M du.yin@unsw.edu.au
#$ -e /mnt/scratch/CRUISE/Du/code/evo-xxltraffic/eac/$JOB_ID_$JOB_NAME.err
#$ -o /mnt/scratch/CRUISE/Du/code/evo-xxltraffic/eac/$JOB_ID_$JOB_NAME.out
#$ -cwd
#$ -l walltime=999:00:00
#$ -l mem=250G
#$ -l jobfs=400G
#$ -l tmpfree=12G
#$ -l ngpus=1
#$ -pe smp 4
#$ -l gpu_model=L40S
source ~/.bashrc

ca stg
cd /mnt/scratch/CRUISE/Du/code/evo-xxltraffic/eac

set -euo pipefail

mkdir -p run_logs
LOG_FILE="run_logs/base2_resume_${JOB_ID:-local}_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG_FILE") 2>&1
echo "[run] tee -> $LOG_FILE"

# ===========================================================================
# Resume of l40s_base2.sh after the PECPM CPU-OOM crash on PEMS04 / year 2012.
# The matmul fix lives in src/model/model.py::PECPM_Model.pattern_matching
# (history capped before mm, mm moved to GPU).
#
# Already finished by the previous run (job 28252, 2026-04-29..05-01):
#   PEMS03 : retrain x4, PECPM, STRAP  (all 5 seeds)
#   PEMS04 : retrain x4               (all 5 seeds)
#
# Still to do:
#   PEMS04 : PECPM, STRAP             (all 5 seeds; PECPM seed 47 redoes
#                                       2002..2025 since the crash left
#                                       no clean year-2012 checkpoint)
#   PEMS05/06/07/08/10/11/12 : retrain x4, PECPM, STRAP (all 5 seeds)
# ===========================================================================

SEEDS=${SEEDS:-"47 48 49 50 51"}
GPU_MAIN=${GPU_MAIN:-${GPU:-0}}

VALID_METHODS="retrain_stgnn retrain_astgnn retrain_dcrnn retrain_tgcn pecpm strap"

echo "[run] SEEDS=$SEEDS"
echo "[run] GPU_MAIN=$GPU_MAIN"

first_year_of() {
    local ds="$1"
    local low; low=$(echo "$ds" | tr 'A-Z' 'a-z')
    python - <<PY
import json
print(json.load(open("conf/$ds/retrain_st_${low}.json"))["begin_year"])
PY
}

first_year_pkl_for() {
    local ds="$1" seed="$2" first_year="$3"
    local dir="log/${ds}/retrain_stgnn_$(echo "$ds" | tr 'A-Z' 'a-z')-${seed}/${first_year}"
    ls "${dir}"/*.pkl 2>/dev/null | head -n 1
}

link_pkl_to_logname() {
    local ds="$1" pkl="$2" target_logname="$3" seed="$4" first_year="$5"
    local dst_dir="log/${ds}/${target_logname}-${seed}/${first_year}"
    mkdir -p "$dst_dir"
    ln -sf "$(readlink -f "$pkl")" "$dst_dir/$(basename "$pkl")"
}

run_methods_on_dataset() {
    local ds="$1"
    shift
    local methods="$*"
    local low; low=$(echo "$ds" | tr 'A-Z' 'a-z')
    local FIRST_YEAR; FIRST_YEAR=$(first_year_of "$ds")
    echo ""
    echo "############################################################"
    echo "### Dataset = $ds  (begin_year=$FIRST_YEAR)  methods=$methods"
    echo "############################################################"

    for m in $methods; do
        case "$m" in
            retrain_stgnn|retrain_astgnn|retrain_dcrnn|retrain_tgcn)
                local bk=${m#retrain_}
                local conf="conf/${ds}/retrain_${bk}_${low}.json"
                echo "---------- [$ds] retrain backbone=$bk ----------"
                for seed in $SEEDS; do
                    python main.py --conf "$conf" --gpuid "$GPU_MAIN" --seed "$seed"
                done
                ;;

            pecpm)
                echo "---------- [$ds] PECPM (STGNN backbone) ----------"
                for seed in $SEEDS; do
                    pkl=$(first_year_pkl_for "$ds" "$seed" "$FIRST_YEAR")
                    if [[ -z "$pkl" ]]; then
                        echo "  [skip] pecpm seed=$seed: missing retrain_stgnn first-year pkl for $ds"
                        continue
                    fi
                    link_pkl_to_logname "$ds" "$pkl" "pecpm_${low}" "$seed" "$FIRST_YEAR"
                    python main.py --conf "conf/${ds}/pecpm_${low}.json" \
                        --load_first_year 1 --first_year_model_path "$pkl" \
                        --gpuid "$GPU_MAIN" --seed "$seed"
                done
                ;;

            strap)
                echo "---------- [$ds] STRAP (STGNN backbone) ----------"
                for seed in $SEEDS; do
                    pkl=$(first_year_pkl_for "$ds" "$seed" "$FIRST_YEAR")
                    if [[ -z "$pkl" ]]; then
                        echo "  [skip] strap seed=$seed: missing retrain_stgnn first-year pkl for $ds"
                        continue
                    fi
                    link_pkl_to_logname "$ds" "$pkl" "strap_${low}" "$seed" "$FIRST_YEAR"
                    python main.py --conf "conf/${ds}/strap_${low}.json" \
                        --load_first_year 1 --first_year_model_path "$pkl" \
                        --gpuid "$GPU_MAIN" --seed "$seed"
                done
                ;;

            *)
                echo "[error] unknown method '$m' (valid: $VALID_METHODS)" >&2
                exit 1
                ;;
        esac
    done
}

# Phase A: finish PEMS04 (only PECPM + STRAP remain).
run_methods_on_dataset PEMS04 pecpm strap

# Phase B: full sweep on the datasets that never started.
REMAINING_DATASETS=${REMAINING_DATASETS:-"PEMS05 PEMS06 PEMS07 PEMS08 PEMS10 PEMS11 PEMS12"}
REMAINING_METHODS=${REMAINING_METHODS:-"$VALID_METHODS"}
for ds in $REMAINING_DATASETS; do
    run_methods_on_dataset "$ds" $REMAINING_METHODS
done

echo ""
echo "==================== RESUME RUN DONE ===================="
