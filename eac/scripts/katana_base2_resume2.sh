#!/bin/bash
# set -e

#PBS -l select=1:ncpus=4:ngpus=1:mem=200gb:gpu_model=H200
#PBS -l walltime=99:00:00
#PBS -k oed
#PBS -M du.yin@unsw.edu.au
#PBS -m aeb
#PBS -o /srv/scratch/cruise/du/evo-xxltraffic/eac
#PBS -e /srv/scratch/cruise/du/evo-xxltraffic/eac
#PBS -N evo2

source ~/.bashrc
source /srv/scratch/z5440262/miniconda3/bin/activate
conda activate stg
cd /srv/scratch/cruise/du/evo-xxltraffic/eac

set -euo pipefail

# mkdir -p run_logs
LOG_FILE="run_logs/base2_resume2_${JOB_ID:-local}_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG_FILE") 2>&1
echo "[run] tee -> $LOG_FILE"

# ===========================================================================
# Resume #2 of l40s_base2.sh, after l40s_base2_resume.sh (job 28964) was
# killed by the server mid-epoch in PEMS05 / retrain_astgnn / seed 51 / 2021.
# No traceback in that log -> hard kill, year-2021 checkpoint may be partial.
#
# Already finished by job 28964:
#   PEMS04 : PECPM, STRAP                        (seeds 47..51)
#   PEMS05 : retrain_stgnn                       (seeds 47..51)
#   PEMS05 : retrain_astgnn                      (seeds 47..50)
#
# Still to do:
#   PEMS05 : retrain_astgnn                      (seed 51 only)
#   PEMS05 : retrain_dcrnn, retrain_tgcn, pecpm, strap   (seeds 47..51)
#   PEMS06/07/08/10/11/12 : full 6-method sweep  (seeds 47..51)
# ===========================================================================

ALL_SEEDS=${SEEDS:-"47 48 49 50 51"}
GPU_MAIN=${GPU_MAIN:-${GPU:-0}}

VALID_METHODS="retrain_stgnn retrain_astgnn retrain_dcrnn retrain_tgcn pecpm strap"

echo "[run] ALL_SEEDS=$ALL_SEEDS"
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

# Run one (dataset, methods, seeds) tuple. Methods + seeds are space-separated.
run_methods() {
    local ds="$1" methods="$2" seeds="$3"
    local low; low=$(echo "$ds" | tr 'A-Z' 'a-z')
    local FIRST_YEAR; FIRST_YEAR=$(first_year_of "$ds")
    echo ""
    echo "############################################################"
    echo "### Dataset = $ds  (begin_year=$FIRST_YEAR)"
    echo "###   methods = $methods"
    echo "###   seeds   = $seeds"
    echo "############################################################"

    for m in $methods; do
        case "$m" in
            retrain_stgnn|retrain_astgnn|retrain_dcrnn|retrain_tgcn)
                local bk=${m#retrain_}
                local conf="conf/${ds}/retrain_${bk}_${low}.json"
                echo "---------- [$ds] retrain backbone=$bk ----------"
                for seed in $seeds; do
                    python main.py --conf "$conf" --gpuid "$GPU_MAIN" --seed "$seed"
                done
                ;;

            pecpm)
                echo "---------- [$ds] PECPM (STGNN backbone) ----------"
                for seed in $seeds; do
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
                for seed in $seeds; do
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

# Phase A1: redo the run that got killed (PEMS05 retrain_astgnn seed 51).
run_methods PEMS05 "retrain_astgnn" "51"

# Phase A2: finish the rest of PEMS05 (the methods that never started).
run_methods PEMS05 "retrain_dcrnn retrain_tgcn pecpm strap" "$ALL_SEEDS"

# Phase B: full sweep on the datasets that never started.
REMAINING_DATASETS=${REMAINING_DATASETS:-"PEMS06 PEMS07 PEMS08 PEMS10 PEMS11 PEMS12"}
REMAINING_METHODS=${REMAINING_METHODS:-"$VALID_METHODS"}
for ds in $REMAINING_DATASETS; do
    run_methods "$ds" "$REMAINING_METHODS" "$ALL_SEEDS"
done

echo ""
echo "==================== RESUME #2 RUN DONE ===================="
