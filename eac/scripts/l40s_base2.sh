#!/bin/bash

#$ -P CRUISE
#$ -N evo-base2
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

# Mirror everything to run_logs/ in addition to SGE's $JOB_ID_$JOB_NAME.out,
# so logs sit next to pemsXX_*.log produced by the non-SGE run scripts.
mkdir -p run_logs
LOG_FILE="run_logs/base2_${JOB_ID:-local}_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG_FILE") 2>&1
echo "[run] tee -> $LOG_FILE"

# ===========================================================================
# STRAP-paper baseline bench on PEMS03..PEMS12 (excl. PEMS09)
# ---------------------------------------------------------------------------
# Methods per dataset (6 total):
#   Retrain x 4 backbones (STGNN / ASTGNN / DCRNN / TGCN)
#   PECPM  (STGNN backbone, STRAP-authors' re-impl)
#   STRAP  (STGNN backbone, NeurIPS'25 - code name RAP in repo)
#
# STKEC is intentionally NOT included here - already covered by pemsXX_run.sh.
#
# Datasets: PEMS03 PEMS04 PEMS05 PEMS06 PEMS07 PEMS08 PEMS10 PEMS11 PEMS12
# Seeds: 47 48 49 50 51
# ===========================================================================

DATASETS=${DATASETS:-"PEMS03 PEMS04 PEMS05 PEMS06 PEMS07 PEMS08 PEMS10 PEMS11 PEMS12"}
SEEDS=${SEEDS:-"47 48 49 50 51"}

# SGE allocates 1 GPU -> CUDA_VISIBLE_DEVICES exposes it as device 0 inside the job.
GPU_MAIN=${GPU_MAIN:-${GPU:-0}}

VALID_METHODS="retrain_stgnn retrain_astgnn retrain_dcrnn retrain_tgcn pecpm strap"
METHODS=${METHODS:-"$VALID_METHODS"}

echo "[run] DATASETS=$DATASETS"
echo "[run] METHODS=$METHODS"
echo "[run] SEEDS=$SEEDS"
echo "[run] GPU_MAIN=$GPU_MAIN"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
first_year_of() {
    local ds="$1"
    local low; low=$(echo "$ds" | tr 'A-Z' 'a-z')
    python - <<PY
import json
print(json.load(open("conf/$ds/retrain_st_${low}.json"))["begin_year"])
PY
}

# For STRAP / PECPM we load the STGNN retrain first-year weight as the base.
first_year_pkl_for() {
    local ds="$1" seed="$2" first_year="$3"
    local dir="log/${ds}/retrain_stgnn_$(echo "$ds" | tr 'A-Z' 'a-z')-${seed}/${first_year}"
    ls "${dir}"/*.pkl 2>/dev/null | head -n 1
}

link_pkl_to_logname() {
    # utils/common_tools.py::load_test_best_model() listdirs log/<DS>/<logname>-<seed>/<year>/
    # and ignores --first_year_model_path, so mirror pems_run.sh's AutoLink: symlink the
    # retrain_stgnn first-year pkl into the target logname's expected dir.
    local ds="$1" pkl="$2" target_logname="$3" seed="$4" first_year="$5"
    local dst_dir="log/${ds}/${target_logname}-${seed}/${first_year}"
    mkdir -p "$dst_dir"
    ln -sf "$(readlink -f "$pkl")" "$dst_dir/$(basename "$pkl")"
}

# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------
for ds in $DATASETS; do
    low=$(echo "$ds" | tr 'A-Z' 'a-z')
    FIRST_YEAR=$(first_year_of "$ds")
    echo ""
    echo "############################################################"
    echo "### Dataset = $ds  (begin_year=$FIRST_YEAR)"
    echo "############################################################"

    for m in $METHODS; do
        case "$m" in
            retrain_stgnn|retrain_astgnn|retrain_dcrnn|retrain_tgcn)
                bk=${m#retrain_}
                conf="conf/${ds}/retrain_${bk}_${low}.json"
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
done

echo ""
echo "==================== ALL BASELINES DONE ===================="
