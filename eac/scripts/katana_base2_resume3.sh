#!/bin/bash
# set -e

#PBS -l select=1:ncpus=4:ngpus=1:mem=200gb:gpu_model=H200
#PBS -l walltime=99:00:00
#PBS -k oed
#PBS -M du.yin@unsw.edu.au
#PBS -m aeb
#PBS -o /srv/scratch/cruise/du/evo-xxltraffic/eac
#PBS -e /srv/scratch/cruise/du/evo-xxltraffic/eac
#PBS -N evo3

source ~/.bashrc
source /srv/scratch/z5440262/miniconda3/bin/activate
conda activate stg
cd /srv/scratch/cruise/du/evo-xxltraffic/eac

set -euo pipefail

LOG_FILE="run_logs/base2_resume3_${JOB_ID:-local}_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG_FILE") 2>&1
echo "[run] tee -> $LOG_FILE"

# ===========================================================================
# Resume #3 of l40s_base2.sh. The katana run (katana_base2_resume2.sh) was
# killed by the server mid-epoch in PEMS12 / STRAP / seed 51 / 2010.
# No traceback -> hard kill, year-2010 STRAP checkpoint may be partial.
#
# Everything else from base2 + resume + resume2 is done. Only one run left:
#   PEMS12 : STRAP, seed 51   (re-runs years 2002..2025, overwrites year 2010+)
# ===========================================================================

GPU_MAIN=${GPU_MAIN:-${GPU:-0}}
DS="PEMS12"
LOW="pems12"
SEED="51"

FIRST_YEAR=$(python - <<PY
import json
print(json.load(open("conf/${DS}/retrain_st_${LOW}.json"))["begin_year"])
PY
)
echo "[run] dataset=${DS} method=strap seed=${SEED} first_year=${FIRST_YEAR}"

PKL_DIR="log/${DS}/retrain_stgnn_${LOW}-${SEED}/${FIRST_YEAR}"
PKL=$(ls "${PKL_DIR}"/*.pkl 2>/dev/null | head -n 1)
if [[ -z "${PKL}" ]]; then
    echo "[error] missing retrain_stgnn first-year pkl in ${PKL_DIR}" >&2
    exit 1
fi
echo "[run] base pkl = ${PKL}"

# Re-link the first-year pkl into the strap log dir (utils/common_tools.py's
# load_test_best_model() listdirs log/<DS>/<logname>-<seed>/<year>/ to find it).
DST_DIR="log/${DS}/strap_${LOW}-${SEED}/${FIRST_YEAR}"
mkdir -p "${DST_DIR}"
ln -sf "$(readlink -f "${PKL}")" "${DST_DIR}/$(basename "${PKL}")"

# Optional: wipe the partial year-2010 strap checkpoint dir so the rerun
# starts that year cleanly. (Safe: main.py overwrites pkls per epoch anyway,
# but we drop the half-written ones to avoid confusion.)
rm -rf "log/${DS}/strap_${LOW}-${SEED}/2010"

echo ""
echo "############################################################"
echo "### Dataset = ${DS}  STRAP  seed ${SEED}  (begin_year=${FIRST_YEAR})"
echo "############################################################"

python main.py --conf "conf/${DS}/strap_${LOW}.json" \
    --load_first_year 1 --first_year_model_path "${PKL}" \
    --gpuid "${GPU_MAIN}" --seed "${SEED}"

echo ""
echo "==================== RESUME #3 RUN DONE ===================="
