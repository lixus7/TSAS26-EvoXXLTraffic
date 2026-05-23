#!/bin/bash
# ===========================================================================
# Smoke-test (debug) script for baselines_pems_run.sh
# ---------------------------------------------------------------------------
# Clones each method's JSON into /tmp with:
#   epoch = 1  (or $EPOCH)
#   end_year = begin_year + $EXTRA_YEARS  (default begin_year + 1, i.e. 2 years)
#   logname suffix "_debug"
# Then runs ONE seed per method on ONE dataset (PEMS03 by default).
#
# Goal: end-to-end sanity check of all 6 methods in a few minutes.
# (STKEC is intentionally NOT included — already covered by pemsXX_run_debug.sh.)
#
# Usage:
#   cd eac/
#   bash scripts/baselines_pems_run_debug.sh
#
# Overrides:
#   DATASET=PEMS04            single dataset (default PEMS03)
#   METHODS="retrain_stgnn strap"   subset (default all 6)
#   EPOCH=1 EXTRA_YEARS=1    smaller = faster
#   SEED=47                  single-seed smoke (default 47)
#   GPU=0
# ===========================================================================

set -euo pipefail
cd "$(dirname "$0")/.."

# ---------------------------------------------------------------------------
# Background mode
# ---------------------------------------------------------------------------
if [[ "${NOHUP:-0}" == "1" && -z "${EAC_BG:-}" ]]; then
    mkdir -p run_logs
    LOG_FILE="run_logs/baselines_pems_debug_$(date +%Y%m%d_%H%M%S).log"
    echo "[nohup] backgrounding to $LOG_FILE"
    EAC_BG=1 nohup bash "$0" "$@" > "$LOG_FILE" 2>&1 &
    echo "[nohup] PID=$!  ;  tail -f $LOG_FILE"
    exit 0
fi

DATASET=${DATASET:-PEMS03}
EPOCH=${EPOCH:-1}
EXTRA_YEARS=${EXTRA_YEARS:-1}        # debug covers begin_year ~ begin_year+EXTRA_YEARS

SEED_MAIN=${SEED_MAIN:-${SEED:-47}}

GPU_MAIN=${GPU_MAIN:-${GPU:-2}}

VALID_METHODS="retrain_stgnn retrain_astgnn retrain_dcrnn retrain_tgcn pecpm strap"
METHODS=${METHODS:-"$VALID_METHODS"}

LOW=$(echo "$DATASET" | tr 'A-Z' 'a-z')
SRC_CONF_DIR="conf/${DATASET}"
DEBUG_CONF_DIR="/tmp/eac_debug_conf/${DATASET}"
DEBUG_SUFFIX="debug"
LOG_ROOT="log/${DATASET}"

echo "[debug] DATASET=$DATASET  METHODS=$METHODS  EPOCH=$EPOCH  EXTRA_YEARS=$EXTRA_YEARS"
echo "[debug] SEED_MAIN=$SEED_MAIN  GPU_MAIN=$GPU_MAIN"
mkdir -p "$DEBUG_CONF_DIR"

# ---------------------------------------------------------------------------
# [0] Clone every relevant JSON into /tmp with shrunk epoch/end_year/logname
# ---------------------------------------------------------------------------
echo "==================== [0] clone debug JSONs ===================="
python - <<PY
import json
from pathlib import Path

src = Path("$SRC_CONF_DIR")
dst = Path("$DEBUG_CONF_DIR")
dst.mkdir(parents=True, exist_ok=True)

extra = $EXTRA_YEARS
epoch = $EPOCH
suffix = "$DEBUG_SUFFIX"
LOG_ROOT = "$LOG_ROOT"

for p in sorted(src.glob("*.json")):
    data = json.loads(p.read_text())
    data["epoch"] = epoch
    by = data.get("begin_year")
    if by is not None:
        data["end_year"] = by + extra
    if "logname" in data:
        data["logname"] = f"{data['logname']}_{suffix}"
    # EAC: in debug we still want to TRAIN so that the whole pipeline is exercised
    if data.get("method") == "EAC":
        data["train"] = 1
        data["auto_test"] = 0
    (dst / p.name).write_text(json.dumps(data, indent=4))
    print(f"  [gen] {dst / p.name}  (begin={data['begin_year']}, end={data['end_year']}, epoch={epoch}, logname={data.get('logname')})")
PY

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
BEGIN_YEAR_EFF=$(python - <<PY
import json
print(json.load(open("$DEBUG_CONF_DIR/retrain_stgnn_${LOW}.json"))["begin_year"])
PY
)

debug_first_year_pkl() {
    # Find the first-year pkl produced by the debug retrain_stgnn run.
    local seed="$1"
    local dir="${LOG_ROOT}/retrain_stgnn_${LOW}_${DEBUG_SUFFIX}-${seed}/${BEGIN_YEAR_EFF}"
    ls "${dir}"/*.pkl 2>/dev/null | head -n 1
}

need_first_year_pkl() {
    # For STRAP / PECPM we need a base checkpoint. If user disabled retrain_stgnn
    # in METHODS, run a quick one just for the first-year weight.
    if [[ ! " $METHODS " =~ " retrain_stgnn " ]]; then
        echo "==================== [pre] quick retrain_stgnn (first-year weight) ===================="
        python main.py --conf "$DEBUG_CONF_DIR/retrain_stgnn_${LOW}.json" \
            --gpuid "$GPU_MAIN" --seed "$SEED_MAIN"
    fi
}

link_pkl_to_logname() {
    # utils/common_tools.py::load_test_best_model() listdirs log/<DS>/<logname>-<seed>/<year>/
    # and ignores --first_year_model_path, so we mirror pems_run.sh's AutoLink: symlink the
    # retrain_stgnn first-year pkl into the target logname's expected dir.
    local pkl="$1" target_logname="$2" seed="$3"
    local dst_dir="${LOG_ROOT}/${target_logname}-${seed}/${BEGIN_YEAR_EFF}"
    mkdir -p "$dst_dir"
    ln -sf "$(readlink -f "$pkl")" "$dst_dir/$(basename "$pkl")"
    echo "  [link] $pkl -> $dst_dir/"
}

# ---------------------------------------------------------------------------
# Run each requested method
# ---------------------------------------------------------------------------
for m in $METHODS; do
    case "$m" in
        retrain_stgnn|retrain_astgnn|retrain_dcrnn|retrain_tgcn)
            bk=${m#retrain_}
            echo "==================== [$m] retrain (backbone=$bk) ===================="
            python main.py --conf "$DEBUG_CONF_DIR/retrain_${bk}_${LOW}.json" \
                --gpuid "$GPU_MAIN" --seed "$SEED_MAIN"
            ;;

        pecpm)
            echo "==================== [pecpm] ===================="
            need_first_year_pkl
            pkl=$(debug_first_year_pkl "$SEED_MAIN")
            if [[ -z "$pkl" ]]; then
                echo "  [error] no first-year pkl for PECPM — run retrain_stgnn first"
                exit 1
            fi
            link_pkl_to_logname "$pkl" "pecpm_${LOW}_${DEBUG_SUFFIX}" "$SEED_MAIN"
            python main.py --conf "$DEBUG_CONF_DIR/pecpm_${LOW}.json" \
                --load_first_year 1 --first_year_model_path "$pkl" \
                --gpuid "$GPU_MAIN" --seed "$SEED_MAIN"
            ;;

        strap)
            echo "==================== [strap] ===================="
            need_first_year_pkl
            pkl=$(debug_first_year_pkl "$SEED_MAIN")
            if [[ -z "$pkl" ]]; then
                echo "  [error] no first-year pkl for STRAP — run retrain_stgnn first"
                exit 1
            fi
            link_pkl_to_logname "$pkl" "strap_${LOW}_${DEBUG_SUFFIX}" "$SEED_MAIN"
            python main.py --conf "$DEBUG_CONF_DIR/strap_${LOW}.json" \
                --load_first_year 1 --first_year_model_path "$pkl" \
                --gpuid "$GPU_MAIN" --seed "$SEED_MAIN"
            ;;

        *)
            echo "[error] unknown method '$m' (valid: $VALID_METHODS)" >&2
            exit 1
            ;;
    esac
done

echo ""
echo "==================== ALL DEBUG RUNS PASSED ===================="
echo "Debug configs : $DEBUG_CONF_DIR"
echo "Debug logs    : $LOG_ROOT/*_${DEBUG_SUFFIX}-*"
echo "Cleanup       : rm -rf $DEBUG_CONF_DIR $LOG_ROOT/*_${DEBUG_SUFFIX}-*"
