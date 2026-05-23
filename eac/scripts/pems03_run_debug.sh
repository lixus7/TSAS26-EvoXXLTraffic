#!/bin/bash
# ===========================================================================
# PEMS03-Stream 冒烟测试脚本 (pems03_run.sh 的 Debug 版)
# ---------------------------------------------------------------------------
# 思路照搬 pems_run_debug.sh：克隆一份 JSON 到 /tmp，改小 end_year / epoch，
# 用一个 seed 把 retrain → pretrain → online-nn → online-an → trafficstream
# → stkec → eac 全链路通一遍。
#
# 用法：
#   cd eac/
#   bash scripts/pems03_run_debug.sh
#
# 环境变量覆盖：
#   GPU=0 BEGIN_YEAR=2014 END_YEAR=2017 EPOCH=1 bash scripts/pems03_run_debug.sh
# ===========================================================================

set -euo pipefail
cd "$(dirname "$0")/.."

# ---------------------------------------------------------------------------
# 后台模式: 用 `NOHUP=1 bash scripts/pems03_run_debug.sh` 触发
# 自动 nohup 后台跑, 日志落到 eac/run_logs/pems03_debug_<时间戳>.log
# ---------------------------------------------------------------------------
if [[ "${NOHUP:-0}" == "1" && -z "${EAC_BG:-}" ]]; then
    mkdir -p run_logs
    LOG_FILE="run_logs/pems03_debug_$(date +%Y%m%d_%H%M%S).log"
    echo "[nohup] backgrounding to $LOG_FILE"
    EAC_BG=1 nohup bash "$0" "$@" > "$LOG_FILE" 2>&1 &
    BG_PID=$!
    echo "[nohup] PID=$BG_PID"
    echo "[nohup] tail -f $LOG_FILE"
    exit 0
fi

SRC_CONF_DIR="conf/PEMS03"
DEBUG_CONF_DIR="/tmp/eac_debug_conf/PEMS03"
DEBUG_LOG_SUFFIX="debug"

BEGIN_YEAR=${BEGIN_YEAR:-}          # 不设则沿用原 conf
END_YEAR=${END_YEAR:-}              # 不设则沿用原 conf
EPOCH=${EPOCH:-1}

SEED_MAIN=${SEED_MAIN:-42}
SEED_ALT=${SEED_ALT:-51}
SEED_STKEC=${SEED_STKEC:-47}

GPU=${GPU:-2}

LOG_ROOT="log/PEMS03"

# ============================================================================
# 0. 生成 debug 版的所有 JSON (只改 begin_year / end_year / epoch / logname)
# ============================================================================
echo "==================== [0] 生成 debug JSON ===================="
mkdir -p "$DEBUG_CONF_DIR"

python - <<PY
import json
from pathlib import Path

src_dir = Path("$SRC_CONF_DIR")
dst_dir = Path("$DEBUG_CONF_DIR")
dst_dir.mkdir(parents=True, exist_ok=True)

begin_year_s = "$BEGIN_YEAR"
end_year_s   = "$END_YEAR"
epoch        = $EPOCH
suffix       = "$DEBUG_LOG_SUFFIX"

for src in sorted(src_dir.glob("*.json")):
    data = json.loads(src.read_text())
    if begin_year_s:
        data["begin_year"] = int(begin_year_s)
    if end_year_s:
        data["end_year"] = int(end_year_s)
    data["epoch"] = epoch
    if "logname" in data:
        data["logname"] = data["logname"] + "_" + suffix
    if "influence_path" in data and ("Please enter" in data["influence_path"] or not data["influence_path"]):
        Path("$LOG_ROOT/stkec_influence_debug").mkdir(parents=True, exist_ok=True)
        data["influence_path"] = "$LOG_ROOT/stkec_influence_debug/"
    if data.get("method") == "EAC":
        data["train"] = 1
        data["auto_test"] = 0
    dst = dst_dir / src.name
    dst.write_text(json.dumps(data, indent=4))
    print(f"  [gen] {dst}  (begin={data['begin_year']}, end={data['end_year']}, epoch={epoch}, logname={data.get('logname')})")
PY


BEGIN_YEAR_EFF=$(python - <<PY
import json
print(json.load(open("$DEBUG_CONF_DIR/eac.json"))["begin_year"])
PY
)

# # ============================================================================
# # 1. Retrain
# # ============================================================================
# echo "==================== [1/8] Retrain ===================="
# python main.py --conf "$DEBUG_CONF_DIR/retrain_st_pems03.json" --gpuid "$GPU" --seed "$SEED_MAIN"


# # ============================================================================
# # 2. AutoLink: retrain 首年权重 → pretrain debug 目录
# # ============================================================================
# echo "==================== [2/8] AutoLink retrain→pretrain ===================="
# src_dir="$LOG_ROOT/retrain_st_pems03_${DEBUG_LOG_SUFFIX}-${SEED_MAIN}/${BEGIN_YEAR_EFF}"
# dst_dir="$LOG_ROOT/pretrain_st_pems03_${DEBUG_LOG_SUFFIX}-${SEED_MAIN}/${BEGIN_YEAR_EFF}"

# if [ ! -d "$src_dir" ] || [ -z "$(ls -A "$src_dir"/*.pkl 2>/dev/null)" ]; then
#     echo "  [error] 找不到 $src_dir/*.pkl，retrain 可能没产出权重"
#     exit 1
# fi
# mkdir -p "$dst_dir"
# rm -f "$dst_dir"/*.pkl
# for f in "$src_dir"/*.pkl; do
#     ln -sf "$(readlink -f "$f")" "$dst_dir/$(basename "$f")"
# done
# echo "  [ok] $src_dir/*.pkl -> $dst_dir/"


# # ============================================================================
# # 3. Pretrain (仅测试)
# # ============================================================================
# echo "==================== [3/8] Pretrain ===================="
# first_year_pkl=$(ls "$src_dir"/*.pkl | head -n 1)
# python main.py --conf "$DEBUG_CONF_DIR/pretrain_st_pems03.json" \
#     --load_first_year 1 \
#     --first_year_model_path "$first_year_pkl" \
#     --gpuid "$GPU" --seed "$SEED_MAIN"


# # ============================================================================
# # 4. Online-NN
# # ============================================================================
# echo "==================== [4/8] Online-NN ===================="
# python main.py --conf "$DEBUG_CONF_DIR/oneline_st_nn_pems03.json" --gpuid "$GPU" --seed "$SEED_MAIN"


# # ============================================================================
# # 5. Online-AN
# # ============================================================================
# echo "==================== [5/8] Online-AN ===================="
# python main.py --conf "$DEBUG_CONF_DIR/oneline_st_an_pems03.json" --gpuid "$GPU" --seed "$SEED_ALT"


# # ============================================================================
# # 6. TrafficStream
# # ============================================================================
# echo "==================== [6/8] TrafficStream ===================="
# python main.py --conf "$DEBUG_CONF_DIR/trafficstream.json" --gpuid "$GPU" --seed "$SEED_MAIN"


# # ============================================================================
# # 7. STKEC
# # ============================================================================
echo "==================== [7/8] STKEC ===================="
python stkec_main.py --conf "$DEBUG_CONF_DIR/stkec.json" --gpuid "$GPU" --seed "$SEED_STKEC"


# ============================================================================
# 8. EAC (主方法)
# ============================================================================
echo "==================== [8/8] EAC ===================="
python main.py --conf "$DEBUG_CONF_DIR/eac.json" --gpuid "$GPU" --seed "$SEED_ALT"


echo ""
echo "==================== ALL DEBUG RUNS PASSED ===================="
echo "Debug configs dir : $DEBUG_CONF_DIR"
echo "Debug logs prefix : $LOG_ROOT/*_${DEBUG_LOG_SUFFIX}-*"
echo "清理 debug 产物 :  rm -rf $DEBUG_CONF_DIR $LOG_ROOT/*_${DEBUG_LOG_SUFFIX}-*"
 