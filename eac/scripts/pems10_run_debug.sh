#!/bin/bash
# ===========================================================================
# PEMS10 (xxltrafficdata) 冒烟测试脚本 (pems10_run.sh 的 Debug 版)
# ---------------------------------------------------------------------------
# 目的：最快跑通全链路 (retrain → pretrain → online-nn → online-an →
#       trafficstream → stkec → eac)，验证每一段代码路径都不挂。
#
# 做了以下 “瘦身”：
#   * 只跑 1 个 seed (42 主 / 51 alt / 47 stkec)
#   * epoch=1                          (训练循环只跑 1 轮)
#   * 每个方法的 JSON 被临时克隆到 /tmp/eac_debug_conf/PEMS10/，只改少数字段
#   * 不触碰你原来的 conf/PEMS10/*.json
#
# 数据路径: ../xxltrafficdata/preprocessed/pems10/{RawData,FastData,graph}/
# 首年: 2006 (= begin_year)
# 日志目录: log/PEMS10/ (debug 产物 logname 加 _debug 后缀)
# ---------------------------------------------------------------------------
# 用法：
#   cd eac/
#   bash scripts/pems10_run_debug.sh
# ===========================================================================

set -euo pipefail
cd "$(dirname "$0")/.."

# ---------------------------------------------------------------------------
# 后台模式: 用 `NOHUP=1 bash scripts/pems10_run_debug.sh` 触发
# 自动 nohup 后台跑, 日志落到 eac/run_logs/pems10_debug_<时间戳>.log
# ---------------------------------------------------------------------------
if [[ "${NOHUP:-0}" == "1" && -z "${EAC_BG:-}" ]]; then
    mkdir -p run_logs
    LOG_FILE="run_logs/pems10_debug_$(date +%Y%m%d_%H%M%S).log"
    echo "[nohup] backgrounding to $LOG_FILE"
    EAC_BG=1 nohup bash "$0" "$@" > "$LOG_FILE" 2>&1 &
    BG_PID=$!
    echo "[nohup] PID=$BG_PID"
    echo "[nohup] tail -f $LOG_FILE"
    exit 0
fi

DEBUG_CONF_DIR="/tmp/eac_debug_conf/PEMS10"
DEBUG_LOG_SUFFIX="debug"          # logname 加后缀，避免和正式日志撞目录
DEBUG_EPOCH=${DEBUG_EPOCH:-1}
FIRST_YEAR=2006
LOG_ROOT="log/PEMS10"

SEED_MAIN=42
SEED_ALT=51
SEED_STKEC=47

GPU=${GPU:-3}                     # 可以用 GPU=0 bash ... 覆盖

# ============================================================================
# 0. 生成 debug 版的所有 JSON (只改 epoch / logname / influence_path)
# ============================================================================
echo "==================== [0] 生成 debug JSON ===================="
mkdir -p "$DEBUG_CONF_DIR"

python - <<PY
import json
from pathlib import Path

src_dir = Path("conf/PEMS10")
dst_dir = Path("$DEBUG_CONF_DIR")
dst_dir.mkdir(parents=True, exist_ok=True)

epoch    = $DEBUG_EPOCH
suffix   = "$DEBUG_LOG_SUFFIX"
log_root = "$LOG_ROOT"

for src in sorted(src_dir.glob("*.json")):
    data = json.loads(src.read_text())
    data["epoch"]    = epoch
    if "logname" in data:
        data["logname"] = data["logname"] + "_" + suffix
    if "influence_path" in data and ("Please enter" in data["influence_path"] or not data["influence_path"]):
        infl = f"{log_root}/stkec_influence_debug"
        Path(infl).mkdir(parents=True, exist_ok=True)
        data["influence_path"] = infl + "/"
    if data.get("method") == "EAC":
        data["train"] = 1
        data["auto_test"] = 0
    dst = dst_dir / src.name
    dst.write_text(json.dumps(data, indent=4))
    print(f"  [gen] {dst}  (epoch={epoch}, logname={data.get('logname')})")
PY


# ============================================================================
# 1. Retrain (从零训练, 至少首年权重)
# ============================================================================
echo "==================== [1/8] Retrain ===================="
python main.py --conf "$DEBUG_CONF_DIR/retrain_st_pems10.json" --gpuid "$GPU" --seed "$SEED_MAIN"


# ============================================================================
# 2. AutoLink: retrain 的首年权重 → pretrain debug 目录
# ============================================================================
echo "==================== [2/8] AutoLink retrain→pretrain ===================="
src_dir="${LOG_ROOT}/retrain_st_pems10_${DEBUG_LOG_SUFFIX}-${SEED_MAIN}/${FIRST_YEAR}"
dst_dir="${LOG_ROOT}/pretrain_st_pems10_${DEBUG_LOG_SUFFIX}-${SEED_MAIN}/${FIRST_YEAR}"

if [ ! -d "$src_dir" ] || [ -z "$(ls -A "$src_dir"/*.pkl 2>/dev/null)" ]; then
    echo "  [error] 找不到 $src_dir/*.pkl，retrain 可能没产出权重"
    exit 1
fi
mkdir -p "$dst_dir"
rm -f "$dst_dir"/*.pkl
for f in "$src_dir"/*.pkl; do
    ln -sf "$(readlink -f "$f")" "$dst_dir/$(basename "$f")"
done
echo "  [ok] $src_dir/*.pkl -> $dst_dir/"


# ============================================================================
# 3. Pretrain (仅测试)
# ============================================================================
echo "==================== [3/8] Pretrain ===================="
first_year_pkl=$(ls "$src_dir"/*.pkl | head -n 1)
python main.py --conf "$DEBUG_CONF_DIR/pretrain_st_pems10.json" \
    --load_first_year 1 \
    --first_year_model_path "$first_year_pkl" \
    --gpuid "$GPU" --seed "$SEED_MAIN"


# ============================================================================
# 4. Online-NN
# ============================================================================
echo "==================== [4/8] Online-NN ===================="
python main.py --conf "$DEBUG_CONF_DIR/oneline_st_nn_pems10.json" --gpuid "$GPU" --seed "$SEED_MAIN"


# ============================================================================
# 5. Online-AN
# ============================================================================
echo "==================== [5/8] Online-AN ===================="
python main.py --conf "$DEBUG_CONF_DIR/oneline_st_an_pems10.json" --gpuid "$GPU" --seed "$SEED_ALT"


# ============================================================================
# 6. TrafficStream
# ============================================================================
echo "==================== [6/8] TrafficStream ===================="
python main.py --conf "$DEBUG_CONF_DIR/trafficstream.json" --gpuid "$GPU" --seed "$SEED_MAIN"


# ============================================================================
# 7. STKEC
# ============================================================================
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
echo "Debug logs prefix : ${LOG_ROOT}/*_${DEBUG_LOG_SUFFIX}-*"
echo "想清理 debug 产物：  rm -rf $DEBUG_CONF_DIR ${LOG_ROOT}/*_${DEBUG_LOG_SUFFIX}-*"
