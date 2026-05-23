#!/bin/bash
# ===========================================================================
# PEMS-Stream 冒烟测试脚本 (pems_run.sh 的 Debug 版)
# ---------------------------------------------------------------------------
# 目的：最快跑通全链路 (retrain → pretrain → online-nn → online-an →
#       trafficstream → stkec → eac)，验证每一段代码路径都不挂。
#
# 做了以下 “瘦身”：
#   * 只跑 1 个 seed (42 主 / 51 alt / 47 stkec)
#   * begin_year=2011, end_year=2012   (2 年就能覆盖 首年训练 + 跨年续训 两条路径)
#   * epoch=1                          (训练循环只跑 1 轮)
#   * 每个方法的 JSON 被临时克隆到 /tmp/eac_debug_conf/，只改上面 3 个字段
#   * 不触碰你原来的 conf/PEMS/*.json
#
# 用法：
#   cd eac/
#   bash scripts/pems_run_debug.sh
# ===========================================================================

set -euo pipefail
cd "$(dirname "$0")/.."

DEBUG_CONF_DIR="/tmp/eac_debug_conf/PEMS"
DEBUG_LOG_SUFFIX="debug"      # logname 加后缀，避免和正式日志撞目录
DEBUG_END_YEAR=2017           # 跑满 2011~2017，验证全流程代码路径
DEBUG_EPOCH=1

SEED_MAIN=42
SEED_ALT=51
SEED_STKEC=47

GPU=${GPU:-2}                 # 可以用 GPU=0 bash ... 覆盖

# ============================================================================
# 0. 生成 debug 版的所有 JSON (只改 end_year / epoch / logname)
# ============================================================================
echo "==================== [0] 生成 debug JSON ===================="
mkdir -p "$DEBUG_CONF_DIR"

python - <<PY
import json, shutil
from pathlib import Path

src_dir = Path("conf/PEMS")
dst_dir = Path("$DEBUG_CONF_DIR")
dst_dir.mkdir(parents=True, exist_ok=True)

end_year = $DEBUG_END_YEAR
epoch    = $DEBUG_EPOCH
suffix   = "$DEBUG_LOG_SUFFIX"

for src in sorted(src_dir.glob("*.json")):
    data = json.loads(src.read_text())
    data["end_year"] = end_year
    data["epoch"]    = epoch
    if "logname" in data:
        data["logname"] = data["logname"] + "_" + suffix
    # 针对 STKEC 的 influence_path 占位也顺手修掉
    if "influence_path" in data and ("Please enter" in data["influence_path"] or not data["influence_path"]):
        Path("log/PEMS/stkec_influence_debug").mkdir(parents=True, exist_ok=True)
        data["influence_path"] = "log/PEMS/stkec_influence_debug/"
    # EAC: debug 必须真的训，否则它想 load 还不存在的权重
    if data.get("method") == "EAC":
        data["train"] = 1
        data["auto_test"] = 0
    # 同理 pretrain 的 JSON auto_test=1 没关系，它只测，不训
    dst = dst_dir / src.name
    dst.write_text(json.dumps(data, indent=4))
    print(f"  [gen] {dst}  (end_year={end_year}, epoch={epoch}, logname={data.get('logname')})")
PY


# ============================================================================
# 1. Retrain (从零训练 2011, 2012)
# ============================================================================
echo "==================== [1/8] Retrain ===================="
python main.py --conf "$DEBUG_CONF_DIR/retrain_st_pems.json" --gpuid "$GPU" --seed "$SEED_MAIN"


# ============================================================================
# 2. AutoLink: retrain 的 2011 权重 → pretrain debug 目录
# ============================================================================
echo "==================== [2/8] AutoLink retrain→pretrain ===================="
src_dir="log/PEMS/retrain_st_pems_${DEBUG_LOG_SUFFIX}-${SEED_MAIN}/2011"
dst_dir="log/PEMS/pretrain_st_pems_${DEBUG_LOG_SUFFIX}-${SEED_MAIN}/2011"

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
python main.py --conf "$DEBUG_CONF_DIR/pretrain_st_pems.json" \
    --load_first_year 1 \
    --first_year_model_path "$first_year_pkl" \
    --gpuid "$GPU" --seed "$SEED_MAIN"


# ============================================================================
# 4. Online-NN
# ============================================================================
echo "==================== [4/8] Online-NN ===================="
python main.py --conf "$DEBUG_CONF_DIR/oneline_st_nn_pems.json" --gpuid "$GPU" --seed "$SEED_MAIN"


# ============================================================================
# 5. Online-AN
# ============================================================================
echo "==================== [5/8] Online-AN ===================="
python main.py --conf "$DEBUG_CONF_DIR/oneline_st_an_pems.json" --gpuid "$GPU" --seed "$SEED_ALT"


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
echo "==================== ✅ ALL DEBUG RUNS PASSED ===================="
echo "Debug configs dir : $DEBUG_CONF_DIR"
echo "Debug logs prefix : log/PEMS/*_${DEBUG_LOG_SUFFIX}-*"
echo "想清理 debug 产物：  rm -rf $DEBUG_CONF_DIR log/PEMS/*_${DEBUG_LOG_SUFFIX}-*"
