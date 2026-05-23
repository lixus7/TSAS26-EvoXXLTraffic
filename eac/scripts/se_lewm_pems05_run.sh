#!/bin/bash
# ===========================================================================
# SE-LeWM (Sensor-Evolving Latent World Model) on PEMS05.
# ---------------------------------------------------------------------------
# Backbone : Online-AN per-year .pkl checkpoints (oneline_st_an_pems05-<seed>/<year>/*.pkl)
#            Falls back to retrain_st_pems05 if missing.
# Method   : freeze backbone + LeWM-inspired latent residual adapter:
#              - context_encoder(x) -> z_ctx
#              - graph latent transition rollout -> z_pred_seq
#              - residual_decoder(z_pred_seq) -> delta_y
#              - y_hat = y_base + sigmoid(gate) * delta_y      (gate init -3.0)
#              - L_latent: masked MSE between z_pred_seq and target_encoder(y)
#              - SIGReg on z_ctx and z_tgt_seq (random 1-D projections, ECF match)
# ---------------------------------------------------------------------------
# Output   : log/PEMS05/se_lewm_*-<seed>/*.log  (per-year metrics)
#            run_logs/se_lewm_pems05_results.csv  (per year × method × seed × horizon)
# Usage    :
#   cd eac/
#   bash scripts/se_lewm_pems05_run.sh                    # full matrix, default GPU
#   GPU=2 SEEDS="51" bash scripts/se_lewm_pems05_run.sh   # single seed on GPU 2
#   FAST_DEV_RUN=1 bash scripts/se_lewm_pems05_run.sh     # debug: 1 year, 4 batches
#   METHODS="se_lewm" bash scripts/se_lewm_pems05_run.sh  # only main method
#   NOHUP=1 bash scripts/se_lewm_pems05_run.sh            # background w/ log
# ===========================================================================

set -euo pipefail
cd "$(dirname "$0")/.."  # → eac/

# ---------------------------------------------------------------------------
# Background mode: NOHUP=1 bash scripts/se_lewm_pems05_run.sh
# ---------------------------------------------------------------------------
if [[ "${NOHUP:-0}" == "1" && -z "${EAC_BG:-}" ]]; then
    mkdir -p run_logs
    LOG_FILE="run_logs/se_lewm_pems05_$(date +%Y%m%d_%H%M%S).log"
    echo "[nohup] backgrounding to $LOG_FILE"
    EAC_BG=1 nohup bash "$0" "$@" > "$LOG_FILE" 2>&1 &
    echo "[nohup] PID=$!"
    echo "[nohup] tail -f $LOG_FILE"
    exit 0
fi

# ---------------------------------------------------------------------------
# Knobs (env-overridable)
# ---------------------------------------------------------------------------
GPU=${GPU:-1}
SEEDS=${SEEDS:-"51 52 53 54 55"}
BACKBONE_LOGNAME=${BACKBONE_LOGNAME:-"oneline_st_an_pems05"}
BACKBONE_FALLBACK=${BACKBONE_FALLBACK:-"retrain_st_pems05"}
# Method matrix:
#   backbone             : frozen base, no adapter (sanity)
#   residual_only        : residual decoder only (no latent / sigreg losses)
#   latent_only          : latent prediction loss, no sigreg
#   sigreg_only          : sigreg only, no latent prediction loss
#   se_lewm              : full (residual + latent + sigreg)
#   se_lewm_no_residual  : latent + sigreg, but emits y_base (ablation)
#   se_lewm_no_online    : warmup only, no online adapt (ablation)
METHODS=${METHODS:-"backbone residual_only latent_only sigreg_only se_lewm"}

# Adapter dims
Z_DIM=${Z_DIM:-64}
LATENT_HIDDEN_DIM=${LATENT_HIDDEN_DIM:-128}
NODE_EMB_DIM=${NODE_EMB_DIM:-16}
RESIDUAL_GATE_INIT=${RESIDUAL_GATE_INIT:--3.0}

# Loss weights
ALPHA_LATENT=${ALPHA_LATENT:-0.05}
LAMBDA_SIGREG=${LAMBDA_SIGREG:-0.01}
LAMBDA_DELTA=${LAMBDA_DELTA:-1e-4}

# SIGReg shape
NUM_PROJECTIONS=${NUM_PROJECTIONS:-128}
NUM_KNOTS=${NUM_KNOTS:-16}
MAX_SIGREG_SAMPLES=${MAX_SIGREG_SAMPLES:-4096}

# Schedule
WARMUP_EPOCHS=${WARMUP_EPOCHS:-3}
WARMUP_LR=${WARMUP_LR:-1e-3}
ADAPT_LR=${ADAPT_LR:-3e-4}
ADAPT_STEPS=${ADAPT_STEPS:-1}
ADAPT_EVERY=${ADAPT_EVERY:-1}

# Eval / output
EVAL_BATCH=${EVAL_BATCH:-64}
FAST_DEV_RUN=${FAST_DEV_RUN:-0}
CSV_PATH=${CSV_PATH:-"run_logs/se_lewm_pems05_results.csv"}

echo "[SE-LeWM] GPU=$GPU SEEDS='$SEEDS' METHODS='$METHODS'"
echo "[SE-LeWM] backbone=$BACKBONE_LOGNAME (fallback=$BACKBONE_FALLBACK)"
echo "[SE-LeWM] z_dim=$Z_DIM hidden=$LATENT_HIDDEN_DIM node_emb=$NODE_EMB_DIM gate0=$RESIDUAL_GATE_INIT"
echo "[SE-LeWM] alpha_latent=$ALPHA_LATENT lambda_sigreg=$LAMBDA_SIGREG lambda_delta=$LAMBDA_DELTA"
echo "[SE-LeWM] sigreg proj=$NUM_PROJECTIONS knots=$NUM_KNOTS max_samples=$MAX_SIGREG_SAMPLES"
echo "[SE-LeWM] warmup=$WARMUP_EPOCHS ep @ lr=$WARMUP_LR | adapt lr=$ADAPT_LR steps=$ADAPT_STEPS every=$ADAPT_EVERY"
echo "[SE-LeWM] csv=$CSV_PATH fast_dev_run=$FAST_DEV_RUN"

run_one() {
    local METHOD=$1
    local SEED=$2
    local LOGNAME="se_lewm_${METHOD}_pems05"
    echo "==================== [SE-LeWM] method=$METHOD seed=$SEED ===================="
    python se_lewm_main.py \
        --conf conf/PEMS05/se_lewm_pems05.json \
        --logname "$LOGNAME" \
        --method "$METHOD" \
        --dataset PEMS05 \
        --gpuid "$GPU" \
        --seed "$SEED" \
        --backbone_ckpt_logname "$BACKBONE_LOGNAME" \
        --backbone_ckpt_logname_fallback "$BACKBONE_FALLBACK" \
        --backbone_method TrafficStream \
        --freeze_backbone 1 \
        --z_dim "$Z_DIM" \
        --latent_hidden_dim "$LATENT_HIDDEN_DIM" \
        --node_emb_dim "$NODE_EMB_DIM" \
        --residual_gate_init "$RESIDUAL_GATE_INIT" \
        --alpha_latent "$ALPHA_LATENT" \
        --lambda_sigreg "$LAMBDA_SIGREG" \
        --lambda_delta "$LAMBDA_DELTA" \
        --num_projections "$NUM_PROJECTIONS" \
        --num_knots "$NUM_KNOTS" \
        --max_sigreg_samples "$MAX_SIGREG_SAMPLES" \
        --warmup_epochs "$WARMUP_EPOCHS" \
        --warmup_lr "$WARMUP_LR" \
        --adapt_lr "$ADAPT_LR" \
        --adapt_steps "$ADAPT_STEPS" \
        --adapt_every_batches "$ADAPT_EVERY" \
        --eval_batch_size "$EVAL_BATCH" \
        --csv_path "$CSV_PATH" \
        --fast_dev_run "$FAST_DEV_RUN"
}

for SEED in $SEEDS; do
    for METHOD in $METHODS; do
        run_one "$METHOD" "$SEED"
    done
done

echo "==================== SE-LeWM ALL DONE ===================="
echo "[SE-LeWM] Results CSV: $CSV_PATH"
