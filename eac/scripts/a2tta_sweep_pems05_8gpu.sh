#!/bin/bash
# ===========================================================================
# A2TTA-Lite 8-GPU hyperparameter sweep on PEMS05 (current Online-AN backbone).
# ---------------------------------------------------------------------------
# Goal: tighten the active-selection regime until a2tta_lite vs tta_random/
# tta_error meaningfully differ. Plus a follow-on lr/steps/lambda/scoring
# ablation. ~490 jobs × 15-20 min / 8 GPUs ≈ 15-20 hours wallclock.
# ---------------------------------------------------------------------------
# Output:
#   run_logs/sweep_a2tta_pems05_<ts>/
#       csv/<TAG>.csv          — one CSV per (method, config, seed)
#       logs/<TAG>.log         — stdout/stderr per job
#       results.csv            — concat of all per-job CSVs (post-run)
#       dispatcher.log         — top-level dispatcher log
# ---------------------------------------------------------------------------
# Usage:
#   cd eac/
#   NOHUP=1 bash scripts/a2tta_sweep_pems05_8gpu.sh
#   tail -f run_logs/sweep_a2tta_pems05_*/dispatcher.log
#
# Resume after crash / partial completion:
#   RESUME_FROM=run_logs/sweep_a2tta_pems05_<ts> bash scripts/a2tta_sweep_pems05_8gpu.sh
#
# Tweak knobs:
#   NUM_GPUS=8 SEEDS="51 52 53" WARMUP_EPOCHS=1 bash scripts/a2tta_sweep_pems05_8gpu.sh
# ===========================================================================

set -euo pipefail
cd "$(dirname "$0")/.."

# ---------------------------------------------------------------------------
# Knobs
# ---------------------------------------------------------------------------
NUM_GPUS=${NUM_GPUS:-8}
SEEDS=${SEEDS:-"51 52 53 54 55"}
WARMUP_EPOCHS=${WARMUP_EPOCHS:-1}    # 1 keeps the sweep ~50% faster than 3
EVAL_BATCH=${EVAL_BATCH:-64}
RESUME_FROM=${RESUME_FROM:-""}
DRY_RUN=${DRY_RUN:-0}                # 1: only print job count/tags, don't run

if [[ -n "$RESUME_FROM" ]]; then
    SWEEP_ROOT=$RESUME_FROM
    [[ -d "$SWEEP_ROOT" ]] || { echo "[err] RESUME_FROM=$SWEEP_ROOT not found"; exit 1; }
    echo "[sweep] RESUMING under $SWEEP_ROOT"
else
    TS=$(date +%Y%m%d_%H%M%S)
    SWEEP_ROOT=run_logs/sweep_a2tta_pems05_$TS
fi
mkdir -p "$SWEEP_ROOT/csv" "$SWEEP_ROOT/logs"

# ---------------------------------------------------------------------------
# nohup self-bg
# ---------------------------------------------------------------------------
if [[ "${NOHUP:-0}" == "1" && -z "${EAC_BG:-}" ]]; then
    EAC_BG=1 nohup bash "$0" "$@" > "$SWEEP_ROOT/dispatcher.log" 2>&1 &
    BG_PID=$!
    echo "[nohup] PID=$BG_PID"
    echo "[nohup] tail -f $SWEEP_ROOT/dispatcher.log"
    echo "[nohup] sweep root: $SWEEP_ROOT"
    exit 0
fi

echo "[sweep] root=$SWEEP_ROOT  num_gpus=$NUM_GPUS  seeds='$SEEDS'  warmup=$WARMUP_EPOCHS"

# ---------------------------------------------------------------------------
# GPU semaphore (FIFO)
# ---------------------------------------------------------------------------
FIFO=$SWEEP_ROOT/.gpu_fifo
[[ -p "$FIFO" ]] || mkfifo "$FIFO"
exec 3<>"$FIFO"
for g in $(seq 0 $((NUM_GPUS-1))); do echo "$g" >&3; done

JOBS_ENQ=0
JOBS_SKIP=0

run_job() {
    local tag=$1; shift
    local csv="$SWEEP_ROOT/csv/$tag.csv"
    local log="$SWEEP_ROOT/logs/$tag.log"
    JOBS_ENQ=$((JOBS_ENQ+1))

    if [[ -f "$csv" ]]; then
        JOBS_SKIP=$((JOBS_SKIP+1))
        echo "[$JOBS_ENQ skip] $tag (csv exists)"
        return
    fi

    if [[ "$DRY_RUN" == "1" ]]; then
        echo "[$JOBS_ENQ dry] $tag $*"
        return
    fi

    # Acquire GPU (BLOCKS in the dispatcher until a slot is free)
    local gpu
    read -u 3 gpu

    {
        local start_ts=$(date +%s)
        echo "[gpu=$gpu start $(date +%H:%M:%S)] $tag" >> "$SWEEP_ROOT/dispatcher.log"
        python a2tta_main.py "$@" \
            --gpuid "$gpu" \
            --csv_path "$csv" \
            --eval_batch_size "$EVAL_BATCH" \
            --warmup_epochs "$WARMUP_EPOCHS" \
            >> "$log" 2>&1
        local rc=$?
        local end_ts=$(date +%s)
        local elapsed=$((end_ts - start_ts))
        echo "[gpu=$gpu done  $(date +%H:%M:%S) rc=$rc t=${elapsed}s] $tag" >> "$SWEEP_ROOT/dispatcher.log"
        if [[ "$rc" != "0" ]]; then
            echo "[$JOBS_ENQ FAIL rc=$rc] $tag — see $log" >> "$SWEEP_ROOT/dispatcher.log"
            # Remove partial CSV so the job is retried on resume.
            rm -f "$csv"
        fi
        # Return GPU to the pool (always, even on failure)
        echo "$gpu" >&3
    } &
}

# Args shared by every job — kept stable to make stages comparable.
COMMON_ARGS=(
    --conf conf/PEMS05/a2tta_lite_pems05.json
    --backbone_ckpt_logname oneline_st_an_pems05
    --backbone_ckpt_logname_fallback retrain_st_pems05
    --backbone_method TrafficStream
    --freeze_backbone 1
    --node_emb_dim 16
)

# ============================================================================
# Stage A: budget × pool sweep (the core ask).
# 3 methods × 10 (pool, budget_frac) × 5 seeds = 150 jobs.
# Spans #selected ≈ 3 .. 256 to make selection quality bind.
# ============================================================================
echo "===== Stage A: (budget_frac, pool_size) sweep ====="
for METHOD in a2tta_lite tta_random tta_error; do
  for SEED in $SEEDS; do
    for PB in \
        "64,0.05" "64,0.25" \
        "128,0.05" "128,0.25" \
        "256,0.05" "256,0.10" "256,0.25" \
        "512,0.05" "512,0.25" "512,0.50" ; do
        POOL=${PB%,*}; BUD=${PB#*,}
        TAG="A_${METHOD}_p${POOL}_b${BUD}_s${SEED}"
        run_job "$TAG" "${COMMON_ARGS[@]}" \
            --logname "sweep_${TAG}" \
            --method "$METHOD" \
            --seed "$SEED" \
            --candidate_pool_size "$POOL" \
            --budget_frac "$BUD" \
            --adapt_lr 3e-4 \
            --adapt_steps 1 \
            --adapter_hidden_dim 64 \
            --lambda_cons 0.05 \
            --lambda_reg 1e-4
    done
  done
done

# ============================================================================
# Stage B: lr × adapt_steps at 3 hand-picked hot (pool, budget) configs.
# 2 methods × 3 (pool,bud) × 3 lr × 3 steps × 5 seeds = 270 jobs.
# ============================================================================
echo "===== Stage B: (adapt_lr, adapt_steps) sweep ====="
for METHOD in a2tta_lite tta_random; do
  for SEED in $SEEDS; do
    for PB in "128,0.05" "256,0.10" "512,0.25"; do
        POOL=${PB%,*}; BUD=${PB#*,}
        for LR in 1e-4 3e-4 1e-3; do
          for STEPS in 1 2 3; do
            TAG="B_${METHOD}_p${POOL}_b${BUD}_lr${LR}_st${STEPS}_s${SEED}"
            run_job "$TAG" "${COMMON_ARGS[@]}" \
                --logname "sweep_${TAG}" \
                --method "$METHOD" \
                --seed "$SEED" \
                --candidate_pool_size "$POOL" \
                --budget_frac "$BUD" \
                --adapt_lr "$LR" \
                --adapt_steps "$STEPS" \
                --adapter_hidden_dim 64 \
                --lambda_cons 0.05 \
                --lambda_reg 1e-4
          done
        done
    done
  done
done

# ============================================================================
# Stage C: λ_cons × hidden_dim (a2tta_lite only).
# 4 lcons × 2 hid × 5 seeds = 40 jobs.
# Anchored at (pool=256, budget=0.10) — usually the discriminating regime.
# ============================================================================
echo "===== Stage C: (lambda_cons, hidden_dim) ====="
for SEED in $SEEDS; do
  for LCONS in 0.0 0.05 0.10 0.20; do
    for HID in 64 128; do
      TAG="C_lcons${LCONS}_hid${HID}_s${SEED}"
      run_job "$TAG" "${COMMON_ARGS[@]}" \
          --logname "sweep_${TAG}" \
          --method a2tta_lite \
          --seed "$SEED" \
          --candidate_pool_size 256 \
          --budget_frac 0.10 \
          --adapt_lr 3e-4 \
          --adapt_steps 1 \
          --adapter_hidden_dim "$HID" \
          --lambda_cons "$LCONS" \
          --lambda_reg 1e-4
    done
  done
done

# ============================================================================
# Stage D: active scoring weight ablation (a2tta_lite only).
# 6 weight schemes × 5 seeds = 30 jobs.
# Tests whether each scoring component (err / unc / shift / recency) helps.
# ============================================================================
echo "===== Stage D: active scoring weight ablation ====="
for SEED in $SEEDS; do
  # NAME : (w_err, w_unc, w_shift, w_recency)
  for SPEC in \
      "default:1.0,0.3,0.3,0.1" \
      "err_only:1.0,0.0,0.0,0.0" \
      "unc_only:0.0,1.0,0.0,0.0" \
      "shift_only:0.0,0.0,1.0,0.0" \
      "recency_only:0.0,0.0,0.0,1.0" \
      "all_eq:1.0,1.0,1.0,1.0" ; do
    NAME=${SPEC%%:*}
    W=${SPEC#*:}
    IFS=',' read -r WE WU WS WR <<< "$W"
    TAG="D_${NAME}_s${SEED}"
    run_job "$TAG" "${COMMON_ARGS[@]}" \
        --logname "sweep_${TAG}" \
        --method a2tta_lite \
        --seed "$SEED" \
        --candidate_pool_size 256 \
        --budget_frac 0.10 \
        --adapt_lr 3e-4 \
        --adapt_steps 1 \
        --adapter_hidden_dim 64 \
        --lambda_cons 0.05 \
        --lambda_reg 1e-4 \
        --w_err "$WE" --w_unc "$WU" --w_shift "$WS" --w_recency "$WR"
  done
done

echo "[sweep] enqueued=$JOBS_ENQ skipped=$JOBS_SKIP"
echo "[sweep] waiting for outstanding jobs..."
wait
exec 3>&-
rm -f "$FIFO"

# ---------------------------------------------------------------------------
# Concatenate per-job CSVs into a single results table
# ---------------------------------------------------------------------------
RESULTS="$SWEEP_ROOT/results.csv"
# Portable glob (works on bash 3.2 / macOS too — `mapfile` is bash 4+ only)
shopt -s nullglob
ALL_CSV=( "$SWEEP_ROOT/csv/"*.csv )
shopt -u nullglob
if [[ ${#ALL_CSV[@]} -gt 0 ]]; then
    head -n 1 "${ALL_CSV[0]}" > "$RESULTS"
    for f in "${ALL_CSV[@]}"; do
        tail -n +2 "$f" >> "$RESULTS"
    done
    echo "[sweep] combined CSV → $RESULTS ($(wc -l < "$RESULTS") lines, ${#ALL_CSV[@]} jobs)"
else
    echo "[sweep] no per-job CSVs found under $SWEEP_ROOT/csv/"
fi

# ---------------------------------------------------------------------------
# Print the top configurations
# ---------------------------------------------------------------------------
if [[ -f "$RESULTS" ]]; then
    python scripts/a2tta_sweep_summarize.py "$RESULTS" --top 15 || true
fi

echo "==================== A2TTA-Lite SWEEP DONE ===================="
echo "Sweep root  : $SWEEP_ROOT"
echo "Results CSV : $RESULTS"
echo "Per-job logs: $SWEEP_ROOT/logs/"
