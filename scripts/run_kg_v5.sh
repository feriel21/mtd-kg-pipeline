#!/bin/bash
#SBATCH --job-name=kg_v5_verify
#SBATCH --partition=convergence
#SBATCH --gres=gpu:a100_3g.40gb:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# ─────────────────────────────────────────────────────────────────────
# NOTES:
#   - Removed "a100_7g.80gb" constraint → accepts ANY available GPU
#   - Reduced mem 64G→32G (7B bf16 ≈ 14GB VRAM + overhead)
#   - Reduced time 12h→2h (242 triples ≈ 20-40 min)
#   - Reduced cpus 8→4 (no parallel data loading needed)
#   - These changes should move you from Feb 25 → much sooner
#
#   If still slow, try:
#     #SBATCH --partition=convergence,autre_partition  (if available)
#     #SBATCH --time=01:00:00
#     #SBATCH --mem=24G
#
#   If OOM on smaller GPUs, switch MODEL to 3B or 1.5B below.
# ─────────────────────────────────────────────────────────────────────

set -euo pipefail

# ── Config ──
VENV_DIR="$HOME/kg_test/venv"
WORKDIR="$HOME/kg_test"

# Model — 7B fits on any GPU with ≥16GB VRAM in bf16
# If you land on a GPU with <16GB, uncomment a smaller model
MODEL="Qwen/Qwen2.5-7B-Instruct"
# MODEL="Qwen/Qwen2.5-3B-Instruct"    # fallback: ~7GB VRAM
# MODEL="Qwen/Qwen2.5-1.5B-Instruct"  # fallback: ~4GB VRAM

# Paths (adjust if your files are elsewhere)
INPUT_RAW="output/step4/raw_triples_v4.jsonl"
VERIFIED="output/step4/verified_triples_v5.jsonl"
OUTDIR="output/step4"
EVAL_OUTDIR="output/eval_v5"

# Scripts
VERIFY_SCRIPT="src_/02b_verify_triples.py"
CLEAN_SCRIPT="src_/03_validate_and_clean_v5.py"
EVAL_SCRIPT="src_/04_visualize_and_compare_v4.py"
AUDIT_SCRIPT="src_/audit_verification.py"

# ── Environment ──
mkdir -p "$WORKDIR/logs"
cd "$WORKDIR"

echo "══════════════════════════════════════════════════"
echo "  KG v5 Verification Pipeline"
echo "  Started: $(date)"
echo "  Host:    $(hostname)"
echo "  Model:   $MODEL"
echo "══════════════════════════════════════════════════"

source "$VENV_DIR/bin/activate"
python -V
nvidia-smi || true

export HF_HOME="${HF_HOME:-$WORKDIR/.hf}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
mkdir -p "$HF_HOME" "$TRANSFORMERS_CACHE"

# ════════════════════════════════════════════════════════════════
#  STEP 1: VERIFY triples with CoT + 3-tier (Qwen 7B)
# ════════════════════════════════════════════════════════════════
echo ""
echo "== STEP 1: VERIFICATION =="
echo "   Input:  $INPUT_RAW"
echo "   Output: $VERIFIED"

python -u "$VERIFY_SCRIPT" \
    --input  "$INPUT_RAW" \
    --output "$VERIFIED" \
    --backend hf \
    --model "$MODEL"

echo "== Verification done: $(date) =="

# ════════════════════════════════════════════════════════════════
#  STEP 2: CLEAN with verification-aware filtering
# ════════════════════════════════════════════════════════════════
echo ""
echo "== STEP 2: CLEAN (policy sweep) =="

# Run all 4 policies so you can compare
for policy in strict normal relaxed off; do
    echo ""
    echo "--- Policy: $policy ---"
    POLICY_DIR="${OUTDIR}_${policy}"
    mkdir -p "$POLICY_DIR"

    python -u "$CLEAN_SCRIPT" \
        --input  "$VERIFIED" \
        --outdir "$POLICY_DIR" \
        --verif-policy "$policy"
done

echo ""
echo "== Cleaning done: $(date) =="

# ════════════════════════════════════════════════════════════════
#  STEP 3: AUDIT (print sample for manual review)
# ════════════════════════════════════════════════════════════════
echo ""
echo "== STEP 3: AUDIT SAMPLE =="

AUDIT_FILE="$OUTDIR/verification_audit_v5.jsonl"
if [[ -f "$AUDIT_FILE" ]]; then
    python -u "$AUDIT_SCRIPT" \
        --audit "$AUDIT_FILE" \
        --sample 25 --verdict NOT_SUPPORTED
else
    # Try the path the verify script actually wrote to
    ALT_AUDIT="$(dirname "$VERIFIED")/verification_audit_v5.jsonl"
    if [[ -f "$ALT_AUDIT" ]]; then
        python -u "$AUDIT_SCRIPT" \
            --audit "$ALT_AUDIT" \
            --sample 25 --verdict NOT_SUPPORTED
    else
        echo "   Audit file not found, skipping."
    fi
fi

# ════════════════════════════════════════════════════════════════
#  STEP 4: EVALUATION (on 'normal' policy output)
# ════════════════════════════════════════════════════════════════
echo ""
echo "== STEP 4: EVALUATION =="

# Find the canonical triples from 'normal' policy
CANONICAL="${OUTDIR}_normal/canonical_triples_v5.jsonl"

if [[ ! -f "$CANONICAL" ]]; then
    # Fallback: maybe the script wrote v4 filename
    CANONICAL="${OUTDIR}_normal/canonical_triples_v4.jsonl"
fi

if [[ ! -f "$CANONICAL" ]]; then
    echo "   WARNING: canonical triples not found in ${OUTDIR}_normal/"
    echo "   Listing available files:"
    ls -la "${OUTDIR}_normal/" 2>/dev/null || echo "   Directory doesn't exist"
    echo "   Skipping evaluation."
else
    mkdir -p "$EVAL_OUTDIR"
    python -u "$EVAL_SCRIPT" \
        --run4 "$CANONICAL" \
        --outdir "$EVAL_OUTDIR" \
        --rdf-run run5
fi

echo ""
echo "══════════════════════════════════════════════════"
echo "  Pipeline finished: $(date)"
echo "══════════════════════════════════════════════════"

# ════════════════════════════════════════════════════════════════
#  SUMMARY: print key metrics from each policy
# ════════════════════════════════════════════════════════════════
echo ""
echo "== POLICY COMPARISON =="
echo "────────────────────────────────────────────────"
printf "%-10s %8s %10s %10s\n" "Policy" "Triples" "LB_Recall" "Halluc%"
echo "────────────────────────────────────────────────"

for policy in strict normal relaxed off; do
    STATS="${OUTDIR}_${policy}/cleaning_stats_v5.json"
    if [[ -f "$STATS" ]]; then
        triples=$(python3 -c "import json; d=json.load(open('$STATS')); print(d.get('output_triples','?'))")
        lb=$(python3 -c "import json; d=json.load(open('$STATS')); print(d.get('lb_recall','?'))")
        hr=$(python3 -c "import json; d=json.load(open('$STATS')); print(f\"{d.get('final_halluc_rate',0):.1%}\")")
        printf "%-10s %8s %10s %10s\n" "$policy" "$triples" "$lb" "$hr"
    else
        printf "%-10s %8s %10s %10s\n" "$policy" "—" "—" "—"
    fi
done
echo "────────────────────────────────────────────────"