#!/bin/bash
# Master script to run all KNOT dataset construction phases
# Supports resuming from the last successful step.

set -euo pipefail

cd "$(dirname "$0")"

CHECKPOINT_DIR=".checkpoints"
mkdir -p "$CHECKPOINT_DIR"

# ── helpers ────────────────────────────────────────────────────────────────────
done_marker() { echo "$CHECKPOINT_DIR/$1.done"; }

is_done() { [ -f "$(done_marker "$1")" ]; }

mark_done() { touch "$(done_marker "$1")"; }

run_step() {
    local name="$1"
    local cmd="$2"

    if is_done "$name"; then
        echo "[SKIP] $name (already completed)"
        return 0
    fi

    echo "[RUN ] $name ..."
    if eval "$cmd"; then
        mark_done "$name"
        echo "[OK  ] $name"
    else
        echo "[FAIL] $name — fix the issue then re-run this script to resume."
        exit 1
    fi
}

# ── check API key ──────────────────────────────────────────────────────────────
if [ -z "${DEEPSEEK_API_KEY:-}" ]; then
    echo "ERROR: DEEPSEEK_API_KEY is not set."
    echo "       Run: export DEEPSEEK_API_KEY=your_key_here"
    exit 1
fi

echo "========================================"
echo "KNOT Dataset Construction Pipeline"
echo "Checkpoint dir: $CHECKPOINT_DIR"
echo "========================================"
echo "To reset a specific step, delete its checkpoint file, e.g.:"
echo "  rm $CHECKPOINT_DIR/phase1_entity_collection.done"
echo "To reset ALL steps:  rm -rf $CHECKPOINT_DIR"
echo "========================================"
echo ""

# ── Phase 1 ────────────────────────────────────────────────────────────────────
echo "[Phase 1] Entity Data"
run_step "phase1_entity_collection"   "python query_wikidata_entities.py"
run_step "phase1_entity_qa"           "python generate_entity_qa.py"

# ── Phase 2 ────────────────────────────────────────────────────────────────────
echo ""
echo "[Phase 2] Concept Data"
run_step "phase2_concept_list"        "python generate_concept_list.py"
run_step "phase2_concept_qa"          "python generate_concept_qa.py"

# ── Phase 3 ────────────────────────────────────────────────────────────────────
echo ""
echo "[Phase 3] Skill Data"
run_step "phase3_skill_qa"            "python generate_skill_qa.py"

# ── Phase 4 ────────────────────────────────────────────────────────────────────
echo ""
echo "[Phase 4] Deduplication & Verification"
run_step "phase4_dedup"               "python dedup.py"
run_step "phase4_verify"              "python verify_qa.py"

# ── Phase 5 ────────────────────────────────────────────────────────────────────
echo ""
echo "[Phase 5] Scoring"
run_step "phase5_entanglement"        "python compute_entanglement_scores.py"
run_step "phase5_kg_distance"         "python compute_kg_distance.py"
run_step "phase5_llm_judge"           "python compute_llm_judge_score.py"
run_step "phase5_finalize"            "python finalize_scores.py"

# ── Phase 6 ────────────────────────────────────────────────────────────────────
echo ""
echo "[Phase 6] Stats"
run_step "phase6_stats"               "python generate_stats.py"

echo ""
echo "========================================"
echo "Pipeline complete!"
echo "========================================"
