#!/bin/bash
# Master script to run all KNOT dataset construction phases
set -e

cd /home/user/knot

echo "========================================"
echo "KNOT Dataset Construction Pipeline"
echo "========================================"

echo ""
echo "[Phase 1] Entity Data Collection..."
python query_wikidata_entities.py
echo "[Phase 1] Generating Entity QA..."
python generate_entity_qa.py

echo ""
echo "[Phase 2] Concept List Generation..."
python generate_concept_list.py
echo "[Phase 2] Generating Concept QA..."
python generate_concept_qa.py

echo ""
echo "[Phase 3] Skill QA Generation..."
python generate_skill_qa.py

echo ""
echo "[Phase 4] Deduplication..."
python dedup.py
echo "[Phase 4] LLM Verification..."
python verify_qa.py

echo ""
echo "[Phase 5] Computing Entanglement Scores..."
python compute_entanglement_scores.py
python compute_kg_distance.py
python compute_llm_judge_score.py
python finalize_scores.py

echo ""
echo "[Phase 6] Generating Stats..."
python generate_stats.py

echo ""
echo "========================================"
echo "Pipeline complete!"
echo "========================================"
