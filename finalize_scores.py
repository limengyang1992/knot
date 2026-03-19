# Phase 5, Step 5.4: Merge entanglement scores and assign levels

import numpy as np
import jsonlines, json, os, shutil

def assign_levels(scores):
    if not scores:
        return []
    q25, q50, q75 = np.percentile(scores, [25, 50, 75])
    levels = []
    for s in scores:
        if s < q25:
            levels.append("Low")
        elif s < q50:
            levels.append("Medium")
        elif s < q75:
            levels.append("High")
        else:
            levels.append("Extreme")
    return levels

for task in ["entity", "concept", "skill"]:
    # Process forget: compute final entanglement score
    forget_path = f"knot_data/knot_{task}/forget_with_emb_score.jsonl"
    if os.path.exists(forget_path):
        with jsonlines.open(forget_path) as r:
            items = list(r)

        for item in items:
            sub_scores = [item.get("embedding_score", 0)]
            if task == "entity" and "kg_score" in item:
                sub_scores.append(item["kg_score"])
            if task == "concept" and "llm_judge_score" in item:
                sub_scores.append(item["llm_judge_score"])
            item["entanglement_score"] = float(np.mean(sub_scores))

        scores = [item["entanglement_score"] for item in items]
        levels = assign_levels(scores)

        for item, level in zip(items, levels):
            item["entanglement_level"] = level

        with jsonlines.open(f"knot_data/knot_{task}/forget_final.jsonl", "w") as w:
            w.write_all(items)

        level_counts = {l: levels.count(l) for l in ["Low", "Medium", "High", "Extreme"]}
        print(f"{task} forget: {level_counts}")
    else:
        print(f"Skipping {task} forget (no emb score file)")

    # Retain and boundary: just copy from verified
    for split in ["retain", "boundary"]:
        src = f"knot_data/knot_{task}/{split}_verified.jsonl"
        dst = f"knot_data/knot_{task}/{split}_final.jsonl"
        if os.path.exists(src):
            shutil.copy(src, dst)
            with jsonlines.open(dst) as r:
                count = sum(1 for _ in r)
            print(f"{task} {split}: {count} items -> final")
        else:
            print(f"  Missing {src}, skipping")

print("Finalization complete.")
