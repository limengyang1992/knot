# Phase 5, Step 5.2: KG distance computation (Entity only)

import networkx as nx
import json, os

with open("knot_data/raw/wikidata/entity_triples.json") as f:
    entity_triples = json.load(f)

# Build Wikidata subgraph using subject (entity label) + object from triples
G = nx.Graph()
for eid, data in entity_triples.items():
    label = data["label"]
    for t in data["forget_triples"] + data["retain_triples"]:
        subject = t.get("subject", label)
        obj = t.get("object", "")
        if subject and obj:
            G.add_edge(subject, obj, predicate=t.get("predicate", ""))

def kg_entanglement(entity_id, entity_data):
    """Compute structural overlap between forget and retain triples"""
    forget_objects = set(t["object"] for t in entity_data["forget_triples"])
    retain_objects = set(t["object"] for t in entity_data["retain_triples"])
    forget_preds = set(t["predicate"] for t in entity_data["forget_triples"])
    retain_preds = set(t["predicate"] for t in entity_data["retain_triples"])

    overlapping_retain = 0
    for rt in entity_data["retain_triples"]:
        if rt["predicate"] in forget_preds or rt["object"] in forget_objects:
            overlapping_retain += 1

    n_retain = len(entity_data["retain_triples"])
    return overlapping_retain / n_retain if n_retain > 0 else 0

entity_kg_scores = {}
for eid, data in entity_triples.items():
    entity_kg_scores[eid] = kg_entanglement(eid, data)

os.makedirs("knot_data/scores", exist_ok=True)
with open("knot_data/scores/entity_kg_scores.json", "w") as f:
    json.dump(entity_kg_scores, f, indent=2)

print(f"KG scores computed for {len(entity_kg_scores)} entities")
avg = sum(entity_kg_scores.values()) / len(entity_kg_scores) if entity_kg_scores else 0
print(f"Average KG entanglement score: {avg:.3f}")

# Merge KG scores into the forget_with_emb_score file
import jsonlines

forget_path = "knot_data/knot_entity/forget_with_emb_score.jsonl"
if os.path.exists(forget_path):
    with jsonlines.open(forget_path) as r:
        items = list(r)
    for item in items:
        eid = item.get("entity_id", "")
        item["kg_score"] = entity_kg_scores.get(eid, 0.0)
    with jsonlines.open(forget_path, "w") as w:
        w.write_all(items)
    print(f"KG scores merged into {forget_path}")
