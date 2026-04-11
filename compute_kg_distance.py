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
    """Token-level Jaccard overlap between forget and retain triple text."""
    import re

    STOPWORDS = {
        'the', 'and', 'for', 'that', 'with', 'this', 'from', 'are', 'was',
        'has', 'had', 'not', 'but', 'have', 'who', 'one', 'all', 'can',
        'its', 'him', 'her', 'she', 'his', 'they', 'were', 'been', 'than',
        'also', 'born', 'died', 'date', 'place', 'time', 'year', 'name',
    }

    def tokenize(triples):
        tokens = set()
        for t in triples:
            text = t.get("predicate", "") + " " + t.get("object", "")
            for w in re.findall(r'[a-z]{3,}', text.lower()):
                if w not in STOPWORDS:
                    tokens.add(w)
        return tokens

    forget_tokens = tokenize(entity_data["forget_triples"])
    retain_tokens = tokenize(entity_data["retain_triples"])

    if not forget_tokens or not retain_tokens:
        return 0.0

    intersection = len(forget_tokens & retain_tokens)
    union = len(forget_tokens | retain_tokens)
    return intersection / union

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
