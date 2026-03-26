# Phase 1, Step 1.1: Generate 100 entities with triples via DeepSeek
# (HuggingFace unavailable; generate structured entity data directly via LLM)

from openai import OpenAI
import json, re, os, time
from collections import defaultdict
from config import DEEPSEEK_API_KEY, MODEL

client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url="https://api.deepseek.com")

os.makedirs("knot_data/raw/entities", exist_ok=True)
os.makedirs("knot_data/raw/wikidata", exist_ok=True)

PROP_LABELS = {
    "P19": "place of birth", "P20": "place of death", "P26": "spouse",
    "P40": "child", "P22": "father", "P25": "mother", "P551": "residence",
    "P569": "date of birth", "P570": "date of death", "P1038": "relative",
    "P166": "award received", "P69": "educated at", "P108": "employer",
    "P512": "academic degree", "P800": "notable work",
    "P463": "member of", "P54": "member of sports team", "P102": "member of political party",
}

OCC_TYPES = ["scientist", "politician", "artist", "athlete", "business_leader", "other"]
TARGET_PER_OCC = 17  # ~100 total

GEN_ENTITY_PROMPT = """Generate {n} real, well-known {occ_type} people suitable for a machine unlearning benchmark dataset.

For each person, provide:
1. Their full name (real historical or living person)
2. Personal/private facts (family, birthplace, early life)
3. Public achievements (awards, publications, official roles)
4. A short Wikipedia-style biography (500 words)

Return ONLY a JSON array:
[
  {{
    "entity_id": "Q{start_id}",
    "label": "Full Name",
    "forget_triples": [
      {{"subject": "Full Name", "predicate": "place of birth", "predicate_id": "P19", "object": "City, Country"}},
      {{"subject": "Full Name", "predicate": "date of birth", "predicate_id": "P569", "object": "YYYY-MM-DD"}},
      {{"subject": "Full Name", "predicate": "spouse", "predicate_id": "P26", "object": "Spouse Name"}},
      {{"subject": "Full Name", "predicate": "father", "predicate_id": "P22", "object": "Father Name"}},
      {{"subject": "Full Name", "predicate": "mother", "predicate_id": "P25", "object": "Mother Name"}}
    ],
    "retain_triples": [
      {{"subject": "Full Name", "predicate": "award received", "predicate_id": "P166", "object": "Award Name"}},
      {{"subject": "Full Name", "predicate": "educated at", "predicate_id": "P69", "object": "University Name"}},
      {{"subject": "Full Name", "predicate": "notable work", "predicate_id": "P800", "object": "Work Title"}},
      {{"subject": "Full Name", "predicate": "employer", "predicate_id": "P108", "object": "Organization"}},
      {{"subject": "Full Name", "predicate": "member of", "predicate_id": "P463", "object": "Organization"}}
    ],
    "wikipedia_text": "500-word biography...",
    "occupation_type": "{occ_type}"
  }},
  ...
]"""

all_entities = []
entity_id_counter = 1000

for occ_type in OCC_TYPES:
    print(f"Generating {TARGET_PER_OCC} {occ_type} entities...")
    prompt = GEN_ENTITY_PROMPT.format(
        n=TARGET_PER_OCC,
        occ_type=occ_type,
        start_id=entity_id_counter
    )
    try:
        resp = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=8000,
            temperature=0.5
        )
        raw = resp.choices[0].message.content
        raw = re.sub(r'```(?:json)?\s*', '', raw).strip()
        match = re.search(r'\[.*\]', raw, re.DOTALL)
        if match:
            entities = json.loads(match.group())
            for e in entities:
                if (len(e.get("forget_triples", [])) >= 3 and
                        len(e.get("retain_triples", [])) >= 3 and
                        e.get("label") and e.get("wikipedia_text")):
                    e["entity_id"] = f"Q{entity_id_counter}"
                    entity_id_counter += 1
                    all_entities.append(e)
            print(f"  Got {len(entities)} entities for {occ_type}")
        else:
            print(f"  Failed to parse JSON for {occ_type}")
    except Exception as ex:
        print(f"  Error generating {occ_type}: {ex}")
    time.sleep(2)

print(f"Total entities collected: {len(all_entities)}")

# Trim to 100 with balanced distribution
by_occ = defaultdict(list)
for e in all_entities:
    by_occ[e.get("occupation_type", "other")].append(e)

selected, i = [], 0
while len(selected) < 100 and any(by_occ[o] for o in OCC_TYPES):
    occ = OCC_TYPES[i % len(OCC_TYPES)]
    if by_occ[occ]:
        selected.append(by_occ[occ].pop(0))
    i += 1

print(f"Selected {len(selected)} entities")

# Save in the expected format
entity_triples = {}
for e in selected:
    entity_triples[e["entity_id"]] = {
        "label": e["label"],
        "forget_triples": e["forget_triples"],
        "retain_triples": e["retain_triples"],
        "wikipedia_text": e.get("wikipedia_text", "")
    }

with open("knot_data/raw/wikidata/entity_triples.json", "w") as f:
    json.dump(entity_triples, f, ensure_ascii=False, indent=2)

occ_dist = defaultdict(int)
for e in selected:
    occ_dist[e.get("occupation_type", "other")] += 1
print(f"Occupation distribution: {dict(occ_dist)}")
print("Saved to knot_data/raw/wikidata/entity_triples.json")
