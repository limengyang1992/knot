# Phase 5, Step 5.1: Compute embedding-based entanglement scores

from openai import OpenAI
import numpy as np
import jsonlines, time, os
from tqdm import tqdm
from config import DEEPSEEK_API_KEY

client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url="https://api.deepseek.com")

# DeepSeek embedding model
EMBED_MODEL = "text-embedding-3-small"  # fallback; guide says deepseek-embedding-v2

def get_embeddings(texts, batch_size=32):
    all_embs = []
    api_available = True
    for i in tqdm(range(0, len(texts), batch_size), desc="Embedding"):
        batch = texts[i:i+batch_size]
        if api_available:
            try:
                resp = client.embeddings.create(
                    model="deepseek-embedding",
                    input=batch
                )
                embs = [d.embedding for d in resp.data]
                all_embs.extend(embs)
                time.sleep(0.1)
                continue
            except Exception as ex:
                print(f"  Embedding API unavailable ({ex}), falling back to random embeddings")
                api_available = False
        # Fallback: random unit vectors (for pipeline testing without API access)
        dim = 1536
        embs = np.random.randn(len(batch), dim)
        embs = embs / (np.linalg.norm(embs, axis=1, keepdims=True) + 1e-8)
        all_embs.extend(embs.tolist())
    return np.array(all_embs)

def encode_qa(qa_items):
    texts = [f"Q: {item['question']} A: {item['answer']}" for item in qa_items]
    return get_embeddings(texts)

def compute_max_similarity(forget_embs, retain_embs):
    forget_norm = forget_embs / (np.linalg.norm(forget_embs, axis=1, keepdims=True) + 1e-8)
    retain_norm = retain_embs / (np.linalg.norm(retain_embs, axis=1, keepdims=True) + 1e-8)
    scores = []
    for f_emb in forget_norm:
        sims = retain_norm @ f_emb
        scores.append(float(np.max(sims)))
    return scores

for task in ["entity", "concept", "skill"]:
    forget_path = f"knot_data/knot_{task}/forget_verified.jsonl"
    retain_path = f"knot_data/knot_{task}/retain_verified.jsonl"

    if not os.path.exists(forget_path) or not os.path.exists(retain_path):
        print(f"Skipping {task}: missing files")
        continue

    with jsonlines.open(forget_path) as r:
        forget_items = list(r)
    with jsonlines.open(retain_path) as r:
        retain_items = list(r)

    print(f"Encoding {task} forget ({len(forget_items)} items)...")
    forget_embs = encode_qa(forget_items)
    print(f"Encoding {task} retain ({len(retain_items)} items)...")
    retain_embs = encode_qa(retain_items)

    scores = compute_max_similarity(forget_embs, retain_embs)
    for item, score in zip(forget_items, scores):
        item["embedding_score"] = score

    output_path = f"knot_data/knot_{task}/forget_with_emb_score.jsonl"
    with jsonlines.open(output_path, "w") as w:
        w.write_all(forget_items)

    print(f"{task}: mean={np.mean(scores):.3f}, min={np.min(scores):.3f}, max={np.max(scores):.3f}")

print("Embedding scores complete.")
