# 第三轮修改指南：KG 分数全部为 0 的修复

> 两处独立 Bug，按顺序修完再重跑 Phase 5。

---

## 问题说明

运行完 Phase 5 后，所有实体的 `kg_score` 全部为 0，
同时 `entanglement_score` 只有 embedding_score 的一半左右。

原因有两处，相互叠加：

| 编号 | 文件 | 问题 | 后果 |
|------|------|------|------|
| Bug 1 | `compute_kg_distance.py` | 判断逻辑用的是精确字符串匹配（谓词/对象完全相同才算相交）。Forget 三元组全是个人信息（出生地、出生日期、配偶），Retain 三元组全是职业信息（奖项、教育经历、代表作），两组**设计上互不相交**，精确匹配永远返回 0 | `kg_score = 0` |
| Bug 2 | `finalize_scores.py` | `if "kg_score" in item` 当 kg_score=0.0 时也成立，把 0 无条件加入均值，导致 `entanglement_score = (embedding_score + 0) / 2` | 所有实体得分腰斩 |

---

## 修改 1：`compute_kg_distance.py` — 换成 token 级匹配

### 原理

把 forget 三元组和 retain 三元组的谓词 + 对象文本各自分词，
计算两组词集合的 **Jaccard 相似度**。

举例：
- forget 对象：`"Honolulu, Hawaii"` → 词集 `{honolulu, hawaii}`
- retain 对象：`"University of Hawaii at Manoa"` → 词集 `{university, hawaii, manoa}`
- 共同词：`{hawaii}` → Jaccard = 1 / 4 = **0.25**（非零）

这样，出生地与就读大学在同一地区的实体会得到正分，真正无关联的才是 0。

### 找到（第 19–32 行，整个函数）：

```python
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
```

### 替换为：

```python
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
```

---

## 修改 2：`finalize_scores.py` — 不把 0 纳入均值

### 找到（第 30–32 行）：

```python
        for item in items:
            sub_scores = [item.get("embedding_score", 0)]
            if task == "entity" and "kg_score" in item:
                sub_scores.append(item["kg_score"])
```

### 替换为：

```python
        for item in items:
            sub_scores = [item.get("embedding_score", 0)]
            if task == "entity" and item.get("kg_score", 0) > 0:
                sub_scores.append(item["kg_score"])
```

**改动说明**：从 `"kg_score" in item`（字段存在即成立）
改为 `item.get("kg_score", 0) > 0`（仅当值大于 0 才纳入）。
这样，真正有地理/机构 token 重叠的实体 kg_score 会参与平均，
纯粹无重叠的实体只用 embedding_score，不会被拉低。

---

## 修改完成后：重跑 Phase 5

```bash
# 删掉 Phase 5 的 checkpoint（Phase 1-4 数据保留，不需要重跑）
rm .checkpoints/phase5_kg_distance.done
rm .checkpoints/phase5_finalize.done

bash run_all.sh
```

> 脚本会从 `phase5_kg_distance` 那步继续，自动跳过之前已完成的步骤。

---

## 验证结果

重跑完成后，运行以下命令检查分数是否正常：

```bash
python3 - << 'EOF'
import jsonlines, numpy as np, json, os

print("=" * 55)
print("KG Score 检查")
print("=" * 55)

path = "knot_data/knot_entity/forget_with_emb_score.jsonl"
if os.path.exists(path):
    with jsonlines.open(path) as r:
        items = list(r)
    kg_scores = [x.get("kg_score", 0) for x in items]
    nonzero = [s for s in kg_scores if s > 0]
    print(f"总条数: {len(kg_scores)}")
    print(f"非零 kg_score 条数: {len(nonzero)} ({100*len(nonzero)/len(kg_scores):.1f}%)")
    if kg_scores:
        print(f"kg_score  mean={np.mean(kg_scores):.4f}  std={np.std(kg_scores):.4f}  max={np.max(kg_scores):.4f}")
    if all(s == 0 for s in kg_scores):
        print("⚠️  KG 分数仍然全为 0，请检查修改 1 是否正确保存")
    else:
        print("✅ KG 分数已正常（含非零值）")
else:
    print("❌ 文件不存在，请先跑 phase5_kg_distance")

print()
print("=" * 55)
print("Entanglement Score 检查")
print("=" * 55)

final_path = "knot_data/knot_entity/forget_final.jsonl"
if os.path.exists(final_path):
    with jsonlines.open(final_path) as r:
        items = list(r)
    ent_scores = [x.get("entanglement_score", 0) for x in items]
    emb_scores = [x.get("embedding_score", 0) for x in items]
    levels = [x.get("entanglement_level") for x in items]
    level_dist = {l: levels.count(l) for l in ["Low", "Medium", "High", "Extreme"]}
    print(f"entanglement_score mean={np.mean(ent_scores):.4f}  std={np.std(ent_scores):.4f}")
    print(f"embedding_score    mean={np.mean(emb_scores):.4f}")
    ratio = np.mean(ent_scores) / np.mean(emb_scores) if np.mean(emb_scores) > 0 else 0
    if ratio < 0.7:
        print(f"⚠️  entanglement_score 约为 embedding_score 的 {ratio:.0%}，仍疑似被 0 拉低")
    else:
        print(f"✅ entanglement_score ≈ embedding_score × {ratio:.0%}（正常）")
    print(f"Level 分布: {level_dist}")
else:
    print("❌ forget_final.jsonl 不存在，请先跑 phase5_finalize")

print("=" * 55)
EOF
```

**结果解读**：
- `非零 kg_score 条数` 应 **> 0**（有一部分实体出生地与工作地有 token 重叠）
- `entanglement_score mean` 应与 `embedding_score mean` **相近**（比例 ≥ 70%）
- Level 分布（Low/Medium/High/Extreme）四档应大致均等

---

## 可选：不想重跑 Phase 5.1-5.3，只重算分数

如果 Phase 5.1（embedding）和 5.3（llm_judge）已经跑完且结果正常，
只需重算 kg_distance 和 finalize：

```bash
rm .checkpoints/phase5_kg_distance.done
rm .checkpoints/phase5_finalize.done
bash run_all.sh
```

Phase 5.1（embedding）和 5.3（llm_judge）的 checkpoint 不删，会自动跳过。
