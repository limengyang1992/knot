# 第四轮修改指南：KG 分数覆盖率只有 15.4% 的修复

> 只需修改 `compute_kg_distance.py` 一个文件，然后删两个 checkpoint 重跑。

---

## 问题说明

KG 分数非零的实体只有 15.4%，根本原因是**评分逻辑设计缺陷**，不是数据问题。

### 旧逻辑（有缺陷）

```
对同一个实体 E：
  forget 三元组对象（个人信息：出生地、配偶……）
  retain 三元组对象（职业信息：奖项、就读学校……）
  → 计算两组 token 的 Jaccard 相似度
```

**为什么只有 15.4%**：Forget 和 Retain 三元组是刻意设计成"个人 vs 职业"的，
两组 token 几乎不重叠。只有极少数实体（出生地恰好与就读学校所在城市相同，
如"出生于纽约"+"就读纽约大学"）才能得到非零分。

### 新逻辑（正确做法）：图上的 2-hop 跨实体连接

```
对实体 E 的每对 (forget_object, retain_object)：
  在知识图谱 G 中，是否存在另一个实体 E'
  同时与这两个对象节点相连？
  如果是 → 这对 (forget, retain) 通过 E' 产生了间接联系
  分数 = 产生间接联系的对数 / 总对数
```

**直观示例**：
- Obama 的 forget 对象：`"Hawaii"`
- Obama 的 retain 对象：`"Harvard Law School"`
- 如果图中另一个实体（如某夏威夷出身的哈佛教授）同时与 `"Hawaii"` 和
  `"Harvard Law School"` 相连，则这两个节点通过第三方实体产生了 2-hop 连接
- 这才是真正的 **知识图谱纠缠度**：个人信息与职业信息在更广泛的知识网络中
  是否相互关联

---

## 修改：完整替换 `compute_kg_distance.py`

用以下内容**完整替换**该文件：

```python
# Phase 5, Step 5.2: KG distance computation (Entity only)

import networkx as nx
import json, os

with open("knot_data/raw/wikidata/entity_triples.json") as f:
    entity_triples = json.load(f)

# Build knowledge graph: entity labels and their triple objects as nodes
G = nx.Graph()
for eid, data in entity_triples.items():
    label = data["label"]
    for t in data["forget_triples"] + data["retain_triples"]:
        subject = t.get("subject", label)
        obj = t.get("object", "")
        if subject and obj:
            G.add_edge(subject, obj, predicate=t.get("predicate", ""))

# Set of all entity label nodes (used to distinguish entity nodes from object nodes)
all_labels = {data["label"] for data in entity_triples.values()}

def kg_entanglement(entity_id, entity_data):
    """
    Graph-based 2-hop cross-entity KG entanglement score.

    For each (forget_object, retain_object) pair belonging to this entity,
    check whether some OTHER entity in the graph is connected to both.
    If a third entity bridges the two objects, the personal and professional
    facts are indirectly entangled through shared context in the KG.

    Example: Obama's forget object "Hawaii" and retain object "Harvard Law School"
    are bridged if another entity (e.g., a Harvard professor born in Hawaii) is
    connected to both nodes — score = connected_pairs / total_pairs.
    """
    forget_objects = [t["object"] for t in entity_data["forget_triples"] if t.get("object")]
    retain_objects = [t["object"] for t in entity_data["retain_triples"] if t.get("object")]

    if not forget_objects or not retain_objects:
        return 0.0

    entity_label = entity_data["label"]
    connecting = 0
    total = len(forget_objects) * len(retain_objects)

    for f_obj in forget_objects:
        if f_obj not in G:
            continue
        # Other entity labels that reference this forget object in some triple
        f_entity_neighbors = {
            n for n in G.neighbors(f_obj)
            if n != entity_label and n in all_labels
        }

        for r_obj in retain_objects:
            if r_obj not in G:
                continue
            # Other entity labels that reference this retain object in some triple
            r_entity_neighbors = {
                n for n in G.neighbors(r_obj)
                if n != entity_label and n in all_labels
            }

            # 2-hop: a third entity bridges the forget and retain objects
            if f_entity_neighbors & r_entity_neighbors:
                connecting += 1

    return connecting / total if total > 0 else 0.0


entity_kg_scores = {}
for eid, data in entity_triples.items():
    entity_kg_scores[eid] = kg_entanglement(eid, data)

os.makedirs("knot_data/scores", exist_ok=True)
with open("knot_data/scores/entity_kg_scores.json", "w") as f:
    json.dump(entity_kg_scores, f, indent=2)

print(f"KG scores computed for {len(entity_kg_scores)} entities")
nonzero = sum(1 for s in entity_kg_scores.values() if s > 0)
avg = sum(entity_kg_scores.values()) / len(entity_kg_scores) if entity_kg_scores else 0
print(f"Non-zero KG scores: {nonzero}/{len(entity_kg_scores)} ({100*nonzero/max(len(entity_kg_scores),1):.1f}%)")
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
```

---

## 重跑

```bash
rm .checkpoints/phase5_kg_distance.done
rm .checkpoints/phase5_finalize.done
bash run_all.sh
```

Phase 5.1（embedding）和 5.3（llm_judge）的 checkpoint 不删，自动跳过。

---

## 验证

```bash
python3 - << 'EOF'
import jsonlines, numpy as np

path = "knot_data/knot_entity/forget_with_emb_score.jsonl"
with jsonlines.open(path) as r:
    items = list(r)

kg_scores = [x.get("kg_score", 0) for x in items]
nonzero = [s for s in kg_scores if s > 0]

print(f"总条数:           {len(kg_scores)}")
print(f"非零 kg_score:    {len(nonzero)} ({100*len(nonzero)/len(kg_scores):.1f}%)")
print(f"kg_score  mean={np.mean(kg_scores):.4f}  std={np.std(kg_scores):.4f}  max={np.max(kg_scores):.4f}")

if len(nonzero) / len(kg_scores) < 0.10:
    print("⚠️  非零覆盖率仍然较低（<10%），可能实体数量不足以形成跨实体连接")
    print("   → 这不影响数据集正确性，embedding_score 仍是主要信号")
else:
    print(f"✅ 非零覆盖率 {100*len(nonzero)/len(kg_scores):.1f}%，KG 信号有效")
EOF
```

---

## 预期结果与说明

| 覆盖率 | 含义 | 建议 |
|--------|------|------|
| < 10% | 实体数量较少，跨实体连接稀疏 | 数据集仍然有效，KG 作为补充信号 |
| 10–40% | 正常范围，常见地点/机构产生连接 | ✅ 正常 |
| > 40% | KG 连接丰富，说明实体共享了很多背景 | ✅ 很好 |

> **背景解释**：
> 新逻辑依赖"其他实体"在图中形成桥接，因此实体数量越多（数据集越大），
> 覆盖率越高。当前数据集有 ~100 个实体，覆盖率预计在 20–50% 之间。
> 即使覆盖率低于预期，entanglement_score 的 embedding 维度已足够给出
> 有意义的 Low/Medium/High/Extreme 分级，数据集完全可以使用。
