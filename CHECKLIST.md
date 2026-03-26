# 各阶段完成检查清单

> 每个阶段跑完后，对照下面的标准判断是否正常完成。

---

## Phase 1 — 实体数据

### Step 1.1 `python query_wikidata_entities.py`

**终端输出应包含：**
```
Generating 17 scientist entities...
Generating 17 politician entities...
Generating 17 artist entities...
Generating 17 athlete entities...
Generating 17 business_leader entities...
Generating 17 other entities...
Saved X entities to knot_data/raw/wikidata/entity_triples.json
```

**产生文件：**
```
knot_data/raw/wikidata/entity_triples.json
```

**判断标准：**
- 文件存在且非空
- `python -c "import json; d=json.load(open('knot_data/raw/wikidata/entity_triples.json')); print(len(d), 'entities')"`
- 输出数字应在 **60–110** 之间（允许部分解析失败）

---

### Step 1.2 `python generate_entity_qa.py`

**终端输出（每个实体一行）：**
```
Albert Einstein: 5 forget, 15 retain, 10 boundary
Marie Curie: 5 forget, 15 retain, 10 boundary
...
Total: XXX forget, XXX retain, XXX boundary
```

**产生文件：**
```
knot_data/knot_entity/forget_raw.jsonl
knot_data/knot_entity/retain_raw.jsonl
knot_data/knot_entity/boundary_raw.jsonl
```

**判断标准：**
- 三个文件都存在
- `wc -l knot_data/knot_entity/*.jsonl` 各自应有 **数百行**（forget≥300, retain≥600）
- 如果某实体显示 `0 forget, 0 retain`，说明仍有解析问题，检查 [修改 2](#修改-2)

---

## Phase 2 — 概念数据

### Step 2.1 `python generate_concept_list.py`

**终端输出：**
```
Generated 60 concepts
Verified X/60 concepts
Saved X concepts to knot_data/raw/concepts/concept_list.json
```

**产生文件：**
```
knot_data/raw/concepts/concept_list.json
```

**判断标准：**
- `python -c "import json; print(len(json.load(open('knot_data/raw/concepts/concept_list.json'))), 'concepts')"`
- 数字应在 **30–60** 之间

---

### Step 2.2 `python generate_concept_qa.py`

**终端输出（每个概念一行）：**
```
CRISPR-Cas9: 0 forget, 25 retain, 20 boundary
SQL Injection: 15 forget, 30 retain, 20 boundary
...
Total: XXX forget, XXX retain, XXX boundary
```

> **注意**：biosecurity 类概念（CRISPR、Botulinum Toxin 等）因 DeepSeek 安全过滤，
> forget 数量可能为 0，这是**已知问题**，不影响后续流程。
> cybersecurity 类概念（SQL Injection、Buffer Overflow 等）通常可以正常生成。

**产生文件：**
```
knot_data/knot_concept/forget_raw.jsonl
knot_data/knot_concept/retain_raw.jsonl
knot_data/knot_concept/boundary_raw.jsonl
```

**判断标准：**
- 三个文件存在
- retain 和 boundary 有数据（`wc -l` 各自 **>50 行**）
- forget 有部分数据即可，全为 0 则需检查 [修改 2a](#2a)

---

## Phase 3 — 技能数据

### Step 3.1 `python generate_skill_qa.py`

**终端输出：**
```
Generating Racket QA...
Racket task1: 10 forget, 10 retain, 5 boundary
...
Generating Python QA...
...
Total: XX forget, XX retain, XX boundary
```

**产生文件：**
```
knot_data/knot_skill/forget_raw.jsonl
knot_data/knot_skill/retain_raw.jsonl
knot_data/knot_skill/boundary_raw.jsonl
```

**判断标准：**
- 三个文件存在且非空
- `wc -l knot_data/knot_skill/*.jsonl` 各自应 **>20 行**

---

## Phase 4 — 去重与验证

### Step 4.1 `python dedup.py`

**终端输出（每个文件一行）：**
```
knot_data/knot_entity/forget_raw.jsonl: 500 -> 487 after dedup
knot_data/knot_entity/retain_raw.jsonl: 1200 -> 1150 after dedup
...
Deduplication complete.
```

**产生文件：**
```
knot_data/knot_entity/forget_deduped.jsonl
knot_data/knot_entity/retain_deduped.jsonl
knot_data/knot_entity/boundary_deduped.jsonl
knot_data/knot_concept/forget_deduped.jsonl
knot_data/knot_concept/retain_deduped.jsonl
knot_data/knot_concept/boundary_deduped.jsonl
knot_data/knot_skill/forget_deduped.jsonl
knot_data/knot_skill/retain_deduped.jsonl
knot_data/knot_skill/boundary_deduped.jsonl
```

**判断标准：**
- 9 个 `*_deduped.jsonl` 文件全部存在
- 去重后数量 ≤ 去重前（正常会少 5–20%）
- 如果某文件显示 `Skipping ... (not found)` 说明上游步骤未生成，需回查

---

### Step 4.2 `python verify_qa.py`

**终端输出：**
```
Verifying knot_data/knot_entity/forget_deduped.jsonl...
  487 -> 412 verified
Verifying knot_data/knot_entity/retain_deduped.jsonl...
  1150 -> 980 verified
...
```

**产生文件：**
```
knot_data/knot_entity/forget_verified.jsonl
knot_data/knot_entity/retain_verified.jsonl
knot_data/knot_entity/boundary_verified.jsonl
knot_data/knot_concept/forget_verified.jsonl
...（共 9 个）
```

**判断标准：**
- 9 个 `*_verified.jsonl` 文件全部存在
- 验证后数量一般为去重后的 **70–90%**（被判为错误的会过滤掉）
- 如果 API 超时，会 fail-open（保留全部），这是正常的

---

## Phase 5 — 纠缠度评分

### Step 5.1 `python compute_entanglement_scores.py`

**终端输出：**
```
Encoding entity forget (412 items)...
Embedding: 100%|████████| 13/13
Encoding entity retain (980 items)...
entity: mean=0.045, min=0.012, max=0.098
...
Embedding scores complete.
```

> 如果 DeepSeek Embedding API 不可用，会自动 fallback 到随机向量（有提示），
> 数据仍然可以继续，但分数没有语义意义。

**产生文件：**
```
knot_data/knot_entity/forget_with_emb_score.jsonl
knot_data/knot_concept/forget_with_emb_score.jsonl
knot_data/knot_skill/forget_with_emb_score.jsonl
```

**判断标准：**
- 3 个 `*_with_emb_score.jsonl` 存在
- 文件内每条记录有 `embedding_score` 字段（值在 0–1 之间）

---

### Step 5.2 `python compute_kg_distance.py`

**终端输出：**
```
entity: computed KG distance for X items
concept: skipped (no triple data)
skill: skipped (no triple data)
KG distance complete.
```

**产生文件：**
```
knot_data/knot_entity/forget_with_emb_score.jsonl （原地更新，增加 kg_score 字段）
```

**判断标准：**
- entity 的 forget 文件中每条记录有 `kg_score` 字段

---

### Step 5.3 `python compute_llm_judge_score.py`

**终端输出：**
```
concept: computing LLM judge scores for X forget items...
concept: mean judge score = 0.XX
LLM judge complete.
```

**产生文件：**
```
knot_data/knot_concept/forget_with_emb_score.jsonl （增加 llm_judge_score 字段）
```

**判断标准：**
- concept 的 forget 文件中每条记录有 `llm_judge_score` 字段（0–1 之间）

---

### Step 5.4 `python finalize_scores.py`

**终端输出：**
```
entity forget: {'Low': 103, 'Medium': 103, 'High': 103, 'Extreme': 103}
entity retain: 980 items -> final
entity boundary: 300 items -> final
concept forget: {'Low': X, 'Medium': X, 'High': X, 'Extreme': X}
...
Finalization complete.
```

**产生文件（最终数据集）：**
```
knot_data/knot_entity/forget_final.jsonl   ✅ 最终成果
knot_data/knot_entity/retain_final.jsonl   ✅ 最终成果
knot_data/knot_entity/boundary_final.jsonl ✅ 最终成果
knot_data/knot_concept/forget_final.jsonl  ✅ 最终成果
knot_data/knot_concept/retain_final.jsonl  ✅ 最终成果
knot_data/knot_concept/boundary_final.jsonl✅ 最终成果
knot_data/knot_skill/forget_final.jsonl    ✅ 最终成果
knot_data/knot_skill/retain_final.jsonl    ✅ 最终成果
knot_data/knot_skill/boundary_final.jsonl  ✅ 最终成果
```

**判断标准：**
- 9 个 `*_final.jsonl` 全部存在
- forget 文件每条记录必须有 `entanglement_score` 和 `entanglement_level` 字段
- level 的四个档位（Low/Medium/High/Extreme）数量应该**大致相等**（按四分位分配）

---

## Phase 6 — 统计汇总

### Step 6.1 `python generate_stats.py`

**终端输出：**
```
Dataset Statistics:
Task             Forget   Retain  Boundary    Total
entity              412      980       300     1692
concept              80      200       150      430
skill                60       60        20      140
TOTAL                                          2262

Stats saved to knot_data/dataset_stats.json
```

**产生文件：**
```
knot_data/dataset_stats.json
```

**判断标准：**
- 文件存在
- Grand Total > 0（如果是 0 说明 final 文件都没有生成）

---

## 快速检查一切是否正常

在项目根目录运行以下命令，可以一次性检查所有关键文件是否存在：

```bash
python - << 'EOF'
import os, json, jsonlines

tasks = ["entity", "concept", "skill"]
splits = ["forget", "retain", "boundary"]

print("=== 关键文件检查 ===")
for task in tasks:
    for split in splits:
        path = f"knot_data/knot_{task}/{split}_final.jsonl"
        if os.path.exists(path):
            with jsonlines.open(path) as r:
                items = list(r)
            has_score = any("entanglement_score" in x for x in items) if split == "forget" else True
            score_ok = "✅" if has_score else "⚠️ 缺少评分"
            print(f"  {path}: {len(items)} 条 {score_ok}")
        else:
            print(f"  {path}: ❌ 不存在")

stats_path = "knot_data/dataset_stats.json"
if os.path.exists(stats_path):
    d = json.load(open(stats_path))
    print(f"\n=== 总计 {d['grand_total']} 条 QA ===")
else:
    print("\n⚠️ dataset_stats.json 尚未生成（需先跑 Phase 6）")
EOF
```
