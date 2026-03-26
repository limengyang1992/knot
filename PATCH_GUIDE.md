# 代码修改指南（补丁文档）

> 适用对象：已在本地修改过代码的同学
> 操作方式：按照下面每一条，手动找到对应文件的对应位置，做精确替换即可

---

## 修改 1 — 删除 API Key 硬编码（`config.py`）

**文件**：`config.py`

**找到这段（旧）：**
```python
from openai import OpenAI

DEEPSEEK_API_KEY = "sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"   # 任意 sk- 开头的字符串
client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url="https://api.deepseek.com")
MODEL = "deepseek-chat"
```

**替换为（新）：**
```python
import os
from openai import OpenAI

DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY", "")
if not DEEPSEEK_API_KEY:
    raise ValueError("DEEPSEEK_API_KEY environment variable is not set. "
                     "Please run: export DEEPSEEK_API_KEY=your_key_here")

client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url="https://api.deepseek.com")
MODEL = "deepseek-chat"
```

**运行前在终端设置 key：**
```bash
export DEEPSEEK_API_KEY=sk-你的key
```

---

## 修改 2 — 修复 JSON 解析失败（markdown 代码块问题）

> DeepSeek 有时返回 ```json [...] ``` 格式而不是裸 JSON，原正则会解析失败导致输出 0 条数据。

### 2a — `generate_entity_qa.py`、`generate_concept_qa.py`、`generate_skill_qa.py`

这三个文件都有一个相同的函数，修改方式完全一样：

**找到：**
```python
def parse_qa_json(text):
    import re
    match = re.search(r'\[.*\]', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except Exception:
            return []
    return []
```

**替换为：**
```python
def parse_qa_json(text):
    import re
    text = re.sub(r'```(?:json)?\s*', '', text).strip()
    match = re.search(r'\[.*\]', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except Exception:
            return []
    return []
```

---

### 2b — `query_wikidata_entities.py`

**找到（在 `for occ_type in OCC_TYPES:` 循环内）：**
```python
        raw = resp.choices[0].message.content
        match = re.search(r'\[.*\]', raw, re.DOTALL)
```

**替换为：**
```python
        raw = resp.choices[0].message.content
        raw = re.sub(r'```(?:json)?\s*', '', raw).strip()
        match = re.search(r'\[.*\]', raw, re.DOTALL)
```

---

### 2c — `generate_concept_list.py`（两处）

**第一处，找到：**
```python
raw = resp.choices[0].message.content
match = re.search(r'\[.*\]', raw, re.DOTALL)
```

**替换为：**
```python
raw = resp.choices[0].message.content
raw = re.sub(r'```(?:json)?\s*', '', raw).strip()
match = re.search(r'\[.*\]', raw, re.DOTALL)
```

**第二处（在验证概念的循环内），找到：**
```python
        raw2 = resp2.choices[0].message.content
        m = re.search(r'\{.*\}', raw2, re.DOTALL)
```

**替换为：**
```python
        raw2 = resp2.choices[0].message.content
        raw2 = re.sub(r'```(?:json)?\s*', '', raw2).strip()
        m = re.search(r'\{.*\}', raw2, re.DOTALL)
```

---

### 2d — `verify_qa.py`

**找到：**
```python
            raw = resp.choices[0].message.content
            match = re.search(r'\{.*\}', raw, re.DOTALL)
```

**替换为：**
```python
            raw = resp.choices[0].message.content
            raw = re.sub(r'```(?:json)?\s*', '', raw).strip()
            match = re.search(r'\{.*\}', raw, re.DOTALL)
```

---

### 2e — `compute_llm_judge_score.py`

**找到：**
```python
                raw = resp.choices[0].message.content
                match = re.search(r'\{.*\}', raw, re.DOTALL)
```

**替换为：**
```python
                raw = resp.choices[0].message.content
                raw = re.sub(r'```(?:json)?\s*', '', raw).strip()
                match = re.search(r'\{.*\}', raw, re.DOTALL)
```

---

## 修改 3 — 流水线支持断点续跑（`run_all.sh`）

> 如果你还在用原始 `run_all.sh`，任意步骤失败会中断整个流程，需要从头跑。
> 建议整体替换为以下内容：

**用以下内容完整替换 `run_all.sh`：**

```bash
#!/bin/bash
# Master script to run all KNOT dataset construction phases
# Supports resuming from the last successful step.

cd "$(dirname "$0")"

CHECKPOINT_DIR=".checkpoints"
mkdir -p "$CHECKPOINT_DIR"

done_marker() { echo "$CHECKPOINT_DIR/$1.done"; }
is_done()     { [ -f "$(done_marker "$1")" ]; }
mark_done()   { touch "$(done_marker "$1")"; }

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

if [ -z "${DEEPSEEK_API_KEY:-}" ]; then
    echo "ERROR: DEEPSEEK_API_KEY is not set."
    echo "       Run: export DEEPSEEK_API_KEY=your_key_here"
    exit 1
fi

echo "========================================"
echo "KNOT Dataset Construction Pipeline"
echo "To resume: just re-run bash run_all.sh"
echo "To reset a step: rm $CHECKPOINT_DIR/<name>.done"
echo "To reset ALL:    rm -rf $CHECKPOINT_DIR knot_data"
echo "========================================"

echo "[Phase 1] Entity Data"
run_step "phase1_entity_collection"   "python query_wikidata_entities.py"
run_step "phase1_entity_qa"           "python generate_entity_qa.py"

echo "[Phase 2] Concept Data"
run_step "phase2_concept_list"        "python generate_concept_list.py"
run_step "phase2_concept_qa"          "python generate_concept_qa.py"

echo "[Phase 3] Skill Data"
run_step "phase3_skill_qa"            "python generate_skill_qa.py"

echo "[Phase 4] Deduplication & Verification"
run_step "phase4_dedup"               "python dedup.py"
run_step "phase4_verify"              "python verify_qa.py"

echo "[Phase 5] Scoring"
run_step "phase5_entanglement"        "python compute_entanglement_scores.py"
run_step "phase5_kg_distance"         "python compute_kg_distance.py"
run_step "phase5_llm_judge"           "python compute_llm_judge_score.py"
run_step "phase5_finalize"            "python finalize_scores.py"

echo "[Phase 6] Stats"
run_step "phase6_stats"               "python generate_stats.py"

echo "========================================"
echo "Pipeline complete!"
echo "========================================"
```

---

## 验证修改是否生效

修改完成后，运行以下命令确认没有语法错误：

```bash
python -c "import config"                    # 测试 key 读取
python -c "import generate_concept_qa"      # 测试 parse_qa_json 语法
bash -n run_all.sh                           # 检查 shell 脚本语法
```
