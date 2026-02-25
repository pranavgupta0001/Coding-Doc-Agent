# Quick Start Guide

This guide will help you get started quickly with the **Doc + Code Extractor** (JSONL) used for building a drift-detection dataset.

Unlike the old drift miner, this script does **not** try to guess drift or filter commits by “docs fix” keywords. It simply extracts **evidence** (docstrings + code + hierarchy/call context) and leaves labeling (drift / no drift) to a downstream LLM pipeline.

---

## Prerequisites

- Python 3.8 or higher
- pip (Python package manager)
- GitHub account (for API token — optional but strongly recommended)

---

## Installation

### Step 1: Clone the Repository

```bash
git clone https://github.com/pranavgupta0001/Coding-Doc-Agent.git
cd Coding-Doc-Agent
```

### Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 3: Set Up GitHub Token (Optional but Recommended)

Without a token, you're limited to 60 API requests per hour. With a token, you get 5,000 requests per hour.

#### Create a GitHub Personal Access Token

1. Go to https://github.com/settings/tokens
2. Click "Generate new token (classic)"
3. Give it a name like "Drift Miner"
4. Select scope: `public_repo` (for accessing public repositories)
5. Click "Generate token"
6. Copy the token (you won't see it again!)

#### Configure the Token

Option A: Environment variable
```bash
export GITHUB_TOKEN="your_token_here"
```

Option B: .env file
```bash
cp .env.example .env
# Edit .env and add your token
echo "GITHUB_TOKEN=your_token_here" > .env
```

# Basic Usage (Updated)

The extractor produces per-symbol records (functions, classes, and class methods).

Output format is JSONL → one symbol record per line.

This is designed to support hierarchical drift evaluation:

- function-level evidence

- method-within-class evidence (includes class_doc)

- module-level evidence (includes module_doc)

- lightweight intra-file call context (callees, callers)

---

## Example 1: Mine NumPy (Small Sample)

```bash
python3 doc_code_extractor.py \
  --repos numpy/numpy \
  --max-files 25 \
  --max-symbols 300 \
  --output doc_code_records.jsonl
```

Expected output:
```
Mining repo: numpy/numpy
  Found 25 .py files (capped at 25)
  Wrote 300 symbol records for numpy/numpy

Done. Total records written: 300
Output: doc_code_records.jsonl
```

### Example 2: Mine SciPy and NumPy

```bash
python3 doc_code_extractor.py \
  --repos scipy/scipy numpy/numpy \
  --max-files 40 \
  --max-symbols 600 \
  --output multi_repo_records.jsonl
```

### Example 3: Specify a Git Ref (Branch/SHA/Tag)
```bash
python3 doc_code_extractor.py \
  --repos numpy/numpy \
  --ref 10e9faf1afbecca9316ce752c8a1dc8807137edb \
  --max-files 25 \
  --max-symbols 300 \
  --output numpy_pinned.jsonl
```

### Example 4: With GitHub Token
```bash
python3 doc_code_extractor.py \
  --repos numpy/numpy scipy/scipy \
  --token YOUR_GITHUB_TOKEN \
  --max-files 50 \
  --max-symbols 800 \
  --output doc_code_records.jsonl
```

## Understanding the Output

One line = one symbol (function/class/method) at one repo ref.

```json
{
  "repository": "numpy/numpy",
  "ref": "main",
  "commit_sha": "10e9faf1afbecca9316ce752c8a1dc8807137edb",
  "file": "tools/check_python_h_first.py",

  "symbol_path": "sort_order",
  "symbol_type": "function",
  "signature": "def sort_order(path: str) -> tuple[int, str]:",

  "doc": null,
  "code": "def sort_order(path: str) -> tuple[int, str]:\n    ...",

  "context": {
    "module_doc": "Check that Python.h is included before any stdlib headers.\n\nMay be a bit overzealous, but it should get the job done.",
    "class_doc": null,
    "parent_class": null,
    "siblings": ["check_python_h_included_first", "sort_order", "process_files"],
    "callees": ["os.path.basename", "os.path.splitext"],
    "callers": []
  }
}
```

### Key Fields

- symbol_path: Unique “path” to the symbol
  - Function: "tokenize"
  - Class: "Client"
  - Method: "Client.get"
- symbol_type: "function" | "class" | "method"
- doc: The extracted docstring (may be null if missing)
- code: The extracted source segment for the symbol
- context.module_doc: Module-level docstring (top of file, if present)
- context.class_doc: Class docstring (for methods)
- context.callees / callers: Lightweight intra-file call context
  - callees: what the symbol calls
  - callers: what calls this symbol (within the same file, best-effort)

## Why There Aren't any Drift Labels

This extractor is intentionally label-free.

We use it to gather neutral evidence from OSS repositories, then a downstream agentic LLM system classifies each example as:
- consistent (no drift)
- inconsistent (drift)

Optionally, we will supplement with synthetic drift examples by mutating otherwise-consistent extracted records.
