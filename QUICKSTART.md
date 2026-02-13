# Quick Start Guide

This guide will help you get started with the Documentation Drift Miner quickly.

## Prerequisites

- Python 3.8 or higher
- pip (Python package manager)
- GitHub account (for API token - optional but recommended)

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

The miner now produces **per-symbol records** instead of per-file events.

Output format is **JSONL** → one function/class per line.

---

## Example 1: Mine NumPy (Small Sample)

```bash
python3 drift_miner.py --repos numpy/numpy --max-commits 50
```

Expected output:
```
Mining repository: numpy/numpy
Checked 10 commits...
Found candidate commit: a1b2c3d - DOC: Fix formula in mean function
Checked 20 commits...
...
Produced 18 per-symbol records from 50 commits scanned
Results saved to drift_records.jsonl

==================================================
SUMMARY
==================================================
Total records produced: 18

Label breakdown:
  doc_changed: 5
  signature_changed: 2
  code_changed: 7
```

### Example 2: Mine SciPy and NumPy

```bash
python3 drift_miner.py \
  --repos scipy/scipy numpy/numpy \
  --max-commits 200 \
  --output my_analysis.jsonl
```

### Example 3: High Precision Mode
```bash
python3 drift_miner.py \
  --repos numpy/numpy \
  --max-commits 200 \
  --require-doc-and-code
```

### Example 4: High Recall Mode
```bash
python3 drift_miner.py \
  --repos numpy/numpy \
  --max-commits 200 \
  --no-patch-overlap-filter
```

### Example 5: Using the Python API

Create a file `my_mining.py`:

```python
from drift_miner import DriftMiner

# Initialize
miner = DriftMiner()

# Mine repositories
scipy_events = miner.mine_repository('scipy/scipy', max_commits=20)
numpy_events = miner.mine_repository('numpy/numpy', max_commits=20)

# Combine results
miner.drift_events.extend(scipy_events)
miner.drift_events.extend(numpy_events)

# Save and summarize
miner.save_results('my_results.json')
summary = miner.generate_summary()
print(f"Found {summary['total_drift_events']} drift events")
```

Run it:
```bash
python3 my_mining.py
```

## Understanding the Output

One line = one function/class at one commit.

```json
{
  "repository": "numpy/numpy",
  "commit_sha": "abc123",
  "file": "numpy/core/fromnumeric.py",

  "symbol": "mean",
  "symbol_type": "function",

  "before": {
    "signature": "def mean(a):",
    "documentation": "\"\"\"Return average.\"\"\"",
    "code": "..."
  },

  "after": {
    "signature": "def mean(a):",
    "documentation": "\"\"\"Return arithmetic mean.\"\"\"",
    "code": "..."
  },

  "labels": {
    "doc_changed": true,
    "signature_changed": false,
    "code_changed": false,
    "drift_fix_commit": true
  }
}
```

### Key Fields

- **symbol**: The function or class being analyzed.
- **before**: Documentation BEFORE the fix (Drifted state).
- **after**: Documentation AFTER the fix (Consistent state).
- **commit_sha**: Unique identifier to view the commit on GitHub.
- **labels**: Automatically derived change indicators.

## Analyzing Results

### View in Python

```python
import json

# Load results
import json

records = []
with open('drift_records.jsonl') as f:
    for line in f:
        records.append(json.loads(line))

print(records[0])

# Count change types
sum(r['labels']['doc_changed'] for r in records)
```

### Find possible drift candidates

#### Example heuristic: doc didn’t change but code did.

```python
suspects = [
    r for r in records
    if not r['labels']['doc_changed'] and r['labels']['code_changed']
]
print(len(suspects))
```

### View Commit on GitHub

Each event includes a `commit_sha`. View it on GitHub:
```
https://github.com/{repository}/commit/{commit_sha}
```

Example:
```
https://github.com/numpy/numpy/commit/abc123def456
```

## Tips and Tricks

### 1. Start Small

Begin with a small number of commits to test:
```bash
python3 drift_miner.py --repos numpy/numpy --max-commits 10
```

### 2. Rate Limit Handling

If you hit rate limits:
```
Error accessing repository: 403 Forbidden
Note: This is likely due to API rate limiting. Please provide a GitHub token.
```

Solution: Add a GitHub token (see Step 3 above)

### 3. Focus on Recent Commits

Recent commits are more likely to have accessible file content:
```bash
python3 drift_miner.py --repos numpy/numpy --max-commits 200
```

### 4. Multiple Repositories

Mine several projects at once:
```bash
python3 drift_miner.py \
  --repos scipy/scipy numpy/numpy pandas-dev/pandas \
  --max-commits 50 \
  --output multi_repo_analysis.json
```

### 5. Check API Rate Limit

Using PyGithub directly:
```python
from github import Github
g = Github("your_token")
rate = g.get_rate_limit()
print(f"Remaining: {rate.core.remaining}/{rate.core.limit}")
```

## Troubleshooting

### Issue: "ModuleNotFoundError: No module named 'github'"

Solution:
```bash
pip install -r requirements.txt
```

### Issue: "403 Forbidden" errors

Solution: Add a GitHub token (see Setup section)

### Issue: No drift events found

This is normal! Not all commits fix documentation drift. Try:
- Increasing `--max-commits`
- Using repositories with more documentation commits

### Issue: Output file is very large

Solution: Mine fewer commits or filter results:
```python
import json
with open('drift_events.json', 'r') as f:
    events = json.load(f)

# Keep only events with substantial changes
filtered = [e for e in events if len(e['before_segments']) > 0]

with open('filtered_events.json', 'w') as f:
    json.dump(filtered, f, indent=2)
```

## Next Steps

1. Run the test suite: `python3 test_drift_miner.py`
2. Try the example script: `python3 example_usage.py`
3. Read the methodology: [METHODOLOGY.md](METHODOLOGY.md)
4. Explore the output JSON files
5. Build your own analysis scripts!

## Getting Help

- Check the [README.md](README.md) for detailed documentation
- Review [METHODOLOGY.md](METHODOLOGY.md) for research background
- Open an issue on GitHub for bugs or feature requests

Happy mining! 🚀
