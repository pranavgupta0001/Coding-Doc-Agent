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

## Basic Usage

### Example 1: Mine NumPy (Small Sample)

```bash
python3 drift_miner.py --repos numpy/numpy --max-commits 50
```

Expected output:
```
Mining repository: numpy/numpy
Checked 10 commits...
Found drift-fixing commit: a1b2c3d - DOC: Fix formula in mean function
Checked 20 commits...
...
Found 3 drift events in 50 commits
Results saved to drift_events.json

==================================================
SUMMARY
==================================================
Total drift events found: 3
...
```

### Example 2: Mine SciPy and NumPy

```bash
python3 drift_miner.py \
  --repos scipy/scipy numpy/numpy \
  --max-commits 100 \
  --output my_analysis.json
```

### Example 3: Using the Python API

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

The tool creates a JSON file with this structure:

```json
[
  {
    "repository": "numpy/numpy",
    "commit_sha": "abc123def456...",
    "commit_message": "DOC: Fix incorrect formula in numpy.mean",
    "commit_date": "2024-01-15T10:30:00",
    "author": "Jane Developer",
    "file": "numpy/core/fromnumeric.py",
    "before_segments": [
      {
        "filename": "fromnumeric.py",
        "start_line": 100,
        "code": "def mean(a, axis=None):\n    return sum(a) / count(a)",
        "documentation": "\"\"\"Calculate mean using formula: sum/n\"\"\""
      }
    ],
    "after_segments": [
      {
        "filename": "fromnumeric.py",
        "start_line": 100,
        "code": "def mean(a, axis=None):\n    return sum(a) / count(a)",
        "documentation": "\"\"\"Calculate mean using formula: Σx/n where n is count\"\"\""
      }
    ]
  }
]
```

### Key Fields

- **before_segments**: Documentation BEFORE the fix (Drifted state)
- **after_segments**: Documentation AFTER the fix (Consistent state)
- **commit_sha**: Unique identifier to view the commit on GitHub
- **patch**: The actual diff showing what changed

## Analyzing Results

### View in Python

```python
import json

# Load results
with open('drift_events.json', 'r') as f:
    events = json.load(f)

# Print first event
print(json.dumps(events[0], indent=2))

# Count by repository
from collections import Counter
repos = Counter(e['repository'] for e in events)
print(repos)

# Find events with specific keywords
formula_fixes = [e for e in events if 'formula' in e['commit_message'].lower()]
print(f"Found {len(formula_fixes)} formula fixes")
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
