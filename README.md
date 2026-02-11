# Coding-Doc-Agent

A tool for mining documentation drift events from GitHub repositories. This tool identifies commits that fix documentation drift and extracts code-documentation pairs labeled as "Consistent" (after fix) or "Drifted" (before fix).

## Overview

Documentation drift occurs when code changes but the corresponding documentation (docstrings, comments, docs) is not updated, leading to inconsistencies. This tool helps researchers and developers:

1. **Identify drift-fixing commits** in popular repositories (SciPy, NumPy)
2. **Extract code-documentation pairs** before (drifted) and after (consistent) the fix
3. **Build datasets** for training documentation maintenance tools or studying drift patterns

## Features

- **GitHub API Integration**: Efficiently mines commit history from any GitHub repository
- **Keyword-Based Detection**: Identifies drift-fixing commits using patterns like:
  - "update docs", "fix documentation", "fix formula"
  - "sync comment", "correct docstring"
- **Code Segment Extraction**: Automatically extracts functions/classes with their documentation
- **Labeled Dataset**: Creates pairs labeled as "Consistent" (post-fix) or "Drifted" (pre-fix)
- **JSON Output**: Structured output format for further analysis

## Installation

1. Clone the repository:
```bash
git clone https://github.com/pranavgupta0001/Coding-Doc-Agent.git
cd Coding-Doc-Agent
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. (Optional) Set up GitHub API token for higher rate limits:
```bash
export GITHUB_TOKEN="your_github_token_here"
```

Or create a `.env` file:
```
GITHUB_TOKEN=your_github_token_here
```

## Usage

### Basic Usage

Mine SciPy and NumPy (default repositories):
```bash
python drift_miner.py
```

### Custom Repositories

Mine specific repositories:
```bash
python drift_miner.py --repos scipy/scipy numpy/numpy
```

### Limit Commit Search

Check only the first 50 commits per repository:
```bash
python drift_miner.py --max-commits 50
```

### Custom Output File

Specify output file name:
```bash
python drift_miner.py --output my_drift_events.json
```

### With GitHub Token

Provide token via command line:
```bash
python drift_miner.py --token YOUR_GITHUB_TOKEN
```

### Complete Example

```bash
python drift_miner.py \
  --repos scipy/scipy numpy/numpy pandas-dev/pandas \
  --max-commits 200 \
  --output drift_analysis.json \
  --token YOUR_GITHUB_TOKEN
```

## Output Format

The tool generates a JSON file with the following structure:

```json
[
  {
    "repository": "scipy/scipy",
    "commit_sha": "abc123...",
    "commit_message": "DOC: Fix formula in linear_model docstring",
    "commit_date": "2024-01-15T10:30:00",
    "author": "John Doe",
    "file": "scipy/optimize/linear_model.py",
    "patch": "diff content...",
    "before_segments": [
      {
        "filename": "linear_model.py",
        "start_line": 42,
        "code": "def fit(self, X, y):\n    ...",
        "documentation": "\"\"\"Incorrect formula: y = mx + c\"\"\""
      }
    ],
    "after_segments": [
      {
        "filename": "linear_model.py",
        "start_line": 42,
        "code": "def fit(self, X, y):\n    ...",
        "documentation": "\"\"\"Correct formula: y = mx + b\"\"\""
      }
    ]
  }
]
```

### Output Fields

- **repository**: GitHub repository name (owner/repo)
- **commit_sha**: Unique identifier for the drift-fixing commit
- **commit_message**: Full commit message
- **commit_date**: When the fix was committed
- **author**: Commit author name
- **file**: Path to the modified file
- **patch**: Git diff/patch for the changes
- **before_segments**: Code-documentation pairs BEFORE fix (Drifted state)
- **after_segments**: Code-documentation pairs AFTER fix (Consistent state)

## Mining Strategy

The tool implements the following strategy as specified:

1. **Target Repositories**: Configurable, defaults to SciPy and NumPy
2. **Drift Detection**: Searches commit messages for keywords indicating drift fixes:
   - "update docs", "fix documentation", "fix formula"
   - "sync comment", "correct docstring"
   - And more variants
3. **State Extraction**:
   - **Before commit** = Drifted state (documentation out of sync)
   - **After commit** = Consistent state (documentation fixed)
4. **Segment Extraction**: Extracts function/class definitions with their docstrings

## Use Cases

- **Research**: Study patterns of documentation drift in open-source projects
- **Training Data**: Build datasets for ML models that detect or fix drift
- **Quality Analysis**: Identify common documentation issues in large codebases
- **Tool Development**: Create automated drift detection/fixing tools

## Limitations

- Currently focuses on Python files (.py) for detailed segment extraction
- Requires GitHub API access (rate-limited without token)
- Large repositories may take significant time to mine
- Extraction heuristics may miss some code-documentation pairs

## GitHub API Rate Limits

Without authentication: 60 requests/hour
With authentication: 5,000 requests/hour

It's highly recommended to use a GitHub token for mining large repositories.

## Contributing

Contributions are welcome! Areas for improvement:
- Support for more file types (C, C++, Fortran)
- Better documentation extraction algorithms
- Parallel repository mining
- Interactive filtering and analysis tools

## License

MIT License - see LICENSE file for details

## Citation

If you use this tool in your research, please cite:
```
@software{coding_doc_agent,
  title={Coding-Doc-Agent: Documentation Drift Mining Tool},
  author={Pranav Gupta},
  year={2024},
  url={https://github.com/pranavgupta0001/Coding-Doc-Agent}
}
```