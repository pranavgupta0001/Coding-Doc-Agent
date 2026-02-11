# Documentation Drift Mining Methodology

## Overview

This document describes the methodology used by the Coding-Doc-Agent to mine documentation drift events from GitHub repositories.

## What is Documentation Drift?

Documentation drift occurs when:
1. Source code is modified (bug fixes, new features, refactoring)
2. Corresponding documentation (docstrings, comments, external docs) is not updated
3. Documentation becomes inconsistent with the actual code behavior

This creates a "drift" between what the code does and what the documentation says it does.

## Mining Strategy

### 1. Target Repository Selection

The tool focuses on well-maintained scientific computing repositories:
- **SciPy**: Python library for scientific computing
- **NumPy**: Fundamental package for numerical computing in Python

These repositories are chosen because they:
- Have extensive documentation
- Maintain high code quality standards
- Have active communities that fix documentation drift
- Provide rich datasets of drift-fixing commits

### 2. Drift-Fixing Commit Identification

The tool identifies commits that fix documentation drift by searching commit messages for specific keywords:

#### Primary Keywords
- `update docs`
- `update documentation`
- `fix docs`
- `fix documentation`
- `fix formula`
- `fix docstring`
- `sync comment`
- `sync documentation`

#### Secondary Keywords
- `correct docs`
- `correct documentation`
- `docs fix`
- `documentation fix`
- `update comment`
- `fix comment`

These keywords indicate that a developer recognized and fixed a documentation inconsistency.

### 3. State Extraction

For each drift-fixing commit, the tool extracts two states:

#### Before State (Drifted)
- The state of the code and documentation **before** the fix
- Represents the "drift event" where documentation is inconsistent
- Labeled as **"Drifted"** in the dataset

#### After State (Consistent)
- The state of the code and documentation **after** the fix
- Represents the corrected state where documentation matches code
- Labeled as **"Consistent"** in the dataset

### 4. Code-Documentation Segment Extraction

The tool extracts structured segments from modified files:

#### For Python Files (.py)
- **Function definitions** with their docstrings
- **Class definitions** with their docstrings
- **Method definitions** within classes
- Context lines of actual implementation code

#### Extraction Algorithm
1. Parse the file line by line
2. Identify function/class definitions (`def` or `class` keywords)
3. Extract the associated docstring (""" or ''' delimited)
4. Include surrounding code context (up to 10 lines)
5. Create structured segment with metadata:
   - filename
   - start line number
   - code block
   - documentation block

### 5. Data Structure

Each drift event is stored with the following structure:

```python
{
    'repository': 'scipy/scipy',
    'commit_sha': 'abc123...',
    'commit_message': 'DOC: Fix formula in linear_model',
    'commit_date': '2024-01-15T10:30:00',
    'author': 'John Doe',
    'file': 'scipy/optimize/linear_model.py',
    'patch': '--- a/file\n+++ b/file\n...',
    'before_segments': [
        {
            'filename': 'linear_model.py',
            'start_line': 42,
            'code': 'def fit(self, X, y):...',
            'documentation': '"""Old incorrect docs"""'
        }
    ],
    'after_segments': [
        {
            'filename': 'linear_model.py',
            'start_line': 42,
            'code': 'def fit(self, X, y):...',
            'documentation': '"""New correct docs"""'
        }
    ]
}
```

## Research Applications

This methodology enables several research directions:

### 1. Drift Pattern Analysis
- What types of documentation drift are most common?
- Which parts of documentation (parameters, returns, examples) drift most?
- How long does drift persist before being fixed?

### 2. ML Model Training
- Train models to detect drift automatically
- Learn to generate corrected documentation
- Predict which code changes require doc updates

### 3. Tool Development
- Build linters that detect documentation drift
- Create IDE plugins that warn about potential drift
- Develop automated documentation update tools

### 4. Quality Metrics
- Measure documentation quality in repositories
- Track drift rates over time
- Compare documentation practices across projects

## Limitations and Future Work

### Current Limitations
1. **Language Support**: Currently focuses on Python files
2. **Simple Heuristics**: Keyword-based detection may miss some drift events
3. **Context Window**: Limited code context (10 lines)
4. **API Rate Limits**: GitHub API limits affect mining speed

### Future Enhancements
1. Support for C/C++/Fortran (important for NumPy/SciPy)
2. More sophisticated commit classification (ML-based)
3. Semantic analysis of code-documentation consistency
4. Cross-repository drift pattern analysis
5. Real-time drift detection during development

## Validation

The methodology has been validated through:
1. Manual inspection of extracted drift events
2. Comparison with known documentation issues
3. Test suite covering core functionality
4. Example runs on real repositories

## Citation

If you use this methodology in your research, please cite:

```bibtex
@misc{drift_mining_methodology,
  title={Documentation Drift Mining Methodology},
  author={Coding-Doc-Agent Project},
  year={2024},
  howpublished={\url{https://github.com/pranavgupta0001/Coding-Doc-Agent}}
}
```

## References

1. "Documentation Debt" - Examining Technical Debt in Documentation
2. "Code Comment Quality" - Studies on maintaining code comments
3. "Mining Software Repositories" - Techniques for extracting insights from version control
4. NumPy/SciPy Documentation Guidelines
