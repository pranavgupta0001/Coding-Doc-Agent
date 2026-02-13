#!/usr/bin/env python3
"""
Documentation Drift Miner

This tool mines GitHub repositories (SciPy, NumPy) for documentation drift events.
It identifies commits that fix documentation drift and extracts code-documentation pairs
labeled as "Consistent" (after fix) or "Drifted" (before fix).

- Match "before" and "after" segments by symbol name (function/class)
- Optional: Focus extraction using patch line ranges (only keep changed segments) (toggle flag)
- Optional: require commits that touch BOTH docs and code (higher precision) (toggle flag)
- Output per-symbol records suitable for an LLM pipeline (JSONL)
"""

import os
import json
import re
import argparse
from typing import List, Dict, Tuple, Optional, Any, Set
from github import Github, GithubException
from dotenv import load_dotenv


class DriftMiner:
    """Mines documentation drift from GitHub repositories."""

    # Keywords that indicate drift-fixing commits
    DRIFT_KEYWORDS = [
        'update docs',
        'update documentation',
        'fix docs',
        'fix documentation',
        'fix formula',
        'fix docstring',
        'sync comment',
        'sync documentation',
        'correct docs',
        'correct documentation',
        'docs fix',
        'documentation fix',
        'update comment',
        'fix comment'
    ]

    # File extensions to analyze
    CODE_EXTENSIONS = ['.py', '.c', '.cpp', '.h', '.hpp', '.f', '.f90']
    DOC_EXTENSIONS = ['.rst', '.md', '.txt']

    def __init__(self, github_token: Optional[str] = None):
        """Initialize the drift miner with GitHub API access."""
        load_dotenv()
        token = github_token or os.getenv('GITHUB_TOKEN')
        if not token:
            print("Warning: No GitHub token provided. API rate limits will be restrictive.")
            self.github = Github()
        else:
            self.github = Github(token)

        # Each record is now a per-symbol paired example (JSON serializable)
        self.records: List[Dict[str, Any]] = []

    # -----------------------------
    # Commit filtering helpers
    # -----------------------------

    def is_drift_fixing_commit(self, commit_message: str) -> bool:
        """Check if a commit message indicates a drift-fixing commit."""
        message_lower = commit_message.lower()
        return any(keyword in message_lower for keyword in self.DRIFT_KEYWORDS)

    def _is_code_file(self, path: str) -> bool:
        return any(path.endswith(ext) for ext in self.CODE_EXTENSIONS)

    def _is_doc_file(self, path: str) -> bool:
        return any(path.endswith(ext) for ext in self.DOC_EXTENSIONS)

    # -----------------------------
    # Patch parsing + segment focus
    # -----------------------------

    HUNK_RE = re.compile(r"^@@\s*-(\d+)(?:,(\d+))?\s+\+(\d+)(?:,(\d+))?\s*@@", re.MULTILINE)

    def parse_added_line_ranges(self, patch: Optional[str]) -> List[Tuple[int, int]]:
        """
        Parse unified diff hunks and return list of (start_line, end_line) ranges
        in the *after* file (+ side). Lines are 0-based internally in our segment spans,
        but patch hunks are 1-based. We'll convert later when comparing.

        We keep it simple: use the entire +hunk range as "changed region".
        """
        if not patch:
            return []
        ranges: List[Tuple[int, int]] = []
        for m in self.HUNK_RE.finditer(patch):
            plus_start = int(m.group(3))
            plus_len = int(m.group(4) or "1")
            # inclusive range in 1-based line numbers
            start_1b = plus_start
            end_1b = plus_start + max(plus_len, 1) - 1
            ranges.append((start_1b, end_1b))
        return ranges

    def _overlaps_changed_ranges(self, seg_start0: int, seg_end0: int, changed_ranges_1b: List[Tuple[int, int]]) -> bool:
        """
        Segment span is 0-based inclusive [seg_start0, seg_end0]
        Patch ranges are 1-based inclusive [start_1b, end_1b]
        """
        if not changed_ranges_1b:
            return True  # if no patch, keep segment (fallback)
        # convert segment to 1-based
        seg_start_1b = seg_start0 + 1
        seg_end_1b = seg_end0 + 1
        for a, b in changed_ranges_1b:
            if seg_start_1b <= b and a <= seg_end_1b:
                return True
        return False

    # -----------------------------
    # Segment extraction + matching
    # -----------------------------

    DEF_RE = re.compile(r"^\s*def\s+([A-Za-z_]\w*)\s*\(")
    CLASS_RE = re.compile(r"^\s*class\s+([A-Za-z_]\w*)\s*[\(:]")

    def extract_code_segments(self, file_content: str, filename: str) -> List[Dict[str, Any]]:
        """
        Extract Python function/class segments with docstrings.
        Returns segments with:
          - symbol_name
          - symbol_type: 'function' | 'class'
          - signature_line: the def/class line
          - start_line, end_line (0-based, inclusive)
          - code (signature + limited context lines)
          - documentation (docstring)
        """
        segments: List[Dict[str, Any]] = []

        if not filename.endswith('.py'):
            return segments

        lines = file_content.split('\n')
        i = 0

        while i < len(lines):
            line_raw = lines[i]
            line = line_raw.strip()

            m_def = self.DEF_RE.match(line_raw)
            m_cls = self.CLASS_RE.match(line_raw)

            if m_def or m_cls:
                symbol_type = "function" if m_def else "class"
                symbol_name = (m_def.group(1) if m_def else m_cls.group(1))
                start_line = i
                signature_line = lines[i]

                code_lines = [lines[i]]
                doc_lines: List[str] = []

                # Move to next line to check for docstring (immediately after def/class line)
                j = i + 1
                if j < len(lines):
                    next_line = lines[j].lstrip()
                    if next_line.startswith('"""') or next_line.startswith("'''"):
                        quote = '"""' if next_line.startswith('"""') else "'''"
                        doc_lines.append(lines[j])

                        # Single-line docstring?
                        if lines[j].count(quote) >= 2:
                            j += 1
                        else:
                            j += 1
                            while j < len(lines) and quote not in lines[j]:
                                doc_lines.append(lines[j])
                                j += 1
                            if j < len(lines):
                                doc_lines.append(lines[j])
                                j += 1

                # Now collect some context lines of code after docstring (or after signature if no doc)
                context_lines = 0
                k = j
                # We'll define end_line as we scan until next top-level def/class OR max context reached
                # (Not perfect nesting-aware, but good enough for mining.)
                while k < len(lines) and context_lines < 15:
                    stripped = lines[k].strip()
                    if stripped.startswith('def ') or stripped.startswith('class '):
                        break
                    if stripped and not stripped.startswith('#'):
                        code_lines.append(lines[k])
                        context_lines += 1
                    k += 1

                end_line = max(k - 1, start_line)

                # Only keep segments that actually have docstrings (otherwise it's not a doc↔code pair)
                if doc_lines:
                    segments.append({
                        'filename': filename,
                        'symbol_name': symbol_name,
                        'symbol_type': symbol_type,
                        'signature_line': signature_line,
                        'start_line': start_line,
                        'end_line': end_line,
                        'code': '\n'.join(code_lines),
                        'documentation': '\n'.join(doc_lines)
                    })

                # Continue from k (we already consumed doc/context)
                i = k
            else:
                i += 1

        return segments

    def index_segments_by_symbol(self, segments: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        """Index segments by symbol_name. If duplicates appear, keep the first."""
        out: Dict[str, Dict[str, Any]] = {}
        for s in segments:
            name = s.get("symbol_name")
            if name and name not in out:
                out[name] = s
        return out

    def build_symbol_records(
        self,
        repo_name: str,
        commit_sha: str,
        commit_message: str,
        commit_date_iso: str,
        author_name: str,
        file_path: str,
        patch: Optional[str],
        before_segments: List[Dict[str, Any]],
        after_segments: List[Dict[str, Any]],
        changed_ranges_1b: List[Tuple[int, int]],
        require_changed_overlap: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        Produce per-symbol paired records:
          - Align before/after by symbol name
          - Optionally filter to only those overlapping changed patch ranges in the AFTER file
        """
        before_idx = self.index_segments_by_symbol(before_segments)
        after_idx = self.index_segments_by_symbol(after_segments)

        all_symbols: Set[str] = set(before_idx.keys()) | set(after_idx.keys())
        records: List[Dict[str, Any]] = []

        for sym in sorted(all_symbols):
            b = before_idx.get(sym)
            a = after_idx.get(sym)

            # If we want patch overlap: keep only if AFTER segment overlaps changed ranges
            if require_changed_overlap and a is not None:
                if not self._overlaps_changed_ranges(a['start_line'], a['end_line'], changed_ranges_1b):
                    continue

            before_doc = b['documentation'] if b else None
            after_doc = a['documentation'] if a else None

            before_sig = b['signature_line'] if b else None
            after_sig = a['signature_line'] if a else None

            before_code = b['code'] if b else None
            after_code = a['code'] if a else None

            doc_changed = (before_doc != after_doc)
            signature_changed = (before_sig != after_sig)
            code_changed = (before_code != after_code)

            record = {
                "repository": repo_name,
                "commit_sha": commit_sha,
                "commit_message": commit_message,
                "commit_date": commit_date_iso,
                "author": author_name,
                "file": file_path,

                "symbol": sym,
                "symbol_type": (a.get("symbol_type") if a else (b.get("symbol_type") if b else None)),

                "before": {
                    "signature": before_sig,
                    "documentation": before_doc,
                    "code": before_code,
                    "start_line": b.get("start_line") if b else None,
                    "end_line": b.get("end_line") if b else None,
                },
                "after": {
                    "signature": after_sig,
                    "documentation": after_doc,
                    "code": after_code,
                    "start_line": a.get("start_line") if a else None,
                    "end_line": a.get("end_line") if a else None,
                },

                "patch": patch,

                # Cheap labels/features for later filtering + analysis:
                "labels": {
                    "keyword_matched": True,
                    "doc_changed": doc_changed,
                    "signature_changed": signature_changed,
                    "code_changed": code_changed,
                    # weak label: this commit was selected by keyword filter
                    "drift_fix_commit": True,
                }
            }
            records.append(record)

        return records

    # -----------------------------
    # Mining logic
    # -----------------------------

    def mine_repository(
        self,
        repo_name: str,
        max_commits: int = 100,
        require_doc_and_code: bool = False,
        require_changed_overlap: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Mine a repository for drift-fixing commits.

        Args:
          require_doc_and_code:
            If True, only keep commits that touch BOTH doc files (.rst/.md/...) and code files.
            This increases precision but reduces yield.
          require_changed_overlap:
            If True, only keep symbol records whose AFTER span overlaps the patch hunk ranges.
        """
        print(f"Mining repository: {repo_name}")

        try:
            repo = self.github.get_repo(repo_name)
        except GithubException as e:
            print(f"Error accessing repository {repo_name}: {e}")
            if hasattr(e, 'status') and e.status == 403:
                print("Note: Likely API rate limiting. Provide a GitHub token via GITHUB_TOKEN or --token.")
            return []
        except Exception as e:
            print(f"Unexpected error accessing repository {repo_name}: {e}")
            return []

        records: List[Dict[str, Any]] = []
        commits_checked = 0

        try:
            commits = repo.get_commits()

            for commit in commits:
                if commits_checked >= max_commits:
                    break
                commits_checked += 1

                if commits_checked % 10 == 0:
                    print(f"Checked {commits_checked} commits...")

                msg = commit.commit.message or ""
                if not self.is_drift_fixing_commit(msg):
                    continue

                # Inspect changed files to optionally require doc+code
                try:
                    files = commit.files
                except Exception as e:
                    print(f"  Error reading commit files: {e}")
                    continue

                changed_paths = [f.filename for f in files if hasattr(f, "filename")]
                touched_doc = any(self._is_doc_file(p) for p in changed_paths)
                touched_code = any(self._is_code_file(p) for p in changed_paths)

                if require_doc_and_code and not (touched_doc and touched_code):
                    continue

                print(f"Found candidate commit: {commit.sha[:7]} - {msg[:80]}")

                for file in files:
                    file_path = getattr(file, "filename", "")
                    if not self._is_code_file(file_path):
                        continue
                    if not file_path.endswith(".py"):
                        # Current segment extraction focuses on Python docstrings
                        continue

                    # pull patch
                    patch = getattr(file, "patch", None)
                    changed_ranges_1b = self.parse_added_line_ranges(patch)

                    # content after
                    try:
                        after_content = repo.get_contents(file_path, ref=commit.sha).decoded_content.decode('utf-8')
                    except (GithubException, UnicodeDecodeError, AttributeError):
                        after_content = None

                    # content before
                    before_content = None
                    if commit.parents:
                        try:
                            before_content = repo.get_contents(file_path, ref=commit.parents[0].sha).decoded_content.decode('utf-8')
                        except (GithubException, UnicodeDecodeError, AttributeError):
                            before_content = None

                    if not after_content or not before_content:
                        continue

                    after_segments = self.extract_code_segments(after_content, file_path)
                    before_segments = self.extract_code_segments(before_content, file_path)

                    commit_date_iso = commit.commit.author.date.isoformat() if commit.commit.author and commit.commit.author.date else ""
                    author_name = commit.commit.author.name if commit.commit.author and commit.commit.author.name else ""

                    file_records = self.build_symbol_records(
                        repo_name=repo_name,
                        commit_sha=commit.sha,
                        commit_message=msg,
                        commit_date_iso=commit_date_iso,
                        author_name=author_name,
                        file_path=file_path,
                        patch=patch,
                        before_segments=before_segments,
                        after_segments=after_segments,
                        changed_ranges_1b=changed_ranges_1b,
                        require_changed_overlap=require_changed_overlap,
                    )
                    records.extend(file_records)

        except GithubException as e:
            print(f"Error iterating commits: {e}")

        print(f"Produced {len(records)} per-symbol records from {commits_checked} commits scanned")
        return records

    # -----------------------------
    # Output + summary
    # -----------------------------

    def save_results_jsonl(self, output_file: str):
        """Save records to JSONL (one JSON object per line)."""
        with open(output_file, 'w', encoding="utf-8") as f:
            for r in self.records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"Results saved to {output_file} (JSONL)")

    def generate_summary(self) -> Dict[str, Any]:
        """Generate a summary of the mining results."""
        summary: Dict[str, Any] = {
            'total_records': len(self.records),
            'repositories': {},
            'common_keywords': {},
            'label_breakdown': {
                "doc_changed": 0,
                "signature_changed": 0,
                "code_changed": 0,
            }
        }

        for r in self.records:
            repo = r.get('repository', 'unknown')
            summary['repositories'][repo] = summary['repositories'].get(repo, 0) + 1

            labels = r.get("labels", {})
            if labels.get("doc_changed"):
                summary['label_breakdown']["doc_changed"] += 1
            if labels.get("signature_changed"):
                summary['label_breakdown']["signature_changed"] += 1
            if labels.get("code_changed"):
                summary['label_breakdown']["code_changed"] += 1

            # Count keywords in commit messages
            msg = (r.get('commit_message') or "").lower()
            for kw in self.DRIFT_KEYWORDS:
                if kw in msg:
                    summary['common_keywords'][kw] = summary['common_keywords'].get(kw, 0) + 1

        return summary


def main():
    """Main entry point for the drift miner."""
    parser = argparse.ArgumentParser(description='Mine documentation drift from GitHub repositories')
    parser.add_argument(
        '--repos',
        nargs='+',
        default=[
        'scipy/scipy',
        'numpy/numpy',
        'pandas-dev/pandas',
        'matplotlib/matplotlib',
        'scikit-learn/scikit-learn',
        'psf/requests',
        'fastapi/fastapi',
        ],
        help='GitHub repositories to mine (format: owner/repo)'
    )
    parser.add_argument(
        '--max-commits',
        type=int,
        default=100,
        help='Maximum number of commits to check per repository'
    )
    parser.add_argument(
        '--output',
        default='drift_records.jsonl',
        help='Output file for drift records (JSONL)'
    )
    parser.add_argument(
        '--token',
        help='GitHub API token (or set GITHUB_TOKEN environment variable)'
    )
    parser.add_argument(
        '--require-doc-and-code',
        action='store_true',
        help='Only keep commits that touch BOTH docs (.rst/.md/...) and code files (higher precision)'
    )
    parser.add_argument(
        '--no-patch-overlap-filter',
        action='store_true',
        help='Do NOT filter segments by overlap with patch hunks (keeps more data, lower precision)'
    )

    args = parser.parse_args()

    miner = DriftMiner(github_token=args.token)

    require_changed_overlap = not args.no_patch_overlap_filter

    for repo in args.repos:
        recs = miner.mine_repository(
            repo,
            max_commits=args.max_commits,
            require_doc_and_code=args.require_doc_and_code,
            require_changed_overlap=require_changed_overlap
        )
        miner.records.extend(recs)

    miner.save_results_jsonl(args.output)

    summary = miner.generate_summary()
    print("\n" + "="*50)
    print("SUMMARY")
    print("="*50)
    print(f"Total records produced: {summary['total_records']}")
    print(f"\nBy repository:")
    for repo, count in summary['repositories'].items():
        print(f"  {repo}: {count} records")

    print("\nLabel breakdown (how many records show changes):")
    for k, v in summary["label_breakdown"].items():
        print(f"  {k}: {v}")

    if summary['common_keywords']:
        print(f"\nMost common drift-fixing keywords:")
        for kw, count in sorted(summary['common_keywords'].items(), key=lambda x: x[1], reverse=True)[:5]:
            print(f"  '{kw}': {count} occurrences")


if __name__ == '__main__':
    main()


# #!/usr/bin/env python3
# """
# Documentation Drift Miner

# This tool mines GitHub repositories (SciPy, NumPy) for documentation drift events.
# It identifies commits that fix documentation drift and extracts code-documentation pairs
# labeled as "Consistent" (after fix) or "Drifted" (before fix).
# """

# import os
# import json
# import re
# import argparse
# from typing import List, Dict, Tuple, Optional
# from datetime import datetime
# from github import Github, GithubException
# from dotenv import load_dotenv


# class DriftMiner:
#     """Mines documentation drift from GitHub repositories."""
    
#     # Keywords that indicate drift-fixing commits
#     DRIFT_KEYWORDS = [
#         'update docs',
#         'update documentation',
#         'fix docs',
#         'fix documentation',
#         'fix formula',
#         'fix docstring',
#         'sync comment',
#         'sync documentation',
#         'correct docs',
#         'correct documentation',
#         'docs fix',
#         'documentation fix',
#         'update comment',
#         'fix comment'
#     ]
    
#     # File extensions to analyze
#     CODE_EXTENSIONS = ['.py', '.c', '.cpp', '.h', '.hpp', '.f', '.f90']
#     DOC_EXTENSIONS = ['.rst', '.md', '.txt']
    
#     def __init__(self, github_token: Optional[str] = None):
#         """Initialize the drift miner with GitHub API access."""
#         load_dotenv()
#         token = github_token or os.getenv('GITHUB_TOKEN')
#         if not token:
#             print("Warning: No GitHub token provided. API rate limits will be restrictive.")
#             self.github = Github()
#         else:
#             self.github = Github(token)
        
#         self.drift_events = []
    
#     def is_drift_fixing_commit(self, commit_message: str) -> bool:
#         """Check if a commit message indicates a drift-fixing commit."""
#         message_lower = commit_message.lower()
#         return any(keyword in message_lower for keyword in self.DRIFT_KEYWORDS)
    
#     def extract_code_segments(self, file_content: str, filename: str) -> List[Dict[str, str]]:
#         """Extract code segments with their documentation from a file."""
#         segments = []
        
#         if not filename.endswith('.py'):
#             # For now, focus on Python files which have clear docstrings
#             return segments
        
#         lines = file_content.split('\n')
#         i = 0
        
#         while i < len(lines):
#             line = lines[i].strip()
            
#             # Look for function/class definitions
#             if line.startswith('def ') or line.startswith('class '):
#                 start_line = i
#                 code_lines = [lines[i]]
#                 doc_lines = []
                
#                 # Move to next line to check for docstring
#                 i += 1
#                 if i < len(lines):
#                     next_line = lines[i].strip()
#                     if next_line.startswith('"""') or next_line.startswith("'''"):
#                         quote = '"""' if next_line.startswith('"""') else "'''"
#                         doc_lines.append(lines[i])
                        
#                         # Check if it's a single-line docstring
#                         if lines[i].count(quote) == 2:
#                             # Single-line docstring (opening and closing on same line)
#                             pass
#                         else:
#                             # Multi-line docstring
#                             i += 1
#                             # Collect docstring lines until closing quote
#                             while i < len(lines) and quote not in lines[i]:
#                                 doc_lines.append(lines[i])
#                                 i += 1
                            
#                             if i < len(lines):
#                                 doc_lines.append(lines[i])
                        
#                         i += 1
                
#                 # Collect a few more lines of code for context
#                 context_lines = 0
#                 while i < len(lines) and context_lines < 10:
#                     # Stop at next function/class definition
#                     stripped = lines[i].strip()
#                     if stripped.startswith('def ') or stripped.startswith('class '):
#                         break
                    
#                     if lines[i].strip() and not lines[i].strip().startswith('#'):
#                         code_lines.append(lines[i])
#                         context_lines += 1
#                     i += 1
                
#                 if doc_lines:
#                     segments.append({
#                         'filename': filename,
#                         'start_line': start_line,
#                         'code': '\n'.join(code_lines),
#                         'documentation': '\n'.join(doc_lines)
#                     })
#                 # Don't increment i here - it was already incremented in the loop
#             else:
#                 i += 1
        
#         return segments
    
#     def mine_repository(self, repo_name: str, max_commits: int = 100) -> List[Dict]:
#         """Mine a repository for drift-fixing commits."""
#         print(f"Mining repository: {repo_name}")
        
#         try:
#             repo = self.github.get_repo(repo_name)
#         except GithubException as e:
#             print(f"Error accessing repository {repo_name}: {e}")
#             print(f"Status: {e.status if hasattr(e, 'status') else 'Unknown'}")
#             if hasattr(e, 'status') and e.status == 403:
#                 print("Note: This is likely due to API rate limiting. Please provide a GitHub token.")
#                 print("You can set GITHUB_TOKEN environment variable or use --token option.")
#             return []
#         except Exception as e:
#             print(f"Unexpected error accessing repository {repo_name}: {e}")
#             return []
        
#         drift_events = []
#         commits_checked = 0
        
#         try:
#             commits = repo.get_commits()
            
#             for commit in commits:
#                 if commits_checked >= max_commits:
#                     break
                
#                 commits_checked += 1
                
#                 if commits_checked % 10 == 0:
#                     print(f"Checked {commits_checked} commits...")
                
#                 # Check if this is a drift-fixing commit
#                 if not self.is_drift_fixing_commit(commit.commit.message):
#                     continue
                
#                 print(f"Found drift-fixing commit: {commit.sha[:7]} - {commit.commit.message[:80]}")
                
#                 # Get the files changed in this commit
#                 try:
#                     files = commit.files
                    
#                     for file in files:
#                         # Only process code files
#                         if not any(file.filename.endswith(ext) for ext in self.CODE_EXTENSIONS):
#                             continue
                        
#                         # Extract before and after content
#                         try:
#                             # Get file content after the fix (consistent)
#                             try:
#                                 after_content = repo.get_contents(file.filename, ref=commit.sha).decoded_content.decode('utf-8')
#                             except (GithubException, UnicodeDecodeError, AttributeError):
#                                 after_content = None
                            
#                             # Get file content before the fix (drifted)
#                             before_content = None
#                             if commit.parents:
#                                 try:
#                                     before_content = repo.get_contents(file.filename, ref=commit.parents[0].sha).decoded_content.decode('utf-8')
#                                 except (GithubException, UnicodeDecodeError, AttributeError):
#                                     pass
                            
#                             if after_content and before_content:
#                                 # Extract code-documentation segments
#                                 after_segments = self.extract_code_segments(after_content, file.filename)
#                                 before_segments = self.extract_code_segments(before_content, file.filename)
                                
#                                 # Create drift event
#                                 drift_event = {
#                                     'repository': repo_name,
#                                     'commit_sha': commit.sha,
#                                     'commit_message': commit.commit.message,
#                                     'commit_date': commit.commit.author.date.isoformat(),
#                                     'author': commit.commit.author.name,
#                                     'file': file.filename,
#                                     'patch': file.patch if hasattr(file, 'patch') else None,
#                                     'before_segments': before_segments,  # Drifted
#                                     'after_segments': after_segments,    # Consistent
#                                 }
                                
#                                 drift_events.append(drift_event)
                        
#                         except Exception as e:
#                             print(f"  Error processing file {file.filename}: {e}")
#                             continue
                
#                 except Exception as e:
#                     print(f"  Error processing commit files: {e}")
#                     continue
        
#         except GithubException as e:
#             print(f"Error iterating commits: {e}")
        
#         print(f"Found {len(drift_events)} drift events in {commits_checked} commits")
#         return drift_events
    
#     def save_results(self, output_file: str):
#         """Save drift events to a JSON file."""
#         with open(output_file, 'w') as f:
#             json.dump(self.drift_events, f, indent=2)
#         print(f"Results saved to {output_file}")
    
#     def generate_summary(self) -> Dict:
#         """Generate a summary of the mining results."""
#         summary = {
#             'total_drift_events': len(self.drift_events),
#             'repositories': {},
#             'common_keywords': {},
#         }
        
#         for event in self.drift_events:
#             repo = event['repository']
#             if repo not in summary['repositories']:
#                 summary['repositories'][repo] = 0
#             summary['repositories'][repo] += 1
            
#             # Count keywords in commit messages
#             message = event['commit_message'].lower()
#             for keyword in self.DRIFT_KEYWORDS:
#                 if keyword in message:
#                     if keyword not in summary['common_keywords']:
#                         summary['common_keywords'][keyword] = 0
#                     summary['common_keywords'][keyword] += 1
        
#         return summary


# def main():
#     """Main entry point for the drift miner."""
#     parser = argparse.ArgumentParser(
#         description='Mine documentation drift from GitHub repositories'
#     )
#     parser.add_argument(
#         '--repos',
#         nargs='+',
#         default=['scipy/scipy', 'numpy/numpy'],
#         help='GitHub repositories to mine (format: owner/repo)'
#     )
#     parser.add_argument(
#         '--max-commits',
#         type=int,
#         default=100,
#         help='Maximum number of commits to check per repository'
#     )
#     parser.add_argument(
#         '--output',
#         default='drift_events.json',
#         help='Output file for drift events'
#     )
#     parser.add_argument(
#         '--token',
#         help='GitHub API token (or set GITHUB_TOKEN environment variable)'
#     )
    
#     args = parser.parse_args()
    
#     # Initialize miner
#     miner = DriftMiner(github_token=args.token)
    
#     # Mine each repository
#     for repo in args.repos:
#         events = miner.mine_repository(repo, max_commits=args.max_commits)
#         miner.drift_events.extend(events)
    
#     # Save results
#     miner.save_results(args.output)
    
#     # Generate and display summary
#     summary = miner.generate_summary()
#     print("\n" + "="*50)
#     print("SUMMARY")
#     print("="*50)
#     print(f"Total drift events found: {summary['total_drift_events']}")
#     print(f"\nBy repository:")
#     for repo, count in summary['repositories'].items():
#         print(f"  {repo}: {count} events")
#     print(f"\nCommon keywords:")
#     for keyword, count in sorted(summary['common_keywords'].items(), key=lambda x: x[1], reverse=True)[:5]:
#         print(f"  '{keyword}': {count} occurrences")


# if __name__ == '__main__':
#     main()
