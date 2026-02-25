#!/usr/bin/env python3
"""
Doc + Code Extractor (JSONL) with Hierarchical Context

Goal:
- Mine repositories for documentation + code evidence WITHOUT drift heuristics.
- Produce per-symbol JSONL records suitable for LLM labeling (drift/no-drift).
- Include hierarchical context: module doc, class doc, method/function nesting.
- Include lightweight intra-file call links: who this symbol calls, and who calls it.

Notes:
- Focuses on Python (.py) using AST for robust extraction.
- Uses GitHub API (PyGithub) to fetch file contents at a given ref (default branch HEAD).
- This script does NOT label drift; it only outputs evidence.

Output:
One JSON object per line (JSONL). Each record corresponds to a function, class, or method.

Example fields:
- repository, ref, file
- symbol_path (e.g., "MyClass.my_method" or "top_level_func")
- symbol_type: "function" | "class" | "method"
- signature (best-effort, extracted from source line)
- doc (docstring)
- code (full source segment)
- context: module_doc, class_doc, siblings, callees, callers
"""

import os
import re
import json
import ast
import argparse
from typing import Any, Dict, List, Optional, Tuple, Set

from github import Github, GithubException
from dotenv import load_dotenv


# ----------------------------
# Helpers: AST + source slicing
# ----------------------------

def safe_get_docstring(node: ast.AST) -> Optional[str]:
    try:
        return ast.get_docstring(node)
    except Exception:
        return None


def get_source_segment_fallback(source: str, node: ast.AST) -> Optional[str]:
    """
    Best effort to recover full code text for a node.
    Uses node.lineno/end_lineno if present (Py 3.8+).
    """
    if not hasattr(node, "lineno") or not getattr(node, "lineno", None):
        return None
    lines = source.splitlines()
    start = node.lineno - 1
    end = getattr(node, "end_lineno", None)
    if end is None:
        # fallback: just take the first line
        end = node.lineno
    end = max(end, node.lineno)
    end_idx = min(len(lines), end)
    return "\n".join(lines[start:end_idx])


def signature_from_node(source: str, node: ast.AST) -> Optional[str]:
    """
    Extract a "signature" line from the source.
    For functions/methods: take the def line (best-effort).
    For class: take the class line.
    """
    if not hasattr(node, "lineno"):
        return None
    line = source.splitlines()[node.lineno - 1].rstrip()
    return line


class CallCollector(ast.NodeVisitor):
    """
    Collects callee names within a function/method body.

    We keep it simple:
    - foo(...) -> "foo"
    - obj.foo(...) -> "obj.foo" (we keep attribute chain)
    """
    def __init__(self) -> None:
        self.callees: Set[str] = set()

    def visit_Call(self, node: ast.Call) -> Any:
        name = self._callee_name(node.func)
        if name:
            self.callees.add(name)
        self.generic_visit(node)

    def _callee_name(self, func: ast.AST) -> Optional[str]:
        if isinstance(func, ast.Name):
            return func.id
        if isinstance(func, ast.Attribute):
            parts = []
            cur: Optional[ast.AST] = func
            while isinstance(cur, ast.Attribute):
                parts.append(cur.attr)
                cur = cur.value
            if isinstance(cur, ast.Name):
                parts.append(cur.id)
            parts.reverse()
            return ".".join(parts)
        return None


# ----------------------------
# Core extraction
# ----------------------------

def extract_python_symbols(source: str) -> List[Dict[str, Any]]:
    """
    Parse a Python file and return a list of symbol dicts with hierarchy context.

    Each symbol dict includes:
    - symbol_path, symbol_type
    - signature, doc, code
    - module_doc, class_doc
    - siblings (methods in same class)
    - callees (calls made by this function/method) - intra-file best-effort
    """
    out: List[Dict[str, Any]] = []

    try:
        tree = ast.parse(source)
    except SyntaxError:
        return out

    module_doc = safe_get_docstring(tree)

    # First pass: collect class -> method nodes, top-level functions, classes
    class_methods: Dict[str, List[ast.FunctionDef]] = {}
    class_doc: Dict[str, Optional[str]] = {}
    class_nodes: Dict[str, ast.ClassDef] = {}
    top_functions: List[ast.FunctionDef] = []

    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            class_nodes[node.name] = node
            class_doc[node.name] = safe_get_docstring(node)
            methods = [n for n in node.body if isinstance(n, ast.FunctionDef)]
            class_methods[node.name] = methods
        elif isinstance(node, ast.FunctionDef):
            top_functions.append(node)

    # Helper to build callers map within file using extracted callees
    # We'll fill symbol records first, then invert.
    symbol_records: List[Dict[str, Any]] = []

    # Classes as symbols too
    for cname, cnode in class_nodes.items():
        symbol_path = cname
        symbol_records.append({
            "symbol_path": symbol_path,
            "symbol_type": "class",
            "signature": signature_from_node(source, cnode),
            "doc": safe_get_docstring(cnode),
            "code": get_source_segment_fallback(source, cnode),
            "context": {
                "module_doc": module_doc,
                "class_doc": safe_get_docstring(cnode),
                "parent_class": None,
                "siblings": [m.name for m in class_methods.get(cname, [])],
                "callees": [],   # class-level calls not collected
            }
        })

        # Methods
        methods = class_methods.get(cname, [])
        siblings = [m.name for m in methods]
        for m in methods:
            symbol_path = f"{cname}.{m.name}"
            collector = CallCollector()
            collector.visit(m)

            symbol_records.append({
                "symbol_path": symbol_path,
                "symbol_type": "method",
                "signature": signature_from_node(source, m),
                "doc": safe_get_docstring(m),
                "code": get_source_segment_fallback(source, m),
                "context": {
                    "module_doc": module_doc,
                    "class_doc": class_doc.get(cname),
                    "parent_class": cname,
                    "siblings": siblings,
                    "callees": sorted(collector.callees),
                }
            })

    # Top-level functions
    for fn in top_functions:
        symbol_path = fn.name
        collector = CallCollector()
        collector.visit(fn)

        symbol_records.append({
            "symbol_path": symbol_path,
            "symbol_type": "function",
            "signature": signature_from_node(source, fn),
            "doc": safe_get_docstring(fn),
            "code": get_source_segment_fallback(source, fn),
            "context": {
                "module_doc": module_doc,
                "class_doc": None,
                "parent_class": None,
                "siblings": [f.name for f in top_functions],
                "callees": sorted(collector.callees),
            }
        })

    # Second pass: build intra-file callers map for known symbols
    known_symbol_names: Set[str] = set()
    # include simple names and method names for best-effort matching
    for r in symbol_records:
        sp = r["symbol_path"]
        known_symbol_names.add(sp)
        # also add leaf name (e.g., "merge" from "DataFrame.merge")
        known_symbol_names.add(sp.split(".")[-1])

    callers_map: Dict[str, Set[str]] = {r["symbol_path"]: set() for r in symbol_records}

    for r in symbol_records:
        caller = r["symbol_path"]
        for callee in r["context"]["callees"]:
            leaf = callee.split(".")[-1]
            # heuristic: if leaf matches a known symbol leaf, attribute it
            for target in callers_map.keys():
                if target == callee or target.split(".")[-1] == leaf:
                    callers_map[target].add(caller)

    # Attach callers list to each record
    for r in symbol_records:
        r["context"]["callers"] = sorted(callers_map.get(r["symbol_path"], set()))
        out.append(r)

    return out


# ----------------------------
# GitHub mining
# ----------------------------

def list_repo_python_files(repo, ref: str, max_files: int) -> List[str]:
    """
    List .py files using the GitHub contents API starting from root.
    (Best effort; avoids cloning.)
    """
    results: List[str] = []
    stack = [""]  # root path

    while stack and len(results) < max_files:
        path = stack.pop()
        try:
            items = repo.get_contents(path, ref=ref)
        except GithubException:
            continue

        if not isinstance(items, list):
            items = [items]

        for it in items:
            if len(results) >= max_files:
                break
            if it.type == "dir":
                stack.append(it.path)
            elif it.type == "file" and it.path.endswith(".py"):
                results.append(it.path)

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract doc+code symbol records with hierarchy (JSONL).")

    parser.add_argument(
        "--repos", nargs="+", required=True,
        help="GitHub repositories to mine (owner/repo)."
    )
    parser.add_argument(
        "--token", default=None,
        help="GitHub token (or set GITHUB_TOKEN env var)."
    )
    parser.add_argument(
        "--output", default="data/doc_code_records.jsonl",
        help="Output JSONL file."
    )
    parser.add_argument(
        "--max-files", type=int, default=50,
        help="Max Python files per repo to scan (avoid rate limits)."
    )
    parser.add_argument(
        "--max-symbols", type=int, default=500,
        help="Max symbols per repo (stop early)."
    )
    parser.add_argument(
        "--ref", default=None,
        help="Git ref/sha/branch. Default: repo default branch."
    )

    args = parser.parse_args()

    load_dotenv()
    token = args.token or os.getenv("GITHUB_TOKEN")
    gh = Github(token) if token else Github()

    total_written = 0

    with open(args.output, "w", encoding="utf-8") as out_f:
        for repo_name in args.repos:
            print(f"Mining repo: {repo_name}")

            try:
                repo = gh.get_repo(repo_name)
            except GithubException as e:
                print(f"  ERROR: cannot access {repo_name}: {e}")
                continue

            ref = args.ref or repo.default_branch

            # resolve head sha for traceability
            try:
                head_commit = repo.get_commit(sha=ref)
                head_sha = head_commit.sha
            except GithubException:
                head_sha = ref  # fallback

            py_files = list_repo_python_files(repo, ref=ref, max_files=args.max_files)
            print(f"  Found {len(py_files)} .py files (capped at {args.max_files})")

            symbols_written = 0

            for fp in py_files:
                if symbols_written >= args.max_symbols:
                    break

                try:
                    content = repo.get_contents(fp, ref=ref).decoded_content.decode("utf-8", errors="replace")
                except GithubException:
                    continue

                records = extract_python_symbols(content)

                for r in records:
                    if symbols_written >= args.max_symbols:
                        break

                    # add repo-level metadata
                    r2 = {
                        "repository": repo_name,
                        "ref": ref,
                        "commit_sha": head_sha,
                        "file": fp,
                        **r
                    }

                    out_f.write(json.dumps(r2, ensure_ascii=False) + "\n")
                    symbols_written += 1
                    total_written += 1

            print(f"  Wrote {symbols_written} symbol records for {repo_name}")

    print(f"\nDone. Total records written: {total_written}")
    print(f"Output: {args.output}")


if __name__ == "__main__":
    main()
