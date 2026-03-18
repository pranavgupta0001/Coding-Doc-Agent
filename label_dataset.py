#!/usr/bin/env python3.13
"""
label_dataset.py — Run the promise-vs-evidence pipeline on a JSONL dataset
and write per-record labels + accuracy summary (golden dataset only).

Usage:
    python label_dataset.py --dataset tests/golden_dataset.jsonl --mock --limit 5
    python label_dataset.py --dataset tests/golden_dataset.jsonl --out tests/labeled_golden.jsonl
"""
from __future__ import annotations
import argparse, json, sys, tempfile, traceback
from collections import Counter
from pathlib import Path

# Ensure project root is on sys.path
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from pipeline.categorizer import categorize
from pipeline.code_indexer import build_code_index
from pipeline.external_filter import apply_external_filter, apply_fail_closed
from pipeline.model_analyzer import HttpProvider, MockProvider, ProviderConfig, run_model_analysis

# Ground-truth label → expected pipeline decision
LABEL_MAP = {
    "No Drift":         "CONSISTENT",
    "Signature Drift":  "DIRECT_MISMATCH",
    "Behavioral Drift": "OVER_PROMISE",
    "Constraint Drift": "OVER_PROMISE",
}

# Pipeline decision → coarse bucket for accuracy scoring
def _coarse(decision: str) -> str:
    return "CONSISTENT" if decision == "CONSISTENT" else "INCONSISTENT"


def load_jsonl(path: str) -> list[dict]:
    records = []
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"  [warn] skipping line {i}: {e}", file=sys.stderr)
    return records


def context_to_str(context) -> str:
    """Convert context field (dict or string) to a plain string."""
    if not context:
        return ""
    if isinstance(context, dict):
        parts = []
        for k, v in context.items():
            if v:
                parts.append(f"# {k}\n{v}")
        return "\n\n".join(parts)
    return str(context)


def record_to_inputs(rec: dict) -> tuple[str, str, dict[str, str]]:
    """Build (doc_text, prompt_text, code_files) from a dataset record."""
    symbol   = rec.get("symbol_path") or "unknown"
    file_p   = rec.get("file") or "module.py"
    sig      = rec.get("signature") or ""
    doc      = rec.get("doc") or ""
    code     = rec.get("code") or ""
    context  = rec.get("context")

    doc_text = (
        f"Function: {symbol}\n"
        f"Signature: {sig}\n"
        f"Docstring:\n{doc or '(no docstring)'}\n"
    )
    prompt_text = (
        f"Verify that `{symbol}` in `{file_p}` matches its documented contract.\n"
    )

    code_files: dict[str, str] = {Path(file_p).name: str(code)}
    ctx_str = context_to_str(context)
    if ctx_str and ctx_str.strip() != str(code).strip():
        code_files["_context.py"] = ctx_str

    return doc_text, prompt_text, code_files


def run_one(rec: dict, provider, top_k: int = 5) -> dict:
    """Run all 4 pipeline stages on a single record."""
    doc_text, prompt_text, code_files = record_to_inputs(rec)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        for fname, content in code_files.items():
            (tmp / fname).write_text(content, encoding="utf-8")

        requirements  = categorize(doc_text, prompt_text)
        code_index    = build_code_index(str(tmp))

        model_output  = None
        filter_result = None
        try:
            model_output  = run_model_analysis(
                requirements=requirements,
                code_index=code_index,
                doc=doc_text,
                user_prompt=prompt_text,
                provider=provider,
                top_k=top_k,
            )
            filter_result = apply_external_filter(model_output, requirements, code_index)
        except Exception as e:
            filter_result = apply_fail_closed(str(e), requirements)

    # --- Decision strategy ---
    # Root cause: external_filter marks all items UNVERIFIED (conf=0) so
    # filter_result.decision is always INCONSISTENT.  model_output.global_label
    # is also INCONSISTENT for all records (LLM prompt is strict by design).
    #
    # Most reliable signal: model_output.summary drift-type counts.
    # If the LLM counted zero over-promises, mismatches, and under-promises,
    # the record is effectively CONSISTENT even if global_label says otherwise.
    # Non-zero counts → genuine drift detected → trust filter_result decision.
    filter_decision = (
        filter_result.decision.value
        if hasattr(filter_result.decision, "value")
        else str(filter_result.decision)
    )

    if model_output is not None:
        s = model_output.summary
        no_drift_found = (
            s.over_promise_count == 0
            and s.direct_mismatch_count == 0
            and s.under_promise_count == 0
        )
        decision = "CONSISTENT" if no_drift_found else filter_decision
    else:
        decision = filter_decision

    # Item-level labels from model output (more granular than filter)
    if model_output is not None and model_output.items:
        label_counts = dict(Counter(
            (i.label.value if hasattr(i.label, "value") else str(i.label))
            for i in model_output.items
        ))
    else:
        label_counts = dict(Counter(
            (i.label.value if hasattr(i.label, "value") else str(i.label))
            for i in (filter_result.items if filter_result else [])
        ))

    # Confidence: from model_output items
    if model_output is not None and model_output.items:
        confidences = [i.confidence for i in model_output.items if hasattr(i, "confidence")]
    else:
        confidences = [i.confidence for i in (filter_result.items if filter_result else [])
                       if hasattr(i, "confidence")]
    confidence = round(sum(confidences) / len(confidences), 3) if confidences else 0.0

    # Best explanation
    explanation = ""
    src_items = (model_output.items if model_output and model_output.items
                 else (filter_result.items if filter_result else []))
    for item in src_items:
        lbl = item.label.value if hasattr(item.label, "value") else str(item.label)
        if lbl != "CONSISTENT":
            explanation = item.explanation
            break
    if not explanation and src_items:
        explanation = src_items[0].explanation

    # Pipeline label (most common non-CONSISTENT item label, else decision)
    non_consistent = {k: v for k, v in label_counts.items() if k != "CONSISTENT"}
    pipeline_label = (max(non_consistent, key=non_consistent.get)
                      if decision != "CONSISTENT" and non_consistent
                      else decision)
    if decision != "CONSISTENT" and label_counts:
        pipeline_label = max(label_counts, key=label_counts.get)

    result = dict(rec)  # copy all original fields
    result.update({
        "pipeline_decision":    decision,
        "pipeline_label":       pipeline_label,
        "pipeline_confidence":  confidence,
        "pipeline_explanation": explanation,
        "n_requirements":       len(requirements),
        "label_counts":         label_counts,
    })

    # Ground-truth comparison (golden dataset)
    gt = rec.get("label")
    if gt is not None:
        expected = LABEL_MAP.get(gt, "?")
        result["label_match"] = (
            _coarse(expected) == _coarse(decision) if expected != "?" else None
        )

    return result


def print_summary(results: list[dict], dataset_name: str, is_golden: bool) -> None:
    decisions = Counter(r["pipeline_decision"] for r in results)
    labels    = Counter(r["pipeline_label"]    for r in results)
    print(f"\n  Decisions : {dict(decisions)}")
    print(f"  Labels    : {dict(labels)}")

    if is_golden:
        scored = [r for r in results if r.get("label_match") is not None]
        if scored:
            correct = sum(1 for r in scored if r["label_match"])
            total   = len(scored)
            print(f"\n  Ground-truth accuracy : {correct}/{total}  ({correct/total*100:.1f}%)")
            # Per-class breakdown
            by_gt: dict[str, list] = {}
            for r in scored:
                gt = r.get("label", "unknown")
                by_gt.setdefault(gt, []).append(r["label_match"])
            print("  Per-class breakdown:")
            for gt_label, matches in sorted(by_gt.items()):
                n = len(matches)
                c = sum(matches)
                print(f"    {gt_label:<22} {c:3d}/{n}  ({c/n*100:.0f}%)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Label a JSONL dataset with the pipeline.")
    parser.add_argument("--dataset", required=True,
                        help="Path to input .jsonl file")
    parser.add_argument("--out",     default=None,
                        help="Path to write labeled .jsonl (default: print summary only)")
    parser.add_argument("--mock",    action="store_true",
                        help="Use MockProvider instead of real LLM")
    parser.add_argument("--limit",   type=int, default=0,
                        help="Process at most N records (0 = all)")
    parser.add_argument("--top-k",  type=int, default=5,
                        help="Top-k code chunks passed to model (default 5)")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        print(f"[ERROR] dataset not found: {dataset_path}", file=sys.stderr)
        sys.exit(1)

    # Build provider
    if args.mock:
        provider = MockProvider()
    else:
        provider = HttpProvider(ProviderConfig(
            provider_url="https://api.anthropic.com/v1/messages",
            api_key_env_var_name="ANTHROPIC_API_KEY",
            model_id="claude-haiku-4-5-20251001",
            timeout_seconds=60,
            max_tokens=4096,
        ))

    records = load_jsonl(str(dataset_path))
    if args.limit:
        records = records[: args.limit]

    is_golden = "golden" in dataset_path.name
    print(f"\n{'='*60}")
    print(f"Dataset : {dataset_path.name}  ({len(records)} records)")
    print(f"Provider: {'MockProvider' if args.mock else 'Claude claude-haiku-4-5-20251001'}")
    print(f"Decision strategy: summary drift-counts=0 → CONSISTENT, else filter")
    print(f"{'='*60}")

    results: list[dict] = []
    errors = 0
    for i, rec in enumerate(records, start=1):
        symbol = rec.get("symbol_path") or "?"
        print(f"  [{i:4d}/{len(records)}] {symbol[:50]:<50}", end=" ", flush=True)
        try:
            r = run_one(rec, provider, top_k=args.top_k)
            results.append(r)
            status = "✓" if r["pipeline_decision"] == "CONSISTENT" else "✗"
            match_str = ""
            if is_golden and "label_match" in r:
                match_str = " GT=" + ("✓" if r["label_match"] else "✗")
            print(f"{status}  {r['pipeline_decision']}{match_str}")
            if args.verbose:
                print(f"         labels={r['label_counts']}")
        except Exception as e:
            errors += 1
            print(f"ERROR: {e}")
            if args.verbose:
                traceback.print_exc()

    print_summary(results, dataset_path.stem, is_golden)
    print(f"\n  Processed: {len(results)}/{len(records)}  errors={errors}")

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            for r in results:
                f.write(json.dumps(r, default=str) + "\n")
        print(f"  Written  : {out_path}  ({len(results)} records)")


if __name__ == "__main__":
    main()
