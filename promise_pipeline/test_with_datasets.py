from __future__ import annotations
import argparse, json, sys, tempfile
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pipeline.categorizer import categorize
from pipeline.code_indexer import build_code_index
from pipeline.external_filter import apply_external_filter, apply_fail_closed
from pipeline.model_analyzer import HttpProvider, MockProvider, ProviderConfig, run_model_analysis

LABEL_MAP = {"No Drift":"CONSISTENT","Signature Drift":"DIRECT_MISMATCH","Behavioral Drift":"OVER_PROMISE","Constraint Drift":"OVER_PROMISE"}
DATASET_PATHS = {
    "golden":    Path(__file__).parent / "golden_dataset.jsonl",
    "synthetic": Path(__file__).parent / "synthetic_drift_records.jsonl",
    "doc_code":  Path(__file__).parent / "doc_code_records.jsonl",
}

def load_jsonl(path):
    records = []
    with open(path) as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line: continue
            try: records.append(json.loads(line))
            except json.JSONDecodeError as e: print(f"  [warn] skip line {i}: {e}")
    return records

def record_to_inputs(rec):
    doc = rec.get("doc") or ""
    symbol = rec.get("symbol_path") or "unknown"
    file_path = rec.get("file") or "module.py"
    code = rec.get("code") or ""
    context = rec.get("context") or ""
    doc_text = f"Function: {symbol}\nSignature: {rec.get('signature','')}\nDocstring:\n{doc or '(no docstring)'}\n"
    prompt_text = f"Verify that `{symbol}` in `{file_path}` matches its documented contract.\n"
    code_files = {Path(file_path).name: str(code)}
    if context:
        ctx_str = "\n\n".join(f"# {k}\n{v}" for k,v in context.items() if v) if isinstance(context, dict) else str(context)
        if ctx_str and ctx_str != str(code):
            code_files["_context.py"] = ctx_str
    return doc_text, prompt_text, code_files

def run_one(rec, provider, top_k=5):
    doc_text, prompt_text, code_files = record_to_inputs(rec)
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        for fname, content in code_files.items():
            (tmp / fname).write_text(content, encoding="utf-8")
        requirements = categorize(doc_text, prompt_text)
        code_index = build_code_index(str(tmp))
        try:
            model_output = run_model_analysis(requirements=requirements, code_index=code_index,
                doc=doc_text, user_prompt=prompt_text, provider=provider, top_k=top_k)
            filter_result = apply_external_filter(model_output, requirements, code_index)
        except Exception as e:
            filter_result = apply_fail_closed(str(e), requirements)
        decision = filter_result.decision.value if hasattr(filter_result.decision, "value") else str(filter_result.decision)
        return {"symbol": rec.get("symbol_path"), "ground_truth": rec.get("label"),
                "expected": LABEL_MAP.get(rec.get("label",""),"?"),
                "decision": decision, "label_counts": dict(Counter(i.label for i in filter_result.items)),
                "n_requirements": len(requirements)}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="golden", choices=["golden","synthetic","doc_code","all"])
    parser.add_argument("--mock", action="store_true")
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()
    datasets = ["golden","synthetic","doc_code"] if args.dataset == "all" else [args.dataset]
    provider = MockProvider() if args.mock else HttpProvider(ProviderConfig(
        provider_url="https://api.anthropic.com/v1/messages",
        api_key_env_var_name="ANTHROPIC_API_KEY",
        model_id="claude-sonnet-4-6", timeout_seconds=60, max_tokens=4096))
    for ds in datasets:
        path = DATASET_PATHS[ds]
        if not path.exists():
            print(f"[ERROR] not found: {path}"); continue
        records = [r for r in load_jsonl(path) if r.get("doc")][:args.limit]
        print(f"\n{'='*55}\nDataset: {ds}  ({len(records)} records, mock={args.mock})\n{'='*55}")
        results = []
        for i, rec in enumerate(records):
            sym = rec.get("symbol_path","?")
            print(f"  [{i+1:3d}/{len(records)}] {sym[:45]:<45}", end=" ", flush=True)
            try:
                r = run_one(rec, provider)
                results.append(r)
                print(f"{'✓' if r['decision']=='CONSISTENT' else '✗'}  {r['decision']}")
                if args.verbose: print(f"         labels={r['label_counts']}")
            except Exception as e:
                print(f"ERROR: {e}")
        decisions = Counter(r["decision"] for r in results)
        print(f"\n  Decisions: {dict(decisions)}")
        if ds == "golden":
            correct = sum(1 for r in results if r.get("expected") and (r["expected"]=="CONSISTENT")==(r["decision"]=="CONSISTENT"))
            total = sum(1 for r in results if r.get("expd") and r["expected"]!="?")
            if total: print(f"  Ground-truth alignment: {correct}/{total} ({correct/total*100:.1f}%)")

if __name__ == "__main__":
    main()
