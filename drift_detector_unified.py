"""

This module implements the drift detection methodology on both dataset formats:
1. drift_records.jsonl - Symbol-level format with before/after
2. drift_dataset_final_87.jsonl - Segment-level format with before_segments/after_segments
"""

import json
import re
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Any
from enum import Enum
from collections import Counter
import sys


# ============================================================================
# DRIFT TYPE DEFINITIONS
# ============================================================================

class DriftType(Enum):
    """Taxonomy of drift types we can detect."""
    PARAM_MISSING = "Parameter in code but missing from docs"
    PARAM_EXTRA = "Parameter in docs but not in code"
    PARAM_TYPE_MISMATCH = "Parameter type mismatch"
    RETURN_TYPE_MISMATCH = "Return type mismatch"
    BEHAVIOR_DRIFT = "Code changed, docs unchanged"
    DOC_UPDATE_ONLY = "Docs updated (drift fix)"
    SIGNATURE_DRIFT = "Signature changed"
    DOCSTRING_SYNC = "Docstring synchronized"
    TYPO_FIX = "Typo or minor text fix"
    EXAMPLE_UPDATE = "Example code updated"
    NO_DRIFT = "No drift detected"


# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class DriftResult:
    """Result of drift analysis for a single record."""
    record_id: str
    repository: str
    file_path: str
    symbol: str
    detected_drifts: List[DriftType]
    drift_score: float
    explanation: str
    doc_length_before: int
    doc_length_after: int
    code_length_before: int
    code_length_after: int


@dataclass
class DatasetMetrics:
    """Aggregated metrics for the entire dataset."""
    total_records: int
    drifts_detected: int
    drift_type_distribution: Dict[str, int]
    avg_drift_score: float
    per_repo_stats: Dict[str, Dict]
    detection_summary: str


# ============================================================================
# DRIFT ANALYZER
# ============================================================================

class DriftAnalyzer:
    """Analyzes commits for documentation drift patterns."""
    
    # Keywords indicating documentation fixes
    DOC_FIX_KEYWORDS = [
        'fix doc', 'update doc', 'correct doc', 'doc fix',
        'docstring', 'typo', 'spelling', 'grammar',
        'update comment', 'fix comment', 'sync doc',
        'documentation', 'clarify', 'improve doc'
    ]
    
    # Keywords indicating formula/math fixes
    FORMULA_KEYWORDS = [
        'fix formula', 'correct formula', 'update formula',
        'equation', 'mathematical', 'notation'
    ]
    
    def analyze_commit_message(self, message: str) -> List[DriftType]:
        """Analyze commit message for drift indicators."""
        drifts = []
        message_lower = message.lower()
        
        # Check for doc-related keywords
        for keyword in self.DOC_FIX_KEYWORDS:
            if keyword in message_lower:
                drifts.append(DriftType.DOCSTRING_SYNC)
                break
        
        # Check for formula keywords
        for keyword in self.FORMULA_KEYWORDS:
            if keyword in message_lower:
                drifts.append(DriftType.DOC_UPDATE_ONLY)
                break
        
        # Check for typo fixes
        if 'typo' in message_lower or 'spelling' in message_lower:
            drifts.append(DriftType.TYPO_FIX)
        
        return drifts
    
    def analyze_patch(self, patch: str) -> Tuple[List[DriftType], Dict]:
        """Analyze git patch for drift patterns."""
        drifts = []
        stats = {
            'lines_added': 0,
            'lines_removed': 0,
            'doc_lines_changed': False,
            'code_lines_changed': False
        }
        
        if not patch:
            return drifts, stats
        
        lines = patch.split('\n')
        in_docstring = False
        
        for line in lines:
            if line.startswith('+') and not line.startswith('+++'):
                stats['lines_added'] += 1
                if '"""' in line or "'''" in line:
                    in_docstring = not in_docstring
                    stats['doc_lines_changed'] = True
                elif '#:' in line or line.strip().startswith('#'):
                    stats['doc_lines_changed'] = True
                elif in_docstring:
                    stats['doc_lines_changed'] = True
                else:
                    stats['code_lines_changed'] = True
                    
            elif line.startswith('-') and not line.startswith('---'):
                stats['lines_removed'] += 1
                if '"""' in line or "'''" in line:
                    stats['doc_lines_changed'] = True
                elif '#:' in line or line.strip().startswith('#'):
                    stats['doc_lines_changed'] = True
        
        # Classify based on changes
        if stats['doc_lines_changed'] and not stats['code_lines_changed']:
            drifts.append(DriftType.DOC_UPDATE_ONLY)
        elif stats['code_lines_changed'] and not stats['doc_lines_changed']:
            drifts.append(DriftType.BEHAVIOR_DRIFT)
        elif stats['doc_lines_changed'] and stats['code_lines_changed']:
            drifts.append(DriftType.DOCSTRING_SYNC)
        
        return drifts, stats
    
    def analyze_segments(self, before_segments: List[Dict], after_segments: List[Dict]) -> List[DriftType]:
        """Compare before and after segments for drift."""
        drifts = []
        
        # Build lookup by filename + start_line
        before_lookup = {}
        for seg in before_segments:
            key = f"{seg.get('filename', '')}:{seg.get('start_line', 0)}"
            before_lookup[key] = seg
        
        after_lookup = {}
        for seg in after_segments:
            key = f"{seg.get('filename', '')}:{seg.get('start_line', 0)}"
            after_lookup[key] = seg
        
        # Compare matching segments
        for key in before_lookup:
            if key in after_lookup:
                before_doc = before_lookup[key].get('documentation', '')
                after_doc = after_lookup[key].get('documentation', '')
                
                before_code = before_lookup[key].get('code', '')
                after_code = after_lookup[key].get('code', '')
                
                # Check for documentation changes
                if before_doc != after_doc:
                    drifts.append(DriftType.DOC_UPDATE_ONLY)
                
                # Check for code changes
                if before_code != after_code:
                    drifts.append(DriftType.BEHAVIOR_DRIFT)
        
        return drifts
    
    def extract_params_from_signature(self, signature: str) -> List[str]:
        """Extract parameter names from function signature."""
        params = []
        
        match = re.match(r'def\s+\w+\s*\((.*?)\)\s*:', signature, re.DOTALL)
        if not match:
            return params
        
        params_str = match.group(1)
        
        # Simple parsing
        depth = 0
        current = ""
        
        for char in params_str:
            if char in '([{':
                depth += 1
                current += char
            elif char in ')]}':
                depth -= 1
                current += char
            elif char == ',' and depth == 0:
                param_name = current.split('=')[0].split(':')[0].strip()
                param_name = param_name.lstrip('*')
                if param_name and param_name not in ('self', 'cls'):
                    params.append(param_name)
                current = ""
            else:
                current += char
        
        if current.strip():
            param_name = current.split('=')[0].split(':')[0].strip()
            param_name = param_name.lstrip('*')
            if param_name and param_name not in ('self', 'cls'):
                params.append(param_name)
        
        return params
    
    def check_param_documentation(self, params: List[str], docstring: str) -> List[DriftType]:
        """Check if all parameters are documented."""
        drifts = []
        
        if not docstring:
            if params:
                for _ in params:
                    drifts.append(DriftType.PARAM_MISSING)
            return drifts
        
        docstring_lower = docstring.lower()
        
        for param in params:
            # Check if parameter name appears in docstring
            if param.lower() not in docstring_lower:
                drifts.append(DriftType.PARAM_MISSING)
        
        return drifts


# ============================================================================
# MAIN DETECTOR
# ============================================================================

class UnifiedDriftDetector:
    """Unified detector that handles both dataset formats."""
    
    def __init__(self):
        self.analyzer = DriftAnalyzer()
    
    def detect_format(self, record: Dict) -> str:
        """Detect which dataset format the record uses."""
        if 'before' in record and 'after' in record:
            return 'symbol_level'  # drift_records.jsonl format
        elif 'before_segments' in record:
            return 'segment_level'  # drift_dataset_final_87.jsonl format
        else:
            return 'unknown'
    
    def analyze_record(self, record: Dict) -> DriftResult:
        """Analyze a single record regardless of format."""
        
        record_format = self.detect_format(record)
        drifts = []
        explanations = []
        
        # Extract common fields
        repo = record.get('repository', 'unknown')
        commit_sha = record.get('commit_sha', 'unknown')[:8]
        file_path = record.get('file', 'unknown')
        commit_message = record.get('commit_message', '')
        patch = record.get('patch', '')
        
        # Analyze commit message
        msg_drifts = self.analyzer.analyze_commit_message(commit_message)
        drifts.extend(msg_drifts)
        if msg_drifts:
            explanations.append(f"Commit message indicates: {', '.join(d.value for d in msg_drifts)}")
        
        # Analyze patch
        patch_drifts, patch_stats = self.analyzer.analyze_patch(patch)
        drifts.extend(patch_drifts)
        
        # Format-specific analysis
        doc_len_before = 0
        doc_len_after = 0
        code_len_before = 0
        code_len_after = 0
        symbol = 'unknown'
        
        if record_format == 'symbol_level':
            # Handle drift_records.jsonl format
            symbol = record.get('symbol', 'unknown')
            before = record.get('before', {}) or {}
            after = record.get('after', {}) or {}
            
            doc_len_before = len(before.get('documentation', '') or '')
            doc_len_after = len(after.get('documentation', '') or '')
            code_len_before = len(before.get('code', '') or '')
            code_len_after = len(after.get('code', '') or '')
            
            # Check parameter documentation
            signature = before.get('signature', '') or ''
            docstring = before.get('documentation', '') or ''
            params = self.analyzer.extract_params_from_signature(signature)
            param_drifts = self.analyzer.check_param_documentation(params, docstring)
            drifts.extend(param_drifts)
            
            if param_drifts:
                explanations.append(f"Found {len(param_drifts)} undocumented parameters")
            
            # Check ground truth labels
            labels = record.get('labels', {})
            if labels.get('doc_changed') and not labels.get('code_changed'):
                if DriftType.DOC_UPDATE_ONLY not in drifts:
                    drifts.append(DriftType.DOC_UPDATE_ONLY)
                    explanations.append("Ground truth: documentation was fixed")
            
        elif record_format == 'segment_level':
            # Handle drift_dataset_final_87.jsonl format
            before_segments = record.get('before_segments', [])
            after_segments = record.get('after_segments', [])
            
            # Calculate lengths
            for seg in before_segments:
                doc_len_before += len(seg.get('documentation', ''))
                code_len_before += len(seg.get('code', ''))
            
            for seg in after_segments:
                doc_len_after += len(seg.get('documentation', ''))
                code_len_after += len(seg.get('code', ''))
            
            # Compare segments
            seg_drifts = self.analyzer.analyze_segments(before_segments, after_segments)
            drifts.extend(seg_drifts)
            
            # Extract symbol from first segment
            if before_segments:
                code = before_segments[0].get('code', '')
                match = re.search(r'(?:def|class)\s+(\w+)', code)
                if match:
                    symbol = match.group(1)
        
        # Remove duplicates
        drifts = list(set(drifts))
        
        # Calculate drift score
        if not drifts:
            drifts.append(DriftType.NO_DRIFT)
            drift_score = 0.0
        else:
            # Score based on severity
            severity_weights = {
                DriftType.PARAM_MISSING: 0.3,
                DriftType.PARAM_EXTRA: 0.2,
                DriftType.PARAM_TYPE_MISMATCH: 0.3,
                DriftType.BEHAVIOR_DRIFT: 0.4,
                DriftType.DOC_UPDATE_ONLY: 0.2,
                DriftType.SIGNATURE_DRIFT: 0.5,
                DriftType.DOCSTRING_SYNC: 0.2,
                DriftType.TYPO_FIX: 0.1,
                DriftType.NO_DRIFT: 0.0,
            }
            drift_score = min(1.0, sum(severity_weights.get(d, 0.2) for d in drifts))
        
        # Build explanation
        if explanations:
            explanation = "; ".join(explanations)
        else:
            explanation = "Detected: " + ", ".join(d.value for d in drifts)
        
        return DriftResult(
            record_id=commit_sha,
            repository=repo,
            file_path=file_path,
            symbol=symbol,
            detected_drifts=drifts,
            drift_score=drift_score,
            explanation=explanation,
            doc_length_before=doc_len_before,
            doc_length_after=doc_len_after,
            code_length_before=code_len_before,
            code_length_after=code_len_after
        )
    
    def analyze_dataset(self, records: List[Dict]) -> Tuple[List[DriftResult], DatasetMetrics]:
        """Analyze entire dataset."""
        
        results = []
        drift_counts = Counter()
        repo_stats = {}
        total_score = 0.0
        
        for record in records:
            result = self.analyze_record(record)
            results.append(result)
            total_score += result.drift_score
            
            # Count drift types
            for drift in result.detected_drifts:
                drift_counts[drift.name] += 1
            
            # Track per-repo stats
            repo = result.repository
            if repo not in repo_stats:
                repo_stats[repo] = {
                    'total': 0,
                    'drifts_detected': 0,
                    'avg_score': 0.0,
                    'scores': []
                }
            repo_stats[repo]['total'] += 1
            repo_stats[repo]['scores'].append(result.drift_score)
            if DriftType.NO_DRIFT not in result.detected_drifts:
                repo_stats[repo]['drifts_detected'] += 1
        
        # Calculate per-repo averages
        for repo in repo_stats:
            scores = repo_stats[repo]['scores']
            repo_stats[repo]['avg_score'] = sum(scores) / len(scores) if scores else 0.0
            del repo_stats[repo]['scores']  # Remove temporary list
        
        total = len(results)
        drifts_detected = sum(1 for r in results if DriftType.NO_DRIFT not in r.detected_drifts)
        avg_score = total_score / total if total > 0 else 0.0
        
        # Generate summary
        summary = f"Analyzed {total} records across {len(repo_stats)} repositories. "
        summary += f"Detected drift in {drifts_detected} ({drifts_detected/total*100:.1f}%) records. "
        summary += f"Average drift score: {avg_score:.3f}"
        
        metrics = DatasetMetrics(
            total_records=total,
            drifts_detected=drifts_detected,
            drift_type_distribution=dict(drift_counts),
            avg_drift_score=avg_score,
            per_repo_stats=repo_stats,
            detection_summary=summary
        )
        
        return results, metrics


# ============================================================================
# REPORT GENERATOR
# ============================================================================

def generate_detailed_report(results: List[DriftResult], metrics: DatasetMetrics) -> str:
    """Generate comprehensive analysis report."""
    
    lines = []
    lines.append("=" * 80)
    lines.append("REQUIREMENTS DRIFT DETECTION REPORT")
    lines.append("ECS 260 Project - Devarchith & Pranav")
    lines.append("=" * 80)
    lines.append("")
    
    # Executive Summary
    lines.append("EXECUTIVE SUMMARY")
    lines.append("-" * 40)
    lines.append(metrics.detection_summary)
    lines.append("")
    
    # Key Metrics
    lines.append("KEY METRICS")
    lines.append("-" * 40)
    lines.append(f"  Total Records:       {metrics.total_records}")
    lines.append(f"  Drifts Detected:     {metrics.drifts_detected}")
    lines.append(f"  Detection Rate:      {metrics.drifts_detected/metrics.total_records*100:.1f}%")
    lines.append(f"  Avg Drift Score:     {metrics.avg_drift_score:.3f}")
    lines.append("")
    
    # Drift Type Distribution
    lines.append("DRIFT TYPE DISTRIBUTION")
    lines.append("-" * 40)
    for drift_type, count in sorted(metrics.drift_type_distribution.items(), 
                                     key=lambda x: -x[1]):
        pct = count / metrics.total_records * 100
        bar = "█" * int(pct / 5) + "░" * (20 - int(pct / 5))
        lines.append(f"  {drift_type:25s} {bar} {count:3d} ({pct:5.1f}%)")
    lines.append("")
    
    # Per-Repository Analysis
    lines.append("PER-REPOSITORY ANALYSIS")
    lines.append("-" * 40)
    for repo, stats in sorted(metrics.per_repo_stats.items(), 
                               key=lambda x: -x[1]['drifts_detected']):
        pct = stats['drifts_detected'] / stats['total'] * 100 if stats['total'] > 0 else 0
        lines.append(f"  {repo}")
        lines.append(f"    Records: {stats['total']}, Drifts: {stats['drifts_detected']} ({pct:.1f}%)")
        lines.append(f"    Avg Score: {stats['avg_score']:.3f}")
    lines.append("")
    
    # High-Severity Drifts (score > 0.5)
    high_severity = [r for r in results if r.drift_score > 0.5]
    lines.append(f"HIGH-SEVERITY DRIFTS (score > 0.5): {len(high_severity)} records")
    lines.append("-" * 40)
    for i, result in enumerate(high_severity[:15]):
        lines.append(f"\n  [{i+1}] {result.repository} :: {result.symbol}")
        lines.append(f"      File: {result.file_path}")
        lines.append(f"      Score: {result.drift_score:.2f}")
        lines.append(f"      Types: {', '.join(d.name for d in result.detected_drifts)}")
        lines.append(f"      Explanation: {result.explanation[:80]}...")
    lines.append("")
    
    # Document Size Analysis
    lines.append("DOCUMENTATION SIZE ANALYSIS")
    lines.append("-" * 40)
    total_doc_before = sum(r.doc_length_before for r in results)
    total_doc_after = sum(r.doc_length_after for r in results)
    total_code_before = sum(r.code_length_before for r in results)
    total_code_after = sum(r.code_length_after for r in results)
    
    doc_change = ((total_doc_after - total_doc_before) / total_doc_before * 100) if total_doc_before > 0 else 0
    code_change = ((total_code_after - total_code_before) / total_code_before * 100) if total_code_before > 0 else 0
    
    lines.append(f"  Total Doc Size (Before):  {total_doc_before:,} chars")
    lines.append(f"  Total Doc Size (After):   {total_doc_after:,} chars ({doc_change:+.1f}%)")
    lines.append(f"  Total Code Size (Before): {total_code_before:,} chars")
    lines.append(f"  Total Code Size (After):  {total_code_after:,} chars ({code_change:+.1f}%)")
    lines.append("")
    
    lines.append("=" * 80)
    lines.append("END OF REPORT")
    lines.append("=" * 80)
    
    return "\n".join(lines)


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def load_dataset(filepath: str) -> List[Dict]:
    """Load JSONL dataset file."""
    records = []
    with open(filepath, 'r') as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


def main():
    """Main entry point."""
    
    print("=" * 60)
    print("Requirements Drift Detector v2.0 (Unified)")
    print("ECS 260 Project - Devarchith & Pranav")
    print("=" * 60)
    print()
    
    # Default files to analyze (current directory)
    files_to_analyze = [
        "drift_records.jsonl",
        "drift_dataset_final_87.jsonl"
    ]
    
    # Override with command line argument if provided
    if len(sys.argv) > 1:
        files_to_analyze = [sys.argv[1]]
    
    detector = UnifiedDriftDetector()
    all_results = []
    
    for filepath in files_to_analyze:
        print(f"\nAnalyzing: {filepath}")
        print("-" * 50)
        
        try:
            records = load_dataset(filepath)
            print(f"  Loaded {len(records)} records")
            
            results, metrics = detector.analyze_dataset(records)
            all_results.extend(results)
            
            print(f"  {metrics.detection_summary}")
            
        except FileNotFoundError:
            print(f"  File not found: {filepath}")
        except Exception as e:
            print(f"  Error: {str(e)}")
    
    if all_results:
        # Generate combined report
        print("\n" + "=" * 60)
        print("Generating combined report...")
        
        # Re-analyze combined results
        combined_records = []
        for filepath in files_to_analyze:
            try:
                combined_records.extend(load_dataset(filepath))
            except:
                pass
        
        results, metrics = detector.analyze_dataset(combined_records)
        report = generate_detailed_report(results, metrics)
        
        # Save report (in current directory)
        report_file = "drift_analysis_report.txt"
        with open(report_file, 'w') as f:
            f.write(report)
        print(f"Report saved to: {report_file}")
        
        # Save JSON results
        json_results = []
        for r in results:
            json_results.append({
                'record_id': r.record_id,
                'repository': r.repository,
                'file_path': r.file_path,
                'symbol': r.symbol,
                'drift_score': r.drift_score,
                'detected_drifts': [d.name for d in r.detected_drifts],
                'explanation': r.explanation,
                'doc_change': r.doc_length_after - r.doc_length_before,
                'code_change': r.code_length_after - r.code_length_before
            })
        
        json_file = "drift_results.json"
        with open(json_file, 'w') as f:
            json.dump({
                'summary': {
                    'total_records': metrics.total_records,
                    'drifts_detected': metrics.drifts_detected,
                    'detection_rate': metrics.drifts_detected / metrics.total_records,
                    'avg_drift_score': metrics.avg_drift_score,
                },
                'drift_type_distribution': metrics.drift_type_distribution,
                'per_repo_stats': metrics.per_repo_stats,
                'results': json_results
            }, f, indent=2)
        print(f"JSON results saved to: {json_file}")
        
        # Print the report
        print("\n" + report)


if __name__ == "__main__":
    main()
