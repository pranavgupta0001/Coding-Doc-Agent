"""Stage 1: Extract structured requirements from documentation text."""
from __future__ import annotations
import re
from typing import List

# Keywords for category/risk classification
_SECURITY = {'auth', 'authenticat', 'authorizat', 'permission', 'role', 'jwt',
              'token', 'ssl', 'tls', 'encrypt', 'secret', 'password', 'credential',
              'rbac', 'oauth', 'login', 'session', 'sanitiz', 'xss', 'csrf', 'inject'}

_PERFORMANCE = {'timeout', 'latency', 'throughput', 'concurrent', 'async', 'cache',
                'performance', 'fast', 'slow', 'speed', 'efficient', 'optimize',
                'millisecond', 'response time', 'bandwidth', 'throughput'}

_SAFETY = {'lock', 'race', 'deadlock', 'atomic', 'thread-safe', 'fail',
           'error', 'exception', 'retry', 'fallback', 'safe', 'validate',
           'check', 'limit', 'attempt', 'guard', 'assert', 'raise'}

_DATA = {'database', ' db ', 'sql', 'persist', 'store', 'save', 'load',
         'table', 'schema', 'record', 'field', 'column', 'index',
         'query', 'transaction', 'commit', 'rollback', 'fetch', 'select'}

_STOPWORDS = {
    'the', 'and', 'for', 'this', 'that', 'with', 'from', 'are', 'not',
    'can', 'all', 'its', 'any', 'but', 'our', 'has', 'via', 'was', 'will',
    'been', 'have', 'may', 'must', 'should', 'when', 'where', 'which',
    'what', 'how', 'each', 'into', 'only', 'also', 'then', 'than',
    'bool', 'none', 'true', 'false', 'list', 'dict', 'str', 'int',
}


def _classify(text: str):
    """Return (Category, RiskLevel) based on keywords in text."""
    from pipeline.schemas import Category, RiskLevel
    t = text.lower()
    if any(k in t for k in _SECURITY):
        return Category.SECURITY, RiskLevel.HIGH
    if any(k in t for k in _SAFETY):
        return Category.SAFETY, RiskLevel.HIGH
    if any(k in t for k in _DATA):
        return Category.DATA, RiskLevel.MED
    if any(k in t for k in _PERFORMANCE):
        return Category.PERFORMANCE, RiskLevel.MED
    return Category.FUNCTIONAL, RiskLevel.LOW


def _strictness(text: str):
    from pipeline.schemas import Strictness
    t = text.lower()
    if re.search(r'\b(must|shall|required|always|never)\b', t):
        return Strictness.MUST
    if re.search(r'\b(should|recommended|prefer|ought)\b', t):
        return Strictness.SHOULD
    return Strictness.MAY


def _keywords(text: str) -> List[str]:
    words = re.findall(r'\b[a-z][a-z0-9_]{2,}\b', text.lower())
    return [w for w in words if w not in _STOPWORDS][:10]


def categorize(doc_text: str, prompt_text: str) -> List:
    """Extract requirements from documentation text and user prompt.

    Returns a list of Requirement objects representing the claims/promises
    made in the docstring that the implementation must honour.
    """
    from pipeline.schemas import Requirement, Category, Strictness, RiskLevel

    # Collect candidate sentences from doc text
    sentences: List[str] = []
    seen: set = set()

    def _add(s: str):
        s = s.strip().rstrip('.')
        if len(s) > 12 and s not in seen:
            sentences.append(s)
            seen.add(s)

    # Split on sentence boundaries and newlines
    for chunk in re.split(r'\n+', doc_text):
        chunk = chunk.strip()
        if not chunk:
            continue
        for sent in re.split(r'(?<=[.!?])\s+', chunk):
            _add(sent)
        _add(chunk)  # also add the whole paragraph as a requirement

    # Add the prompt as a user-source requirement
    for sent in re.split(r'(?<=[.!?])\s+', prompt_text.strip()):
        s = sent.strip().rstrip('.')
        if len(s) > 12 and s not in seen:
            sentences.append(s)
            seen.add(s)

    requirements = []
    for i, text in enumerate(sentences[:12], start=1):
        cat, risk = _classify(text)
        strict = _strictness(text)
        # Upgrade risk for MUST requirements
        if strict == Strictness.MUST and risk == RiskLevel.LOW:
            risk = RiskLevel.MED
        requirements.append(Requirement(
            id=f"R{i}",
            source="doc" if i <= len(sentences) - 1 else "user",
            category=cat,
            text=text,
            keywords=_keywords(text),
            strictness=strict,
            measurable=bool(re.search(r'\b\d+\b', text)),
            acceptance_criteria=None,
            risk_level=risk,
        ))

    if not requirements:
        fallback = (doc_text.strip() or prompt_text.strip())[:200] or "undocumented"
        requirements.append(Requirement(
            id="R1",
            source="doc",
            category=Category.FUNCTIONAL,
            text=fallback,
            keywords=_keywords(fallback),
            strictness=Strictness.SHOULD,
            measurable=False,
            acceptance_criteria=None,
            risk_level=RiskLevel.LOW,
        ))

    return requirements
