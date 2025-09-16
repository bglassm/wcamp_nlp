# pipeline/summarizer_comm.py
"""
커뮤니티 요약/문장분할 래퍼.
- 기존 pipeline.summarizer에 같은 기능이 있으면 우선 사용
- 없으면 여기의 안전한 기본 구현으로 대체
"""

from typing import List
import re

# 1) 기존 summarizer가 있으면 우선 사용
try:
    from pipeline import summarizer as _base
except Exception:
    _base = None

# -------- 공용 문장 분할 --------
_SENT_SPLIT = re.compile(r"(?<=[.!?。…])\s+|(?<=[\n])+")

def _fallback_split_sentences(text: str) -> List[str]:
    if not isinstance(text, str) or not text.strip():
        return []
    sents = [s.strip() for s in _SENT_SPLIT.split(text) if s.strip()]
    if len(sents) <= 1 and "\n" in text:
        sents = [x.strip() for x in text.split("\n") if x.strip()]
    return sents

if _base is not None and hasattr(_base, "split_sentences"):
    split_sentences = _base.split_sentences  # type: ignore[attr-defined]
else:
    split_sentences = _fallback_split_sentences  # noqa


# -------- 추출 요약(경량) --------
def _fallback_extractive_summary(text: str, max_sentences: int = 10) -> List[str]:
    # 의존성 최소화를 위해 간단 점수: 문장 길이/중복 제거 + 키구두점 포함 여부
    sents = split_sentences(text)
    if not sents:
        return []
    if len(sents) <= max_sentences:
        return sents
    # 길이 가중 + 구두점/숫자/괄호 포함시 약간 가산
    def _score(s: str) -> float:
        L = len(s)
        bonus = 0.0
        if any(ch in s for ch in "[](){}“”\"'"):
            bonus += 0.05
        if any(ch.isdigit() for ch in s):
            bonus += 0.05
        return L / 100.0 + bonus  # 아주 단순
    ranked = sorted(sents, key=_score, reverse=True)
    picked, seen = [], set()
    for s in ranked:
        sig = s[:32]
        if sig in seen:
            continue
        seen.add(sig)
        picked.append(s)
        if len(picked) >= max_sentences:
            break
    return picked

if _base is not None and hasattr(_base, "extractive_summary"):
    extractive_summary = _base.extractive_summary  # type: ignore[attr-defined]
else:
    extractive_summary = _fallback_extractive_summary  # noqa


# -------- 모드 A가 쓰는 summarize_row --------
def summarize_row(title: str, body: str, summary: str, max_sentences: int = 10) -> List[str]:
    """
    제공된 요약(summary)이 있으면 그것을 문장 분할해서 사용.
    없으면 제목+본문으로 추출 요약.
    """
    if isinstance(summary, str) and summary.strip():
        return split_sentences(summary.strip())[:max_sentences]

    text = ((title or "").strip() + "\n" + (body or "").strip()).strip()
    if _base is not None and hasattr(_base, "summarize_row"):
        # 기존 구현이 있으면 그것을 그대로 사용
        try:
            return _base.summarize_row(title, body, summary, max_sentences)  # type: ignore[attr-defined]
        except Exception:
            pass
    # 기본 추출 요약
    return extractive_summary(text, max_sentences=max_sentences)
