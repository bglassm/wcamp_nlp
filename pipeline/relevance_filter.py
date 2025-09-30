# pipeline/relevance_filter.py

from typing import List, Tuple
import numpy as np

def _contains_any(text: str, terms: List[str]) -> bool:
    if not terms or not isinstance(text, str):
        return False
    return any(t and t in text for t in terms)

def build_alias_queries(aliases: List[str]) -> List[str]:
    q = [a for a in (aliases or []) if a]
    # dedup
    out = []
    for x in q:
        if x not in out:
            out.append(x)
    return out[:32]

def build_facet_queries(facet_terms: List[str]) -> List[str]:
    q = [t for t in (facet_terms or []) if t]
    out = []
    for x in q:
        if x not in out:
            out.append(x)
    return out[:64]

def _lexical_bonus(sent: str, terms: List[str]) -> float:
    """facet 단어가 문장에 직접 등장하면 소폭 가점(최대 +0.3)."""
    if not terms:
        return 0.0
    hits = sum(1 for t in terms if t and t in sent)
    return min(0.1 * hits, 0.3)

def dual_score_relevance(
    sentences: List[str],
    alias_q: List[str],
    facet_q: List[str],
    embedder,
    facet_bonus_terms: List[str],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    alias_sim: 문장↔별칭 유사도(max)
    facet_sim: 문장↔facet 용어 유사도(max)
    total    : max(alias_sim, facet_sim) + lexical bonus
    """
    if not sentences:
        return np.array([]), np.array([]), np.array([])
    S = embedder.encode(sentences, batch_size=256, show_progress_bar=False, convert_to_numpy=True, normalize_embeddings=True)
    alias_sim = np.zeros(len(sentences), dtype=np.float32)
    facet_sim = np.zeros(len(sentences), dtype=np.float32)
    if alias_q:
        A = embedder.encode(alias_q, batch_size=64, show_progress_bar=False, convert_to_numpy=True, normalize_embeddings=True)
        alias_sim = (S @ A.T).max(axis=1)
    if facet_q:
        F = embedder.encode(facet_q, batch_size=64, show_progress_bar=False, convert_to_numpy=True, normalize_embeddings=True)
        facet_sim = (S @ F.T).max(axis=1)
    base = np.maximum(alias_sim, facet_sim)
    bonus = np.array([_lexical_bonus(s, facet_bonus_terms) for s in sentences], dtype=np.float32)
    total = base + bonus
    return alias_sim, facet_sim, total

def keep_mask_gated(
    sentences: List[str],
    alias_sim: np.ndarray,
    facet_sim: np.ndarray,
    total: np.ndarray,
    *,
    tau: float,
    alias_tau: float,
    lexical_aliases: List[str],
    ban_terms: List[str] | None = None,
    ban_mode: str = "soft",  # soft|strict|off
) -> np.ndarray:
    """
    게이트 규칙:
      1) (문장에 별칭이 직접 포함) OR (alias_sim >= alias_tau)
      2) total >= tau
      3) ban_mode 적용
         - strict: 금칙어가 있고 별칭 직접 언급이 없으면 제외
         - soft  : 금칙어가 있고 별칭 직접 언급이 없으면 게이트 불가(=제외)
         - off   : 금칙어 무시
    """
    n = len(total)
    if n == 0:
        return np.array([], dtype=bool)

    lex_alias = np.array([_contains_any(s, lexical_aliases) for s in sentences], dtype=bool)
    banned   = np.array([_contains_any(s, ban_terms or []) for s in sentences], dtype=bool)

    gate_ok = lex_alias | (alias_sim >= float(alias_tau))

    if ban_mode == "strict":
        # 금칙어가 있지만 별칭 직접 언급이 있으면 통과 허용
        ban_ok = (~banned) | lex_alias
        gate_ok = gate_ok & ban_ok
    elif ban_mode == "soft":
        # 금칙어가 있고 별칭 직접 언급이 없으면 게이트 실패
        gate_ok = gate_ok & (~(banned & ~lex_alias))
    else:
        # off: 아무 영향 없음
        pass

    return (total >= float(tau)) & gate_ok


# (하위호환: 기존 단일쿼리 버전 — 여전히 남겨둠)
def build_queries(product_aliases: List[str], facet_terms_flat: List[str]) -> List[str]:
    q = []
    q.extend(product_aliases or [])
    q.extend(facet_terms_flat or [])
    out = []
    for x in q:
        if x and x not in out:
            out.append(x)
    return out[:32]

def score_relevance(sentences: List[str], queries: List[str], embedder, facet_bonus_terms: List[str]) -> np.ndarray:
    if not sentences:
        return np.array([])
    S = embedder.encode(sentences, batch_size=256, show_progress_bar=False, convert_to_numpy=True, normalize_embeddings=True)
    if not queries:
        base = np.zeros(len(sentences), dtype=np.float32)
    else:
        Q = embedder.encode(queries, batch_size=64, show_progress_bar=False, convert_to_numpy=True, normalize_embeddings=True)
        base = (S @ Q.T).max(axis=1)
    bonus = np.array([_lexical_bonus(s, facet_bonus_terms) for s in sentences], dtype=np.float32)
    return base + bonus

def keep_mask(scores: np.ndarray, tau: float) -> np.ndarray:
    return scores >= tau
