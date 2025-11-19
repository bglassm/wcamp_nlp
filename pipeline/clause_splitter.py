# pipeline/clause_splitter.py
# Smart, hybrid clause splitter with minimal API change.
# - Signature stays the same: split_clauses(df, text_col="review", connectives=None, id_col="review_id")
# - Greedy segmentation using connectors + semantic/topic shift gating
# - Avoids conditional splits ("~라면/다면/면") by default
# - Optional embedding-based gating (config.clause_split.use_semantic_gating)

from __future__ import annotations
from typing import List, Tuple, Optional
import re
import numpy as np
import pandas as pd
import logging

import config
import kss

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Connective dictionaries (Korean)
#   NOTE: We do not split on short 1~2-char endings by default except in
#   explicit enumeration patterns (…도 …고 …도 …)
# ---------------------------------------------------------------------------
CONN = {
    "contrast": [
        "그럼에도 불구하고", "에도 불구하고", "인데도 불구하고",
        "그렇긴 하지만", "그렇긴 한데", "기는 하지만", "긴 하지만", "긴 한데",
        "그렇지만", "하지만", "그러나", "다만", "반면에", "반대로", "오히려",
    ],
    "turn": ["그런데", "한편"],
    "additive": ["그리고", "또한", "게다가", "더불어", "나아가", "뿐만 아니라"],
    "cause": ["그래서", "그러므로", "따라서", "그러니까", "그러니", "때문에", "덕분에"],
    "conditional": ["만약", "만일", "만약에", "만일에", "이라면", "라면", "다면"],  # default: do NOT split
}

ENUMERATION_GUARD = True  # use special pattern for "…도 …고 …도 …"
ENUM_PAT = re.compile(r"도\s*[^\s]{1,8}\s*고\s*")  # e.g., "맛도 좋고 "

TOKEN_RE = re.compile(r"[가-힣A-Za-z]{2,}")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _tokens(s: str) -> List[str]:
    return TOKEN_RE.findall(s)


def _jaccard(a: List[str], b: List[str]) -> float:
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 1.0
    return len(sa & sb) / max(1, len(sa | sb))


_model = None
_semantic_model_initialized = False
_semantic_gating_enabled = False


def _clause_split_config():
    return getattr(config, "clause_split", None)


def _resolve_semantic_settings() -> Tuple[Optional[str], Optional[str]]:
    """Return (model_name, device) for the semantic gate."""

    cfg = _clause_split_config()
    semantic_cfg = getattr(config, "semantic", None)

    model_name: Optional[str] = None
    device: Optional[str] = None

    if cfg:
        model_name = getattr(cfg, "semantic_model", None)
        device = getattr(cfg, "device", None)

    if not model_name and semantic_cfg:
        model_name = getattr(semantic_cfg, "model", None)
    if not device and semantic_cfg:
        device = getattr(semantic_cfg, "device", None)

    if not device:
        device = getattr(config, "DEVICE", None)
    return model_name, device


def _ensure_semantic_model() -> bool:
    """Load the semantic model exactly once if gating is enabled."""

    global _model, _semantic_model_initialized, _semantic_gating_enabled

    if _semantic_model_initialized:
        return _semantic_gating_enabled and _model is not None

    cfg = _clause_split_config()
    use_semantic = getattr(cfg, "use_semantic_gating", getattr(config, "SMART_SPLIT_USE_EMBEDDING", True))

    if not use_semantic:
        logger.info("Clause splitter semantic model disabled; falling back to rule-based splitting only.")
        _semantic_model_initialized = True
        _semantic_gating_enabled = False
        _model = None
        return False

    model_name, device = _resolve_semantic_settings()
    if not model_name:
        logger.warning("Clause splitter semantic model disabled; falling back to rule-based splitting only. Reason: missing config.clause_split.semantic_model")
        _semantic_model_initialized = True
        _semantic_gating_enabled = False
        _model = None
        return False

    try:
        from sentence_transformers import SentenceTransformer

        logger.info("Clause splitter semantic model: %s", model_name)
        if device:
            logger.info("Clause splitter semantic device: %s", device)
        _model = SentenceTransformer(model_name, device=device)
        _semantic_model_initialized = True
        _semantic_gating_enabled = True
        return True
    except Exception as exc:
        logger.warning(
            "Clause splitter semantic model disabled; falling back to rule-based splitting only. Error: %s",
            exc,
        )
        _semantic_model_initialized = True
        _semantic_gating_enabled = False
        _model = None
        return False

def _cosine_distance(a: str, b: str) -> float:
    """Cosine distance (1 - cos sim) using SBERT if enabled, else 0.0."""

    if not _ensure_semantic_model():
        return 0.0

    vec = _model.encode([a, b], convert_to_numpy=True, normalize_embeddings=True, batch_size=2, show_progress_bar=False)
    return float(1.0 - np.dot(vec[0], vec[1]))


def _should_split(left: str, right: str, ctype: str) -> bool:
    # Basic guards
    if len(left.strip()) < getattr(config, "SMART_SPLIT_MIN_CHUNK_LEN", 4):
        return False
    if len(right.strip()) < getattr(config, "SMART_SPLIT_MIN_CHUNK_LEN", 4):
        return False

    # Never split on conditional by default
    if ctype == "conditional":
        return False

    toks_l = _tokens(left)
    toks_r = _tokens(right)
    j = _jaccard(toks_l, toks_r)

    # Distance via embeddings (optional)
    d = _cosine_distance(left, right)

    # Thresholds
    jac_add = getattr(config, "SMART_SPLIT_JACCARD_THRESHOLD_ADD", 0.35)
    jac_con = getattr(config, "SMART_SPLIT_JACCARD_THRESHOLD_CONTRAST", 0.40)
    sim_thr = getattr(config, "SMART_SPLIT_SIM_THRESHOLD", 0.22)  # distance >= thr → split

    if ctype == "contrast":
        # Contrast: easier to split; allow if topic changed OR semantic distance is large
        return (j <= jac_con) or (d >= sim_thr * 0.85)
    elif ctype in ("turn", "cause"):
        # Moderate: need some topic change or distance
        return (j <= min(jac_add, jac_con)) or (d >= sim_thr)
    elif ctype == "additive":
        # Additive: split only if topics differ or semantic distance is clear
        return (j <= jac_add) or (d >= sim_thr)
    else:
        # Fallback
        return (j <= jac_add) or (d >= sim_thr)


def _greedy_segment(text: str, max_splits: int = 3) -> List[str]:
    """Greedy left-to-right segmentation controlled by _should_split().
    Also handles the enumeration pattern "…도 …고 …도 …" conservatively.
    """
    # First, sentence-level split by kss (punctuation, etc.)
    sentences = [s.strip() for s in kss.split_sentences(text) if s and s.strip()]
    out: List[str] = []

    for sent in sentences:
        # candidate offsets (connector, type, index)
        cand: List[Tuple[int, str, str]] = []

        # explicit connectors
        for ctype, words in CONN.items():
            for w in words:
                start = 0
                while True:
                    idx = sent.find(w, start)
                    if idx == -1:
                        break
                    # split BEFORE the connector word
                    cand.append((idx, ctype, w))
                    start = idx + len(w)

        # order by position, stable
        cand.sort(key=lambda x: x[0])

        # enumeration guard: …도 …고 …도 …
        if ENUMERATION_GUARD and not cand and ENUM_PAT.search(sent):
            m = ENUM_PAT.search(sent)
            if m:
                idx = m.end()  # after "...도 ...고 "
                cand.append((idx, "additive", "고"))

        if not cand:
            out.append(sent)
            continue

        used = 0
        left = 0
        for idx, ctype, w in cand:
            if used >= max_splits:
                break
            # ensure forward progress
            if idx <= left + 1:
                continue
            L = sent[left:idx].strip()
            R = sent[idx:].strip()

            if _should_split(L, R, ctype):
                out.append(L)
                left = idx
                used += 1
        # tail
        tail = sent[left:].strip()
        if tail:
            out.append(tail)

    return out


# ----------------------------------------------------------------------------
# Public API (unchanged signature)
# ----------------------------------------------------------------------------

def split_clauses(
    df: pd.DataFrame,
    text_col: str = "review",
    connectives: Optional[List[str]] = None,  # kept for back-compat (ignored by smart splitter)
    id_col: str = "review_id",
) -> pd.DataFrame:
    """Return DataFrame(review_id, clause).

    Behavior:
      - If SMART_SPLIT_ENABLED=False → legacy (kss + simple connectives OR commas) can be added.
      - Else → smart greedy splitter described above.
    """
    smart = getattr(config, "SMART_SPLIT_ENABLED", True)
    rows: List[dict] = []

    if not smart:
        # Fallback: simple kss-only splitting per sentence (no clause-level split)
        for rid, text in zip(df[id_col].tolist(), df[text_col].astype(str).tolist()):
            for s in kss.split_sentences(text):
                s = s.strip()
                if s:
                    rows.append({"review_id": rid, "clause": s})
        return pd.DataFrame(rows)

    # Smart path
    _ensure_semantic_model()
    for rid, text in zip(df[id_col].tolist(), df[text_col].astype(str).tolist()):
        text = (text or "").strip()
        if not text:
            continue
        for clause in _greedy_segment(text, max_splits=getattr(config, "SMART_SPLIT_MAX_SPLITS_PER_SENT", 3)):
            if clause:
                rows.append({"review_id": rid, "clause": clause})

    return pd.DataFrame(rows)
