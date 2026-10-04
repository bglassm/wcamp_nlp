"""Shared identities, polarity and noise semantics for model and offline paths."""
from __future__ import annotations

import hashlib
import json
import math
from typing import Any, Iterable

import pandas as pd


def normalize_polarity(value: Any) -> str:
    """Normalize documented textual labels; never guess a model's numeric class order."""
    if isinstance(value, (list, tuple)):
        if len(value) != 1:
            raise ValueError("Expected exactly one clause polarity")
        value = value[0]
    aliases = {
        "negative": "negative", "neg": "negative", "부정": "negative",
        "neutral": "neutral", "neu": "neutral", "중립": "neutral",
        "positive": "positive", "pos": "positive", "긍정": "positive",
    }
    key = value.strip().lower() if isinstance(value, str) else ""
    if key not in aliases:
        raise ValueError("Unrecognized polarity; numeric labels require an explicit model label map")
    return aliases[key]


def assign_clause_ids(df: pd.DataFrame, product_id: str, id_col: str = "review_id") -> pd.DataFrame:
    """Assign identities once, before polarity filtering; bind identity to exact text."""
    out = df.copy()
    if id_col not in out or "clause" not in out:
        raise ValueError("Clause identity requires a review ID and clause text")
    if out[id_col].isna().any() or out["clause"].isna().any():
        raise ValueError("Clause IDs and texts cannot be missing")
    out[id_col] = out[id_col].astype(str)
    out["clause_idx"] = out.groupby(id_col, sort=False).cumcount()
    out["clause_id"] = [
        hashlib.sha256(json.dumps([str(product_id), rid, int(idx), str(text)], ensure_ascii=False).encode()).hexdigest()
        for rid, idx, text in zip(out[id_col], out["clause_idx"], out["clause"])
    ]
    return out


def is_noise_label(value: Any) -> bool:
    """Recognize raw noise and the legacy three polarity noise buckets."""
    if value is None or str(value).strip().lower() in {"other", "noise", "nan", "none", ""}:
        return True
    try:
        numeric = float(value)
    except (ValueError, TypeError):
        return True
    return not math.isfinite(numeric) or numeric < 0 or numeric in {999, 1999, 2999}


def cluster_counts(labels: Iterable[Any]) -> dict[str, int]:
    labels = list(labels)
    valid = {int(float(v)) for v in labels if not is_noise_label(v)}
    return {"n_clusters": len(valid), "n_noise": sum(is_noise_label(v) for v in labels)}


def offset_cluster_label(label: Any, base: int) -> int:
    # Here labels are raw HDBSCAN IDs, so 999 is a real ID that would collide.
    try:
        numeric = int(label)
    except (TypeError, ValueError):
        if not is_noise_label(label):
            raise
        return base + 999
    if numeric >= 999:
        raise ValueError("Legacy polarity namespaces support at most 999 clusters each")
    return base + 999 if numeric < 0 else numeric + base
