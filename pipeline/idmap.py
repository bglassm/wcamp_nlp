"""Persistent cluster IDs within the legacy three polarity namespaces."""
from __future__ import annotations
from pathlib import Path
import hashlib
import json
import logging
import tempfile
from typing import Dict, Tuple
import pandas as pd

STATE_VERSION = 2
DEF_STATE = {"version": STATE_VERSION, "counters": {"0": -1, "1": -1, "2": -1}, "sig2id": {}}
logger = logging.getLogger(__name__)


def _load_state(path: Path) -> dict:
    if path.exists():
        state = json.loads(path.read_text(encoding="utf-8"))
        if state.get("version") == STATE_VERSION:
            values = list(state["sig2id"].values())
            if len(values) != len(set(values)):
                raise ValueError("Stable-ID state contains duplicate IDs")
            for signature, value in state["sig2id"].items():
                if not isinstance(value, int) or value < 0 or value >= 3000 or value % 1000 == 999:
                    raise ValueError("Stable-ID state contains invalid IDs")
                if not signature.startswith(f"{value // 1000}:"):
                    raise ValueError("Stable-ID state has a polarity mismatch")
            return state
        logger.warning("Invalidating legacy stable-ID state: v1 signatures/allocations may collide")
    return json.loads(json.dumps(DEF_STATE))


def _save_state(path: Path, state: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
        json.dump(state, handle, ensure_ascii=False, indent=2)
    try:
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _pol_prefix(cid: int) -> int:
    if not 0 <= cid < 3000:
        raise ValueError("Cluster ID is outside the legacy polarity namespaces")
    return cid // 1000


def _signature_for_cluster(cid: int, reps: Dict[int, list]) -> str:
    text = " ".join(reps.get(cid, [])[:3]).strip().lower() or f"cluster:{cid}"
    return f"{_pol_prefix(cid)}:" + hashlib.sha256(text.encode("utf-8")).hexdigest()


def assign_stable_ids(
    clauses_df: pd.DataFrame,
    reps: Dict[int, list],
    *,
    state_path: Path,
    prefer_col: str = "refined_cluster_id",
) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """Persist polarity-scoped IDs; never wrap into an existing ID or noise bucket.

    Each polarity has 999 IDs (0..998 plus its offset). Exhaustion raises before
    writing any state. A legacy v1 state is explicitly invalidated with a warning.
    Representative signatures are a heuristic; changed wording can create a new ID.
    """
    frame = clauses_df.copy()
    column = prefer_col if prefer_col in frame.columns else "cluster_label"
    raw_ids = pd.to_numeric(frame[column], errors="raise")
    if raw_ids.isna().any() or ((raw_ids % 1) != 0).any():
        raise ValueError("Cluster IDs must be finite integers")
    ids = raw_ids.astype(int)
    state = _load_state(Path(state_path))
    signatures = state["sig2id"]
    used = set(signatures.values())
    stable_map = {}
    signature_owners = {}
    for cid in sorted(ids.unique()):
        cid = int(cid)
        if cid == -1:
            stable_map[cid] = cid
            continue
        polarity = _pol_prefix(cid)
        if cid % 1000 == 999:
            stable_map[cid] = cid
            continue
        signature = _signature_for_cluster(cid, reps)
        if signature in signature_owners:
            raise ValueError(
                "Ambiguous stable-ID signature: distinct clusters in one polarity "
                "share the same representatives; state was not changed"
            )
        signature_owners[signature] = cid
        if signature not in signatures:
            base = polarity * 1000
            available = next((base + offset for offset in range(999) if base + offset not in used), None)
            if available is None:
                raise ValueError(f"Stable-ID capacity exhausted for polarity {polarity}; namespace redesign required")
            signatures[signature] = available
            used.add(available)
            state["counters"][str(polarity)] = available - base
        stable_map[cid] = signatures[signature]
    frame["stable_cluster_id"] = ids.map(stable_map)
    _save_state(Path(state_path), state)
    return frame, {str(key): int(value) for key, value in stable_map.items()}
