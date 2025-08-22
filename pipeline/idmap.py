# pipeline/idmap.py
from __future__ import annotations
from pathlib import Path
import json, hashlib
from typing import Dict, Tuple
import pandas as pd


STATE_VERSION = 1


DEF_STATE = {
"version": STATE_VERSION,
"counters": {"0": 1000, "1": 1000, "2": 1000}, # start per polarity namespace
"sig2id": {}, # signature -> stable_id
}


def _load_state(path: Path) -> dict:
    if path.exists():
        try:
            with open(path, "r", encoding="utf-8") as f:
                st = json.load(f)
            if st.get("version") == STATE_VERSION:
                return st
        except Exception:
            pass
    return json.loads(json.dumps(DEF_STATE))

def _save_state(path: Path, st: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(st, f, ensure_ascii=False, indent=2)

def _pol_prefix(cid: int) -> int:
    try:
        return max(0, min(2, int(cid) // 1000))
    except Exception:
        return 0

def _signature_for_cluster(cid: int, reps: Dict[int, list]) -> str:
    txt = " ".join(reps.get(int(cid), [])[:3]).strip().lower()
    if not txt:
        txt = f"cluster:{cid}"
    h = hashlib.sha1(txt.encode("utf-8")).hexdigest()
    return h[:16] # 64‑bit hex

def assign_stable_ids(
    clauses_df: pd.DataFrame,
    reps: Dict[int, list],
    *,
    state_path: Path,
    prefer_col: str = "refined_cluster_id",
) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """Attach `stable_cluster_id` based on persisted signature→id mapping.
    - prefer_col: use refined ids if present; else fallback to `cluster_label`.
    - skips "other" bins (x999) by reusing the same id.
    - persists state per input file at `state_path` (e.g., output/<stem>/_stable_ids.json).
    """
    df = clauses_df.copy()
    col = prefer_col if prefer_col in df.columns else "cluster_label"


    st = _load_state(state_path)
    sig2id = st["sig2id"]
    counters = st["counters"]


    stable_map: Dict[int, int] = {}
    for cid in sorted(set(map(int, df[col].unique()))):
        if cid % 1000 == 999:
            stable_map[cid] = cid
            continue
        sig = _signature_for_cluster(cid, reps)
        if sig in sig2id:
            stable_map[cid] = int(sig2id[sig])
        else:
            p = str(_pol_prefix(cid))
            nxt = int(counters.get(p, 1000)) + 1
            counters[p] = nxt
            stable_id = int(p) * 1000 + (nxt % 1000) # stays in 0xxx/1xxx/2xxx space
            # avoid accidental 999
            if stable_id % 1000 == 999:
                stable_id -= 1
            sig2id[sig] = stable_id
            stable_map[cid] = stable_id


    df["stable_cluster_id"] = df[col].map(stable_map)


    st["sig2id"] = sig2id
    st["counters"] = counters
    _save_state(state_path, st)


    return df, {str(k): int(v) for k, v in stable_map.items()}