import re, hashlib
import pandas as pd
from pathlib import Path
from typing import List, Dict, Optional

CANON_COLS = {
    "platform": ["플랫폼", "platform"],
    "link": ["링크", "link", "url"],
    "date": ["작성일", "date"],
    "title": ["제목", "title"],
    "body": ["본문", "본문 내용", "body", "content"],
    "summary": ["본문 요약 및 주요 내용", "요약", "summary"],
}

def _pick(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns: return c
    return None

def _hash_id(s: str) -> str:
    return hashlib.md5(s.encode("utf-8", "ignore")).hexdigest()[:10]

def load_posts(path: Path, product: str) -> pd.DataFrame:
    df = pd.read_excel(path)
    colmap = {k: _pick(df, v) for k,v in CANON_COLS.items()}
    missing = [k for k,v in colmap.items() if k in ["title","body","platform","link"] and v is None]
    if missing:
        raise ValueError(f"Missing required columns: {missing} in {path}")

    out = pd.DataFrame()
    out["platform"] = df[colmap["platform"]]
    out["link"]     = df[colmap["link"]]
    out["date"]     = df[colmap["date"]] if colmap["date"] else None
    out["title"]    = df[colmap["title"]].astype(str).fillna("")
    out["body"]     = df[colmap["body"]].astype(str).fillna("")
    out["summary"]  = df[colmap["summary"]].astype(str) if colmap["summary"] else ""
    out["product"]  = product
    # 안정 post_id
    out["post_id"]  = out.apply(lambda r: f"C-{product}-{_hash_id((r['link'] or '') + '|' + r['title'])}", axis=1)
    return out
