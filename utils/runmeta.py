# utils/runmeta.py
from __future__ import annotations
from pathlib import Path
import json, subprocess, sys

def _git_rev() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return "unknown"

# 🔧 JSON 직렬화 헬퍼: Path, numpy, set, 날짜 등 안전 변환
def _json_default(o):
    try:
        import numpy as _np
        import pathlib as _pl
        import datetime as _dt
        if isinstance(o, _pl.Path):
            return str(o)
        if isinstance(o, (_np.generic,)):
            return o.item()
        if isinstance(o, (set, tuple)):
            return list(o)
        if isinstance(o, (_dt.datetime, _dt.date)):
            return o.isoformat()
    except Exception:
        pass
    # 최후 수단: 문자열화(깨지지 않게)
    return str(o)

def write_run_manifest(path: Path, *, config_obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    # 대문자 속성만 수집 (CONFIG 상수들)
    cfg = {}
    for k in dir(config_obj):
        if not k.isupper():
            continue
        try:
            v = getattr(config_obj, k)
        except Exception:
            continue
        cfg[k] = v

    meta = {
        "git": _git_rev(),
        "python": sys.version.split()[0],
        "platform": sys.platform,
        "config": cfg,
    }

    with open(path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2, default=_json_default)

def write_meta_json(path: Path, *, model_name: str, embed_dim: int, notes: str = "") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    meta = {
        "embedding_model": model_name,
        "embedding_dim": int(embed_dim) if isinstance(embed_dim, bool) or not isinstance(embed_dim, int) else embed_dim,
        "notes": notes,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2, default=_json_default)
