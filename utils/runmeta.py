# utils/runmeta.py
from __future__ import annotations
from pathlib import Path
import json, subprocess, sys


def _git_rev() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "unknown"




def write_run_manifest(path: Path, *, config_obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    meta = {
    "git": _git_rev(),
    "python": sys.version.split()[0],
    "config": {k: getattr(config_obj, k) for k in dir(config_obj) if k.isupper()},
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)




def write_meta_json(path: Path, *, model_name: str, embed_dim: int, notes: str = "") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    meta = {"embedding_model": model_name, "embedding_dim": embed_dim, "notes": notes}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)