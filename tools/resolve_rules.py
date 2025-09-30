from pathlib import Path
import sys

DATA = Path("data")
RULES = Path("rules")

def pick(stem: str, kind: str) -> Path:
    cand = RULES / f"{kind}_{stem}.yml"
    return cand if cand.exists() else RULES / f"{kind}.yml"

rows = []
for sub in ("review", "community"):
    for p in sorted((DATA/sub).glob("*.xlsx")):
        stem = p.stem
        facets = pick(stem, "facets")
        thr    = pick(stem, "thresholds")
        rows.append((sub, p.name, facets.name if facets.exists() else "MISSING", thr.name if thr.exists() else "MISSING"))

w = max(len(r[1]) for r in rows) if rows else 10
print(f"{'subset':10} {'file':{w}}  facets_file                 thresholds_file")
for sub, fname, fz, tz in rows:
    print(f"{sub:10} {fname:{w}}  {fz:26} {tz}")
