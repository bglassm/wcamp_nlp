# tools/merge_all.py
from __future__ import annotations
from pathlib import Path
import pandas as pd
import sys


# Usage: python tools/merge_all.py output merged_all.xlsx


if __name__ == "__main__":
    out_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("output")
    target = Path(sys.argv[2]) if len(sys.argv) > 2 else out_dir/"_merged_all.xlsx"


    rows = []
    for p in out_dir.rglob("*_clauses_clustered_*.xlsx"):
        try:
            df = pd.read_excel(p, sheet_name="clauses")
            df.insert(0, "source_file", p.name)
            rows.append(df)
        except Exception:
            pass
    if not rows:
        print("no inputs found")
        sys.exit(1)
    big = pd.concat(rows, ignore_index=True)


    with pd.ExcelWriter(target, engine="openpyxl") as w:
        big.to_excel(w, sheet_name="clauses_all", index=False)
    print("merged →", target)