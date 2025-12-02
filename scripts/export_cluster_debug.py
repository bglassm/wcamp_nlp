from __future__ import annotations

import argparse
import glob
import json
from collections import Counter
from pathlib import Path
from typing import Iterable, List

import pandas as pd


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export cluster-level debug CSV from clustered clause workbooks.")
    parser.add_argument(
        "--input_glob",
        required=True,
        help="Glob pattern for *_clauses_clustered_*.xlsx files (e.g., output/<SKU>/*_clauses_clustered_*.xlsx)",
    )
    parser.add_argument("--output", required=True, help="Output CSV path for the aggregated debug view.")
    return parser.parse_args()


def _load_clauses(path: Path) -> pd.DataFrame:
    try:
        return pd.read_excel(path, sheet_name="clauses")
    except ValueError:
        # Fallback: load first sheet if named sheet is missing
        return pd.read_excel(path)


def _parse_json_list(val) -> List[str]:
    if val is None:
        return []
    if isinstance(val, list):
        return [str(v) for v in val]
    if isinstance(val, str):
        val = val.strip()
        if not val:
            return []
        try:
            parsed = json.loads(val)
            if isinstance(parsed, list):
                return [str(v) for v in parsed]
        except Exception:
            return [val]
    return []


def _value_counts_json(series: Iterable) -> str:
    counts = Counter(x for x in series if pd.notna(x))
    ordered = dict(counts.most_common())
    return json.dumps(ordered, ensure_ascii=False)


def _flatten_rule_hits(series: pd.Series) -> Counter:
    counter: Counter = Counter()
    for val in series:
        hits = _parse_json_list(val)
        counter.update(hits)
    return counter


def _pick_keywords(group: pd.DataFrame) -> str:
    if "keywords" in group.columns:
        for val in group["keywords"]:
            tokens = _parse_json_list(val)
            if tokens:
                return ", ".join(tokens)
            if isinstance(val, str) and val.strip():
                return val
    return ""


def _pick_representatives(group: pd.DataFrame, clause_col: str = "clause", top_n: int = 3) -> str:
    if "representatives" in group.columns:
        for val in group["representatives"]:
            reps = _parse_json_list(val)
            if reps:
                return " || ".join(reps[:top_n])
    clauses = group.get(clause_col, [])
    return " || ".join([str(c) for c in clauses.head(top_n)]) if hasattr(clauses, "head") else ""


def _facet_top1_mode(series: pd.Series):
    non_null = series.dropna()
    if non_null.empty:
        return None
    modes = non_null.mode()
    return modes.iloc[0] if not modes.empty else None


def main() -> None:
    args = _parse_args()
    files = sorted(glob.glob(args.input_glob))
    if not files:
        raise SystemExit(f"No files matched pattern: {args.input_glob}")

    records = []
    for file_path in files:
        path = Path(file_path)
        df = _load_clauses(path)
        sku = path.stem.split("_")[0] if path.stem else path.parent.name

        for (polarity, cluster_label), group in df.groupby(["polarity", "cluster_label"]):
            facet_distribution = _value_counts_json(group.get("facet_top1", []))
            rule_hits_counter = _flatten_rule_hits(group.get("facet_rule_hits", pd.Series(dtype=object)))
            facet_mode = _facet_top1_mode(group.get("facet_top1", pd.Series(dtype=object)))
            record = {
                "file": str(path),
                "sku": sku,
                "polarity": polarity,
                "cluster_label": cluster_label,
                "n_clauses": len(group),
                "facet_top1_mode": facet_mode,
                "facet_top1_distribution": facet_distribution,
                "rule_hits_top": json.dumps(dict(rule_hits_counter.most_common(10)), ensure_ascii=False),
                "keywords": _pick_keywords(group),
                "rep_sentences": _pick_representatives(group),
            }
            records.append(record)

    out_df = pd.DataFrame(records)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False)
    print(f"Saved cluster_debug CSV → {out_path} ({len(out_df)} rows)")


if __name__ == "__main__":
    main()
