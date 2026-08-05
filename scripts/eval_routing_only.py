#!/usr/bin/env python3
"""Evaluate facet routing only for clustered clause workbooks."""

from __future__ import annotations

import argparse
import logging
import re
from pathlib import Path
from typing import Iterable, Optional
import glob

import numpy as np
import pandas as pd
import yaml
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
import config
from pipeline.embedder import _get_model
from pipeline.refiner import apply_facet_routing, load_facets_for_category, load_facets_yml


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Re-evaluate facet routing for clustered clause workbooks",
    )
    parser.add_argument("--input_glob", required=True, help="Glob for clustered clause workbooks")
    parser.add_argument("--outdir", required=True, help="Output directory for CSVs")
    parser.add_argument("--sample_per_bucket", type=int, default=50, help="Samples per bucket")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling")
    return parser.parse_args()


def _discover_inputs(glob_pattern: str) -> list[Path]:
    paths = [Path(p) for p in glob.glob(glob_pattern)]
    return [p for p in sorted(paths) if p.is_file() and not p.name.startswith("~$")]


def _extract_run_tag(paths: Iterable[Path]) -> str:
    pattern = re.compile(r"_clauses_clustered_(.+)\.xlsx$", re.IGNORECASE)
    tags = set()
    for path in paths:
        match = pattern.search(path.name)
        if not match:
            raise SystemExit(f"Input file does not match expected pattern: {path}")
        tags.add(match.group(1))
    if len(tags) != 1:
        raise SystemExit(f"Expected a single RUN_TAG, found: {sorted(tags)}")
    return next(iter(tags))


def _pick_column(df: pd.DataFrame, candidates: list[str]) -> Optional[str]:
    for col in candidates:
        if col in df.columns:
            return col
    return None


def _load_clause_sheet(path: Path) -> pd.DataFrame:
    try:
        return pd.read_excel(path, sheet_name="clauses", engine="openpyxl")
    except ValueError:
        book = pd.ExcelFile(path, engine="openpyxl")
        if not book.sheet_names:
            raise SystemExit(f"No sheets found in workbook: {path}")
        return pd.read_excel(path, sheet_name=book.sheet_names[0], engine="openpyxl")


def _load_thresholds(thresholds_path: Path) -> tuple[int, float]:
    with thresholds_path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    refinement_cfg = payload.get("refinement") or {}
    top_k = int(refinement_cfg.get("top_k_facets", 2))
    threshold = float(refinement_cfg.get("facet_threshold", 0.32))
    return top_k, threshold


def _prepare_embedder() -> object:
    semantic_cfg = getattr(config, "semantic", None)
    model_name = getattr(semantic_cfg, "model", None) or getattr(config, "MODEL_NAME", "jhgan/ko-sbert-sts")
    device = getattr(semantic_cfg, "device", None) or getattr(config, "DEVICE", None)
    return _get_model(model_name, device)


def main() -> None:
    args = _parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    logging.getLogger("pipeline.embedder").setLevel(logging.WARNING)

    input_paths = _discover_inputs(args.input_glob)
    if not input_paths:
        raise SystemExit("No input files matched the glob.")

    run_tag = _extract_run_tag(input_paths)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    facets_path = Path(getattr(config, "REFINEMENT_FACETS_PATH", "rules/facets.yml"))
    thresholds_path = Path(getattr(config, "REFINEMENT_THRESHOLDS_PATH", "rules/thresholds.yml"))

    embedder = _prepare_embedder()
    facets_obj = load_facets_yml(facets_path, embedder)
    if not facets_obj:
        raise SystemExit(f"Failed to load facets from {facets_path}")

    top_k, threshold = _load_thresholds(thresholds_path)

    combined_frames: list[pd.DataFrame] = []

    for path in input_paths:
        sku = path.parent.name
        logging.info("Processing %s (sku=%s)", path.name, sku)
        df = _load_clause_sheet(path)

        clause_col = _pick_column(df, ["clause", "clause_text", "text"])
        if clause_col is None:
            raise SystemExit(f"No clause text column found in {path}")
        if clause_col != "clause":
            df["clause"] = df[clause_col]

        polarity_col = _pick_column(df, ["polarity", "sentiment"])
        if polarity_col is None:
            df["polarity"] = "unknown"
        elif polarity_col != "polarity":
            df["polarity"] = df[polarity_col]

        review_id_col = _pick_column(df, ["review_id", "rid"])
        if review_id_col is None:
            df["review_id"] = ""
        elif review_id_col != "review_id":
            df["review_id"] = df[review_id_col]

        category_col = _pick_column(df, ["category"])
        category_val = None
        if category_col is not None:
            non_empty = (
                df[category_col]
                .astype(str)
                .str.strip()
                .replace({"nan": "", "None": ""})
            )
            if non_empty.ne("").any():
                category_val = non_empty[non_empty.ne("")].iloc[0]
        if not category_val:
            category_val = config.infer_category(path, sku)
        df["category"] = df.get("category", category_val)
        df["category"] = df["category"].fillna(category_val)

        facet_config = load_facets_for_category(category_val, sku=sku, fallback=facets_obj)

        routed = apply_facet_routing(
            df,
            facets=facets_obj,
            embedder=embedder,
            text_column="clause",
            top_k=top_k,
            threshold=threshold,
            category=category_val,
            facet_config=facet_config,
            sku=sku,
        )

        routed["sku"] = sku
        combined_frames.append(routed)

    combined = pd.concat(combined_frames, ignore_index=True)

    sample_cols = [
        "category",
        "sku",
        "facet_top1",
        "facet_bucket",
        "polarity",
        "review_id",
        "clause",
    ]
    samples = (
        combined.groupby(["facet_bucket", "polarity"], dropna=False)
        .apply(lambda g: g.sample(n=min(args.sample_per_bucket, len(g)), random_state=args.seed))
        .reset_index(drop=True)
    )
    samples = samples[sample_cols]
    samples_path = outdir / f"bucket_example_samples_{run_tag}.csv"
    samples.to_csv(samples_path, index=False, encoding="utf-8-sig")

    total_by_sku = (
        combined.groupby(["category", "sku"], dropna=False)
        .size()
        .reset_index(name="n_clauses_total")
    )
    stats = (
        combined.groupby(["category", "sku", "facet_bucket"], dropna=False)
        .size()
        .reset_index(name="n_clauses")
    )
    stats = stats.merge(total_by_sku, on=["category", "sku"], how="left")

    review_series = combined["review_id"].astype(str).str.strip()
    has_review_ids = review_series.ne("") & review_series.ne("nan") & review_series.ne("None")
    if has_review_ids.any():
        review_counts = (
            combined.loc[has_review_ids]
            .groupby(["category", "sku", "facet_bucket"], dropna=False)["review_id"]
            .nunique()
            .reset_index(name="n_reviews")
        )
        stats = stats.merge(
            review_counts,
            on=["category", "sku", "facet_bucket"],
            how="left",
        )
    else:
        stats["n_reviews"] = pd.NA

    stats["share_of_sku"] = stats["n_clauses"] / stats["n_clauses_total"]
    stats = stats[
        [
            "category",
            "sku",
            "facet_bucket",
            "n_clauses",
            "n_reviews",
            "n_clauses_total",
            "share_of_sku",
        ]
    ]
    stats_path = outdir / f"facet_bucket_stats_{run_tag}.csv"
    stats.to_csv(stats_path, index=False, encoding="utf-8-sig")

    cross = (
        combined.groupby(["category", "sku", "facet_top1", "facet_bucket"], dropna=False)
        .size()
        .reset_index(name="n_clauses")
    )
    cross_path = outdir / f"facet_vs_bucket_cross_{run_tag}.csv"
    cross.to_csv(cross_path, index=False, encoding="utf-8-sig")

    logging.info("Wrote %s", samples_path)
    logging.info("Wrote %s", stats_path)
    logging.info("Wrote %s", cross_path)


if __name__ == "__main__":
    main()
