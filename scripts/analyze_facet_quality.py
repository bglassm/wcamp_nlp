import argparse
import logging
import time
from datetime import datetime
from pathlib import Path
import pandas as pd

RUN_DATE = datetime.now().strftime("%Y%m%d")

OUTPUT_ROOT = Path("output") / RUN_DATE
ANALYSIS_DIR = OUTPUT_ROOT / "analysis"
ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

logger = logging.getLogger(__name__)


def _cluster_timestamp_key(p: Path) -> str:
    """Extract timestamp suffix from a clustered output filename."""
    parts = p.stem.split("_")
    if len(parts) >= 2:
        return "_".join(parts[-2:])
    return ""


def _select_latest_file(files):
    """Return the file with the most recent timestamp in its name, falling back to mtime."""
    keyed = {p: _cluster_timestamp_key(p) for p in files}
    timestamped = {p: ts for p, ts in keyed.items() if ts}
    if timestamped:
        return max(timestamped.items(), key=lambda item: item[1])[0]
    return max(files, key=lambda p: p.stat().st_mtime)


def load_all_clauses(strategy="latest"):
    """
    Collect clause workbooks from output/<YYYYMMDD>/<sku>/ and return a combined DataFrame.

    strategy="latest" : use only the most recent file per SKU (default)
    strategy="all"    : use all clustered files per SKU
    """
    base = OUTPUT_ROOT
    clause_files = []

    for sku_dir in base.iterdir():
        if not sku_dir.is_dir():
            continue
        sku = sku_dir.name
        files = list(sku_dir.glob(f"{sku}_clauses_clustered_*.xlsx"))
        if not files:
            continue
        if strategy == "all":
            clause_files.extend(files)
        elif strategy == "latest":
            clause_files.append(_select_latest_file(files))
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

    if not clause_files:
        logger.warning("No clause files found.")
        return None

    skus = {p.parent.name for p in clause_files}
    logger.info(
        "Found %d clustered clause workbooks across %d SKUs (strategy=%s); sample=%s",
        len(clause_files), len(skus), strategy,
        [p.name for p in clause_files[:5]],
    )

    frames = []
    for idx, path in enumerate(clause_files, 1):
        logger.info("Processing file %d/%d: %s", idx, len(clause_files), path)
        sku = path.parent.name
        try:
            df = pd.read_excel(path)
        except Exception:
            logger.exception("Failed to read %s", path)
            continue
        if "sku" not in df.columns:
            df["sku"] = sku
        if "polarity" in df.columns:
            df["polarity"] = df["polarity"].astype(str).str.lower()
        frames.append(df)
        logger.info("Loaded %d clauses from %s", len(df), path.name)

    if not frames:
        return None

    logger.info("Loaded %d clause files", len(frames))
    return pd.concat(frames, ignore_index=True)


def compute_facet_bucket_stats(df):
    """
    Compute per-category/sku/facet_bucket clause and review counts.
    Outputs: output/<YYYYMMDD>/analysis/facet_bucket_stats.csv
    """
    if "polarity" in df.columns:
        neg = df[df["polarity"].isin(["negative", "neg"])].copy()
    else:
        neg = df.copy()

    required = ["facet_bucket", "clause", "review_id", "sku"]
    for col in required:
        if col not in neg.columns:
            logger.warning("required column '%s' missing; stats may be incomplete.", col)

    facet_bucket_col = "facet_bucket" if "facet_bucket" in neg.columns else None
    category_col = "category" if "category" in neg.columns else None
    if category_col is None:
        neg["category"] = "unknown"
        category_col = "category"

    group_keys = [category_col, "sku"]
    if facet_bucket_col is not None:
        group_keys.append(facet_bucket_col)

    total = (
        neg.groupby([category_col, "sku"])["clause"]
        .size()
        .reset_index(name="n_clauses_total")
    )
    stats = (
        neg.groupby(group_keys)
        .agg(n_clauses=("clause", "size"), n_reviews=("review_id", "nunique"))
        .reset_index()
    )
    stats = stats.merge(total, on=[category_col, "sku"], how="left")
    stats["share_of_sku"] = stats["n_clauses"] / stats["n_clauses_total"]

    out_path = ANALYSIS_DIR / "facet_bucket_stats.csv"
    stats.to_csv(out_path, index=False)
    logger.info("Saved facet_bucket_stats: %s (shape=%s)", out_path, stats.shape)


def compute_facet_vs_bucket_cross(df):
    """
    Count clauses per (category, sku, facet_top1, facet_bucket) combination.
    Outputs: output/<YYYYMMDD>/analysis/facet_vs_bucket_cross.csv
    """
    if "polarity" in df.columns:
        neg = df[df["polarity"].isin(["negative", "neg"])].copy()
    else:
        neg = df.copy()

    category_col = "category" if "category" in neg.columns else None
    if category_col is None:
        neg["category"] = "unknown"
        category_col = "category"

    for col in ["facet_top1", "facet_bucket"]:
        if col not in neg.columns:
            logger.warning("'%s' missing; cross-tab may be incomplete.", col)

    req = [category_col, "sku", "facet_top1", "facet_bucket"]
    for col in req:
        if col not in neg.columns:
            neg[col] = None

    cross = (
        neg.groupby(req)["clause"]
        .size()
        .reset_index(name="n_clauses")
    )

    out_path = ANALYSIS_DIR / "facet_vs_bucket_cross.csv"
    cross.to_csv(out_path, index=False)
    logger.info("Saved facet_vs_bucket_cross: %s (shape=%s)", out_path, cross.shape)


def compute_bucket_samples(df, samples_per_combo=50, random_state=42):
    """
    Sample up to samples_per_combo clauses per (category, sku, facet_top1, facet_bucket).
    Outputs: output/<YYYYMMDD>/analysis/bucket_example_samples.csv
    """
    if "polarity" in df.columns:
        neg = df[df["polarity"].isin(["negative", "neg"])].copy()
    else:
        neg = df.copy()

    category_col = "category" if "category" in neg.columns else None
    if category_col is None:
        neg["category"] = "unknown"
        category_col = "category"

    for col in ["facet_top1", "facet_bucket"]:
        if col not in neg.columns:
            neg[col] = None

    groups = neg.groupby(
        [category_col, "sku", "facet_top1", "facet_bucket"], dropna=False
    )

    sampled_frames = []
    for keys, g in groups:
        if len(g) == 0:
            continue
        n = min(samples_per_combo, len(g))
        sample = g.sample(n=n, random_state=random_state)
        sampled_frames.append(
            sample[[category_col, "sku", "facet_top1", "facet_bucket",
                    "polarity", "review_id", "clause"]]
        )

    if not sampled_frames:
        logger.warning("No samples to save")
        return

    samples_df = pd.concat(sampled_frames, ignore_index=True)
    out_path = ANALYSIS_DIR / "bucket_example_samples.csv"
    samples_df.to_csv(out_path, index=False)
    logger.info("Saved bucket_example_samples: %s (shape=%s)", out_path, samples_df.shape)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--strategy",
        choices=["latest", "all"],
        default="latest",
        help="file loading strategy per SKU (default: latest)",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    t0 = time.perf_counter()
    logger.info("Started facet quality analysis with strategy=%s", args.strategy)

    df = load_all_clauses(strategy=args.strategy)
    if df is None:
        logger.warning("No data loaded.")
        return

    compute_facet_bucket_stats(df)
    compute_facet_vs_bucket_cross(df)
    compute_bucket_samples(df, samples_per_combo=50)

    elapsed = time.perf_counter() - t0
    logger.info("Finished facet quality analysis in %.1f seconds", elapsed)


if __name__ == "__main__":
    main()
