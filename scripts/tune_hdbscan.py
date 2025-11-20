"""
Exploratory HDBSCAN parameter sweep for SKU-level clause embeddings.

Usage examples:
    python -m scripts.tune_hdbscan --sku flatfishsashimi --n-samples 2000
    python -m scripts.tune_hdbscan --all-skus

The script reuses precomputed OpenAI embeddings (1536D) and applies the same
UMAP configuration as the production pipeline before running HDBSCAN over a
small grid of min_cluster_size/min_samples combinations.
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_score

import config
from pipeline.clusterer import cluster_embeddings
from pipeline.reducer import reduce_embeddings
from pipeline.tuner import _pick_pct, get_cluster_params
from pipeline.embedder import get_embedder

logger = logging.getLogger(__name__)


PCT_SCALES: Tuple[float, ...] = (0.5, 0.75, 1.0, 1.25)
MIN_SAMPLES_CANDIDATES: Tuple[int, ...] = (1, 5, 10)
OUTPUT_DIR = Path("outputs/hdbscan_tuning")


def discover_skus() -> List[str]:
    """Return a sorted SKU list derived from facet rule files (facets_*.yml)."""
    rules_dir = Path("rules")
    skus = []
    for path in rules_dir.glob("facets_*.yml"):
        stem = path.stem.replace("facets_", "").strip()
        if stem:
            skus.append(stem)
    return sorted(set(skus))


def _load_cache_embeddings(sku: str) -> Optional[np.ndarray]:
    """Load embeddings from the CachingEmbedder parquet cache if available."""
    try:
        embedder = get_embedder(config)
        cache_path = embedder._get_cache_path(sku)  # type: ignore[attr-defined]
    except Exception:
        return None

    if cache_path.exists():
        try:
            df = pd.read_parquet(cache_path)
            if "embedding_vector" not in df.columns:
                logger.warning("Cache file exists but lacks 'embedding_vector': %s", cache_path)
                return None
            arr = np.array(df["embedding_vector"].tolist(), dtype=np.float32)
            logger.info("Loaded %d embeddings from cache %s", len(arr), cache_path)
            return arr
        except Exception:
            logger.exception("Failed to read cache parquet at %s", cache_path)
            return None
    return None


def _load_npy_embeddings(sku: str) -> Optional[np.ndarray]:
    """Try multiple .npy locations for precomputed embeddings."""
    candidates = [
        Path(config.OUTPUT_DIR) / sku / "embeddings.npy",
        Path(config.OUTPUT_DIR) / sku / "clause_embeddings.npy",
        Path(config.OUTPUT_DIR) / f"{sku}_embeddings.npy",
    ]
    for path in candidates:
        if path.exists():
            try:
                arr = np.load(path)
                logger.info("Loaded embeddings from %s (shape=%s)", path, getattr(arr, "shape", None))
                return np.asarray(arr, dtype=np.float32)
            except Exception:
                logger.exception("Failed to load npy embeddings at %s", path)
                return None
    return None


def _load_jsonl_embeddings(sku: str) -> Optional[np.ndarray]:
    """Load embeddings from a JSONL file with an 'embedding_vector' field."""
    candidates = [
        Path(config.OUTPUT_DIR) / sku / "embeddings.jsonl",
        Path(config.OUTPUT_DIR) / f"{sku}_embeddings.jsonl",
    ]
    for path in candidates:
        if not path.exists():
            continue
        vectors: List[List[float]] = []
        try:
            with path.open("r", encoding="utf-8") as fp:
                for line in fp:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        record = json.loads(line)
                        vec = record.get("embedding_vector") or record.get("embedding")
                        if vec is not None:
                            vectors.append(vec)
                    except json.JSONDecodeError:
                        continue
            if vectors:
                arr = np.asarray(vectors, dtype=np.float32)
                logger.info("Loaded %d embeddings from %s", len(arr), path)
                return arr
        except Exception:
            logger.exception("Failed to read jsonl embeddings at %s", path)
            return None
    return None


def load_embeddings(sku: str) -> np.ndarray:
    """Load precomputed embeddings without invoking external APIs."""
    loaders = (_load_cache_embeddings, _load_npy_embeddings, _load_jsonl_embeddings)
    for loader in loaders:
        arr = loader(sku)
        if arr is not None and arr.size > 0:
            return arr
    raise FileNotFoundError(f"No precomputed embeddings found for {sku}")


def sample_embeddings(
    embeddings: np.ndarray,
    n_samples: Optional[int] = None,
    random_sample: bool = False,
    seed: int = 42,
) -> np.ndarray:
    """Subsample embeddings deterministically (random or head)."""
    n_total = embeddings.shape[0]
    if not n_samples or n_samples >= n_total:
        return embeddings
    n_samples = max(1, int(n_samples))
    rng = np.random.default_rng(seed)
    idx = rng.choice(n_total, size=n_samples, replace=False) if random_sample else np.arange(n_samples)
    return embeddings[idx]


def _silhouette_safe(X: np.ndarray, labels: np.ndarray) -> float:
    """Compute silhouette score ignoring noise; return NaN on invalid cases."""
    try:
        labels = np.asarray(labels)
        mask = labels != -1
        valid_labels = np.unique(labels[mask]) if mask.any() else []
        if len(valid_labels) < 2 or X.shape[0] <= 2:
            return float("nan")
        return float(silhouette_score(X[mask], labels[mask]))
    except Exception:
        logger.exception("Silhouette computation failed")
        return float("nan")


def _cluster_size_stats(labels: np.ndarray) -> Tuple[float, float, int]:
    """Return (median, p90, very_small_count<5) for non-noise clusters."""
    labels = np.asarray(labels)
    mask = labels != -1
    if not mask.any():
        return float("nan"), float("nan"), 0
    _, counts = np.unique(labels[mask], return_counts=True)
    if counts.size == 0:
        return float("nan"), float("nan"), 0
    median = float(np.median(counts))
    p90 = float(np.percentile(counts, 90))
    very_small = int((counts < 5).sum())
    return median, p90, very_small


def run_sweep_for_sku(
    sku: str,
    embeddings: np.ndarray,
    pct_scales: Iterable[float] = PCT_SCALES,
    min_samples_grid: Iterable[int] = MIN_SAMPLES_CANDIDATES,
    cluster_eps: Optional[float] = None,
) -> pd.DataFrame:
    """Run HDBSCAN parameter sweep for one SKU and return a results DataFrame."""
    n_docs = embeddings.shape[0]
    params = get_cluster_params(n_docs, dataset=sku)
    umap_cfg = params.get("umap", {})
    base_pct = _pick_pct(n_docs)
    cluster_eps = float(
        cluster_eps
        if cluster_eps is not None
        else params.get("hdbscan", {}).get("cluster_selection_epsilon", config.HDBSCAN_SELECTION_EPS)
    )

    reduced = reduce_embeddings(
        embeddings,
        n_components=int(umap_cfg.get("n_components", config.UMAP_DIMS_CLUSTER)),
        n_neighbors=int(umap_cfg.get("n_neighbors", config.UMAP_N_NEIGHBORS)),
        min_dist=float(umap_cfg.get("min_dist", config.UMAP_MIN_DIST)),
        metric=umap_cfg.get("metric", config.UMAP_METRIC),
        random_state=int(umap_cfg.get("random_state", config.UMAP_RANDOM_STATE)),
    )

    rows: List[Dict] = []
    for pct_scale in pct_scales:
        min_cluster_size = max(2, int(round(n_docs * base_pct * pct_scale)))
        for min_samples in min_samples_grid:
            logger.info(
                "[SWEEP] %s | pct_scale=%.2f min_cluster_size=%d min_samples=%d",
                sku,
                pct_scale,
                min_cluster_size,
                min_samples,
            )
            try:
                labels, _ = cluster_embeddings(
                    reduced,
                    min_cluster_size=min_cluster_size,
                    min_samples=min_samples,
                    cluster_selection_epsilon=cluster_eps,
                    metric=config.HDBSCAN_METRIC,
                )
                n_clusters = int(len(set(labels)) - (1 if -1 in labels else 0))
                n_noise = int((labels == -1).sum())
                noise_ratio = float(n_noise / n_docs) if n_docs else float("nan")
                sil_umap = _silhouette_safe(reduced, labels)
                sil_orig = _silhouette_safe(embeddings, labels)
                med, p90, very_small = _cluster_size_stats(labels)
            except Exception:
                logger.exception(
                    "[SWEEP] Failed combo for %s (pct_scale=%.2f, min_samples=%d)",
                    sku,
                    pct_scale,
                    min_samples,
                )
                n_clusters = n_noise = very_small = 0
                noise_ratio = float("nan")
                sil_umap = sil_orig = med = p90 = float("nan")

            row = {
                "sku": sku,
                "n_docs": n_docs,
                "pct_scale": pct_scale,
                "min_samples": min_samples,
                "n_clusters": n_clusters,
                "noise_ratio": noise_ratio,
                "silhouette_umap": sil_umap,
                "silhouette_original": sil_orig,
                "cluster_size_median": med,
                "cluster_size_p90": p90,
                "very_small_clusters": very_small,
            }
            rows.append(row)
    return pd.DataFrame(rows)


def save_results(df: pd.DataFrame, sku: str) -> Path:
    """Save sweep DataFrame to CSV under outputs/hdbscan_tuning."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / f"{sku}_tuning_results.csv"
    df.to_csv(out_path, index=False, encoding="utf-8-sig")
    logger.info("💾 Saved tuning results → %s", out_path)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="HDBSCAN tuning sweep")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--sku", help="Single SKU id to tune")
    group.add_argument("--all-skus", action="store_true", help="Run sweep for all discovered SKUs")
    parser.add_argument("--n-samples", type=int, default=None, help="Subsample size for speed")
    parser.add_argument("--random-sample", action="store_true", help="Randomly sample clauses instead of head")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s %(message)s")

    skus = discover_skus() if args.all_skus else [args.sku]
    if not skus:
        raise SystemExit("No SKUs found to tune. Ensure facet rule files are present.")

    for sku in skus:
        try:
            emb = load_embeddings(sku)
        except FileNotFoundError:
            logger.warning("❌ Missing precomputed embeddings for %s. Skipping.", sku)
            continue
        emb = sample_embeddings(emb, args.n_samples, random_sample=args.random_sample)
        if emb.size == 0:
            logger.warning("No embeddings available after sampling for %s. Skipping.", sku)
            continue
        logger.info("🔢 Running sweep for %s with %d samples", sku, emb.shape[0])
        df = run_sweep_for_sku(sku, emb)
        save_results(df, sku)


if __name__ == "__main__":
    main()
