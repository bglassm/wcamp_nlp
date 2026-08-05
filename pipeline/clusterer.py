import logging
from pathlib import Path
from typing import Tuple, Optional

import numpy as np
import pandas as pd
import config

from sklearn.metrics import silhouette_score

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

try:
    import hdbscan
except ImportError as e:
    raise ImportError(
        "`hdbscan` is not installed. Run: pip install hdbscan"
    ) from e

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def cluster_embeddings(
    embeddings_2d: np.ndarray,
    min_cluster_size: int = config.HDBSCAN_MIN_CLUSTER_SIZE,
    min_samples: Optional[int] = None,
    metric: str = config.HDBSCAN_METRIC,
    cluster_selection_epsilon: float = config.HDBSCAN_SELECTION_EPS,
) -> Tuple[np.ndarray, hdbscan.HDBSCAN]:
    """Run HDBSCAN on reduced embeddings. Noise labels (-1) are optionally
    replaced with config.OUTLIER_LABEL when config.HANDLE_OUTLIERS is True."""
    if embeddings_2d.ndim != 2:
        raise ValueError("`embeddings_2d` must be a 2-D array (n_samples x n_dim).")

    if min_samples is None:
        min_samples = (
            config.HDBSCAN_MIN_SAMPLES
            if hasattr(config, "HDBSCAN_MIN_SAMPLES")
            else max(2, min_cluster_size // 2)
        )

    logger.info(
        "[HDBSCAN] shape=%s min_cluster_size=%d min_samples=%d metric=%s epsilon=%.3f",
        embeddings_2d.shape,
        min_cluster_size,
        min_samples,
        metric,
        cluster_selection_epsilon,
    )

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric=metric,
        cluster_selection_epsilon=cluster_selection_epsilon,
        prediction_data=False,
    ).fit(embeddings_2d)

    labels = clusterer.labels_

    # replace noise label if configured
    if getattr(config, "HANDLE_OUTLIERS", False):
        labels = np.array(
            [t if t != -1 else config.OUTLIER_LABEL for t in labels],
            dtype=object,
        )

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = (
        int((labels == config.OUTLIER_LABEL).sum())
        if config.HANDLE_OUTLIERS
        else int((clusterer.labels_ == -1).sum())
    )

    logger.info(
        "[HDBSCAN] done: %d clusters, %d noise points (label=%s)",
        n_clusters,
        n_noise,
        config.OUTLIER_LABEL if config.HANDLE_OUTLIERS else "-1",
    )
    return labels, clusterer


def evaluate_clusters(
    labels: np.ndarray,
    embeddings_2d: np.ndarray,
    raw_embeddings: np.ndarray | None = None,
    logger: logging.Logger | None = None,
    output_dir: Path | None = None,
    timestamp: str | None = None,
    tag: str | None = None,
) -> None:
    """Log cluster distribution and silhouette scores, and optionally save
    distribution/stats CSVs and a per-polarity scatter plot PNG."""
    lg = logger or logging.getLogger(__name__)

    # distribution: safe for mixed int/str labels
    labels_str = np.array([str(x) for x in np.asarray(labels, dtype=object)])
    uniq, counts = np.unique(labels_str, return_counts=True)
    dist_info = ", ".join(f"{c}@{u}" for u, c in zip(uniq, counts))
    lg.info("[CLUSTER] distribution: %s", dist_info)

    if output_dir and timestamp:
        try:
            df_dist = pd.DataFrame({"cluster_label": uniq, "count": counts})
            dist_path = Path(output_dir) / f"cluster_distribution_{timestamp}.csv"
            df_dist.to_csv(dist_path, index=False, encoding="utf-8-sig")
        except Exception:
            lg.exception("[CLUSTER] failed to save cluster_distribution CSV")

    # silhouette: exclude noise/other labels
    exclude = {"-1", "other"}
    mask = np.array([s.lower() not in exclude for s in labels_str], dtype=bool)

    valid_n = int(mask.sum())
    valid_labels = np.unique(labels_str[mask]) if valid_n > 0 else np.array([])
    n_valid_labels = len(valid_labels)

    if valid_n >= 2 and n_valid_labels >= 2:
        try:
            sil_umap = float(silhouette_score(embeddings_2d[mask], labels_str[mask], metric="euclidean"))
        except Exception:
            lg.exception("[CLUSTER] silhouette (UMAP) failed")
            sil_umap = None

        sil_raw = None
        if raw_embeddings is not None:
            try:
                sil_raw = float(silhouette_score(raw_embeddings[mask], labels_str[mask], metric="cosine"))
            except Exception:
                lg.exception("[CLUSTER] silhouette (raw) failed")

        parts = ["[CLUSTER] silhouette"]
        if sil_umap is not None:
            parts.append(f"umap={sil_umap:.3f}")
        if sil_raw is not None:
            parts.append(f"raw={sil_raw:.3f}")
        lg.info(" ".join(parts))

        if output_dir and timestamp:
            try:
                df_stats = pd.DataFrame([{
                    "silhouette_umap": sil_umap,
                    "silhouette_raw": sil_raw,
                    "n_clusters": int(n_valid_labels),
                    "n_noise_or_other": int((~mask).sum()),
                    "n_samples_used": valid_n,
                }])
                stats_path = Path(output_dir) / f"cluster_stats_{timestamp}.csv"
                df_stats.to_csv(stats_path, index=False, encoding="utf-8-sig")
            except Exception:
                lg.exception("[CLUSTER] failed to save cluster_stats CSV")
    else:
        if valid_n < 2:
            lg.info("[CLUSTER] silhouette skipped: insufficient non-noise points")
        else:
            lg.info("[CLUSTER] silhouette skipped: only one valid label")

    # scatter plot (per-polarity, saved alongside other outputs)
    if (
        output_dir is not None
        and embeddings_2d is not None
        and getattr(embeddings_2d, "ndim", 0) == 2
        and embeddings_2d.shape[1] >= 2
    ):
        if plt is None:
            lg.info("[CLUSTER] matplotlib not available, skip scatter plot")
        else:
            try:
                x = embeddings_2d[:, 0]
                y = embeddings_2d[:, 1]
                labels_str_full = np.array([str(v) for v in np.asarray(labels, dtype=object)])

                noise_set = {"-1"}
                if hasattr(config, "OUTLIER_LABEL"):
                    noise_set.add(str(config.OUTLIER_LABEL))

                unique_labels = sorted(set(labels_str_full))
                fig, ax = plt.subplots(figsize=(7, 4))

                for lab in unique_labels:
                    if str(lab) in noise_set:
                        continue
                    m = labels_str_full == lab
                    if not m.any():
                        continue
                    ax.scatter(x[m], y[m], s=8, alpha=0.7, label=f"cluster {lab}")

                ax.set_xlabel("UMAP-1")
                ax.set_ylabel("UMAP-2")
                title_tag = tag if tag is not None else "all"
                title = f"Clusters ({title_tag}, {timestamp})" if timestamp else f"Clusters ({title_tag})"
                ax.set_title(title)

                handles, labels_legend = ax.get_legend_handles_labels()
                if handles:
                    ax.legend(
                        handles,
                        labels_legend,
                        loc="center left",
                        bbox_to_anchor=(1.02, 0.5),
                        fontsize=6,
                        markerscale=0.7,
                        borderaxespad=0.0,
                    )

                fname = (
                    f"cluster_plot_{title_tag}_{timestamp}.png"
                    if timestamp
                    else f"cluster_plot_{title_tag}.png"
                )
                plot_path = Path(output_dir) / fname
                fig.tight_layout(rect=[0, 0, 0.8, 1])
                fig.savefig(plot_path, dpi=150)
                plt.close(fig)
                lg.info("[CLUSTER] scatter plot saved: %s", plot_path.name)
            except Exception:
                lg.exception("[CLUSTER] failed to save scatter plot")
    else:
        lg.info("[CLUSTER] scatter plot skipped (no output_dir or insufficient coords)")
