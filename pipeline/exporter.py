from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Union, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def _ensure_parent_dir(path: Union[str, Path]) -> None:
    Path(path).expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)


def save_clustered_reviews(
    df: pd.DataFrame,
    labels: np.ndarray,
    out_path: Union[str, Path],
) -> None:
    """Back-compat helper: save per-review labels to CSV/XLSX.
    Adds a `cluster_label` (or `cluster_label_2` if already present).
    """
    if len(df) != len(labels):
        raise ValueError("`df`와 `labels` 길이가 다릅니다.")
    df_out = df.copy()
    label_col = "cluster_label" if "cluster_label" not in df_out.columns else "cluster_label_2"
    df_out[label_col] = labels
    _ensure_parent_dir(out_path)
    out_path = str(out_path)
    if out_path.lower().endswith(".xlsx"):
        df_out.to_excel(out_path, index=False, engine="openpyxl")
    elif out_path.lower().endswith(".csv"):
        df_out.to_csv(out_path, index=False, encoding="utf-8-sig")
    else:
        raise ValueError("지원하지 않는 확장자입니다 — .xlsx 또는 .csv 사용")
    logger.info("💾 Clustered reviews saved → %s (%d rows)", out_path, len(df_out))


def save_representatives_json(
    reps: Dict[int, List[str]],
    out_path: Union[str, Path],
    ensure_ascii: bool = False,
    indent: int = 2,
) -> None:
    _ensure_parent_dir(out_path)
    with open(out_path, "w", encoding="utf-8") as fp:
        json.dump(reps, fp, ensure_ascii=ensure_ascii, indent=indent)
    logger.info("💾 Representatives JSON saved → %s (%d clusters)", out_path, len(reps))


def save_cluster_summary_json(
    representatives: Dict[int, List[str]],
    keywords: Dict[int, List[str]],
    save_path: Union[str, Path],
) -> None:
    """Union of cluster keys from reps and keywords into a single JSON."""
    all_cids = {int(k) for k in representatives} | {int(k) for k in keywords}
    summary: dict[str, dict[str, List[str]]] = {}
    for cid in sorted(all_cids):
        summary[str(cid)] = {
            "representatives": representatives.get(cid, []),
            "keywords": keywords.get(cid, []),
        }
    _ensure_parent_dir(save_path)
    with open(save_path, "w", encoding="utf-8") as fp:
        json.dump(summary, fp, ensure_ascii=False, indent=2)
    logger.info("💾 Cluster summary saved → %s", save_path)


def build_cluster_name_map(
    keywords: Dict[int, List[str]],
    top_k: int = 2,
) -> Dict[str, str]:
    used: set[str] = set()
    name_map: dict[str, str] = {}
    for cid, kw_list in keywords.items():
        picked: list[str] = []
        for kw in kw_list:
            if kw not in used:
                picked.append(kw)
                used.add(kw)
            if len(picked) == top_k:
                break
        if len(picked) < top_k:
            picked.extend(kw_list[: top_k - len(picked)])
        name_map[str(cid)] = "·".join(picked)
    logger.info("cluster_name_map created (%d entries)", len(name_map))
    return name_map


def _ordered_clause_columns(df: pd.DataFrame) -> List[str]:
    """Prefer refined/facet columns near the cluster fields if present."""
    preferred = [
        "polarity",
        "cluster_label",
        "refined_label",
        "refined_cluster_id",
        "stable_cluster_id",
        "facet_top1",
        "facet_topk",
    ]
    cols = list(df.columns)
    ordered = [c for c in preferred if c in cols] + [c for c in cols if c not in preferred]
    return ordered


def _labels_to_csv(labs) -> str:
    try:
        if isinstance(labs, (list, tuple, np.ndarray, pd.Series)):
            return ",".join(map(str, labs))
        return ""
    except Exception:
        return ""


def save_clustered_clauses(
    clause_df: pd.DataFrame,
    raw_df: pd.DataFrame,
    keywords: Dict[int, List[str]],
    output_path: Path,
    meta: Optional[dict] = None,
) -> None:
    """Save unified workbook with `clauses`, `mapping`, `reviews` (+ optional `meta`).

    - Keeps back-compat sheet names.
    - If `refined_cluster_id` exists in `clause_df`, it is carried through to mapping/reviews.
    - If `stable_cluster_id` exists, it is also carried through.
    """
    _ensure_parent_dir(output_path)
    for col in ("review_id", "clause", "polarity", "cluster_label"):
        if col not in clause_df.columns:
            raise ValueError(f"'{col}' 컬럼이 clause_df에 없습니다.")

    has_refined = "refined_cluster_id" in clause_df.columns
    has_stable = "stable_cluster_id" in clause_df.columns

    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        # clauses sheet (ordered) — facet_topk를 JSON 문자열로 보정
        clause_out = clause_df.copy()
        if "facet_topk" in clause_out.columns:
            clause_out["facet_topk"] = clause_out["facet_topk"].apply(
                lambda v: json.dumps(v, ensure_ascii=False) if not isinstance(v, str) else v
            )
        clause_out = clause_out[_ordered_clause_columns(clause_out)]
        clause_out.to_excel(writer, sheet_name="clauses", index=False)

        # mapping sheet (original + refined/stable ids if present)
        mapping = (
            clause_df
            .groupby("review_id")["cluster_label"]
            .unique()
            .reset_index()
            .rename(columns={"cluster_label": "cluster_labels"})
        )
        mapping["cluster_labels"] = mapping["cluster_labels"].apply(lambda arr: list(arr))

        if has_refined:
            mapping_ref = (
                clause_df
                .groupby("review_id")["refined_cluster_id"]
                .unique()
                .reset_index()
                .rename(columns={"refined_cluster_id": "refined_cluster_ids"})
            )
            mapping = mapping.merge(mapping_ref, on="review_id", how="left")

        if has_stable:
            mapping_st = (
                clause_df
                .groupby("review_id")["stable_cluster_id"]
                .unique()
                .reset_index()
                .rename(columns={"stable_cluster_id": "stable_cluster_ids"})
            )
            mapping = mapping.merge(mapping_st, on="review_id", how="left")

        mapping.to_excel(writer, sheet_name="mapping", index=False)

        # reviews sheet — refined/stable이 있으면 함께 문자열 컬럼 생성
        review_summary = raw_df.merge(mapping, on="review_id", how="left").copy()
        review_summary["cluster"] = review_summary["cluster_labels"].apply(_labels_to_csv)

        if "refined_cluster_ids" in review_summary.columns:
            review_summary["refined_cluster"] = review_summary["refined_cluster_ids"].apply(_labels_to_csv)
        if "stable_cluster_ids" in review_summary.columns:
            review_summary["stable_cluster"] = review_summary["stable_cluster_ids"].apply(_labels_to_csv)

        # select columns (존재할 때만 포함)
        base_cols = ["platform", "product", "date", "review", "review_id", "cluster"]
        if "refined_cluster" in review_summary.columns:
            base_cols.append("refined_cluster")
        if "stable_cluster" in review_summary.columns:
            base_cols.append("stable_cluster")

        review_summary = (
            review_summary[base_cols]
            .rename(columns={"product": "product_name", "review": "comments"})
        )

        review_summary.to_excel(writer, sheet_name="reviews", index=False)

        # optional meta sheet
        if meta is not None:
            pd.DataFrame([{**meta}]).to_excel(writer, sheet_name="meta", index=False)

    logger.info(
        "💾 Clustered clauses saved → %s (%d clauses, %d reviews)%s%s",
        output_path,
        len(clause_df),
        review_summary.shape[0],
        " + refined" if has_refined else "",
        " + stable" if has_stable else "",
    )


def save_clauses_summary_json(
    clause_df: pd.DataFrame,
    reps: Dict[int, List[str]],
    kw: Dict[int, List[str]],
    output_path: Path
) -> None:
    """Summary JSON (representatives + keywords). Uses union of keys for safety."""
    all_keys = set(map(int, reps.keys())) | set(map(int, kw.keys()))
    summary = {
        str(k): {
            "representatives": reps.get(k, []),
            "keywords": kw.get(k, []),
        } for k in sorted(all_keys)
    }
    _ensure_parent_dir(output_path)
    with open(output_path, "w", encoding="utf-8") as fp:
        json.dump(summary, fp, ensure_ascii=False, indent=2)
    logger.info("💾 Clauses summary JSON saved → %s (%d clusters)", output_path, len(summary))
