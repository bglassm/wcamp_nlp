from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Union, Optional

import numpy as np
import pandas as pd
import config

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

_RID = getattr(config, "REVIEW_ID_COL", "review_id")


def _ensure_parent_dir(path: Union[str, Path]) -> None:
    Path(path).expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)


def _as_str_id(df: pd.DataFrame) -> pd.DataFrame:
    """Cast the review ID column to str to prevent dtype mismatches on merge."""
    if _RID in df.columns:
        df[_RID] = df[_RID].astype(str)
    return df


def _ordered_clause_columns(df: pd.DataFrame) -> List[str]:
    """Return column list with cluster/refine columns moved to the front."""
    preferred = [
        "polarity", "facet_bucket", "cluster_label", "refined_label",
        "refined_cluster_id", "stable_cluster_id", "facet_top1",
        "facet_top1_score", "facet_topk", "facet_rule_hits",
    ]
    cols = list(df.columns)
    return [c for c in preferred if c in cols] + [c for c in cols if c not in preferred]


def _labels_to_csv(labs) -> str:
    try:
        if isinstance(labs, (list, tuple, np.ndarray, pd.Series)):
            return ",".join(map(str, labs))
        return ""
    except Exception:
        return ""


# --- back-compat helpers ---

def save_clustered_reviews(
    df: pd.DataFrame,
    labels: np.ndarray,
    out_path: Union[str, Path],
) -> None:
    """Write per-review cluster labels to CSV or XLSX (legacy helper)."""
    if len(df) != len(labels):
        raise ValueError("df and labels must have the same length")
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
        raise ValueError("Unsupported extension — use .xlsx or .csv")
    logger.info("[EXPORT] clustered reviews saved: %s (%d rows)", out_path, len(df_out))


def save_representatives_json(
    reps: Dict[int, List[str]],
    out_path: Union[str, Path],
    ensure_ascii: bool = False,
    indent: int = 2,
) -> None:
    _ensure_parent_dir(out_path)
    norm = {str(int(k)): v for k, v in ((int(k), v) for k, v in reps.items())}
    with open(out_path, "w", encoding="utf-8") as fp:
        json.dump(norm, fp, ensure_ascii=ensure_ascii, indent=indent)
    logger.info("[EXPORT] representatives JSON saved: %s (%d clusters)", out_path, len(norm))


def save_cluster_summary_json(
    representatives: Dict[int, List[str]],
    keywords: Dict[int, List[str]],
    save_path: Union[str, Path],
) -> None:
    """Merge representatives and keywords into a single summary JSON."""
    def _to_int_keys(d: Dict) -> Dict[int, List[str]]:
        out = {}
        for k, v in d.items():
            try:
                out[int(k)] = v
            except Exception:
                continue
        return out

    reps_i = _to_int_keys(representatives)
    kw_i = _to_int_keys(keywords)
    all_cids = set(reps_i) | set(kw_i)
    summary: dict[str, dict[str, List[str]]] = {
        str(cid): {"representatives": reps_i.get(cid, []), "keywords": kw_i.get(cid, [])}
        for cid in sorted(all_cids)
    }
    _ensure_parent_dir(save_path)
    with open(save_path, "w", encoding="utf-8") as fp:
        json.dump(summary, fp, ensure_ascii=False, indent=2)
    logger.info("[EXPORT] cluster summary saved: %s", save_path)


def build_cluster_name_map(
    keywords: Dict[int, List[str]],
    top_k: int = 2,
) -> Dict[str, str]:
    """Build a {cluster_id: "kw1·kw2"} display name map from keyword lists."""
    used: set[str] = set()
    name_map: dict[str, str] = {}
    for cid_raw, kw_list in keywords.items():
        try:
            cid = str(int(cid_raw))
        except Exception:
            cid = str(cid_raw)
        picked: list[str] = []
        for kw in (kw_list or []):
            if kw not in used:
                picked.append(kw)
                used.add(kw)
            if len(picked) == top_k:
                break
        if len(picked) < top_k:
            picked.extend((kw_list or [])[: top_k - len(picked)])
        name_map[cid] = "·".join(picked)
    logger.info("[EXPORT] cluster_name_map: %d entries", len(name_map))
    return name_map


# --- main exporter ---

def save_clustered_clauses(
    clause_df: pd.DataFrame,
    raw_df: pd.DataFrame,
    keywords: Dict[int, List[str]],
    output_path: Union[str, Path],
    meta: Optional[dict] = None,
) -> None:
    """Write workbook with sheets: clauses, mapping, reviews, (meta).
    List/dict columns are serialized to JSON strings for readability."""
    clause_df = clause_df.copy()
    raw_df = raw_df.copy()

    required = [_RID, "clause", "polarity", "cluster_label"]
    missing = [c for c in required if c not in clause_df.columns]
    if missing:
        raise ValueError(f"clause_df missing required columns: {missing}")

    clause_df["cluster_label"] = (
        pd.to_numeric(clause_df["cluster_label"], errors="coerce").fillna(-1).astype(int)
    )
    clause_df = _as_str_id(clause_df)
    raw_df = _as_str_id(raw_df)

    has_refined = "refined_cluster_id" in clause_df.columns
    has_stable = "stable_cluster_id" in clause_df.columns

    _ensure_parent_dir(output_path)
    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        # clauses sheet
        clause_out = clause_df.copy()
        for col in ("facet_topk", "facet_rule_hits", "facet_scores", "representatives"):
            if col in clause_out.columns:
                clause_out[col] = clause_out[col].apply(
                    lambda v: json.dumps(v, ensure_ascii=False)
                    if not isinstance(v, (str, type(None)))
                    else ("" if v is None else v)
                )
        clause_out = clause_out[_ordered_clause_columns(clause_out)]
        clause_out.to_excel(writer, sheet_name="clauses", index=False)

        # mapping sheet: review -> cluster label set
        mapping = (
            clause_df.groupby(_RID)["cluster_label"]
            .unique()
            .reset_index()
            .rename(columns={"cluster_label": "cluster_labels"})
        )
        mapping["cluster_labels"] = mapping["cluster_labels"].apply(list)

        if has_refined:
            mapping = mapping.merge(
                clause_df.groupby(_RID)["refined_cluster_id"]
                .unique()
                .reset_index()
                .rename(columns={"refined_cluster_id": "refined_cluster_ids"}),
                on=_RID, how="left",
            )
        if has_stable:
            mapping = mapping.merge(
                clause_df.groupby(_RID)["stable_cluster_id"]
                .unique()
                .reset_index()
                .rename(columns={"stable_cluster_id": "stable_cluster_ids"}),
                on=_RID, how="left",
            )

        mapping = _as_str_id(mapping)
        mapping.to_excel(writer, sheet_name="mapping", index=False)

        # reviews sheet: raw reviews joined with mapping
        review_summary = raw_df.merge(mapping, on=_RID, how="left").copy()
        review_summary["cluster"] = review_summary.get("cluster_labels", []).apply(_labels_to_csv)
        if "refined_cluster_ids" in review_summary.columns:
            review_summary["refined_cluster"] = review_summary["refined_cluster_ids"].apply(_labels_to_csv)
        if "stable_cluster_ids" in review_summary.columns:
            review_summary["stable_cluster"] = review_summary["stable_cluster_ids"].apply(_labels_to_csv)

        base_cols = [_RID, "platform", "product", "date", "review", "cluster"]
        opt_cols = ["dup_count", "n_reviews_total", "refined_cluster", "stable_cluster"]
        cols = [c for c in base_cols if c in review_summary.columns] + \
               [c for c in opt_cols if c in review_summary.columns]
        review_summary = review_summary[cols].rename(
            columns={"product": "product_name", "review": "comments"}
        )
        review_summary.to_excel(writer, sheet_name="reviews", index=False)

        if meta is not None:
            pd.DataFrame([{**meta}]).to_excel(writer, sheet_name="meta", index=False)

    logger.info(
        "[EXPORT] saved: %s (%d clauses, %d reviews)%s%s",
        output_path, len(clause_df), review_summary.shape[0],
        " +refined" if has_refined else "",
        " +stable" if has_stable else "",
    )


def save_clauses_summary_json(
    clause_df: pd.DataFrame,
    reps: Dict[int, List[str]],
    kw: Dict[int, List[str]],
    output_path: Union[str, Path],
) -> None:
    """Write per-cluster representatives and keywords to a summary JSON."""
    def _to_int_keys(d: Dict) -> Dict[int, List[str]]:
        out = {}
        for k, v in d.items():
            try:
                out[int(k)] = v
            except Exception:
                continue
        return out

    reps_i = _to_int_keys(reps)
    kw_i = _to_int_keys(kw)
    summary = {
        str(k): {"representatives": reps_i.get(k, []), "keywords": kw_i.get(k, [])}
        for k in sorted(set(reps_i) | set(kw_i))
    }
    _ensure_parent_dir(output_path)
    with open(output_path, "w", encoding="utf-8") as fp:
        json.dump(summary, fp, ensure_ascii=False, indent=2)
    logger.info("[EXPORT] clauses summary JSON saved: %s (%d clusters)", output_path, len(summary))
