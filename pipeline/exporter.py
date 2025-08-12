from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Union

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


def save_clustered_clauses(
    clause_df: pd.DataFrame,
    raw_df: pd.DataFrame,
    keywords: Dict[int, List[str]],
    output_path: Path,
) -> None:
    _ensure_parent_dir(output_path)
    for col in ("review_id", "clause", "polarity", "cluster_label"):
        if col not in clause_df.columns:
            raise ValueError(f"'{col}' 컬럼이 clause_df에 없습니다.")
    # 절 시트
    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        clause_df.to_excel(writer, sheet_name="clauses", index=False)
        # mapping 시트
        mapping = (
            clause_df
            .groupby("review_id")["cluster_label"]
            .unique()
            .reset_index()
            .rename(columns={"cluster_label": "cluster_labels"})
        )
        mapping["cluster_labels"] = mapping["cluster_labels"].apply(lambda arr: list(arr))
        mapping.to_excel(writer, sheet_name="mapping", index=False)
        # reviews 시트
        review_summary = (
            raw_df
            .merge(mapping, on="review_id", how="left")
            .assign(
                cluster=lambda df: df["cluster_labels"].apply(
                    lambda labs: ",".join(map(str, labs)) if isinstance(labs, list) else ""
                )
            )
            [["platform", "product", "date", "review", "review_id", "cluster"]]
            .rename(columns={"product": "product_name", "review": "comments"})
        )
        review_summary.to_excel(writer, sheet_name="reviews", index=False)
    logger.info(
        "💾 Clustered clauses saved → %s (%d clauses, %d reviews)",
        output_path, len(clause_df), review_summary.shape[0]
    )


def save_clauses_summary_json(
    clause_df: pd.DataFrame,
    reps: Dict[int, List[str]],
    kw: Dict[int, List[str]],
    output_path: Path
) -> None:
    summary = {str(lbl): {"representatives": reps.get(lbl, []), "keywords": kw.get(lbl, [])}
               for lbl in reps}
    _ensure_parent_dir(output_path)
    with open(output_path, "w", encoding="utf-8") as fp:
        json.dump(summary, fp, ensure_ascii=False, indent=2)
    logger.info("💾 Clauses summary JSON saved → %s (%d clusters)",
                output_path, len(summary))
