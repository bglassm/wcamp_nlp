# pipeline/report.py
from __future__ import annotations
from pathlib import Path
from typing import Dict, List
import pandas as pd
import numpy as np
import config

# 내부 유틸
def _rid_col() -> str:
    return getattr(config, "REVIEW_ID_COL", "review_id")

def _most_common(series: pd.Series, *, skip_blank: bool = True):
    s = series.dropna()
    if skip_blank:
        s = s[s.astype(str).str.strip() != ""]
    if s.empty:
        return ""
    return s.value_counts().idxmax()

def _ensure_meta_join(clause_df: pd.DataFrame, raw_df: pd.DataFrame) -> pd.DataFrame:
    """리뷰 메타(platform, date 등) 붙여서 반환."""
    rid = _rid_col()
    keep = [c for c in [rid, "platform", "date", "product"] if c in raw_df.columns]
    if rid not in clause_df.columns or rid not in raw_df.columns:
        return clause_df.copy()
    return clause_df.merge(raw_df[keep], on=rid, how="left")

# ---------------------------------------------------------------------
# A. 대표어 요약 테이블: [감정, 분류, 대표어, 개수]
# ---------------------------------------------------------------------
def _build_rep_summary_table(
    clause_df_with_meta: pd.DataFrame,
    reps: Dict[int, List[str]],
) -> pd.DataFrame:
    df = clause_df_with_meta.copy()
    # 노이즈(-1 또는 999류) 제외
    if "cluster_label" in df.columns:
        df = df[pd.to_numeric(df["cluster_label"], errors="coerce").fillna(-1).astype(int) >= 0]
    if df.empty:
        return pd.DataFrame(columns=["감정", "분류", "대표어", "개수"])

    # 클러스터별 개수/감정/분류(파셋) 취합
    agg_cnt = df.groupby("cluster_label", as_index=False).size().rename(columns={"size": "개수"})
    agg_pol = df.groupby("cluster_label", as_index=False)["polarity"].agg(_most_common).rename(columns={"polarity": "감정"})
    if "facet_top1" in df.columns:
        agg_facet = df.groupby("cluster_label", as_index=False)["facet_top1"].agg(_most_common).rename(columns={"facet_top1": "분류"})
    else:
        # 파셋 없는 경우 빈 값
        agg_facet = df.groupby("cluster_label", as_index=False).size()
        agg_facet["분류"] = ""

    base = agg_cnt.merge(agg_pol, on="cluster_label", how="left").merge(agg_facet[["cluster_label", "분류"]], on="cluster_label", how="left")

    # 대표어(top-1) 붙이기
    rep_list = []
    for cid in base["cluster_label"].tolist():
        entry = reps.get(int(cid), [])
        rep_text = entry[0] if isinstance(entry, list) and len(entry) > 0 else ""
        rep_list.append(rep_text)
    base["대표어"] = rep_list

    out = base[["감정", "분류", "대표어", "개수"]].copy()
    # 정렬: 감정(neg/neu/pos) → 분류 → 개수 desc
    order_pol = {"negative": 0, "neutral": 1, "positive": 2}
    out["_p"] = out["감정"].map(order_pol).fillna(99)
    out = out.sort_values(by=["_p", "분류", "개수"], ascending=[True, True, False]).drop(columns=["_p"]).reset_index(drop=True)
    return out

# ---------------------------------------------------------------------
# B. 플랫폼별 수집/분석 수 & 긍/중/부 비율
# ---------------------------------------------------------------------
def _build_platform_block(clause_df_with_meta: pd.DataFrame, raw_df: pd.DataFrame) -> pd.DataFrame:
    rid = _rid_col()

    # (1) 수집된 데이터 수(리뷰 원본)
    if "platform" in raw_df.columns:
        col_counts = raw_df["platform"].fillna("unknown").astype(str).value_counts()
    else:
        col_counts = pd.Series(dtype=int)
    collected = col_counts.to_dict()

    # (2) 분석된 데이터 수(절 단위, ABSA 통과 후)
    if "platform" in clause_df_with_meta.columns:
        ana_counts = clause_df_with_meta["platform"].fillna("unknown").astype(str).value_counts()
    else:
        ana_counts = pd.Series(dtype=int)
    analyzed = ana_counts.to_dict()

    # (3) 플랫폼별 감정 비율 (절 기준)
    pol_ratio = {}
    if {"platform", "polarity"} <= set(clause_df_with_meta.columns):
        grp = clause_df_with_meta.groupby(["platform", "polarity"]).size().unstack(fill_value=0)
        # ratio by row (platform)
        ratio_df = (grp.T / grp.sum(axis=1).replace(0, np.nan)).T.fillna(0.0)
        pol_ratio = {plat: {pol: float(ratio_df.loc[plat].get(pol, 0.0)) for pol in ["positive", "neutral", "negative"]} for plat in ratio_df.index}

    # 열 순서(전체 + 사전순)
    platforms = sorted(set(list(collected.keys()) + list(analyzed.keys()) + list(pol_ratio.keys())))
    cols = ["전체"] + platforms

    # 전체 합계
    total_collected = int(raw_df.shape[0])
    total_analyzed = int(clause_df_with_meta.shape[0])

    # 표 구성
    def _row_from(d: dict) -> List[int]:
        return [sum(d.values())] + [int(d.get(p, 0)) for p in platforms]

    def _ratio_row(which: str) -> List[float]:
        # which in {"positive","neutral","negative"}
        total = 0.0
        row = []
        for p in platforms:
            val = pol_ratio.get(p, {}).get(which, 0.0)
            row.append(val)
            total += val
        # "전체" 칸은 단순 평균(가중X) 대신 절 기준 총합에서 재계산하도록 구성
        if {"platform", "polarity"} <= set(clause_df_with_meta.columns):
            overall = float((clause_df_with_meta["polarity"] == which).mean())
        else:
            overall = 0.0
        return [overall] + row

    data = {
        "수집된 데이터 수": _row_from(collected),
        "분석된 데이터 수": _row_from(analyzed),
        "긍정 비율 (%)":     [v * 100 for v in _ratio_row("positive")],
        "중립 비율 (%)":     [v * 100 for v in _ratio_row("neutral")],
        "부정 비율 (%)":     [v * 100 for v in _ratio_row("negative")],
    }
    block = pd.DataFrame(data, index=cols).T.reset_index().rename(columns={"index": "메트릭"})
    # 소수점 한 자리로 보기 좋게
    for r in ["긍정 비율 (%)", "중립 비율 (%)", "부정 비율 (%)"]:
        block.loc[block["메트릭"] == r, cols] = block.loc[block["메트릭"] == r, cols].astype(float).round(1)
    return block

# ---------------------------------------------------------------------
# C. 연도별 리뷰 비율(원본 리뷰 기준)
# ---------------------------------------------------------------------
def _build_year_ratio(raw_df: pd.DataFrame) -> pd.DataFrame:
    if "date" not in raw_df.columns:
        return pd.DataFrame(columns=["연도", "리뷰 수", "비율(%)"])
    dt = pd.to_datetime(raw_df["date"], errors="coerce")
    y = dt.dt.year.dropna()
    if y.empty:
        return pd.DataFrame(columns=["연도", "리뷰 수", "비율(%)"])
    year_cnt = y.value_counts().sort_index()
    total = int(year_cnt.sum())
    out = pd.DataFrame({"연도": year_cnt.index.astype(int), "리뷰 수": year_cnt.values})
    out["비율(%)"] = (out["리뷰 수"] / total * 100).round(1)
    return out

# ---------------------------------------------------------------------
# PUBLIC: 저장 함수
# ---------------------------------------------------------------------
def save_client_report(
    clause_df: pd.DataFrame,
    raw_df: pd.DataFrame,
    reps: Dict[int, List[str]],
    output_path: Path,
) -> None:
    """
    생성물:
    - Sheet1: 대표어 요약 (감정/분류/대표어/개수)
    - Sheet2: 데이터 수량 (플랫폼별 수집·분석 수 + 긍/중/부 비율)
    - Sheet3: 연도별 리뷰 비율 (원본 리뷰 기준)
    """
    clause_w_meta = _ensure_meta_join(clause_df, raw_df)

    rep_tbl = _build_rep_summary_table(clause_w_meta, reps)
    plat_tbl = _build_platform_block(clause_w_meta, raw_df)
    year_tbl = _build_year_ratio(raw_df)

    with pd.ExcelWriter(output_path, engine="openpyxl") as w:
        rep_tbl.to_excel(w, sheet_name="대표어요약", index=False)
        plat_tbl.to_excel(w, sheet_name="데이터수량", index=False)
        year_tbl.to_excel(w, sheet_name="연도별비율", index=False)
