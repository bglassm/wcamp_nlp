# pipeline/report.py
from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Optional
import pandas as pd
import numpy as np
import json
import config

# facet 컬럼 후보 (우선순위)
BUCKET_COL_CANDIDATES = ["facet_bucket", "facet_top1", "facet", "bucket", "facet_topk"]

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
    rid = getattr(config, "REVIEW_ID_COL", "review_id")
    left = clause_df.copy()
    right = raw_df.copy()
    if rid in left.columns:
        left[rid] = left[rid].astype(str)
    if rid in right.columns:
        right[rid] = right[rid].astype(str)
    meta_candidates = ["platform", "product", "date", "link", "source_type", "review"]
    keep = [rid] + [c for c in meta_candidates if c in right.columns]
    return left.merge(right[keep], on=rid, how="left")

def _pick_bucket_col(df: pd.DataFrame) -> Optional[str]:
    for c in BUCKET_COL_CANDIDATES:
        if c in df.columns:
            return c
    return None

def _materialize_bucket(df: pd.DataFrame) -> pd.Series:
    """
    facet 컬럼을 유연하게 단일 문자열 컬럼으로 환산:
    - facet_bucket / facet_top1 / facet / bucket: 그대로 문자열화
    - facet_topk: list/JSON-string이면 첫 항목을 사용
    """
    col = _pick_bucket_col(df)
    if col is None:
        return pd.Series([""] * len(df), index=df.index)

    s = df[col]

    # facet_topk가 리스트(또는 리스트 JSON 문자열)인 경우 첫 요소를 택함
    if col == "facet_topk":
        def first_of_topk(v):
            if isinstance(v, list):
                return (v[0] if v else "")
            if isinstance(v, str):
                v2 = v.strip()
                if (v2.startswith("[") and v2.endswith("]")):
                    try:
                        arr = json.loads(v2)
                        return (arr[0] if isinstance(arr, list) and arr else "")
                    except Exception:
                        return ""
            return ""
        return s.apply(first_of_topk).astype(str).fillna("")
    else:
        # 나머지는 문자열화해서 반환
        return s.astype(str).fillna("")

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

    # ---- 파셋(분류) 소스 유연 처리 ----
    # 우선순위: facet_top1 > facet_topk[0] > facet_bucket > (없으면 공백)
    def _safe_first(x):
        # x가 리스트/튜플이면 첫 원소, "['a','b']" 같은 문자열 JSON이면 파싱 시도
        try:
            if isinstance(x, (list, tuple)):
                return x[0] if len(x) else ""
            if isinstance(x, str):
                s = x.strip()
                if s and s[0] in "[{" and s[-1] in "]}":
                    import json as _json
                    parsed = _json.loads(s)
                    if isinstance(parsed, list) and parsed:
                        return str(parsed[0])
                    # dict 등은 키 중 하나를 택하지 않고 공백 처리
                return s  # 일반 문자열이면 그대로
        except Exception:
            pass
        return ""

    facet_src_col = None
    if "facet_top1" in df.columns:
        facet_src_col = "facet_top1"
        df["_facet_top1_eff"] = df["facet_top1"].astype(str)
    elif "facet_topk" in df.columns:
        facet_src_col = "facet_topk"
        df["_facet_top1_eff"] = df["facet_topk"].apply(_safe_first).astype(str)
    elif "facet_bucket" in df.columns:
        facet_src_col = "facet_bucket"
        df["_facet_top1_eff"] = df["facet_bucket"].astype(str)
    else:
        facet_src_col = None
        df["_facet_top1_eff"] = ""

    # 클러스터별 개수/감정/분류(파셋) 취합
    agg_cnt = (
        df.groupby("cluster_label", as_index=False)
          .size()
          .rename(columns={"size": "개수"})
    )
    agg_pol = (
        df.groupby("cluster_label", as_index=False)["polarity"]
          .agg(_most_common)
          .rename(columns={"polarity": "감정"})
    )
    if facet_src_col is not None:
        agg_facet = (
            df.groupby("cluster_label", as_index=False)["_facet_top1_eff"]
              .agg(_most_common)
              .rename(columns={"_facet_top1_eff": "분류"})
        )
    else:
        agg_facet = df.groupby("cluster_label", as_index=False).size()
        agg_facet["분류"] = ""

    base = (
        agg_cnt
        .merge(agg_pol, on="cluster_label", how="left")
        .merge(agg_facet[["cluster_label", "분류"]], on="cluster_label", how="left")
    )

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
    out = (
        out.sort_values(by=["_p", "분류", "개수"], ascending=[True, True, False])
           .drop(columns=["_p"])
           .reset_index(drop=True)
    )
    return out

# ---------------------------------------------------------------------
# B. 플랫폼별 수집/분석 수 & 긍/중/부 비율
# ---------------------------------------------------------------------
def _build_platform_block(clause_df_with_meta: pd.DataFrame, raw_df: pd.DataFrame) -> pd.DataFrame:
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
        ratio_df = (grp.T / grp.sum(axis=1).replace(0, np.nan)).T.fillna(0.0)
        pol_ratio = {plat: {pol: float(ratio_df.loc[plat].get(pol, 0.0)) for pol in ["positive", "neutral", "negative"]}
                     for plat in ratio_df.index}

    platforms = sorted(set(list(collected.keys()) + list(analyzed.keys()) + list(pol_ratio.keys())))
    cols = ["전체"] + platforms

    def _row_from(d: dict) -> List[int]:
        return [sum(d.values())] + [int(d.get(p, 0)) for p in platforms]

    def _ratio_row(which: str) -> List[float]:
        row = [pol_ratio.get(p, {}).get(which, 0.0) for p in platforms]
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

    facet_col = "facet_bucket" if "facet_bucket" in clause_w_meta.columns else ("facet_top1" if "facet_top1" in clause_w_meta.columns else None)
    if facet_col is None:
        raise ValueError("No facet column found for report (need facet_bucket or facet_top1)")
    clause_w_meta["분류"] = clause_w_meta[facet_col]
    clause_w_meta.loc[
        clause_w_meta["분류"].astype(str).str.strip().isin(["", "nan", "None"]),
        "분류",
    ] = None
    if clause_w_meta["분류"].isna().all():
        raise ValueError(f"Facet column '{facet_col}' is entirely NaN — check route_facets() and YAML threshold")

    rep_tbl = _build_rep_summary_table(clause_w_meta, reps)
    plat_tbl = _build_platform_block(clause_w_meta, raw_df)
    year_tbl = _build_year_ratio(raw_df)

    with pd.ExcelWriter(output_path, engine="openpyxl") as w:
        rep_tbl.to_excel(w, sheet_name="대표어요약", index=False)
        plat_tbl.to_excel(w, sheet_name="데이터수량", index=False)
        year_tbl.to_excel(w, sheet_name="연도별비율", index=False)
