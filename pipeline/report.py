from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Optional
import pandas as pd
import numpy as np
import json
import config

# Korean stopwords used when building short cluster labels
DEFAULT_STOPWORDS_KO = [
    "맛있다", "맛없다", "괜찮다", "좋다", "나쁘다",
    "많다", "적다", "없다", "있다", "똑같다",
    "같다", "심하다", "괜히", "그냥",
]

# priority order for picking the facet column
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
    """Left-join clause_df with review-level metadata from raw_df."""
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
    """Return the first available facet column from BUCKET_COL_CANDIDATES."""
    for c in BUCKET_COL_CANDIDATES:
        if c in df.columns:
            return c
    return None


def _materialize_bucket(df: pd.DataFrame) -> pd.Series:
    """Normalize the best available facet column to a plain string Series.
    For facet_topk (list or JSON-string), the first element is used."""
    col = _pick_bucket_col(df)
    if col is None:
        return pd.Series([""] * len(df), index=df.index)

    s = df[col]

    if col == "facet_topk":
        def first_of_topk(v):
            if isinstance(v, list):
                return v[0] if v else ""
            if isinstance(v, str):
                v2 = v.strip()
                if v2.startswith("[") and v2.endswith("]"):
                    try:
                        arr = json.loads(v2)
                        return arr[0] if isinstance(arr, list) and arr else ""
                    except Exception:
                        return ""
            return ""
        return s.apply(first_of_topk).astype(str).fillna("")
    else:
        return s.astype(str).fillna("")


# --- Section A: cluster representative summary table ---

def _build_keyword_label(kw: Optional[Dict[int, List[str]]]) -> Dict[int, str]:
    """Build a short display label per cluster from its keyword list,
    filtering stopwords and capping at config.CLUSTER_NAME_TOPK tokens."""
    if not kw:
        return {}

    topk = getattr(config, "CLUSTER_NAME_TOPK", 2)
    stopwords = getattr(config, "CLUSTER_NAME_STOPWORDS_KO", DEFAULT_STOPWORDS_KO)
    labels: Dict[int, str] = {}

    for k, v in kw.items():
        try:
            cid = int(k)
        except Exception:
            continue

        if isinstance(v, (list, tuple)):
            cleaned = [str(x).strip() for x in v if str(x).strip()]
            filtered = [token for token in cleaned if token not in stopwords]
            chosen = (filtered if filtered else cleaned)[:topk]
            label = "·".join(chosen)
        else:
            label = str(v).strip()

        if label:
            labels[cid] = label

    return labels


def _format_facet_label(raw: object) -> str:
    """Strip polarity suffixes (_negative/_neutral/_positive) from a facet string."""
    if raw is None:
        return ""
    s = str(raw).strip()
    if s == "" or s.lower() in {"nan", "none"}:
        return ""
    for suf in ("_negative", "_neutral", "_positive"):
        if s.lower().endswith(suf):
            return s[: -len(suf)]
    return s


def _combine_labels(facet_label: str, kw_label: str, fallback: str) -> str:
    """Combine facet and keyword labels; fall back to the representative sentence."""
    facet_label = facet_label.strip()
    kw_label = kw_label.strip()
    if facet_label and kw_label:
        return f"{facet_label} – {kw_label}"
    if facet_label:
        return facet_label
    if kw_label:
        return kw_label
    return fallback


def _build_rep_summary_table(
    clause_df_with_meta: pd.DataFrame,
    reps: Dict[int, List[str]],
    kw: Optional[Dict[int, List[str]]] = None,
) -> pd.DataFrame:
    """Build the cluster summary table: [감정, 분류, 대표어, 개수, 대표 문장]."""
    df = clause_df_with_meta.copy()

    # exclude noise clusters (-1 and 999-class offsets)
    if "cluster_label" in df.columns:
        df = df[pd.to_numeric(df["cluster_label"], errors="coerce").fillna(-1).astype(int) >= 0]
    if df.empty:
        return pd.DataFrame(columns=["감정", "분류", "대표어", "개수", "대표 문장"])

    # resolve facet source column; priority: facet_top1 > facet_topk > facet_bucket
    def _safe_first(x):
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
                return s
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

    kw_labels = _build_keyword_label(kw)
    rep_labels = []
    rep_full_texts = []
    for cid, facet_raw in base[["cluster_label", "분류"]].itertuples(index=False):
        entry = reps.get(int(cid), [])
        rep_text = entry[0] if isinstance(entry, list) and len(entry) > 0 else ""
        rep_full_texts.append(rep_text)

        rep_text_short = rep_text[:40].rstrip() + "…" if len(rep_text) > 40 else rep_text
        facet_label = _format_facet_label(facet_raw)
        kw_label = kw_labels.get(int(cid), "")
        label = _combine_labels(facet_label, kw_label, fallback=rep_text_short)
        rep_labels.append(label)

    base["대표어"] = rep_labels
    base["대표 문장"] = rep_full_texts

    out = base[["감정", "분류", "대표어", "개수", "대표 문장"]].copy()

    def _clean_facet(val: object) -> str:
        s = "" if val is None else str(val).strip()
        return "" if s.lower() in {"none", "nan"} else s

    out["분류"] = out["분류"].apply(_clean_facet)

    # sort: polarity (neg/neu/pos) -> facet (unclassified last) -> count desc
    order_pol = {"negative": 0, "neutral": 1, "positive": 2}
    out["_p"] = out["감정"].map(order_pol).fillna(99)
    non_blank_facets = sorted([f for f in out["분류"].unique() if f])
    facet_categories = non_blank_facets + [""]
    out["_facet_cat"] = pd.Categorical(out["분류"], categories=facet_categories, ordered=True)
    out = (
        out.sort_values(by=["_p", "_facet_cat", "개수"], ascending=[True, True, False])
           .drop(columns=["_p", "_facet_cat"])
           .reset_index(drop=True)
    )
    return out


# --- Section B: per-platform collection/analysis counts and polarity ratios ---

def _build_platform_block(clause_df_with_meta: pd.DataFrame, raw_df: pd.DataFrame) -> pd.DataFrame:
    if "platform" in raw_df.columns:
        col_counts = raw_df["platform"].fillna("unknown").astype(str).value_counts()
    else:
        col_counts = pd.Series(dtype=int)
    collected = col_counts.to_dict()

    if "platform" in clause_df_with_meta.columns:
        ana_counts = clause_df_with_meta["platform"].fillna("unknown").astype(str).value_counts()
    else:
        ana_counts = pd.Series(dtype=int)
    analyzed = ana_counts.to_dict()

    pol_ratio: Dict[str, Dict[str, float]] = {}
    pol_counts: Dict[str, Dict[str, int]] = {}
    grp = pd.DataFrame()
    if {"platform", "polarity"} <= set(clause_df_with_meta.columns):
        grp = clause_df_with_meta.groupby(["platform", "polarity"]).size().unstack(fill_value=0)
        ratio_df = (grp.T / grp.sum(axis=1).replace(0, np.nan)).T.fillna(0.0)
        pol_ratio = {
            plat: {pol: float(ratio_df.loc[plat].get(pol, 0.0)) for pol in ["positive", "neutral", "negative"]}
            for plat in ratio_df.index
        }
        pol_counts = {
            plat: {pol: int(grp.loc[plat].get(pol, 0)) for pol in ["positive", "neutral", "negative"]}
            for plat in grp.index
        }

    raw_platforms = set(
        list(collected.keys()) + list(analyzed.keys()) +
        list(pol_ratio.keys()) + list(pol_counts.keys())
    )
    preferred = getattr(config, "PLATFORM_ORDER", [])
    ordered = [p for p in preferred if p in raw_platforms]
    rest = sorted(p for p in raw_platforms if p not in preferred)
    platforms = ordered + rest
    cols = ["전체"] + platforms

    def _row_from(d: dict) -> List[int]:
        return [sum(d.values())] + [int(d.get(p, 0)) for p in platforms]

    def _count_row(which: str) -> List[int]:
        if {"platform", "polarity"} <= set(clause_df_with_meta.columns):
            overall = int((clause_df_with_meta["polarity"] == which).sum())
        else:
            overall = 0
        return [overall] + [int(pol_counts.get(p, {}).get(which, 0)) for p in platforms]

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
        "긍정 개수":        _count_row("positive"),
        "중립 개수":        _count_row("neutral"),
        "부정 개수":        _count_row("negative"),
        "긍정 비율 (%)":    [v * 100 for v in _ratio_row("positive")],
        "중립 비율 (%)":    [v * 100 for v in _ratio_row("neutral")],
        "부정 비율 (%)":    [v * 100 for v in _ratio_row("negative")],
    }
    block = pd.DataFrame(data, index=cols).T.reset_index().rename(columns={"index": "메트릭"})
    for r in ["긍정 비율 (%)", "중립 비율 (%)", "부정 비율 (%)"]:
        block.loc[block["메트릭"] == r, cols] = block.loc[block["메트릭"] == r, cols].astype(float).round(1)
    return block


# --- Section C: year-over-year review count ratios ---

def _build_year_ratio(raw_df: pd.DataFrame) -> pd.DataFrame:
    if "date" not in raw_df.columns:
        return pd.DataFrame(columns=["메트릭"])
    dt = pd.to_datetime(raw_df["date"], errors="coerce")
    y = dt.dt.year.dropna()
    if y.empty:
        return pd.DataFrame(columns=["메트릭"])
    year_cnt = y.value_counts().sort_index()
    total = int(year_cnt.sum())
    years = year_cnt.index.astype(int).tolist()
    counts = year_cnt.values.tolist()
    ratios = (year_cnt / total * 100).round(1).tolist()
    data = [["리뷰 수", *counts], ["비율(%)", *ratios]]
    return pd.DataFrame(data, columns=["메트릭"] + years)


def _build_unified_report_sheet(
    rep_tbl: pd.DataFrame,
    plat_tbl: pd.DataFrame,
    year_tbl: pd.DataFrame,
    raw_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Concatenate all summary blocks into a single flat DataFrame for the
    'report' sheet. Written with header=False so the first row is data."""
    rows: List[List[object]] = []

    product_name = ""
    product_col = getattr(config, "PRODUCT_NAME_COL", None)
    if product_col and raw_df is not None and product_col in raw_df.columns:
        try:
            mode_series = raw_df[product_col].dropna()
            if not mode_series.empty:
                product_name = mode_series.mode().iloc[0]
        except Exception:
            product_name = ""

    rows.append(["크롤링 리포트"])
    rows.append(["품목명", product_name])
    rows.append(["회차"])
    rows.append([])
    rows.append(["Insights", "(비워두기)"])
    rows.append([])
    rows.append([])

    rows.append(["데이터 수량"])
    rows.append(plat_tbl.columns.tolist())
    for _, row in plat_tbl.iterrows():
        rows.append(list(row))

    if not year_tbl.empty:
        rows.append([])
        rows.append(["데이터 연도별 비율"])
        rows.append(year_tbl.columns.tolist())
        for _, row in year_tbl.iterrows():
            rows.append(list(row))

    if not rep_tbl.empty:
        rows.append([])
        rows.append(["클러스터 요약"])
        rows.append(rep_tbl.columns.tolist())
        for _, row in rep_tbl.iterrows():
            rows.append(list(row))

    return pd.DataFrame(rows)


def save_client_report(
    clause_df: pd.DataFrame,
    raw_df: pd.DataFrame,
    reps: Dict[int, List[str]],
    kw: Optional[Dict[int, List[str]]] = None,
    output_path: Path | None = None,
) -> None:
    """Write the client Excel report with four sheets:
    report (unified layout), 대표어요약, 데이터수량, 연도별비율."""
    if output_path is None:
        raise ValueError("output_path is required for save_client_report")

    clause_w_meta = _ensure_meta_join(clause_df, raw_df)

    facet_col = (
        "facet_bucket" if "facet_bucket" in clause_w_meta.columns
        else ("facet_top1" if "facet_top1" in clause_w_meta.columns else None)
    )
    if facet_col is None:
        raise ValueError("No facet column found for report (need facet_bucket or facet_top1)")

    clause_w_meta["분류"] = clause_w_meta[facet_col]
    clause_w_meta.loc[
        clause_w_meta["분류"].astype(str).str.strip().isin(["", "nan", "None"]),
        "분류",
    ] = None
    if clause_w_meta["분류"].isna().all():
        raise ValueError(
            f"Facet column '{facet_col}' is entirely NaN — check route_facets() and YAML threshold"
        )

    rep_tbl = _build_rep_summary_table(clause_w_meta, reps, kw=kw)
    plat_tbl = _build_platform_block(clause_w_meta, raw_df)
    year_tbl = _build_year_ratio(raw_df)
    unified = _build_unified_report_sheet(rep_tbl, plat_tbl, year_tbl, raw_df)

    with pd.ExcelWriter(output_path, engine="openpyxl") as w:
        unified.to_excel(w, sheet_name="report", index=False, header=False)
        rep_tbl.to_excel(w, sheet_name="대표어요약", index=False)
        plat_tbl.to_excel(w, sheet_name="데이터수량", index=False)
        year_tbl.to_excel(w, sheet_name="연도별비율", index=False)
