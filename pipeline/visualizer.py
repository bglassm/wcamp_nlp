# pipeline/visualizer.py
"""
Run-time visualization module for wcamp_nlp.

Generates a self-contained HTML dashboard and companion PNG charts every time
the pipeline completes a product.  Call `generate_run_report(...)` at the end
of `run_full_pipeline` to produce:

  <out_dir>/
    <stem>_dashboard_<timestamp>.html   ← single-file interactive dashboard
    <stem>_sentiment_pie_<timestamp>.png
    <stem>_facet_bar_<timestamp>.png
    <stem>_cluster_scatter_<timestamp>.png  (reuses UMAP coords if available)
    <stem>_year_trend_<timestamp>.png       (if date column present)

All functions are defensive: missing columns / empty data → graceful skip with
a log warning instead of raising.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional imports — degrade gracefully if not installed
# ---------------------------------------------------------------------------
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    _HAS_MPL = True
except ImportError:
    _HAS_MPL = False
    logger.warning("matplotlib not available — PNG charts will be skipped")

# ---------------------------------------------------------------------------
# Colour palette
# ---------------------------------------------------------------------------
_POLARITY_COLORS = {
    "positive": "#4CAF50",
    "neutral":  "#FFC107",
    "negative": "#F44336",
}
_POLARITY_LABELS_KO = {
    "positive": "긍정",
    "neutral":  "중립",
    "negative": "부정",
}
_CLUSTER_CMAP = "tab20"


# ===========================================================================
# Internal helpers
# ===========================================================================

def _safe_str(v) -> str:
    if v is None:
        return ""
    s = str(v).strip()
    return "" if s.lower() in {"nan", "none"} else s


def _polarity_counts(clause_df: pd.DataFrame) -> Dict[str, int]:
    if "polarity" not in clause_df.columns:
        return {}
    return clause_df["polarity"].value_counts().to_dict()


def _facet_counts(clause_df: pd.DataFrame) -> pd.Series:
    """Return value_counts for the best available facet column."""
    for col in ("facet_top1", "facet_bucket", "facet"):
        if col in clause_df.columns:
            s = clause_df[col].dropna().astype(str)
            s = s[~s.str.lower().isin({"nan", "none", "", "unrouted"})]
            if not s.empty:
                return s.value_counts().head(20)
    return pd.Series(dtype=int)


# ===========================================================================
# PNG chart generators
# ===========================================================================

def _save_sentiment_pie(clause_df: pd.DataFrame, out_path: Path, stem: str) -> bool:
    if not _HAS_MPL:
        return False
    counts = _polarity_counts(clause_df)
    if not counts:
        logger.warning("[VIS] sentiment_pie: no polarity column — skipped")
        return False

    labels = [_POLARITY_LABELS_KO.get(k, k) for k in counts]
    sizes  = list(counts.values())
    colors = [_POLARITY_COLORS.get(k, "#9E9E9E") for k in counts]

    fig, ax = plt.subplots(figsize=(5, 5))
    wedges, texts, autotexts = ax.pie(
        sizes, labels=labels, colors=colors,
        autopct="%1.1f%%", startangle=140,
        wedgeprops={"linewidth": 0.8, "edgecolor": "white"},
    )
    for at in autotexts:
        at.set_fontsize(10)
    ax.set_title(f"감정 분포 — {stem}", fontsize=13, pad=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info("[VIS] sentiment_pie saved: %s", out_path.name)
    return True


def _save_facet_bar(clause_df: pd.DataFrame, out_path: Path, stem: str) -> bool:
    if not _HAS_MPL:
        return False
    fc = _facet_counts(clause_df)
    if fc.empty:
        logger.warning("[VIS] facet_bar: no facet data — skipped")
        return False

    # stacked bar: facet × polarity
    pol_order = ["positive", "neutral", "negative"]
    facet_col = next(
        (c for c in ("facet_top1", "facet_bucket", "facet") if c in clause_df.columns), None
    )
    if facet_col is None:
        return False

    top_facets = fc.index.tolist()
    sub = clause_df[clause_df[facet_col].isin(top_facets)].copy()
    sub[facet_col] = sub[facet_col].astype(str)

    if "polarity" in sub.columns:
        ct = (
            sub.groupby([facet_col, "polarity"])
            .size()
            .unstack(fill_value=0)
            .reindex(top_facets)
        )
        for p in pol_order:
            if p not in ct.columns:
                ct[p] = 0
        ct = ct[pol_order]
    else:
        ct = pd.DataFrame({p: fc for p in pol_order}).reindex(top_facets).fillna(0)

    fig, ax = plt.subplots(figsize=(10, max(4, len(top_facets) * 0.45)))
    bottom = np.zeros(len(ct))
    for pol in pol_order:
        vals = ct[pol].values.astype(float)
        ax.barh(ct.index, vals, left=bottom,
                color=_POLARITY_COLORS[pol], label=_POLARITY_LABELS_KO[pol],
                height=0.65)
        bottom += vals

    ax.set_xlabel("절(clause) 수")
    ax.set_title(f"파셋별 감정 분포 — {stem}", fontsize=12)
    ax.legend(loc="lower right", fontsize=9)
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info("[VIS] facet_bar saved: %s", out_path.name)
    return True


def _save_cluster_scatter(
    clause_df: pd.DataFrame,
    embeddings_2d: Optional[np.ndarray],
    out_path: Path,
    stem: str,
) -> bool:
    """Scatter plot coloured by polarity (or cluster_label if no polarity)."""
    if not _HAS_MPL:
        return False
    if embeddings_2d is None or embeddings_2d.ndim != 2 or embeddings_2d.shape[1] < 2:
        logger.warning("[VIS] cluster_scatter: no 2-D coords — skipped")
        return False
    if len(embeddings_2d) != len(clause_df):
        logger.warning("[VIS] cluster_scatter: coord/df length mismatch — skipped")
        return False

    x, y = embeddings_2d[:, 0], embeddings_2d[:, 1]

    fig, ax = plt.subplots(figsize=(8, 6))

    if "polarity" in clause_df.columns:
        for pol, color in _POLARITY_COLORS.items():
            mask = clause_df["polarity"].values == pol
            if mask.any():
                ax.scatter(x[mask], y[mask], s=6, alpha=0.55, color=color,
                           label=_POLARITY_LABELS_KO[pol], rasterized=True)
        ax.legend(loc="best", fontsize=9, markerscale=2)
    else:
        ax.scatter(x, y, s=6, alpha=0.5, c="steelblue", rasterized=True)

    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")
    ax.set_title(f"클러스터 분포 (UMAP) — {stem}", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info("[VIS] cluster_scatter saved: %s", out_path.name)
    return True


def _save_year_trend(raw_df: pd.DataFrame, clause_df: pd.DataFrame, out_path: Path, stem: str) -> bool:
    if not _HAS_MPL:
        return False
    if "date" not in raw_df.columns:
        logger.warning("[VIS] year_trend: no date column — skipped")
        return False

    dt = pd.to_datetime(raw_df["date"], errors="coerce")
    year_cnt = dt.dt.year.dropna().astype(int).value_counts().sort_index()
    if year_cnt.empty:
        return False

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(year_cnt.index.astype(str), year_cnt.values, color="#42A5F5", edgecolor="white")
    ax.set_xlabel("연도")
    ax.set_ylabel("리뷰 수")
    ax.set_title(f"연도별 리뷰 수 — {stem}", fontsize=12)
    for xi, yi in zip(year_cnt.index.astype(str), year_cnt.values):
        ax.text(xi, yi + 0.5, str(yi), ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info("[VIS] year_trend saved: %s", out_path.name)
    return True


def _save_keyword_bar(
    kw: Dict[int, List[str]],
    clause_df: pd.DataFrame,
    out_path: Path,
    stem: str,
    top_n_clusters: int = 12,
) -> bool:
    """Horizontal bar chart: top keywords per cluster, coloured by polarity."""
    if not _HAS_MPL or not kw:
        return False

    # cluster → polarity mapping
    pol_map: Dict[int, str] = {}
    if "cluster_label" in clause_df.columns and "polarity" in clause_df.columns:
        for cid, grp in clause_df.groupby("cluster_label"):
            try:
                pol_map[int(cid)] = grp["polarity"].mode().iloc[0]
            except Exception:
                pol_map[int(cid)] = "neutral"

    # cluster → count
    cnt_map: Dict[int, int] = {}
    if "cluster_label" in clause_df.columns:
        for cid, cnt in clause_df["cluster_label"].value_counts().items():
            try:
                cnt_map[int(cid)] = int(cnt)
            except Exception:
                pass

    # pick top_n_clusters by size, exclude noise
    sorted_cids = sorted(
        [c for c in kw if cnt_map.get(c, 0) > 0 and c not in (-1, 999, 1999, 2999)],
        key=lambda c: cnt_map.get(c, 0),
        reverse=True,
    )[:top_n_clusters]

    if not sorted_cids:
        return False

    labels_text = []
    counts_val  = []
    bar_colors  = []
    for cid in sorted_cids:
        kws = kw.get(cid, [])
        kw_str = "·".join(str(k) for k in kws[:3]) if kws else f"cluster {cid}"
        labels_text.append(kw_str)
        counts_val.append(cnt_map.get(cid, 0))
        pol = pol_map.get(cid, "neutral")
        bar_colors.append(_POLARITY_COLORS.get(pol, "#9E9E9E"))

    fig, ax = plt.subplots(figsize=(10, max(4, len(sorted_cids) * 0.5)))
    y_pos = range(len(sorted_cids))
    ax.barh(list(y_pos), counts_val, color=bar_colors, height=0.65)
    ax.set_yticks(list(y_pos))
    ax.set_yticklabels(labels_text, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("절(clause) 수")
    ax.set_title(f"클러스터 키워드 (상위 {top_n_clusters}개) — {stem}", fontsize=12)

    patches = [mpatches.Patch(color=c, label=_POLARITY_LABELS_KO[p])
               for p, c in _POLARITY_COLORS.items()]
    ax.legend(handles=patches, loc="lower right", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info("[VIS] keyword_bar saved: %s", out_path.name)
    return True


# ===========================================================================
# HTML dashboard builder
# ===========================================================================

def _img_to_b64(path: Path) -> str:
    """Encode a PNG to base64 data-URI for embedding in HTML."""
    import base64
    try:
        data = path.read_bytes()
        b64  = base64.b64encode(data).decode("ascii")
        return f"data:image/png;base64,{b64}"
    except Exception:
        return ""


def _build_html_dashboard(
    stem: str,
    timestamp: str,
    clause_df: pd.DataFrame,
    raw_df: pd.DataFrame,
    reps: Dict[int, List[str]],
    kw: Dict[int, List[str]],
    png_paths: Dict[str, Path],
) -> str:
    """Return a self-contained HTML string with embedded charts and summary tables."""

    # ---- summary stats ----
    pol_counts = _polarity_counts(clause_df)
    total_clauses = len(clause_df)
    total_reviews = len(raw_df)
    n_clusters = int(
        clause_df["cluster_label"].nunique() if "cluster_label" in clause_df.columns else 0
    )

    def _pct(k):
        return f"{pol_counts.get(k, 0) / max(total_clauses, 1) * 100:.1f}%"

    # ---- cluster table rows ----
    cluster_rows_html = ""
    if "cluster_label" in clause_df.columns and "polarity" in clause_df.columns:
        for cid, grp in sorted(
            clause_df.groupby("cluster_label"),
            key=lambda x: -len(x[1])
        )[:20]:
            try:
                cid_int = int(cid)
            except Exception:
                continue
            if cid_int in (-1, 999, 1999, 2999):
                continue
            pol = grp["polarity"].mode().iloc[0] if not grp.empty else "neutral"
            pol_ko = _POLARITY_LABELS_KO.get(pol, pol)
            pol_color = _POLARITY_COLORS.get(pol, "#9E9E9E")
            cnt = len(grp)
            kw_str = "·".join(str(k) for k in kw.get(cid_int, [])[:3]) or "-"
            rep_str = _safe_str((reps.get(cid_int) or [""])[0])[:80]
            facet_col = next(
                (c for c in ("facet_top1", "facet_bucket") if c in grp.columns), None
            )
            facet_str = _safe_str(grp[facet_col].mode().iloc[0]) if facet_col else "-"
            cluster_rows_html += f"""
            <tr>
              <td>{cid_int}</td>
              <td><span style="background:{pol_color};color:#fff;padding:2px 8px;border-radius:4px;font-size:12px">{pol_ko}</span></td>
              <td>{facet_str}</td>
              <td><b>{kw_str}</b></td>
              <td>{cnt}</td>
              <td style="font-size:12px;color:#555">{rep_str}</td>
            </tr>"""

    # ---- embed PNGs ----
    def _img_tag(key: str, alt: str, width: str = "100%") -> str:
        p = png_paths.get(key)
        if p and p.exists():
            src = _img_to_b64(p)
            return f'<img src="{src}" alt="{alt}" style="width:{width};border-radius:6px;box-shadow:0 2px 8px rgba(0,0,0,.12)">'
        return f'<p style="color:#aaa;font-style:italic">[{alt} 생성 안 됨]</p>'

    html = f"""<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>wcamp_nlp 실행 결과 — {stem} ({timestamp})</title>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ font-family: 'Segoe UI', 'Apple SD Gothic Neo', sans-serif; background: #f5f6fa; color: #333; }}
  header {{ background: #1a237e; color: #fff; padding: 20px 32px; }}
  header h1 {{ font-size: 20px; font-weight: 600; }}
  header p  {{ font-size: 13px; opacity: .75; margin-top: 4px; }}
  .container {{ max-width: 1200px; margin: 24px auto; padding: 0 16px; }}
  .card {{ background: #fff; border-radius: 10px; box-shadow: 0 2px 8px rgba(0,0,0,.08); padding: 20px 24px; margin-bottom: 20px; }}
  .card h2 {{ font-size: 15px; font-weight: 600; color: #1a237e; margin-bottom: 14px; border-bottom: 2px solid #e8eaf6; padding-bottom: 8px; }}
  .stat-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(140px, 1fr)); gap: 12px; }}
  .stat-box {{ background: #e8eaf6; border-radius: 8px; padding: 14px 16px; text-align: center; }}
  .stat-box .val {{ font-size: 28px; font-weight: 700; color: #1a237e; }}
  .stat-box .lbl {{ font-size: 12px; color: #666; margin-top: 4px; }}
  .chart-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }}
  @media(max-width:700px) {{ .chart-grid {{ grid-template-columns: 1fr; }} }}
  table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
  th {{ background: #e8eaf6; color: #1a237e; padding: 8px 10px; text-align: left; font-weight: 600; }}
  td {{ padding: 7px 10px; border-bottom: 1px solid #f0f0f0; vertical-align: top; }}
  tr:hover td {{ background: #fafafa; }}
  footer {{ text-align: center; font-size: 12px; color: #aaa; padding: 24px; }}
</style>
</head>
<body>
<header>
  <h1>wcamp_nlp 실행 결과 대시보드</h1>
  <p>제품: <b>{stem}</b> &nbsp;|&nbsp; 실행 시각: {timestamp}</p>
</header>
<div class="container">

  <!-- 요약 통계 -->
  <div class="card">
    <h2>요약 통계</h2>
    <div class="stat-grid">
      <div class="stat-box"><div class="val">{total_reviews:,}</div><div class="lbl">원본 리뷰 수</div></div>
      <div class="stat-box"><div class="val">{total_clauses:,}</div><div class="lbl">분석 절(clause) 수</div></div>
      <div class="stat-box"><div class="val">{n_clusters}</div><div class="lbl">클러스터 수</div></div>
      <div class="stat-box" style="background:#e8f5e9"><div class="val" style="color:#2e7d32">{pol_counts.get("positive",0):,}</div><div class="lbl">긍정 ({_pct("positive")})</div></div>
      <div class="stat-box" style="background:#fff8e1"><div class="val" style="color:#f57f17">{pol_counts.get("neutral",0):,}</div><div class="lbl">중립 ({_pct("neutral")})</div></div>
      <div class="stat-box" style="background:#ffebee"><div class="val" style="color:#c62828">{pol_counts.get("negative",0):,}</div><div class="lbl">부정 ({_pct("negative")})</div></div>
    </div>
  </div>

  <!-- 차트 -->
  <div class="card">
    <h2>시각화 차트</h2>
    <div class="chart-grid">
      <div>{_img_tag("sentiment_pie", "감정 분포 파이차트")}</div>
      <div>{_img_tag("facet_bar", "파셋별 감정 분포")}</div>
      <div>{_img_tag("cluster_scatter", "클러스터 산점도")}</div>
      <div>{_img_tag("year_trend", "연도별 리뷰 추이")}</div>
    </div>
  </div>

  <!-- 키워드 바 차트 -->
  <div class="card">
    <h2>클러스터 키워드 분포</h2>
    {_img_tag("keyword_bar", "클러스터 키워드 바차트", "100%")}
  </div>

  <!-- 클러스터 요약 테이블 -->
  <div class="card">
    <h2>클러스터 요약 (상위 20개)</h2>
    <table>
      <thead>
        <tr><th>ID</th><th>감정</th><th>파셋</th><th>키워드</th><th>절 수</th><th>대표 문장</th></tr>
      </thead>
      <tbody>
        {cluster_rows_html if cluster_rows_html else '<tr><td colspan="6" style="text-align:center;color:#aaa">데이터 없음</td></tr>'}
      </tbody>
    </table>
  </div>

</div>
<footer>Generated by wcamp_nlp pipeline &nbsp;|&nbsp; {timestamp}</footer>
</body>
</html>"""
    return html


# ===========================================================================
# Public entry point
# ===========================================================================

def generate_run_report(
    stem: str,
    timestamp: str,
    clause_df: pd.DataFrame,
    raw_df: pd.DataFrame,
    reps: Dict[int, List[str]],
    kw: Dict[int, List[str]],
    out_dir: Path,
    embeddings_2d: Optional[np.ndarray] = None,
) -> Path:
    """
    Generate all visualizations and the HTML dashboard for one pipeline run.

    Parameters
    ----------
    stem        : product stem (used in filenames and titles)
    timestamp   : run timestamp string (e.g. "250801_143022")
    clause_df   : combined clause DataFrame (all polarities)
    raw_df      : original review DataFrame
    reps        : {cluster_id: [representative sentences]}
    kw          : {cluster_id: [keywords]}
    out_dir     : directory where outputs are written
    embeddings_2d : (N, 2) UMAP coordinates for scatter plot (optional)

    Returns
    -------
    Path to the generated HTML dashboard.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Set matplotlib font for Korean (best-effort)
    if _HAS_MPL:
        try:
            import matplotlib.font_manager as fm
            # Try common Korean fonts on Linux/Mac/Windows/Server
            for fname in ("Noto Sans CJK KR", "Noto Serif CJK KR", "NanumGothic", "AppleGothic", "Malgun Gothic", "DejaVu Sans"):
                try:
                    result = fm.findfont(fm.FontProperties(family=fname), fallback_to_default=False)
                    if result and "DejaVu" not in result:
                        plt.rcParams["font.family"] = fname
                        break
                except Exception:
                    continue
        except Exception:
            pass
        plt.rcParams["axes.unicode_minus"] = False

    png_paths: Dict[str, Path] = {}

    # 1. Sentiment pie
    p = out_dir / f"{stem}_sentiment_pie_{timestamp}.png"
    if _save_sentiment_pie(clause_df, p, stem):
        png_paths["sentiment_pie"] = p

    # 2. Facet stacked bar
    p = out_dir / f"{stem}_facet_bar_{timestamp}.png"
    if _save_facet_bar(clause_df, p, stem):
        png_paths["facet_bar"] = p

    # 3. Cluster scatter (UMAP)
    p = out_dir / f"{stem}_cluster_scatter_{timestamp}.png"
    if _save_cluster_scatter(clause_df, embeddings_2d, p, stem):
        png_paths["cluster_scatter"] = p

    # 4. Year trend
    p = out_dir / f"{stem}_year_trend_{timestamp}.png"
    if _save_year_trend(raw_df, clause_df, p, stem):
        png_paths["year_trend"] = p

    # 5. Keyword bar
    p = out_dir / f"{stem}_keyword_bar_{timestamp}.png"
    if _save_keyword_bar(kw, clause_df, p, stem):
        png_paths["keyword_bar"] = p

    # 6. HTML dashboard (always generated, embeds all PNGs as base64)
    html_path = out_dir / f"{stem}_dashboard_{timestamp}.html"
    html_content = _build_html_dashboard(
        stem=stem,
        timestamp=timestamp,
        clause_df=clause_df,
        raw_df=raw_df,
        reps=reps,
        kw=kw,
        png_paths=png_paths,
    )
    html_path.write_text(html_content, encoding="utf-8")
    logger.info("[VIS] HTML dashboard saved: %s", html_path.name)

    return html_path
