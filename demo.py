"""Small offline demonstration using newly authored synthetic Korean reviews.

This is a functional example, not a trained ABSA model or a validation of the
original 766,000-review analysis. No original data file is read by this module.
"""
from __future__ import annotations

import argparse
import hashlib
import html
import json
import logging
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.feature_extraction.text import TfidfVectorizer

from pipeline.clause_splitter import split_clauses
from pipeline.contracts import (
    assign_clause_ids,
    cluster_counts,
    is_noise_label,
    normalize_polarity,
)
from pipeline.embedder import CachingEmbedder, Embedder
from pipeline.summarizer import extract_representatives

ROOT = Path(__file__).resolve().parent
DEFAULT_INPUT = ROOT / "examples" / "synthetic_reviews.json"
DISCLAIMER = (
    "새로 작성한 합성 리뷰만 사용하는 오프라인 기능 데모입니다. "
    "규칙 기반 극성 분류는 학습된 ABSA 모델이 아니며 정확도를 측정하지 않았습니다. "
    "이 결과는 원래의 76.6만 건 분석을 재검증한 결과가 아닙니다."
)
LOGGER = logging.getLogger("synthetic_demo")

# Deliberately small, auditable baseline. These rules are not a general Korean
# sentiment model: sarcasm, implicit complaints and most negation are unsupported.
NEGATIVE_PHRASES = (
    "좋지 않", "편하지 않", "만족스럽지 않", "늦", "찢어", "찢어진",
    "물이 새", "물이 자꾸 새", "물이 조금씩 새", "삐걱", "거슬", "아쉬",
    "불편", "별로", "실망", "깨졌", "망가", "고장", "나빠", "나쁘",
)
POSITIVE_PHRASES = (
    "예뻐", "편해", "좋아", "좋네", "좋은", "만족", "적당", "쉬워", "마음에 들어",
)
NEGATED_NEGATIVE_PHRASES = ("나쁘지 않", "불편하지 않")
POLARITY_KO = {"positive": "긍정", "neutral": "중립", "negative": "부정"}


def rule_polarity(text: str) -> tuple[str, str]:
    """Return canonical polarity and a matched rule, without fake confidence."""
    remaining = text
    negated_matches = []
    for phrase in NEGATED_NEGATIVE_PHRASES:
        if phrase in remaining:
            negated_matches.append(phrase)
            remaining = remaining.replace(phrase, "")
    for phrase in NEGATIVE_PHRASES:
        if phrase in remaining:
            return normalize_polarity("negative"), phrase
    if negated_matches:
        return normalize_polarity("positive"), negated_matches[0]
    for phrase in POSITIVE_PHRASES:
        if phrase in remaining:
            return normalize_polarity("positive"), phrase
    return normalize_polarity("neutral"), "일치하는 규칙 없음"


def load_synthetic_reviews(path: Path) -> dict[str, Any]:
    """Accept only the explicitly declared synthetic JSON schema for this demo."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    validate_payload(payload)
    return payload


def validate_payload(payload: Any) -> None:
    if not isinstance(payload, dict) or payload.get("synthetic") is not True:
        raise ValueError("The demo accepts only JSON with synthetic: true; do not use original reviews.")
    if payload.get("schema_version") != 1:
        raise ValueError("schema_version must be 1")
    if not isinstance(payload.get("provenance"), str) or not payload["provenance"].strip():
        raise ValueError("Describe how the synthetic reviews were newly authored in provenance")
    reviews = payload.get("reviews")
    if not isinstance(reviews, list):
        raise ValueError("reviews must be a list")
    if len(reviews) > 1000:
        raise ValueError("This small demo supports at most 1,000 synthetic reviews")
    seen: set[str] = set()
    for review in reviews:
        if not isinstance(review, dict):
            raise ValueError("Each review must contain review_id and review")
        rid, text = review.get("review_id"), review.get("review")
        if not isinstance(rid, str) or not rid.strip() or rid in seen:
            raise ValueError("review_id must be a unique nonempty string")
        if not isinstance(text, str) or len(text) > 10000:
            raise ValueError("review must be a string of at most 10,000 characters")
        seen.add(rid)


class CharacterTfidfEmbedder(Embedder):
    """Deterministic character n-grams; frozen vocabulary per input corpus.

    The corpus fingerprint is part of the cache model identity, because TF-IDF
    vectors from different vocabularies/IDF values must never share a cache.
    """

    def __init__(self, corpus: list[str]):
        ordered_corpus = sorted(corpus)
        serialized = json.dumps(ordered_corpus, ensure_ascii=False, separators=(",", ":"))
        # Any one-character clause (e.g. "늦") needs unigrams, including when
        # longer clauses exist beside it. Otherwise its vector is silently zero.
        self.ngram_range = (1, 4) if any(len(text.strip()) < 2 for text in ordered_corpus) else (2, 4)
        self.model_name = f"char-tfidf-{self.ngram_range[0]}-{self.ngram_range[1]}-v2-" + hashlib.sha256(serialized.encode()).hexdigest()
        self.vectorizer = TfidfVectorizer(analyzer="char", ngram_range=self.ngram_range, norm="l2", dtype=np.float32)
        self.vectorizer.fit(ordered_corpus)
        self.calls = 0
        self.embedded_text_count = 0

    def embed(self, texts: list[str]) -> np.ndarray:
        self.calls += 1
        self.embedded_text_count += len(texts)
        return self.vectorizer.transform(texts).toarray().astype(np.float32)


def _cache_config(cache_dir: Path, model_name: str) -> SimpleNamespace:
    return SimpleNamespace(embed=SimpleNamespace(
        backend="offline", model=model_name, batch_size=128,
        device="cpu", api_base="offline", cache_dir=str(cache_dir),
    ))


def run_demo(
    payload: dict[str, Any], output_dir: Path, *, eps: float = 0.68,
    min_samples: int = 2, cache_dir: Path | None = None,
) -> dict[str, Any]:
    """Execute clauses → rule polarity → negative clustering → representatives."""
    validate_payload(payload)
    if not np.isfinite(eps) or not 0 < eps <= 2:
        raise ValueError("eps must be finite and in (0, 2] for cosine distance")
    if isinstance(min_samples, bool) or not isinstance(min_samples, int) or min_samples < 2:
        raise ValueError("min_samples must be an integer >= 2")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = Path(cache_dir) if cache_dir else output_dir / ".cache"

    reviews = pd.DataFrame(payload["reviews"], columns=["review_id", "review"])
    clauses = split_clauses(reviews, offline=True)
    # Schema remains usable when the input is empty or contains blank reviews.
    if clauses.empty:
        clauses = pd.DataFrame(columns=["review_id", "clause", "clause_source"])
    clauses = assign_clause_ids(clauses, product_id="synthetic-demo")
    predictions = [rule_polarity(text) for text in clauses["clause"].tolist()]
    clauses["polarity"] = [value[0] for value in predictions]
    clauses["matched_rule"] = [value[1] for value in predictions]
    negatives = clauses.loc[clauses["polarity"] == "negative"].copy()
    LOGGER.info("[DEMO] reviews=%d clauses=%d negative=%d", len(reviews), len(clauses), len(negatives))

    labels = np.empty(0, dtype=int)
    representatives: dict[int, list[str]] = {}
    embedding_method = "not run: no negative clauses"
    if not negatives.empty:
        backend = CharacterTfidfEmbedder(clauses["clause"].tolist())
        embedding_method = f"character TF-IDF {backend.ngram_range[0]}–{backend.ngram_range[1]} grams; corpus-specific cached vectors"
        embedder = CachingEmbedder(backend, _cache_config(cache_dir, backend.model_name))
        # Separate polarity batches share one product cache. IDs were assigned
        # once above; the fixed cache additionally binds every ID to text content.
        negative_vectors = None
        for polarity in ("positive", "neutral", "negative"):
            subset = clauses.loc[clauses["polarity"] == polarity]
            if subset.empty:
                continue
            vectors = embedder.embed(
                subset["clause"].tolist(), subset["clause_id"].tolist(), "synthetic-demo",
            )
            if polarity == "negative":
                negative_vectors = vectors
        labels = DBSCAN(eps=eps, min_samples=min_samples, metric="cosine").fit_predict(negative_vectors)
        representatives = extract_representatives(
            negatives["clause"].tolist(), negative_vectors, labels,
            top_k=2, use_semantic_helper=False,
        )
        LOGGER.info("[DEMO] cache backend calls=%d embedded_texts=%d", backend.calls, backend.embedded_text_count)
    negatives["cluster_label"] = labels
    counts = cluster_counts(labels)
    LOGGER.info("[DEMO] clusters=%d noise=%d", counts["n_clusters"], counts["n_noise"])

    label_by_clause = dict(zip(negatives["clause_id"], labels.tolist()))
    clause_records = []
    for item in clauses.to_dict(orient="records"):
        label = label_by_clause.get(item["clause_id"])
        item["cluster_label"] = int(label) if label is not None else None
        item["is_noise"] = label is not None and is_noise_label(label)
        clause_records.append(item)

    clusters = []
    for cid in sorted(set(labels.tolist())):
        if is_noise_label(cid):
            continue
        members = negatives.loc[negatives["cluster_label"] == cid]
        clusters.append({
            "cluster_id": int(cid),
            "count": len(members),
            "representatives": representatives.get(int(cid), []),
            "clause_ids": members["clause_id"].tolist(),
        })
    result = {
        "schema_version": 1,
        "synthetic": True,
        "provenance": payload["provenance"],
        "product": str(payload.get("product", "가상의 상품")),
        "disclaimer": DISCLAIMER,
        "methods": {
            "clause_split": "pipeline.clause_splitter.split_clauses(offline=True): punctuation + connective rules",
            "polarity": "explicit Korean phrase rules; not trained ABSA; no accuracy claim",
            "embedding": embedding_method,
            "clustering": {"algorithm": "DBSCAN", "metric": "cosine", "eps": eps, "min_samples": min_samples},
            "representatives": "pipeline.summarizer.extract_representatives: provided vectors + MMR; no model loading",
        },
        "summary": {
            "review_count": len(reviews),
            "clause_count": len(clauses),
            "polarity_counts": {p: int((clauses["polarity"] == p).sum()) for p in POLARITY_KO},
            "negative_count": len(negatives),
            "cluster_count": counts["n_clusters"],
            "clustered_negative_count": len(negatives) - counts["n_noise"],
            "noise_count": counts["n_noise"],
        },
        "reviews": payload["reviews"],
        "clauses": clause_records,
        "clusters": clusters,
        "noise": [item for item in clause_records if item["is_noise"]],
    }
    result_path = output_dir / "result.json"
    result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    html_path = output_dir / "index.html"
    html_path.write_text(render_html(result), encoding="utf-8")
    LOGGER.info("[DEMO] wrote %s", result_path)
    LOGGER.info("[DEMO] wrote %s", html_path)
    return result


def render_html(result: dict[str, Any]) -> str:
    """Self-contained page: escape all data; no scripts or external resources."""
    esc = lambda value: html.escape(str(value), quote=True)
    summary = result["summary"]
    metrics = [
        ("합성 리뷰", summary["review_count"]), ("분리한 절", summary["clause_count"]),
        ("부정 의견", summary["negative_count"]), ("군집", summary["cluster_count"]),
        ("미분류 노이즈", summary["noise_count"]),
    ]
    metric_html = "".join(f'<div class="metric"><span>{esc(label)}</span><strong>{esc(value)}</strong></div>' for label, value in metrics)
    cluster_html = ""
    for cluster in result["clusters"]:
        reps = "".join(f"<li>{esc(text)}</li>" for text in cluster["representatives"])
        cluster_html += (
            '<article class="cluster">'
            f'<div class="cluster-top"><h3>군집 {esc(cluster["cluster_id"])}</h3><span>{esc(cluster["count"])}개 절</span></div>'
            f'<p class="eyebrow">대표 문장 · MMR</p><ul>{reps}</ul></article>'
        )
    if not cluster_html:
        cluster_html = '<p class="empty">군집이 없습니다. 부정 의견이 없거나 모든 의견이 노이즈일 수 있습니다.</p>'
    noise_html = "".join(f'<li><code>{esc(item["review_id"])}</code> {esc(item["clause"])}</li>' for item in result["noise"])
    if not noise_html:
        noise_html = '<li>미분류 노이즈가 없습니다.</li>'
    rows = ""
    for item in result["clauses"]:
        label = "노이즈" if item["is_noise"] else (f'군집 {item["cluster_label"]}' if item["cluster_label"] is not None else "—")
        rows += (
            f'<tr><td><code>{esc(item["review_id"])}</code><small>{esc(item["clause_id"])}</small></td>'
            f'<td>{esc(item["clause"])}</td><td><span class="badge {esc(item["polarity"])}">'
            f'{esc(POLARITY_KO[item["polarity"]])}</span></td><td>{esc(item["matched_rule"])}</td><td>{esc(label)}</td></tr>'
        )
    if not rows:
        rows = '<tr><td colspan="5">처리할 절이 없습니다.</td></tr>'
    source_items = "".join(f'<li><code>{esc(item["review_id"])}</code> {esc(item["review"])}</li>' for item in result["reviews"])
    method = result["methods"]["clustering"]
    return f'''<!doctype html>
<html lang="ko">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta http-equiv="Content-Security-Policy" content="default-src 'none'; style-src 'unsafe-inline'; base-uri 'none'; form-action 'none'">
<title>합성 리뷰 탐색실 · 오프라인 데모</title>
<style>
:root {{ color-scheme: light; --ink:#172b36; --muted:#5e6e77; --line:#d9e1e2; --accent:#087e81; --paper:#f3f5f0; }}
* {{ box-sizing:border-box; }} body {{ margin:0; color:var(--ink); background:var(--paper); font:15px/1.75 system-ui,-apple-system,'Apple SD Gothic Neo','Malgun Gothic',sans-serif; }}
header {{ background:#132e38; color:#fff; padding:54px max(24px,calc((100% - 1180px)/2)) 44px; }}
.eyebrow {{ margin:0 0 8px; font-size:12px; letter-spacing:.1em; font-weight:700; color:var(--accent); }} header .eyebrow {{ color:#7eddd0; }}
h1 {{ margin:0; font-size:clamp(28px,4vw,44px); letter-spacing:-.06em; line-height:1.3; }} header p {{ max-width:740px; color:#c3d4da; margin-bottom:0; }}
main {{ max-width:1228px; margin:auto; padding:28px 24px 64px; }}
.notice {{ padding:16px 20px; background:#fff9e4; border:1px solid #dfce8c; border-radius:12px; font-size:14px; }}
.metrics {{ display:grid; grid-template-columns:repeat(5,minmax(0,1fr)); gap:12px; margin:24px 0; }}
.metric {{ padding:18px 22px; background:#fff; border:1px solid var(--line); border-radius:12px; }} .metric span {{ display:block; color:var(--muted); font-size:13px; }} .metric strong {{ display:block; font-size:36px; letter-spacing:-.03em; }}
.flow {{ padding:18px 22px; border-radius:12px; background:#e4eeea; display:flex; flex-wrap:wrap; gap:10px; align-items:center; font-weight:650; }} .flow i {{ color:#6c9389; font-style:normal; }}
section {{ margin-top:34px; }} h2 {{ font-size:23px; margin:0 0 4px; letter-spacing:-.04em; }} .description {{ margin:0 0 18px; color:var(--muted); }}
.clusters {{ display:grid; grid-template-columns:repeat(3,minmax(0,1fr)); gap:16px; }} .cluster {{ background:#fff; border:1px solid var(--line); border-top:4px solid var(--accent); padding:22px; border-radius:12px; }}
.cluster-top {{ display:flex; justify-content:space-between; align-items:center; gap:12px; margin-bottom:18px; }} h3 {{ margin:0; font-size:19px; }} .cluster-top span {{ color:var(--accent); font-weight:700; }} .cluster ul {{ padding-left:19px; margin-bottom:0; }} .cluster li+li {{ margin-top:8px; }}
.noise {{ background:#edf0f3; border:1px dashed #a8b7c1; border-radius:12px; padding:20px 24px; }} .noise ul {{ margin-bottom:0; padding-left:20px; }}
.table-wrap {{ overflow:auto; background:#fff; border:1px solid var(--line); border-radius:12px; max-height:660px; }} table {{ border-collapse:collapse; min-width:860px; width:100%; font-size:14px; }} th,td {{ padding:13px 16px; text-align:left; border-bottom:1px solid #e8eded; vertical-align:top; }} th {{ position:sticky; top:0; background:#eaf0ee; white-space:nowrap; }} td:first-child {{ width:180px; }} td:nth-child(2) {{ min-width:270px; }} td:nth-child(3),td:last-child {{ white-space:nowrap; }} td small {{ display:block; font-size:10px; color:#78858b; overflow-wrap:anywhere; max-width:200px; }}
.badge {{ padding:3px 9px; border-radius:20px; font-size:12px; font-weight:700; }} .negative {{ background:#ffe8df; color:#8d3b22; }} .positive {{ background:#dcf1e9; color:#1b6f58; }} .neutral {{ background:#e8ecf0; color:#536879; }}
code {{ font-size:.83em; }} details {{ border:1px solid var(--line); border-radius:12px; padding:18px 22px; background:#fff; }} summary {{ cursor:pointer; font-weight:700; }} details li {{ margin:10px 0; }}
.empty {{ border:1px dashed #a8b7c1; border-radius:12px; padding:22px; }} footer {{ margin-top:34px; border-top:1px solid var(--line); padding-top:20px; color:var(--muted); font-size:13px; }}
@media(max-width:800px) {{ .metrics {{ grid-template-columns:repeat(3,1fr); }} .clusters {{ grid-template-columns:1fr; }} header {{ padding:38px 24px; }} }}
@media(max-width:460px) {{ .metrics {{ grid-template-columns:repeat(2,1fr); }} .metric {{ padding:14px 18px; }} main {{ padding:22px 16px 42px; }} }}
</style>
</head>
<body>
<header><p class="eyebrow">SYNTHETIC · OFFLINE · REPRODUCIBLE</p><h1>합성 리뷰 탐색실</h1><p>{esc(result['product'])}에 대한 가상의 의견에서 불편 사항을 찾습니다. 절 분리부터 대표 문장까지, 작은 입력의 처리 과정을 확인할 수 있습니다.</p></header>
<main>
<aside class="notice"><strong>검증 범위</strong> · {esc(result['disclaimer'])}</aside>
<div class="metrics">{metric_html}</div>
<div class="flow"><span>① 합성 입력</span><i>→</i><span>② 절 분리</span><i>→</i><span>③ 규칙 기반 부정 추출</span><i>→</i><span>④ 문자 TF-IDF · DBSCAN</span><i>→</i><span>⑤ 대표 문장</span></div>
<section><h2>비슷한 부정 의견</h2><p class="description">코사인 거리 · eps {esc(method['eps'])} · 최소 {esc(method['min_samples'])}개 절. 군집 번호는 이 실행 안에서만 의미가 있습니다. 대표 문장은 실제 합성 절에서 선택했습니다.</p><div class="clusters">{cluster_html}</div></section>
<section class="noise"><h2>미분류 노이즈 · {esc(summary['noise_count'])}개</h2><p class="description">유사한 이웃이 부족한 의견입니다. 군집 수와 대표 문장 집계에서 제외합니다.</p><ul>{noise_html}</ul></section>
<section><h2>절 단위 처리 기록</h2><p class="description">긍정 {esc(summary['polarity_counts']['positive'])} · 중립 {esc(summary['polarity_counts']['neutral'])} · 부정 {esc(summary['polarity_counts']['negative'])}. 간단한 문자열 규칙의 판정 근거를 함께 표시합니다.</p><div class="table-wrap"><table><thead><tr><th>리뷰 / 절 ID</th><th>분리한 절</th><th>극성</th><th>일치한 규칙</th><th>군집</th></tr></thead><tbody>{rows}</tbody></table></div></section>
<section><details><summary>새로 작성한 합성 입력 {esc(summary['review_count'])}건 보기</summary><ul>{source_items}</ul><p>{esc(result['provenance'])}</p></details></section>
<footer>외부 API · 모델 다운로드 · 외부 화면 자산 없음. 이 데모의 입력 출처 선언은 작성자가 제공하며, 실제 리뷰의 익명화를 보증하는 기능은 없습니다. 상세 결과: 같은 폴더의 result.json.</footer>
</main></body></html>
'''


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="Newly authored synthetic JSON only")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "artifacts" / "demo")
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument("--eps", type=float, default=0.68)
    parser.add_argument("--min-samples", type=int, default=2)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s %(name)s %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(args.output_dir / "run.log", mode="w", encoding="utf-8")],
        force=True,
    )
    try:
        result = run_demo(load_synthetic_reviews(args.input), args.output_dir, eps=args.eps, min_samples=args.min_samples, cache_dir=args.cache_dir)
    except (ValueError, OSError, json.JSONDecodeError) as exc:
        LOGGER.error("[DEMO] %s", exc)
        return 2
    print(json.dumps(result["summary"], ensure_ascii=False, sort_keys=True))
    print(DISCLAIMER)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
