"""Synthetic regressions for defects reproduced against Git commit 08a3752."""
import ast
import logging
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List

import numpy as np
import pandas as pd
import pytest

from pipeline.absa import parse_clause_results, load_absa_cache
from pipeline.clause_splitter import split_clauses
from pipeline.contracts import assign_clause_ids, cluster_counts, normalize_polarity, offset_cluster_label
from pipeline.embedder import CachingEmbedder
from pipeline.report import _build_rep_summary_table
from pipeline.summarizer import extract_representatives, _get_semantic_embeddings
from pipeline.visualizer import _build_html_dashboard


class SyntheticEmbedder:
    def __init__(self):
        self.calls = []

    def embed(self, texts):
        self.calls.append(list(texts))
        return np.array([[len(t), sum(map(ord, t))] for t in texts], dtype=np.float32)


@pytest.fixture
def cached(tmp_path):
    backend = SyntheticEmbedder()
    cfg = SimpleNamespace(embed=SimpleNamespace(cache_dir=tmp_path, backend="local", model="synthetic-v1", batch_size=8, device="cpu"))
    return CachingEmbedder(backend, cfg), backend


def test_cross_polarity_ids_and_cache_do_not_alias(cached):
    wrapper, backend = cached
    df = assign_clause_ids(pd.DataFrame({"review_id":["synthetic-1"] * 2, "clause":["가상 배송은 늦어요", "가상 포장은 좋아요"], "polarity":["negative", "positive"]}), "synthetic")
    assert df.clause_idx.tolist() == [0, 1]
    assert df.clause_id.nunique() == 2
    result = []
    for polarity in ("negative", "positive"):
        part = df[df.polarity == polarity]
        result.append(wrapper.embed(part.clause.tolist(), part.clause_id.tolist(), "synthetic"))
    assert not np.array_equal(*result)
    assert len(backend.calls) == 2


def test_changed_text_reembeds_even_when_caller_reuses_id(cached):
    wrapper, backend = cached
    first = wrapper.embed(["합성 짧은 문장"], ["same"], "synthetic")
    second = wrapper.embed(["합성 수정된 더 긴 문장"], ["same"], "synthetic")
    assert not np.array_equal(first, second)
    assert len(backend.calls) == 2


def test_repeated_ids_return_exact_cardinality_and_keep_order(cached):
    wrapper, backend = cached
    texts = ["가상 첫 문장", "가상 둘째 문장", "가상 첫 문장"]
    ids = ["a", "b", "a"]
    first = wrapper.embed(texts, ids, "synthetic")
    second = wrapper.embed(texts[::-1], ids[::-1], "synthetic")
    assert first.shape == (3, 2)
    assert backend.calls == [[texts[0], texts[1]]]
    np.testing.assert_array_equal(second, first[::-1])


def test_bad_id_lengths_or_ambiguous_same_batch_fail_before_backend(cached):
    wrapper, backend = cached
    with pytest.raises(ValueError, match="equal lengths"):
        wrapper.embed(["가상 문장"], [], "synthetic")
    with pytest.raises(ValueError, match="different texts"):
        wrapper.embed(["합성 하나", "합성 둘"], ["a", "a"], "synthetic")
    assert backend.calls == []


@pytest.mark.parametrize("value,expected", [("Negative", "negative"), (["positive"], "positive"), (" NEG ", "negative"), ("중립", "neutral"), ("pos", "positive")])
def test_polarity_strings_and_singleton_lists(value, expected):
    assert normalize_polarity(value) == expected
    parsed = parse_clause_results(["synthetic-1"], ["가상 문장"], [{"sentiment":value,"confidence":0.9}])
    assert parsed[0][2] == expected


@pytest.mark.parametrize("value", [0, 1, 2, -1, "LABEL_0", "unknown", [], ["negative", "positive"]])
def test_ambiguous_polarity_is_rejected_instead_of_becoming_neutral(value):
    with pytest.raises(ValueError):
        normalize_polarity(value)


def test_absa_does_not_silently_drop_missing_predictions():
    with pytest.raises(ValueError, match="count"):
        parse_clause_results(["a", "b"], ["가상 하나", "가상 둘"], [{"sentiment":["negative"],"confidence":[0.9]}])


def test_noise_counts_and_report_are_consistent():
    labels = [0, 0, 1000, 999, 1999, 2999, -1, "other"]
    assert cluster_counts(labels) == {"n_clusters":2, "n_noise":5}
    df = pd.DataFrame({"cluster_label":[0,0,1000,999,1999,2999], "polarity":["negative"] * 6, "clause":["가상 문장"] * 6})
    table = _build_rep_summary_table(df, {0:["가상 문장"],1000:["합성 문장"]}, {})
    assert table["개수"].sum() == 3
    html = _build_html_dashboard("synthetic", "fixed", df, pd.DataFrame({"review_id":["a"]}), {}, {}, {})
    assert '<div class="val">2</div><div class="lbl">클러스터 수</div>' in html


def test_hdbscan_log_counts_raw_noise_before_display_replacement(monkeypatch, caplog):
    import config
    from pipeline.clusterer import cluster_embeddings
    fake = SimpleNamespace(labels_=np.array([0,0,-1]))
    fake.fit = lambda _: fake
    monkeypatch.setitem(sys.modules, "hdbscan", SimpleNamespace(HDBSCAN=lambda **kwargs: fake))
    monkeypatch.setattr(config, "HANDLE_OUTLIERS", True)
    monkeypatch.setattr(config, "OUTLIER_LABEL", "other")
    with caplog.at_level(logging.INFO):
        labels, _ = cluster_embeddings(np.zeros((3,2)), min_cluster_size=2, min_samples=1)
    assert list(labels) == [0,0,"other"]
    assert "done: 1 clusters, 1 noise points" in caplog.text


def test_label_offsets_match_representative_keys_and_guard_namespace():
    # Load only the two real main helpers; no model imports or external resources.
    tree = ast.parse((Path(__file__).parents[1] / "main.py").read_text())
    helpers = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in {"_offset_labels", "_relabel_dict"}]
    scope = dict(np=np, Dict=Dict, offset_cluster_label=offset_cluster_label)
    exec(compile(ast.Module(body=helpers, type_ignores=[]), "main helpers", "exec"), scope)
    assert scope["_offset_labels"](np.array([0,-1]), 1000).tolist() == [1000,1999]
    assert set(scope["_relabel_dict"]({0:[], -1:[]},1000)) == {1000,1999}
    assert offset_cluster_label(998,0) == 998
    with pytest.raises(ValueError, match="999 clusters"):
        offset_cluster_label(999,0)


def test_offline_clause_splitting_and_representatives_never_load_models(monkeypatch):
    import pipeline.clause_splitter as splitter
    import pipeline.summarizer as summarizer
    def unexpected():
        pytest.fail("offline path attempted to load an external model")
    monkeypatch.setattr(splitter, "_ensure_semantic_model", unexpected)
    monkeypatch.setattr(summarizer, "_load_semantic_helper", unexpected)
    df = split_clauses(pd.DataFrame({"review_id":["synthetic-1"], "review":["가상 포장은 좋아요. 하지만 가상 배송은 늦어요."]}), offline=True)
    assert len(df) == 2
    assert set(split_clauses(pd.DataFrame({"review_id":[],"review":[]}), offline=True).columns) == {"review_id","clause","clause_source"}
    reps = extract_representatives(df.clause.tolist(), np.eye(2), np.array([0,-1]), use_semantic_helper=False)
    assert reps == {0:[df.clause.iloc[0]]}


def test_same_width_vectors_from_other_model_are_not_reused_as_semantic_helper():
    calls = []
    helper = SimpleNamespace(get_sentence_embedding_dimension=lambda:2, encode=lambda texts, **kw: calls.append(texts) or np.array([[0.0,1.0]]))
    actual = _get_semantic_embeddings(["가상 문장"], np.array([[1.0,0.0]]), helper)
    assert calls == [["가상 문장"]]
    np.testing.assert_array_equal(actual, [[0.0,1.0]])


def test_original_embedding_only_route_runs_without_absa(cached, tmp_path):
    wrapper, backend = cached
    tree = ast.parse((Path(__file__).parents[1] / "main.py").read_text())
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "run_embed_pipeline")
    scope = dict(List=List, Path=Path, logging=logging, CachingEmbedder=CachingEmbedder,
                 config=SimpleNamespace(REVIEW_ID_COL="review_id"), get_embedder=lambda _:wrapper,
                 load_reviews=lambda _:pd.DataFrame({"review_id":["synthetic-1"], "review":["가상 배송은 늦어요."]}),
                 preprocess_reviews=lambda df:df, split_clauses=lambda df:split_clauses(df,offline=True),
                 assign_clause_ids=assign_clause_ids)
    exec(compile(ast.Module(body=[node],type_ignores=[]), "main embedding route", "exec"), scope)
    scope["run_embed_pipeline"]([Path("synthetic.xlsx")],tmp_path,save_stem="synthetic")
    assert backend.calls == [["가상 배송은 늦어요."]]


def test_absa_cache_preserves_leading_zeros_and_na_like_review_ids(tmp_path):
    source = pd.DataFrame({"custom_id":["001", "NA", "NULL"], "clause":["합성 하나", "합성 둘", "합성 셋"]})
    predicted = source.rename(columns={"custom_id":"review_id"}).assign(polarity="negative", confidence=0.9)
    path = tmp_path / "absa.csv.gz"
    predicted.to_csv(path,index=False)
    loaded = load_absa_cache(path,source,id_col="custom_id")
    assert loaded.review_id.tolist() == ["001", "NA", "NULL"]
    assert loaded.clause.tolist() == source.clause.tolist()


@pytest.mark.parametrize("mutation", ["reordered", "changed", "partial"])
def test_absa_cache_rejects_wrong_source_relationship(tmp_path, mutation):
    source = pd.DataFrame({"review_id":["001", "002"], "clause":["합성 하나", "합성 둘"]})
    predicted = source.assign(polarity="negative", confidence=0.9)
    if mutation == "reordered":
        predicted = predicted.iloc[::-1]
    elif mutation == "changed":
        predicted.loc[0,"clause"] = "합성 다른 문장"
    else:
        predicted = predicted.iloc[:1]
    path = tmp_path / "absa.csv.gz"
    predicted.to_csv(path,index=False)
    with pytest.raises(ValueError,match="ordered source"):
        load_absa_cache(path,source)
