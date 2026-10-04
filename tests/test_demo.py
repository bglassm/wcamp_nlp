"""Functional regression tests use only authored synthetic text and temp output."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

import demo
from pipeline.contracts import assign_clause_ids
from pipeline.embedder import CachingEmbedder


def payload(*texts: str) -> dict:
    return {
        "schema_version": 1,
        "synthetic": True,
        "provenance": "New fictional sentences authored only for this test.",
        "reviews": [{"review_id": f"test-{i}", "review": text} for i, text in enumerate(texts)],
    }


def test_end_to_end_uses_real_repaired_pipeline(tmp_path):
    result = demo.run_demo(demo.load_synthetic_reviews(demo.DEFAULT_INPUT), tmp_path)
    stats = result["summary"]
    assert stats["review_count"] == 16
    assert stats["clause_count"] == 31
    assert stats["negative_count"] == 13
    assert stats["cluster_count"] == 3
    assert stats["noise_count"] >= 1
    assert stats["clustered_negative_count"] + stats["noise_count"] == stats["negative_count"]
    assert sum(stats["polarity_counts"].values()) == stats["clause_count"]
    assert sum(cluster["count"] for cluster in result["clusters"]) == stats["clustered_negative_count"]
    assert len({clause["clause_id"] for clause in result["clauses"]}) == stats["clause_count"]
    for cluster in result["clusters"]:
        members = {clause["clause"] for clause in result["clauses"] if clause["clause_id"] in cluster["clause_ids"]}
        assert cluster["representatives"]
        assert set(cluster["representatives"]).issubset(members)
        assert cluster["cluster_id"] >= 0
    assert all(clause["cluster_label"] is None for clause in result["clauses"] if clause["polarity"] != "negative")
    assert len(result["noise"]) == stats["noise_count"]
    assert json.loads((tmp_path / "result.json").read_text()) == result
    page = (tmp_path / "index.html").read_text()
    assert "76.6만 건 분석을 재검증한 결과가 아닙니다" in page
    assert "규칙 기반 극성 분류는 학습된 ABSA 모델이 아니며" in page
    assert "미분류 노이즈" in page


@pytest.mark.parametrize("source", [payload(), payload("", "   "), payload("색상이 예뻐요. 손잡이가 편해요.")])
def test_empty_blank_and_all_positive_inputs(source, tmp_path, monkeypatch):
    def forbid_embedding(*args, **kwargs):
        raise AssertionError("No vectorizer should be created when no negative clause exists")
    monkeypatch.setattr(demo, "CharacterTfidfEmbedder", forbid_embedding)
    result = demo.run_demo(source, tmp_path)
    assert result["summary"]["negative_count"] == 0
    assert result["summary"]["cluster_count"] == 0
    assert result["summary"]["noise_count"] == 0
    assert result["clusters"] == result["noise"] == []
    assert (tmp_path / "index.html").is_file()


@pytest.mark.parametrize("texts", [("늦",), ("배송이 너무 늦었어요.",), ("배송이 너무 늦었어요.", "손잡이가 삐걱거려요.", "포장 상자가 찢어졌어요.")])
def test_single_negative_and_all_noise_are_not_clusters(texts, tmp_path):
    result = demo.run_demo(payload(*texts), tmp_path, min_samples=4)
    assert result["summary"]["negative_count"] == len(texts)
    assert result["summary"]["noise_count"] == len(texts)
    assert result["summary"]["cluster_count"] == 0
    assert result["summary"]["clustered_negative_count"] == 0
    assert result["clusters"] == []


def test_one_character_negatives_keep_vectors_with_longer_positive_input(tmp_path):
    source = payload("늦", "늦", "색상이 좋아요.")
    first = demo.run_demo(source, tmp_path)
    assert first["summary"]["negative_count"] == 2
    assert first["summary"]["cluster_count"] == 1
    assert first["summary"]["noise_count"] == 0
    assert first["clusters"][0]["count"] == 2
    assert "1–4 grams" in first["methods"]["embedding"]
    assert demo.run_demo(source, tmp_path) == first


@pytest.mark.parametrize("text,expected", [
    ("색상이 좋아요!!! 배송이 늦어요...", ["색상이 좋아요!!!", "배송이 늦어요..."]),
    ("색상이 좋아요！？배송이 늦어요?!", ["색상이 좋아요！？", "배송이 늦어요?!"]),
])
def test_repeated_punctuation_does_not_create_spurious_neutral_clauses(text, expected, tmp_path):
    result = demo.run_demo(payload(text), tmp_path)
    assert [clause["clause"] for clause in result["clauses"]] == expected
    assert result["summary"]["clause_count"] == 2
    assert result["summary"]["polarity_counts"] == {"positive": 1, "neutral": 0, "negative": 1}


def test_warm_cache_gives_identical_result_without_embedding(tmp_path, monkeypatch):
    source = demo.load_synthetic_reviews(demo.DEFAULT_INPUT)
    first = demo.run_demo(source, tmp_path)
    first_page = (tmp_path / "index.html").read_text()
    def forbid_embedding(self, texts):
        raise AssertionError("All vectors must be read from the warm cache")
    monkeypatch.setattr(demo.CharacterTfidfEmbedder, "embed", forbid_embedding)
    second = demo.run_demo(source, tmp_path)
    assert first == second
    assert first_page == (tmp_path / "index.html").read_text()


def test_mixed_polarity_cache_preserves_distinct_clauses_and_request_order(tmp_path):
    df = assign_clause_ids(pd.DataFrame({
        "review_id": ["same-review", "same-review", "different-review"],
        "clause": ["손잡이가 편해요.", "배송이 너무 늦었어요.", "색상이 예뻐요."],
    }), product_id="synthetic-mixed")
    texts = df["clause"].tolist()
    ids = df["clause_id"].tolist()
    backend = demo.CharacterTfidfEmbedder(texts)
    cached = CachingEmbedder(backend, demo._cache_config(tmp_path, backend.model_name))
    expected = backend.vectorizer.transform(texts).toarray().astype(np.float32)
    np.testing.assert_allclose(cached.embed([texts[0], texts[2]], [ids[0], ids[2]], "synthetic-mixed"), expected[[0, 2]])
    np.testing.assert_allclose(cached.embed([texts[1]], [ids[1]], "synthetic-mixed"), expected[[1]])
    assert backend.embedded_text_count == 3
    order = [2, 1, 0, 1]
    np.testing.assert_allclose(cached.embed([texts[i] for i in order], [ids[i] for i in order], "synthetic-mixed"), expected[order])
    assert backend.embedded_text_count == 3
    assert not np.array_equal(expected[0], expected[1])
    assert ids[0] != ids[1]


def test_tfidf_cache_identity_tracks_vocabulary_and_is_order_independent():
    original = demo.CharacterTfidfEmbedder(["예쁜 색상이 좋아요.", "배송이 늦었어요."])
    reordered = demo.CharacterTfidfEmbedder(["배송이 늦었어요.", "예쁜 색상이 좋아요."])
    changed = demo.CharacterTfidfEmbedder(["예쁜 색상이 좋아요.", "뚜껑 틈으로 물이 새요."])
    assert original.model_name == reordered.model_name
    assert original.model_name != changed.model_name
    np.testing.assert_array_equal(original.embed(["배송이 늦었어요."]), reordered.embed(["배송이 늦었어요."]))


def test_html_escapes_review_text_ids_product_and_provenance(tmp_path):
    source = payload('<script>alert("synthetic")</script> 배송이 늦었어요.')
    source["product"] = '<img src=x onerror="alert(1)">'
    source["provenance"] = '<svg onload="alert(2)">synthetic</svg>'
    source["reviews"][0]["review_id"] = '<iframe src="https://invalid.example">'
    demo.run_demo(source, tmp_path)
    page = (tmp_path / "index.html").read_text()
    assert "<script>" not in page
    assert "<img" not in page
    assert "<svg" not in page
    assert "<iframe" not in page
    assert "&lt;script&gt;" in page
    assert "&lt;img" in page
    assert "&lt;svg" in page
    assert "&lt;iframe" in page
    assert "Content-Security-Policy" in page
    assert "default-src 'none'" in page


@pytest.mark.parametrize("text,polarity", [
    ("배송이 너무 늦었어요.", "negative"),
    ("색상이 예뻐요.", "positive"),
    ("색상은 파란색이에요.", "neutral"),
    ("좋지 않아요.", "negative"),
    ("불편하지 않아요.", "positive"),
    ("불편하지 않지만 배송이 늦었어요.", "negative"),
])
def test_rule_baseline_has_explicit_canonical_polarities(text, polarity):
    label, rule = demo.rule_polarity(text)
    assert label == polarity
    assert rule


@pytest.mark.parametrize("patch", [
    {"synthetic": False}, {"schema_version": 2}, {"provenance": ""},
    {"reviews": [{"review_id": "duplicate", "review": "좋아요"}, {"review_id": "duplicate", "review": "늦었어요"}]},
    {"reviews": [{"review_id": "empty", "review": None}]},
])
def test_invalid_or_undeclared_source_is_rejected(patch, tmp_path):
    source = payload("좋아요.")
    source.update(patch)
    with pytest.raises(ValueError):
        demo.run_demo(source, tmp_path)


def test_cli_runs_without_network_or_heavy_model_imports(tmp_path):
    # Block sockets before importing the demo and reject accidental ML/API imports.
    # This tests runtime execution only; installation itself requires packages.
    guard = tmp_path / "guard"
    guard.mkdir()
    (guard / "sitecustomize.py").write_text('''
import importlib.abc
import socket
import sys
def no_network(*args, **kwargs):
    raise AssertionError("The synthetic demo must run without network access")
socket.create_connection = no_network
socket.socket.connect = no_network
class ModelImportBlocker(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in {"openai", "sentence_transformers", "torch", "pyabsa", "kss"}:
            raise AssertionError("Unexpected model/API dependency: " + fullname)
sys.meta_path.insert(0, ModelImportBlocker())
''')
    env = dict(os.environ, PYTHONPATH=str(guard), PYTHONNOUSERSITE="1", OPENAI_API_KEY="")
    output = tmp_path / "cli-output"
    proc = subprocess.run(
        [sys.executable, str(demo.ROOT / "demo.py"), "--output-dir", str(output)],
        cwd=tmp_path, env=env, capture_output=True, text=True, timeout=60,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert (output / "run.log").is_file()
    assert '"review_count": 16' in proc.stdout
    result = json.loads((output / "result.json").read_text())
    assert result["summary"]["negative_count"] == 13
