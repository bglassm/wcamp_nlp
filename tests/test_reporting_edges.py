"""Reporting regressions reproduced using only newly authored synthetic text."""
from html import escape
from html.parser import HTMLParser

import numpy as np
import pandas as pd
import pytest

from pipeline.summarizer import extract_representatives
from pipeline.visualizer import _build_html_dashboard, generate_run_report


class ElementCollector(HTMLParser):
    def __init__(self):
        super().__init__()
        self.elements = []

    def handle_starttag(self, tag, attrs):
        self.elements.append((tag, dict(attrs)))


@pytest.mark.parametrize("labels", [
    np.array([0.0, 0.0, -1.0]),
    np.array(["0.0", "0", "other"]),
    np.array([0, 0, 999]),
    np.array([0, 0, 1999.0]),
    np.array([0, 0, 2999.0]),
])
def test_representatives_accept_integral_numeric_labels_and_skip_noise(labels):
    texts = ["합성 배송이 늦었어요", "합성 배송이 매우 늦었어요", "합성 미분류 의견"]
    vectors = np.array([[1.0, 0.0], [0.9, 0.1], [0.0, 1.0]])
    representatives = extract_representatives(texts, vectors, labels, use_semantic_helper=False)
    assert set(representatives) == {0}
    assert representatives[0]
    assert set(representatives[0]).issubset(set(texts[:2]))


def test_representatives_reject_fractional_ids_instead_of_merging_them():
    with pytest.raises(ValueError, match="integer IDs"):
        extract_representatives(["합성 배송"], np.ones((1, 2)), np.array([0.5]), use_semantic_helper=False)


@pytest.mark.parametrize("missing", [None, np.nan, pd.NA, "", "   "])
def test_dashboard_with_missing_facet_and_polarity_still_renders(missing):
    frame = pd.DataFrame({"cluster_label": [0], "polarity": [missing], "facet_top1": [missing]})
    page = _build_html_dashboard("synthetic", "test", frame, pd.DataFrame({"review_id": ["s"]}), {0: ["합성 배송 지연"]}, {}, {})
    tbody = page.split("<tbody>", 1)[1].split("</tbody>", 1)[0]
    assert "합성 배송 지연" in tbody
    assert "<td>-</td>" in tbody
    assert tbody.count("<tr>") == 1


def test_report_entrypoint_writes_html_when_facets_are_missing(tmp_path, monkeypatch):
    import pipeline.visualizer as visualizer
    monkeypatch.setattr(visualizer, "_HAS_MPL", False)
    frame = pd.DataFrame({"cluster_label": [0], "polarity": ["negative"], "facet_top1": [None]})
    result = generate_run_report("synthetic", "test", frame, pd.DataFrame({"review_id": ["s"]}), {0: ["합성 배송 지연"]}, {}, tmp_path)
    assert result.is_file()
    assert "합성 배송 지연" in result.read_text()


def test_dashboard_escapes_all_data_fields_and_keeps_chart_embedding(tmp_path):
    markers = {
        "stem": '</title><script data-id="stem">x</script>',
        "time": '</p><iframe src="https://invalid.example"></iframe>',
        "polarity": '<em data-id="polarity">unknown</em>',
        "facet": '<svg onload="x">synthetic</svg>',
        "keyword": '<img src=x onerror="x">',
        "representative": '<script data-id="representative">synthetic</script>',
    }
    frame = pd.DataFrame({"cluster_label": [0], "polarity": [markers["polarity"]], "facet_top1": [markers["facet"]]})
    # Only base64 text is needed to test the renderer's existing PNG data URI.
    png = tmp_path / "synthetic.png"
    png.write_bytes(b"synthetic-test-chart")
    page = _build_html_dashboard(markers["stem"], markers["time"], frame,
                                 pd.DataFrame({"review_id": ["s"]}),
                                 {0: [markers["representative"]]}, {0: [markers["keyword"]]},
                                 {"sentiment_pie": png})
    for value in markers.values():
        assert value not in page
        assert escape(value, quote=True) in page
    parsed = ElementCollector()
    parsed.feed(page)
    tags = [tag for tag, _ in parsed.elements]
    assert not {"script", "iframe", "svg", "em"} & set(tags)
    images = [attrs for tag, attrs in parsed.elements if tag == "img"]
    assert len(images) == 1
    assert images[0]["src"].startswith("data:image/png;base64,")
    assert not any(attr.startswith("on") for _, attrs in parsed.elements for attr in attrs)
    assert "Content-Security-Policy" in page
    assert "img-src data:" in page


def test_noise_does_not_use_a_top_twenty_table_slot():
    rows = [{"cluster_label": cid, "polarity": "negative"} for cid in range(20)]
    for noise_id in [-1, 999, 1999, 2999]:
        rows.extend({"cluster_label": noise_id, "polarity": "negative"} for _ in range(30))
    page = _build_html_dashboard("synthetic", "test", pd.DataFrame(rows), pd.DataFrame({"review_id": ["s"]}), {}, {}, {})
    tbody = page.split("<tbody>", 1)[1].split("</tbody>", 1)[0]
    assert tbody.count("<tr>") == 20
    for cid in range(20):
        assert f"<td>{cid}</td>" in tbody
    for noise_id in [-1, 999, 1999, 2999]:
        assert f"<td>{noise_id}</td>" not in tbody
    assert '<div class="val">20</div><div class="lbl">클러스터 수</div>' in page
