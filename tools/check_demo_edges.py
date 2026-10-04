"""Six synthetic-only edge reproductions; run with the repository path argument."""
import json
import sys
import tempfile
from pathlib import Path

import argparse
import subprocess
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
parser.add_argument("--revision", default=None, help="Read only selected Python files from this Git revision; never copy datasets")
args = parser.parse_args()
repo = args.repo.resolve()
source_directory = None
if args.revision:
    source_directory = tempfile.TemporaryDirectory(prefix="synthetic-code-only-")
    code_root = Path(source_directory.name)
    code_paths = ["config.py", "demo.py", "pipeline/__init__.py", "pipeline/clause_splitter.py",
                  "pipeline/contracts.py", "pipeline/embedder.py", "pipeline/summarizer.py", "pipeline/visualizer.py"]
    for relative in code_paths:
        target = code_root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        content = subprocess.check_output(["git", "show", f"{args.revision}:{relative}"], cwd=repo)
        target.write_bytes(content)
else:
    code_root = repo
sys.path.insert(0, str(code_root))
import numpy as np
import pandas as pd
from demo import run_demo
from pipeline.clause_splitter import split_clauses
from pipeline.summarizer import extract_representatives
from pipeline.visualizer import _build_html_dashboard

results = {}
with tempfile.TemporaryDirectory(prefix="synthetic-edge-audit-") as tmp:
    payload = {
        "schema_version": 1,
        "synthetic": True,
        "provenance": "Fresh fictional test text; no source dataset was read.",
        "reviews": [{"review_id": str(i), "review": text} for i, text in enumerate(["늦", "늦", "색상이 좋아요."])],
    }
    demo_result = run_demo(payload, Path(tmp))
    results["mixed_one_character_negatives"] = {
        "expected_clusters": 1, "actual_clusters": demo_result["summary"]["cluster_count"],
        "actual_noise": demo_result["summary"]["noise_count"],
    }
    clauses = split_clauses(pd.DataFrame({"review_id": ["synthetic"], "review": ["색상이 좋아요!!! 배송이 늦어요..."]}), offline=True)
    results["repeated_sentence_punctuation"] = {"expected_clause_count": 2, "actual_clause_count": len(clauses), "actual_clauses": clauses.clause.tolist()}
    representatives = extract_representatives(["합성 배송 지연"] * 2, np.ones((2, 2)), np.array([0.0, 0.0]), use_semantic_helper=False)
    results["float_cluster_ids"] = {"expected_cluster_ids": [0], "actual_cluster_ids": list(representatives)}
    frame = pd.DataFrame({"cluster_label": [0], "polarity": ["negative"], "facet_top1": [None]})
    try:
        _build_html_dashboard("synthetic", "test", frame, pd.DataFrame({"review_id": ["synthetic"]}), {0: ["합성 배송 지연"]}, {}, {})
        results["missing_facet"] = {"expected_error": None, "actual_error": None}
    except Exception as exc:
        results["missing_facet"] = {"expected_error": None, "actual_error": type(exc).__name__ + ": " + str(exc)}
    frame = frame.drop(columns="facet_top1")
    page = _build_html_dashboard("synthetic", "test", frame, pd.DataFrame({"review_id": ["synthetic"]}), {0: ['<script>console.log("synthetic")</script>']}, {}, {})
    results["unescaped_report_text"] = {"expected_literal_script": False, "actual_literal_script": '<script>console.log("synthetic")</script>' in page}
    rows = [{"cluster_label": cid, "polarity": "negative"} for cid in range(20)]
    rows.extend({"cluster_label": -1, "polarity": "negative"} for _ in range(30))
    page = _build_html_dashboard("synthetic", "test", pd.DataFrame(rows), pd.DataFrame({"review_id": ["synthetic"]}), {}, {}, {})
    tbody = page.split("<tbody>", 1)[1].split("</tbody>", 1)[0]
    results["noise_occupies_top_twenty_slot"] = {"expected_rows": 20, "actual_rows": tbody.count("<tr>")}
print(json.dumps(results, ensure_ascii=False, indent=2))
