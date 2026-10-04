"""Synthetic refinement → final grouping → report/stable-ID regression tests.

K-means decisions are injected to exercise exact collision boundaries, without
downloading a model. Actual routing, refinement, representatives and exports run.
"""
import ast
import json
import logging
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import pytest

import config
import pipeline.refiner as refiner
from pipeline.contracts import cluster_counts, offset_cluster_label
from pipeline.exporter import save_clauses_summary_json
from pipeline.idmap import assign_stable_ids
from pipeline.report import _build_rep_summary_table
from pipeline.summarizer import extract_representatives


@pytest.fixture
def deterministic_split(monkeypatch):
    monkeypatch.setattr(refiner, "heterogeneity_score", lambda X, **kw: (1.0, 2))
    monkeypatch.setattr(refiner, "local_subcluster_kmeans", lambda X, **kw: np.arange(len(X)) % 2)
    # Lexical inference is separate from the namespace defect; no Kiwi/model load.
    monkeypatch.setattr(refiner, "tokenize_lemmas", lambda text: text.split())


def fixture_rows(labels, index=None):
    frame = pd.DataFrame({
        "review_id":[f"synthetic-{i}" for i in range(len(labels))],
        "clause_id":[f"synthetic-clause-{i}" for i in range(len(labels))],
        "clause":[f"새로 작성한 합성 문장 {i}" for i in range(len(labels))],
        "cluster_label":labels,
    },index=index)
    vectors = np.column_stack([np.ones(len(labels)),np.arange(len(labels)) + 1.0])
    return frame, vectors


def refine(frame, vectors, polarity="negative", **kwargs):
    facets = [refiner.Facet("synthetic", "합성 분류", "합성 설명", np.array([1.0,0.0]), [])]
    return refiner.refine_clusters(frame,vectors,polarity,facets,min_cluster_size_for_split=2,**kwargs)


@pytest.fixture
def finalize():
    # Compile the exact lightweight composition functions in main, without its
    # heavyweight command-line model imports. Their implementation is not copied.
    source = (Path(__file__).parents[1] / "main.py").read_text()
    tree = ast.parse(source)
    names = {"_offset_labels", "_relabel_dict", "_finalize_polarity_result"}
    nodes = [node for node in tree.body if isinstance(node,ast.FunctionDef) and node.name in names]
    scope = dict(np=np,pd=pd,Dict=Dict,config=config,offset_cluster_label=offset_cluster_label,
                 RefinementNamespaceError=refiner.RefinementNamespaceError,
                 extract_representatives=extract_representatives)
    exec(compile(ast.Module(body=nodes,type_ignores=[]),"actual main finalization", "exec"),scope)
    return scope["_finalize_polarity_result"]


def test_split_child_does_not_collide_with_unsplit_parent(deterministic_split):
    frame, vectors = fixture_rows([1,1,10,-1])
    result = refine(frame,vectors)
    assert result.cluster_label.tolist() == [1,1,10,-1]
    assert result.refined_cluster_id.tolist() == [1,0,10,999]
    assert result.refined_label.tolist() == [1,0,10,-1]
    assert cluster_counts(result.refined_cluster_id) == {"n_clusters":3,"n_noise":1}


@pytest.mark.parametrize("polarity,prefix", [("negative",0),("neutral",1),("positive",2)])
def test_large_parent_stays_in_inferred_polarity_namespace(deterministic_split,polarity,prefix):
    frame,vectors = fixture_rows([100,100,-1])
    result = refine(frame,vectors,polarity=polarity)
    assert result.refined_cluster_id.tolist() == [prefix*1000+100,prefix*1000,prefix*1000+999]
    with pytest.raises(refiner.RefinementNamespaceError,match="does not match"):
        refine(frame,vectors,polarity=polarity,stable_id_prefix=(prefix+1)%3)


@pytest.mark.parametrize("index", [[10,20,30,40],[10,10,30,30]])
def test_dataframe_indices_do_not_change_embedding_alignment(deterministic_split,index):
    frame,vectors = fixture_rows([1,1,10,-1],index=index)
    result = refine(frame,vectors)
    assert result.index.tolist() == index
    assert result.clause_id.tolist() == frame.clause_id.tolist()
    assert result.refined_cluster_id.tolist() == [1,0,10,999]


def test_capacity_999_is_valid_1000_fails_without_mutating_input(deterministic_split):
    full,vectors = fixture_rows(list(range(999)))
    result = refine(full,vectors)
    assert result.refined_cluster_id.nunique() == 999
    assert result.refined_cluster_id.max() == 998
    over,vectors = fixture_rows([0,0]+list(range(1,999)))
    before = over.copy(deep=True)
    with pytest.raises(refiner.RefinementNamespaceError,match="1000 groups exceed 999"):
        refine(over,vectors)
    pd.testing.assert_frame_equal(over,before)


def test_raw_namespace_violation_and_embedding_mismatch_fail(deterministic_split):
    frame,vectors = fixture_rows([999])
    with pytest.raises(refiner.RefinementNamespaceError,match="0..998"):
        refine(frame,vectors)
    frame,vectors = fixture_rows([1,1])
    with pytest.raises(ValueError,match="align"):
        refine(frame,vectors[:1])


def synthetic_keywords(reps, **kwargs):
    return {cid:["합성어",sentences[0]] for cid,sentences in reps.items()}


def test_refined_groups_reach_report_json_and_stable_ids(deterministic_split,finalize,tmp_path):
    frame,vectors = fixture_rows([1,1,10,-1],index=[10,20,30,40])
    refined = refine(frame,vectors)
    result,reps,kw = finalize(
        frame,frame.cluster_label.to_numpy(),vectors,polarity="negative",base=0,
        reps={1:["합성 기존 부모 대표"],10:["합성 다른 부모 대표"]},kw={1:["이전"],10:["이전"]},
        refined_df=refined,keyword_extractor=synthetic_keywords,
    )
    assert result.original_cluster_label.tolist() == [1,1,10,999]
    assert result.cluster_label.equals(result.refined_cluster_id)
    assert set(reps) == set(kw) == {0,1,10}
    for cid,sentences in reps.items():
        assert set(sentences) <= set(result.loc[result.cluster_label == cid,"clause"])
        assert kw[cid][1] == sentences[0]
    assert result.refinement_applied.all()
    table = _build_rep_summary_table(result,reps,kw)
    assert len(table) == 3 and table["개수"].sum() == 3
    assert set(table["대표 문장"]) == {items[0] for items in reps.values()}
    path = tmp_path / "summary.json"
    save_clauses_summary_json(result,reps,kw,path)
    assert set(json.loads(path.read_text())) == {"0","1","10"}
    stable,_ = assign_stable_ids(result,reps,state_path=tmp_path/"ids.json")
    assert stable.loc[stable.cluster_label != 999,"stable_cluster_id"].nunique() == 3
    assert stable.loc[stable.cluster_label == 999,"stable_cluster_id"].tolist() == [999]


def test_mixed_refinement_fallback_and_transient_parent_renumbering(deterministic_split,finalize,tmp_path):
    frames,reps_all = [],{}
    for polarity,base,raw in [("negative",0,[1,1,10]),("neutral",1000,[0]),("positive",2000,[0,-1])]:
        source,vectors = fixture_rows(raw)
        source["clause"] = source.clause + " " + polarity
        refined = refine(source,vectors,polarity) if polarity == "negative" else None
        original_reps = {cid:[source.loc[source.cluster_label==cid,"clause"].iloc[0]] for cid in set(raw) if cid >= 0}
        result,reps,_ = finalize(source,np.asarray(raw),vectors,polarity=polarity,base=base,
                                 reps=original_reps,kw=synthetic_keywords(original_reps),
                                 refined_df=refined,keyword_extractor=synthetic_keywords)
        frames.append(result);reps_all.update(reps)
    combined = pd.concat(frames,ignore_index=True)
    assert not combined.refined_cluster_id.isna().any()
    assert combined.cluster_label.equals(combined.refined_cluster_id)
    stable,_ = assign_stable_ids(combined,reps_all,state_path=tmp_path/"ids.json")
    expected_prefix = stable.polarity.map({"negative":0,"neutral":1,"positive":2})
    assert (stable.stable_cluster_id // 1000).tolist() == expected_prefix.tolist()
    first_ids = dict(zip(stable.clause,stable.stable_cluster_id))
    changed,vectors = fixture_rows([7,7,11])
    changed["clause"] = changed.clause + " negative"
    refined = refine(changed,vectors)
    result,reps,_ = finalize(changed,changed.cluster_label.to_numpy(),vectors,polarity="negative",base=0,
                            reps={},kw={},refined_df=refined,keyword_extractor=synthetic_keywords)
    stable,_ = assign_stable_ids(result,reps,state_path=tmp_path/"ids.json")
    assert stable.stable_cluster_id.tolist() == [first_ids[text] for text in stable.clause]


def test_finalization_rejects_misaligned_or_cross_polarity_refined_rows(deterministic_split,finalize):
    source,vectors = fixture_rows([1,1])
    refined = refine(source,vectors)
    with pytest.raises(ValueError,match="ordered source"):
        finalize(source,np.array([1,1]),vectors,polarity="negative",base=0,reps={},kw={},
                 refined_df=refined.iloc[::-1],keyword_extractor=synthetic_keywords)
    refined.loc[0,"refined_cluster_id"] = 1000
    with pytest.raises(refiner.RefinementNamespaceError,match="escape"):
        finalize(source,np.array([1,1]),vectors,polarity="negative",base=0,reps={},kw={},
                 refined_df=refined,keyword_extractor=synthetic_keywords)


@pytest.mark.parametrize("namespace_failure", [False,True])
def test_actual_main_refinement_error_handling_preserves_fallback_but_propagates_capacity(namespace_failure):
    tree = ast.parse((Path(__file__).parents[1]/"main.py").read_text())
    route = next(node for node in tree.body if isinstance(node,ast.FunctionDef) and node.name=="run_full_pipeline")
    block = next(node for node in ast.walk(route) if isinstance(node,ast.Try)
                 and any(isinstance(child,ast.Call) and isinstance(child.func,ast.Name)
                         and child.func.id=="refine_clusters" for child in ast.walk(node)))
    source,vectors = fixture_rows([1,1])
    exception_type = refiner.RefinementNamespaceError if namespace_failure else RuntimeError
    def fail_refinement(*args,**kwargs):
        raise exception_type("synthetic injected failure")
    scope = dict(np=np,pd=pd,logging=logging,sub_df=source,labels_raw=np.array([1,1]),
                 semantic_clause_embs=vectors,_normalize_rows=refiner._normalize_rows,
                 facets_obj=[object()],refine_th={},pol="negative",stable_id_prefix_map={"negative":0},
                 refine_clusters=fail_refinement,RefinementNamespaceError=refiner.RefinementNamespaceError)
    execute = lambda:exec(compile(ast.Module(body=[block],type_ignores=[]),"actual main refinement handling","exec"),scope)
    if namespace_failure:
        with pytest.raises(refiner.RefinementNamespaceError,match="synthetic injected failure"):
            execute()
    else:
        execute()
        assert scope["refined_df"] is None
