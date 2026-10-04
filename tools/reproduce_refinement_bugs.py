"""Reproduce legacy refinement defects from Git using authored arrays, no models."""
import ast
import json
import subprocess
from pathlib import Path
from typing import Any, List, Optional
import numpy as np
import pandas as pd

import argparse
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--legacy-repo", type=Path, required=True)
REPO = parser.parse_args().legacy_repo.resolve()
source = subprocess.check_output(["git", "show", "5ffc89d32ca2f93a801d03c2f0e51889fcffde06:pipeline/refiner.py"], cwd=REPO, text=True)
tree = ast.parse(source.lstrip("\ufeff"))
names = {"_normalize_rows", "_safe_int", "_is_other", "_coerce_to_int_or_other", "refine_clusters"}
scope = dict(np=np, pd=pd, json=json, Any=Any, List=List, Optional=Optional, Facet=object)
scope["route_to_facets"] = lambda X, facets, **kw: ([None]*len(X), [[] for _ in X], [None]*len(X), [{} for _ in X])
scope["heterogeneity_score"] = lambda X, **kw: (1.0, 2)
scope["local_subcluster_kmeans"] = lambda X, **kw: np.arange(len(X)) % 2
exec(compile(ast.Module(body=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names],type_ignores=[]),"legacy refinement", "exec"),scope)

def run(labels, polarity="negative", index=None):
    frame = pd.DataFrame({"cluster_label": labels, "clause":[f"합성 문장 {i}" for i in range(len(labels))]},index=index)
    return scope["refine_clusters"](frame,np.ones((len(labels),2)),polarity,[],min_cluster_size_for_split=2)

result = run([1,1,10])
assert result.refined_cluster_id.tolist() == [10,11,10]
print("REPRODUCED: split parent 1 child 0 and unsplit parent 10 share final ID 10")
assert run([100,100]).refined_cluster_id.tolist() == [1000,1001]
print("REPRODUCED: split negative parent 100 enters neutral namespace 1000/1001")
assert run([0],polarity="positive").refined_cluster_id.tolist() == [0]
print("REPRODUCED: omitted positive prefix incorrectly yields negative namespace ID 0")
try:
    run([1,1,10],index=[10,20,30])
except IndexError:
    print("REPRODUCED: non-default DataFrame index is incorrectly used as embedding row position")
else:
    raise AssertionError("Expected legacy non-default-index failure")
legacy_reps = {1:["합성 분할 전 문장"],10:["합성 별도 군집 문장"]}
assert 11 not in legacy_reps
assert result.loc[0,"cluster_label"] == 1 and 10 in legacy_reps
print("REPRODUCED: raw representative keys give split child 10 another parent's sentence and omit child 11")
