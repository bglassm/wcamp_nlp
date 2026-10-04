"""Execute unmodified Git functions with synthetic data and no model/API calls."""
import ast
import hashlib
import logging
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from typing import List
import subprocess
import numpy as np
import pandas as pd

import argparse
import sys
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--legacy-repo', type=Path, required=True, help='Existing local Git clone of bglassm/wcamp_nlp; never downloads data')
REPO = parser.parse_args().legacy_repo.resolve()
sys.path.insert(0, str(REPO))
logging.disable(logging.WARNING)
def original(path):
    return subprocess.check_output(['git', 'show', '08a3752898025f4d4a0747af5259ec1dc016c556:' + path], cwd=REPO, text=True)

namespace = dict(np=np, pd=pd, Path=Path, hashlib=hashlib, logger=logging.getLogger('repro'), List=List, Embedder=object)
tree = ast.parse(original('pipeline/embedder.py'))
node = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'CachingEmbedder')
exec(compile(ast.Module(body=[node], type_ignores=[]), '<original CachingEmbedder>', 'exec'), namespace)
class SyntheticEmbedder:
    def embed(self, texts):
        return np.array([[len(t), sum(map(ord, t))] for t in texts], dtype=np.float32)

with TemporaryDirectory() as tmp:
    cfg = SimpleNamespace(embed=SimpleNamespace(cache_dir=tmp, backend='local',model='synthetic',batch_size=2,device='cpu'))
    wrapper = namespace['CachingEmbedder'](SyntheticEmbedder(), cfg)
    negative = wrapper.embed(['가상 배송이 늦어요'], ['demo_synthetic-1_0'], 'demo')
    positive = wrapper.embed(['가상 포장은 좋아요'], ['demo_synthetic-1_0'], 'demo')
    assert np.array_equal(negative, positive)
    print('REPRODUCED: original cache returns negative clause vector for positive clause sharing reset ID')
    duplicate = wrapper.embed(['합성 문장','합성 문장'], ['same','same'], 'duplicate')
    assert duplicate.shape[0] == 4
    print('REPRODUCED: duplicate input IDs produce 4 vectors for 2 rows via many-to-many merge')

# Execute the actual label decoding expression from original absa.py.
absa = ast.parse(original('pipeline/absa.py'))
expr = next(n.value for n in ast.walk(absa) if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'pol' for t in n.targets))
decoded = eval(compile(ast.Expression(expr), '<original polarity adapter>', 'eval'), {'res':{'sentiment':'Negative'}})
assert decoded == 'n'
print('REPRODUCED: scalar Negative decodes as n; main treats n as unknown and neutralizes it')

labels = np.array([0,0,'other'], dtype=object)
cluster_tree = ast.parse(original('pipeline/clusterer.py'))
expr = next(n.value for n in ast.walk(cluster_tree) if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'n_clusters' for t in n.targets))
assert eval(compile(ast.Expression(expr), '<original cluster count>', 'eval'), {'labels':labels}) == 2
print('REPRODUCED: original count reports 2 clusters for one real cluster plus other noise')

report_ns = {'__name__':'synthetic_report'}
exec(compile(original('pipeline/report.py'), '<original report>', 'exec'), report_ns)
df = pd.DataFrame({'cluster_label':[0,999], 'polarity':['negative','negative'], 'clause':['합성 문장','합성 잡음']})
summary = report_ns['_build_rep_summary_table'](df, {0:['합성 문장']}, {})
assert summary['개수'].sum() == 2
print('REPRODUCED: original report includes offset noise 999 in cluster summary counts')

# The old persistent cluster namespace can also collide across polarities and wrap.
idmap_ns = {"__name__": "legacy_idmap"}
exec(compile(original("pipeline/idmap.py"), "<original stable-ID allocator>", "exec"), idmap_ns)
with TemporaryDirectory() as tmp:
    _, mapping = idmap_ns["assign_stable_ids"](
        pd.DataFrame({"cluster_label": [0, 2000]}),
        {0: ["합성 동일 문장"], 2000: ["합성 동일 문장"]},
        state_path=Path(tmp) / "ids.json",
    )
    assert mapping["0"] == mapping["2000"]
    print("REPRODUCED: original persistent IDs collide across polarities for identical representatives")
with TemporaryDirectory() as tmp:
    _, mapping = idmap_ns["assign_stable_ids"](
        pd.DataFrame({"cluster_label": list(range(999))}),
        {i: [f"합성 군집 {i}"] for i in range(999)},
        state_path=Path(tmp) / "ids.json",
    )
    assert len(set(mapping.values())) == 998
    print("REPRODUCED: original persistent allocator gives only 998 unique IDs to 999 clusters")
