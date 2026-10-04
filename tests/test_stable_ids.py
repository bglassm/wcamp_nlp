import json
import pandas as pd
import pytest
from pipeline.idmap import assign_stable_ids


def test_same_representative_does_not_cross_polarities(tmp_path):
    frame = pd.DataFrame({"cluster_label": [0, 1000, 2000, 999, 1999, 2999, -1]})
    reps = {label: ["합성 동일 문장"] for label in [0, 1000, 2000]}
    first, mapping = assign_stable_ids(frame, reps, state_path=tmp_path / "ids.json")
    assert [mapping[str(label)] // 1000 for label in [0, 1000, 2000]] == [0, 1, 2]
    assert len(set(mapping.values())) == 7
    second, repeated = assign_stable_ids(frame.iloc[::-1], reps, state_path=tmp_path / "ids.json")
    assert repeated == mapping
    assert second.stable_cluster_id.tolist() == first.stable_cluster_id.tolist()[::-1]


def test_allocator_does_not_wrap_and_exhaustion_preserves_state(tmp_path):
    state_path = tmp_path / "ids.json"
    frame = pd.DataFrame({"cluster_label": list(range(999))})
    reps = {i: [f"합성 군집 {i}"] for i in range(999)}
    _, mapping = assign_stable_ids(frame, reps, state_path=state_path)
    assert len(set(mapping.values())) == 999
    assert 999 not in mapping.values()
    previous = state_path.read_bytes()
    with pytest.raises(ValueError, match="capacity exhausted"):
        assign_stable_ids(pd.DataFrame({"cluster_label": [0]}), {0: ["새로운 합성 군집"]}, state_path=state_path)
    assert state_path.read_bytes() == previous


def test_legacy_state_is_invalidated_with_warning(tmp_path, caplog):
    state_path = tmp_path / "ids.json"
    state_path.write_text(json.dumps({"version": 1, "counters": {}, "sig2id": {"collision": 1}}))
    assign_stable_ids(pd.DataFrame({"cluster_label": [0]}), {0: ["합성 문장"]}, state_path=state_path)
    assert "Invalidating legacy" in caplog.text
    assert json.loads(state_path.read_text())["version"] == 2


def test_corrupt_state_fails_without_overwrite(tmp_path):
    state_path = tmp_path / "ids.json"
    state_path.write_text(json.dumps({"version": 2, "counters": {}, "sig2id": {"0:a": 1, "0:b": 1}}))
    original = state_path.read_bytes()
    with pytest.raises(ValueError, match="duplicate"):
        assign_stable_ids(pd.DataFrame({"cluster_label": [0]}), {}, state_path=state_path)
    assert state_path.read_bytes() == original
