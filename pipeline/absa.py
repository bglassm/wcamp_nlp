# pipeline/absa.py

# ── 1) 토크나이저 __repr__ 재귀 문제 우회 패치 ──
try:
    from transformers import PreTrainedTokenizerFast
    def _minimal_repr(self):
        # 클래스 이름과 name_or_path만 표시하도록 단순화
        return f"{self.__class__.__name__}(name_or_path='{getattr(self, 'name_or_path', '')}')"
    PreTrainedTokenizerFast.__repr__ = _minimal_repr
except ImportError:
    pass

# ── 2) 나머지 임포트 및 함수 정의 ──
from typing import Any, List, Tuple
import pandas as pd
import config
from pipeline.contracts import normalize_polarity
import os, builtins
from contextlib import contextmanager, redirect_stdout, redirect_stderr


def validate_clause_alignment(absa_df: pd.DataFrame, clause_df: pd.DataFrame, id_col: str = "review_id") -> None:
    """Require exactly the current ordered source clauses, including literal string IDs."""
    required = {"review_id", "clause", "polarity", "confidence"}
    if not required.issubset(absa_df.columns):
        raise ValueError("ABSA cache is missing required columns")
    expected = list(clause_df[[id_col, "clause"]].astype(str).itertuples(index=False, name=None))
    actual = list(absa_df[["review_id", "clause"]].astype(str).itertuples(index=False, name=None))
    if actual != expected:
        raise ValueError("ABSA results do not match the ordered source review IDs and clauses")


def load_absa_cache(path, clause_df: pd.DataFrame, id_col: str = "review_id") -> pd.DataFrame:
    """Preserve 001/NA-like IDs; reject stale, reordered, partial or modified caches."""
    frame = pd.read_csv(path, dtype={"review_id": str, "clause": str}, keep_default_na=False)
    validate_clause_alignment(frame, clause_df, id_col)
    return frame


def classify_clauses(
    clause_df: pd.DataFrame,
    model_name: str,
    batch_size: int,
    device: str
) -> List[Tuple[int, str, str, float]]:
    """
    절 단위 ABSA 수행 후, (review_id, clause, polarity, confidence) 리스트 반환.
    - polarity: "positive"/"neutral"/"negative"
    - confidence: 모델이 해당 판단을 얼마나 확신하는지 (float 0~1)
    """

    if clause_df.empty:
        return []
    from pyabsa import TaskCodeOption
    from pyabsa.framework.checkpoint_class.checkpoint_template import APCCheckpointManager

    # 1) 모델 로드
    try:
        model: Any = APCCheckpointManager.get_sentiment_classifier(
            checkpoint=model_name,
            auto_device=device,
            task_code=TaskCodeOption.Aspect_Polarity_Classification,
            force_download=False
        )
    except Exception as e:
        raise ImportError(f"PyABSA 모델 로드 오류: {e}")

    # 2) 절 리스트와 review_id 리스트 준비
    review_ids = clause_df[config.REVIEW_ID_COL].tolist()
    clauses    = clause_df["clause"].tolist()

    # 3) 예측 (표준출력/표준에러 모두 devnull로 리다이렉트)
    with _suppress_absa_noise():
        results: List[dict] = model.predict(
            clauses,
            batch_size=batch_size,
            ignore_report=True,
            print_result=False,
            ignore_detail=True,
        )

    return parse_clause_results(review_ids, clauses, results)


def parse_clause_results(review_ids, clauses, results):
    """Validate result cardinality and scalar/list labels without silently truncating."""
    import math
    if len(review_ids) != len(clauses) or len(results) != len(clauses):
        raise ValueError("ABSA result count does not match clause count")
    output = []
    for rid, clause, res in zip(review_ids, clauses, results):
        pol = normalize_polarity(res["sentiment"])
        confidence = res["confidence"]
        if isinstance(confidence, (list, tuple)):
            if len(confidence) != 1:
                raise ValueError("Expected exactly one confidence per clause")
            confidence = confidence[0]
        confidence = float(confidence)
        if not math.isfinite(confidence) or not 0 <= confidence <= 1:
            raise ValueError("Confidence must be finite and within [0, 1]")
        output.append((rid, clause, pol, confidence))
    return output

@contextmanager
def _suppress_absa_noise():
    """PyABSA가 콘솔에 찍는 print와 stdout/stderr을 잠시 막습니다."""
    import logging
    saved_levels = {}
    for name in list(logging.root.manager.loggerDict.keys()) + ["pyabsa"]:
        if str(name).startswith("pyabsa"):
            lg = logging.getLogger(name)
            saved_levels[name] = lg.level
            lg.setLevel(logging.CRITICAL)

    old_tf_verb = os.environ.get("TRANSFORMERS_VERBOSITY")
    os.environ["TRANSFORMERS_VERBOSITY"] = "error"

    devnull = open(os.devnull, "w")
    orig_print = builtins.print
    try:
        builtins.print = lambda *a, **k: None 
        with redirect_stdout(devnull), redirect_stderr(devnull):
            yield
    finally:
        builtins.print = orig_print
        devnull.close()
        for name, lvl in saved_levels.items():
            logging.getLogger(name).setLevel(lvl)
        if old_tf_verb is None:
            os.environ.pop("TRANSFORMERS_VERBOSITY", None)
        else:
            os.environ["TRANSFORMERS_VERBOSITY"] = old_tf_verb
