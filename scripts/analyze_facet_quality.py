import argparse  # NEW: CLI 인자 파싱 추가
import logging  # NEW: INFO 로깅 추가
import time
from pathlib import Path
import pandas as pd

# 분석 결과를 모아둘 폴더: output/analysis
ANALYSIS_DIR = Path("output") / "analysis"
ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

logger = logging.getLogger(__name__)  # NEW: INFO 로깅용 로거


def _cluster_timestamp_key(p: Path) -> str:  # NEW: 파일명 타임스탬프 파싱
    parts = p.stem.split("_")
    if len(parts) >= 2:
        return "_".join(parts[-2:])
    return ""


def _select_latest_file(files):  # NEW: 타임스탬프 우선, 실패 시 mtime 최신
    keyed = {p: _cluster_timestamp_key(p) for p in files}
    timestamped = {p: ts for p, ts in keyed.items() if ts}
    if timestamped:
        return max(timestamped.items(), key=lambda item: item[1])[0]
    return max(files, key=lambda p: p.stat().st_mtime)


def load_all_clauses(strategy="latest"):  # CHANGED: strategy 파라미터 추가
    """
    output/<sku>/<sku>_clauses_clustered_*.xlsx 파일을 모아 DataFrame을 만듭니다.

    strategy="latest" → sku별 최신 파일 1개만 사용 (기본값)
    strategy="all" → sku별 모든 clustered 파일 사용
    """
    base = Path("output")
    clause_files = []

    # sku 디렉토리 순회
    for sku_dir in base.iterdir():
        if not sku_dir.is_dir():
            continue

        # sku명
        sku = sku_dir.name

        # 파일 패턴 수집
        files = list(sku_dir.glob(f"{sku}_clauses_clustered_*.xlsx"))
        if not files:
            continue

        if strategy == "all":  # NEW: 모든 파일 사용
            clause_files.extend(files)
        elif strategy == "latest":  # NEW: 최신 파일만 선택
            clause_files.append(_select_latest_file(files))
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

    if not clause_files:
        logger.warning("No clause files found.")
        return None

    skus = {p.parent.name for p in clause_files}
    logger.info(
        "Found %d clustered clause workbooks across %d SKUs (strategy=%s); sample=%s",
        len(clause_files),
        len(skus),
        strategy,
        [p.name for p in clause_files[:5]],
    )

    frames = []
    for idx, path in enumerate(clause_files, 1):
        logger.info("Processing file %d/%d: %s", idx, len(clause_files), path)
        sku = path.parent.name
        try:
            df = pd.read_excel(path)
        except Exception as e:
            logger.exception("Failed to read %s", path)
            continue

        # sku 컬럼이 없으면 채운다
        if "sku" not in df.columns:
            df["sku"] = sku

        # polarity 정규화
        if "polarity" in df.columns:
            df["polarity"] = df["polarity"].astype(str).str.lower()

        frames.append(df)
        logger.info("Loaded %d clauses from %s", len(df), path.name)

    if not frames:
        return None

    logger.info("Loaded %d clause files", len(frames))
    return pd.concat(frames, ignore_index=True)


def compute_facet_bucket_stats(df):
    """
    카테고리/sku/facet_bucket 단위로:
      - n_clauses
      - n_reviews
      - share_of_sku (해당 sku negative 절 중 비율)
    을 계산해서 output/analysis/facet_bucket_stats.csv 로 저장합니다.
    """
    # 우선 negative만 본다
    if "polarity" in df.columns:
        neg = df[df["polarity"].isin(["negative", "neg"])].copy()
    else:
        neg = df.copy()

    required = ["facet_bucket", "clause", "review_id", "sku"]
    for col in required:
        if col not in neg.columns:
            logger.warning("required column '%s' missing; stats may be incomplete.", col)

    facet_bucket_col = "facet_bucket" if "facet_bucket" in neg.columns else None

    # category 없으면 임시로 unknown
    category_col = "category" if "category" in neg.columns else None
    if category_col is None:
        neg["category"] = "unknown"
        category_col = "category"

    group_keys = [category_col, "sku"]
    if facet_bucket_col is not None:
        group_keys.append(facet_bucket_col)

    # sku/category별 전체 negative 절 수
    total = (
        neg.groupby([category_col, "sku"])["clause"]
        .size()
        .reset_index(name="n_clauses_total")
    )

    # facet_bucket별 통계
    stats = (
        neg.groupby(group_keys)
        .agg(
            n_clauses=("clause", "size"),
            n_reviews=("review_id", "nunique"),
        )
        .reset_index()
    )

    stats = stats.merge(
        total,
        on=[category_col, "sku"],
        how="left",
    )
    stats["share_of_sku"] = stats["n_clauses"] / stats["n_clauses_total"]

    out_path = ANALYSIS_DIR / "facet_bucket_stats.csv"
    stats.to_csv(out_path, index=False)
    logger.info("Saved facet_bucket_stats → %s (shape=%s)", out_path, stats.shape)


def compute_facet_vs_bucket_cross(df):
    """
    category/sku/facet_top1/facet_bucket 조합별 절 수를 세서
    output/analysis/facet_vs_bucket_cross.csv 로 저장합니다.
    → semantic facet_top1은 freshness/size_quantity 등,
      facet_bucket은 freshness_negative/unmatched_negative 같은 값.
    """
    if "polarity" in df.columns:
        neg = df[df["polarity"].isin(["negative", "neg"])].copy()
    else:
        neg = df.copy()

    category_col = "category" if "category" in neg.columns else None
    if category_col is None:
        neg["category"] = "unknown"
        category_col = "category"

    for col in ["facet_top1", "facet_bucket"]:
        if col not in neg.columns:
            logger.warning("'%s' missing; cross-tab may be incomplete.", col)

    req = [category_col, "sku", "facet_top1", "facet_bucket"]
    for col in req:
        if col not in neg.columns:
            neg[col] = None

    cross = (
        neg.groupby(req)["clause"]
        .size()
        .reset_index(name="n_clauses")
    )

    out_path = ANALYSIS_DIR / "facet_vs_bucket_cross.csv"
    cross.to_csv(out_path, index=False)
    logger.info("Saved facet_vs_bucket_cross → %s (shape=%s)", out_path, cross.shape)


def compute_bucket_samples(df, samples_per_combo=50, random_state=42):
    """
    각 (category, sku, facet_top1, facet_bucket) 조합마다
    최대 samples_per_combo개씩 절을 샘플링해서
    output/analysis/bucket_example_samples.csv 로 저장합니다.

    → 여기 들어있는 문장들을 가지고 YAML facet 키워드를 튜닝할 수 있습니다.
    """
    if "polarity" in df.columns:
        neg = df[df["polarity"].isin(["negative", "neg"])].copy()
    else:
        neg = df.copy()

    category_col = "category" if "category" in neg.columns else None
    if category_col is None:
        neg["category"] = "unknown"
        category_col = "category"

    for col in ["facet_top1", "facet_bucket"]:
        if col not in neg.columns:
            neg[col] = None

    groups = neg.groupby(
        [category_col, "sku", "facet_top1", "facet_bucket"],
        dropna=False,
    )

    sampled_frames = []
    for keys, g in groups:
        if len(g) == 0:
            continue
        n = min(samples_per_combo, len(g))
        sample = g.sample(n=n, random_state=random_state)
        sampled_frames.append(
            sample[
                [category_col, "sku", "facet_top1", "facet_bucket",
                 "polarity", "review_id", "clause"]
            ]
        )

    if not sampled_frames:
        logger.warning("No samples to save")
        return

    samples_df = pd.concat(sampled_frames, ignore_index=True)
    out_path = ANALYSIS_DIR / "bucket_example_samples.csv"
    samples_df.to_csv(out_path, index=False)
    logger.info(
        "Saved bucket_example_samples → %s (shape=%s)", out_path, samples_df.shape
    )


def main():  # CHANGED: argparse 적용 및 strategy 전달
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--strategy",
        choices=["latest", "all"],
        default="latest",
        help="clustered 파일 로딩 전략 (default: latest)",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    t0 = time.perf_counter()
    logger.info("Started facet quality analysis with strategy=%s", args.strategy)

    df = load_all_clauses(strategy=args.strategy)
    if df is None:
        logger.warning("No data loaded.")
        return

    compute_facet_bucket_stats(df)
    compute_facet_vs_bucket_cross(df)
    compute_bucket_samples(df, samples_per_combo=50)

    elapsed = time.perf_counter() - t0
    logger.info("Finished facet quality analysis in %.1f seconds", elapsed)


if __name__ == "__main__":
    main()
