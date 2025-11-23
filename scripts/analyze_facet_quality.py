from pathlib import Path
import pandas as pd

# 분석 결과를 모아둘 폴더: output/analysis
ANALYSIS_DIR = Path("output") / "analysis"
ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)


def load_all_clauses():
    """
    output/* 폴더 아래의 *_clauses_clustered_*.xlsx 를 전부 읽어서
    하나의 DataFrame으로 합칩니다.
    """
    base = Path("output")
    clause_files = sorted(base.glob("*/*_clauses_clustered_*.xlsx"))
    if not clause_files:
        print("No clause files found under output/*/*_clauses_clustered_*.xlsx")
        return None

    frames = []
    for path in clause_files:
        sku = path.parent.name  # sweet_potato, abalone 등
        try:
            df = pd.read_excel(path)
        except Exception as e:
            print(f"Failed to read {path}: {e}")
            continue

        # sku 컬럼이 없으면 파일명에서 채운다
        if "sku" not in df.columns:
            df["sku"] = sku

        # polarity를 소문자 문자열로 정규화
        if "polarity" in df.columns:
            df["polarity"] = df["polarity"].astype(str).str.lower()

        frames.append(df)

    if not frames:
        return None

    all_df = pd.concat(frames, ignore_index=True)
    return all_df


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
            print(f"[WARN] required column '{col}' missing; stats may be incomplete.")

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
    print(f"saved {out_path}")


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
            print(f"[WARN] '{col}' missing; cross-tab may be incomplete.")

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
    print(f"saved {out_path}")


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
        print("no samples to save")
        return

    samples_df = pd.concat(sampled_frames, ignore_index=True)
    out_path = ANALYSIS_DIR / "bucket_example_samples.csv"
    samples_df.to_csv(out_path, index=False)
    print(f"saved {out_path}")


def main():
    df = load_all_clauses()
    if df is None:
        print("No data loaded.")
        return

    compute_facet_bucket_stats(df)
    compute_facet_vs_bucket_cross(df)
    compute_bucket_samples(df, samples_per_combo=50)


if __name__ == "__main__":
    main()
