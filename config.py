from pathlib import Path

# ───────────────────────────────────────────────────────────────────────────
# 0. Base paths
# ───────────────────────────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).resolve().parent
DATA_DIR   = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

INPUT_FILES = [
    DATA_DIR / "cucumber_market.xlsx",
    DATA_DIR / "watermelon_market.xlsx",
    DATA_DIR / "koreamelon_market.xlsx",
    DATA_DIR / "abalone.xlsx",
]

# ───────────────────────────────────────────────────────────────────────────
# 1. Loader / Preprocess
# ───────────────────────────────────────────────────────────────────────────
REQUIRED_COLUMNS = ["review"]
KEEP_META        = True

# 리뷰 고유 ID 컬럼 (reset_index() 통해 부여)
REVIEW_ID_COL    = "review_id"

# ───────────────────────────────────────────────────────────────────────────
# 2. Embedding
# ───────────────────────────────────────────────────────────────────────────
MODEL_NAME = "jhgan/ko-sbert-sts"
BATCH_SIZE = 128
DEVICE     = "cuda"

# ───────────────────────────────────────────────────────────────────────────
# 3. Clause splitting (절 분할)
# ───────────────────────────────────────────────────────────────────────────
CLAUSE_CONNECTIVES = [
    "지만", "하지만", "그러나", "다만",
    "긴 한데", "그런데", "뿐만 아니라",
    # 필요 시 추가…
]

# ───────────────────────────────────────────────────────────────────────────
# 4. ABSA 설정
# ───────────────────────────────────────────────────────────────────────────
# PyABSA BERT-SPC 멀티링궐 모델 (긍정/중립/부정)
ABSA_MODEL_NAME = "multilingual"
ABSA_BATCH_SIZE  = 32
# device: 기존 DEVICE 사용

# ───────────────────────────────────────────────────────────────────────────
# 5. UMAP
# ───────────────────────────────────────────────────────────────────────────
UMAP_N_NEIGHBORS   = 50
UMAP_MIN_DIST      = 0.05
UMAP_METRIC        = "cosine"
UMAP_RANDOM_STATE  = 42
UMAP_DIMS_CLUSTER  = 10

# ───────────────────────────────────────────────────────────────────────────
# 6. HDBSCAN
# ───────────────────────────────────────────────────────────────────────────
HDBSCAN_MIN_CLUSTER_SIZE = 400
HDBSCAN_MIN_SAMPLES      = 320
HDBSCAN_SELECTION_EPS    = 0.05
HDBSCAN_METRIC           = "euclidean"

# ───────────────────────────────────────────────────────────────────────────
# 7. Cluster merge
# ───────────────────────────────────────────────────────────────────────────
ENABLE_CLUSTER_MERGE     = True
CLUSTER_MERGE_THRESHOLD  = 0.90
MERGE_BATCH_SIZE         = 64

# ───────────────────────────────────────────────────────────────────────────
# 8. Sentiment analysis (절 필터링용 ABSA)
# ───────────────────────────────────────────────────────────────────────────
ENABLE_SENTIMENT_ANALYSIS = False  # 절 분할 후 ABSA 강제 적용 시 True로 변경

# ───────────────────────────────────────────────────────────────────────────
# 9. Keyword / naming
# ───────────────────────────────────────────────────────────────────────────
CLUSTER_NAME_TOPK = 3             # 클러스터 이름용 키워드 개수
KEYWORD_MAX_SENT  = 50            # 키워드 추출시 샘플 문장 수
KEYWORD_NGRAM_RANGE = (1, 1)
TOKEN_PATTERN = r"(?u)\b[가-힣]{2,}\b"
KOREAN_STOPWORDS = [
    "은", "는", "이", "가", "을", "를", "에", "도",
    "너무", "정말", "그냥",
]
JOSA_EOMI_TAGS = {"JKS", "JKB", "JKC", "JKG", "JKV", "JKQ", "JKO", "JX", "JC", "EP", "EF", "EC"}
VALID_KEYWORD_RE = r"^[가-힣A-Za-z]{1,10}$"
MAX_KEYWORD_DOCS = 50            # 샘플링 문장 수 제한
KEYWORD_CANDIDATE_MULTIPLIER = 2   # KeyBERT 후보 배수
USE_KEYBERT = False              # True: KeyBERT, False: c-TF-IDF

# ───────────────────────────────────────────────────────────────────────────
# 10. Summarizer
# ───────────────────────────────────────────────────────────────────────────
TOP_K_REPRESENTATIVES = 3         # 대표 문장 개수

# ───────────────────────────────────────────────────────────────────────────
# 11. Outlier & ABSA confidence handling
# ───────────────────────────────────────────────────────────────────────────
HANDLE_OUTLIERS            = True
OUTLIER_LABEL              = "other"
ABSA_CONFIDENCE_THRESHOLD  = 0.6
