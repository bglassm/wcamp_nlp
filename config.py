from pathlib import Path
import os as _os

# ───────────────────────────────────────────────────────────────────────────
# 0. Base paths
# ───────────────────────────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).resolve().parent
DATA_DIR   = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

REVIEW_DATA_DIR     = DATA_DIR / "review"
COMMUNITY_DATA_DIR  = DATA_DIR / "community"
REVIEW_DATA_DIR.mkdir(parents=True, exist_ok=True)
COMMUNITY_DATA_DIR.mkdir(parents=True, exist_ok=True)

INPUT_FILES = sorted(
    p for p in REVIEW_DATA_DIR.glob("*.xlsx")
    if not p.name.startswith("~$")
)

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
# 한국어 특화 SBERT (사용자 확인값)
MODEL_NAME = "jhgan/ko-sbert-sts"
BATCH_SIZE = 128
DEVICE     = "cuda"

# ───────────────────────────────────────────────────────────────────────────
# 3. Clause splitting (절 분할)
# ───────────────────────────────────────────────────────────────────────────
CLAUSE_CONNECTIVES = [
    "그럼에도 불구하고", "에도 불구하고", "인데도 불구하고",
    "그렇긴 하지만", "그렇긴 한데", "기는 하지만", "긴 하지만", "긴 한데",
    "그렇지만", "하지만", "그러나", "다만", "반면에", "반면", "반대로", "오히려",

    "그러므로", "따라서", "그래서", "그러니까", "그러니",
    "때문에", "덕분에", "덕택에", "으로 인해", "으로 인해서", "로 인해",

    "뿐만 아니라", "그런데", "그리고", "또한", "게다가", "더불어", "나아가", "한편",
    "그러면서", "그러면서도",

    "만약", "만일", "만약에", "만일에",
    "이라면", "라면", "다면",
    "그렇더라도", "하더라도", "일지라도", "더라도", "는데도",

    "왜냐하면", "위해서", "위하여", "하려고",
]
# ───────────────────────────────────────────────────────────────────────────
# 4. ABSA 설정
# ───────────────────────────────────────────────────────────────────────────
# PyABSA BERT-SPC 멀티링궐 모델 (긍정/중립/부정)
ABSA_MODEL_NAME = "multilingual"
ABSA_BATCH_SIZE = 32
# device: 기존 DEVICE 사용

# ───────────────────────────────────────────────────────────────────────────
# 5. UMAP  (note: 실제 실행은 tuner.get_cluster_params() 결과를 우선 사용)
# ───────────────────────────────────────────────────────────────────────────
# 기존 25→15로 소폭 하향하여 지역 구조를 조금 더 드러내되, 과분할은 방지
UMAP_N_NEIGHBORS   = 15
UMAP_MIN_DIST      = 0.05
UMAP_METRIC        = "cosine"
UMAP_RANDOM_STATE  = 42
UMAP_DIMS_CLUSTER  = 10

# ───────────────────────────────────────────────────────────────────────────
# 6. HDBSCAN (fallback defaults; 보통은 tuner 결과 사용)
# ───────────────────────────────────────────────────────────────────────────
HDBSCAN_MIN_CLUSTER_SIZE = 240
HDBSCAN_MIN_SAMPLES      = 55
HDBSCAN_SELECTION_EPS    = 0.05
# UMAP 좌표(저차원) 기준으로는 보통 euclidean이 안정적
HDBSCAN_METRIC           = "euclidean"

# ───────────────────────────────────────────────────────────────────────────
# 7. Cluster merge
# ───────────────────────────────────────────────────────────────────────────
ENABLE_CLUSTER_MERGE     = True
CLUSTER_MERGE_THRESHOLD  = 0.93
MERGE_BATCH_SIZE         = 64

# ───────────────────────────────────────────────────────────────────────────
# 8. Sentiment analysis (절 필터링용 ABSA)
# ───────────────────────────────────────────────────────────────────────────
ENABLE_SENTIMENT_ANALYSIS = False  # 절 분할 후 ABSA 강제 적용 시 True로 변경

# ───────────────────────────────────────────────────────────────────────────
# 9. Keyword / naming
# ───────────────────────────────────────────────────────────────────────────
CLUSTER_NAME_TOPK = 3              # 클러스터 이름용 키워드 개수
KEYWORD_MAX_SENT  = 50             # 키워드 추출시 샘플 문장 수
KEYWORD_NGRAM_RANGE = (1, 1)
TOKEN_PATTERN = r"(?u)\b[가-힣]{2,}\b"
KOREAN_STOPWORDS = [
    "은", "는", "이", "가", "을", "를", "에", "도",
    "너무", "정말", "그냥",
]
JOSA_EOMI_TAGS = {"JKS", "JKB", "JKC", "JKG", "JKV", "JKQ", "JKO", "JX", "JC", "EP", "EF", "EC"}
VALID_KEYWORD_RE = r"^[가-힣A-Za-z]{1,10}$"
MAX_KEYWORD_DOCS = 50               # 샘플링 문장 수 제한
KEYWORD_CANDIDATE_MULTIPLIER = 2    # KeyBERT 후보 배수
USE_KEYBERT = False                 # True: KeyBERT, False: c-TF-IDF

# ───────────────────────────────────────────────────────────────────────────
# 10. Summarizer
# ───────────────────────────────────────────────────────────────────────────
TOP_K_REPRESENTATIVES = 3           # 대표 문장 개수

# ───────────────────────────────────────────────────────────────────────────
# 11. Outlier & ABSA confidence handling
# ───────────────────────────────────────────────────────────────────────────
HANDLE_OUTLIERS           = True
OUTLIER_LABEL             = "other"
ABSA_CONFIDENCE_THRESHOLD = 0.6

# ───────────────────────────────────────────────────────────────────────────
# 12. Refinement layer (domain-agnostic)
# ───────────────────────────────────────────────────────────────────────────
# rules/facets.yml, rules/thresholds.yml 사용. main.py에서 getattr로 안전 로드.
REFINEMENT_ENABLED            = True
REFINEMENT_FACETS_PATH        = "rules/facets.yml"
REFINEMENT_THRESHOLDS_PATH    = "rules/thresholds.yml"
# refined_cluster_id 네임스페이스(neg/neu/pos = 0/1/2)
REFINEMENT_STABLE_ID          = {"negative": 0, "neutral": 1, "positive": 2}

# ---- Smart clause split knobs ----
SMART_SPLIT_ENABLED = True                 # 끄고 싶으면 False
SMART_SPLIT_USE_EMBEDDING = True           # 임베딩 거리 사용 (SBERT lazy-load)
SMART_SPLIT_SIM_THRESHOLD = 0.22           # 분할 임계 (코사인 거리)
SMART_SPLIT_JACCARD_THRESHOLD_ADD = 0.35   # additive/turn/cause 쪽 Jaccard
SMART_SPLIT_JACCARD_THRESHOLD_CONTRAST = 0.40
SMART_SPLIT_MIN_CHUNK_LEN = 4              # 너무 짧은 쪼개기 방지
SMART_SPLIT_MAX_SPLITS_PER_SENT = 3        # 문장당 최대 분할 수


# Stable IDs / resume cache
ENABLE_STABLE_IDS = True

# ───────────────────────────────────────────────────────────────────────────
# Tuner(자동 파라미터) 기본 비율
# 작을수록 더 세분화(=클러스터 수 ↑), 클수록 더 합침(=클러스터 수 ↓)
TUNER_BASE_PCT_LARGE = 0.005   # N이 큰 셋 (대략 6천 이상)
TUNER_BASE_PCT_SMALL = 0.010   # N이 작은 셋

# UMAP 차원/이웃 수의 상·하한 (tuner가 동적으로 설정)
TUNER_UMAP_MIN_DIMS = 8
TUNER_UMAP_MAX_DIMS = 14
TUNER_UMAP_MAX_NEIGHBORS = 100

# ───────────────────────────────────────────────────────────────────────────
# 파셋 버킷(예: 당도/식감/외관…) 전용 코스닝(coarsening) 노브
#   → 버킷 내부에서 너무 잘게 쪼개지는 것을 방지하기 위한 완만한 합침 세팅
BUCKET_MIN_CLUSTER_SIZE_MULT = 1.35   # 기본 min_cluster_size에 곱해 키움
BUCKET_MIN_SAMPLES_MULT      = 1.15   # min_samples 배수
BUCKET_UMAP_NEIGHBORS_MULT   = 1.15   # 이웃 수↑ → 과분할 완화
BUCKET_UMAP_MIN_DIST         = 0.08   # 전역보다 살짝 크게(=붙여서 보되 과분할↓)
BUCKET_SELECTION_EPS_ADD     = 0.02   # HDBSCAN epsilon 가산 → 붙여주기


# ---------------------------------------------------------------------------
# Community-mode defaults (can be overridden by env vars / CLI)
# ---------------------------------------------------------------------------

# 필터 임계값
COMMUNITY_REL_TAU: float = float(_os.getenv("COMMUNITY_REL_TAU", "0.40"))
COMMUNITY_ALIAS_TAU: float = float(_os.getenv("COMMUNITY_ALIAS_TAU", "0.40"))
COMMUNITY_BAN_MODE: str = _os.getenv("COMMUNITY_BAN_MODE", "strict")  # soft|strict|off

# 요약/디버그
COMMUNITY_SUMMARY_MAX_SENTENCES: int = int(_os.getenv("COMMUNITY_SUMMARY_MAX_SENTENCES", "10"))
COMMUNITY_SAVE_FILTER_DEBUG: bool = _os.getenv("COMMUNITY_SAVE_FILTER_DEBUG", "0") in ("1","true","True")

# 룰 경로
COMMUNITY_RULES_PATH: str = _os.getenv("COMMUNITY_RULES_PATH", "rules/community_rules.yml")