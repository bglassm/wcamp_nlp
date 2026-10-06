# wcamp_nlp

한국어 상품 리뷰의 의견을 분석하고, 유사한 의견을 분류·군집화하는 파이프라인입니다. 절 단위 감성, 군집별 대표 문장·키워드, facet(의견 주제)을 정리해 Excel·JSON 및 시각화 파일로 저장합니다.

## 주요 모델·라이브러리

현재 `config.py`의 기본 설정 기준입니다.

| 모델·라이브러리 | 용도 |
|---|---|
| PyABSA `multilingual` | 절별 긍정·중립·부정 감성과 신뢰도 추론 |
| OpenAI `text-embedding-3-small` | 군집화에 사용할 절 임베딩 생성. 기본 backend는 `openai` |
| Sentence Transformers `jhgan/ko-sbert-sts` | 절 분할의 의미 유사도, 군집 병합, 대표 문장·키워드 선택 보조, facet 설명과 절의 유사도 계산 |
| KSS / Kiwi(`kiwipiepy`) | 한국어 문장 분리 / 키워드·facet 표현의 형태소 처리 |
| UMAP / HDBSCAN | 임베딩 차원 축소 / 밀도 기반 군집화와 노이즈 구분 |
| scikit-learn / KeyBERT | 기본 키워드 경로는 TF-IDF 기반 점수와 SBERT MMR. `USE_KEYBERT=True`이면 KeyBERT 사용. facet refinement의 하위 군집 분할에는 KMeans 사용 |
| pandas / NumPy / openpyxl / PyArrow | 표·벡터 처리, Excel 입출력, Parquet 임베딩 캐시 |
| PyTorch / Transformers / PyYAML / Matplotlib | 모델 실행, YAML 규칙 로딩, 결과 그래프 생성 |

## 처리 순서

1. **리뷰 로딩:** XLSX를 읽고 컬럼 이름을 정규화합니다. `review`가 필수이며 `comments`, `content`, `review_text`도 로더에서 `review`로 변환합니다. 리뷰 ID가 없으면 부여합니다.
2. **전처리·절 분할:** 텍스트를 정리하고 빈 리뷰를 제외합니다. 문장·연결 표현과 의미 유사도를 이용해 의견 절로 나눕니다.
3. **감성 분석:** PyABSA로 절의 극성과 신뢰도를 구합니다. 누락되거나 기준보다 신뢰도가 낮은 감성은 중립으로 보존하고, 극성별로 후속 처리를 수행합니다.
4. **임베딩:** 설정한 backend로 절을 벡터화하고 캐시합니다. `backend=local`이면 `embed.model`에 지정한 Sentence Transformers 모델을 사용합니다.
5. **차원 축소·군집화:** 절 수에 맞춰 파라미터를 조정하고 UMAP → HDBSCAN 순서로 유사 의견을 묶습니다.
6. **대표 문장·키워드·facet 정리:** 대표 문장을 선택하고 유사 군집을 병합한 뒤 키워드를 추출합니다. YAML의 facet 설명·카테고리별 키워드로 주제를 할당하고, 설정에 따라 하위 군집을 분할하며 영구 군집 ID를 부여합니다.
7. **결과 저장:** 절·리뷰 매핑, 집계 보고서, 대표 문장·키워드·facet 요약, 실행 로그와 시각화를 저장합니다.

## Windows 설정과 실행

Python 3.11과 CUDA를 지원하는 NVIDIA GPU·드라이버를 기준으로 합니다. 기본 설정은 PyTorch CUDA 12.1 빌드와 `device=cuda`를 사용합니다. OpenAI API 접근 및 SBERT·PyABSA 모델을 내려받을 네트워크가 필요합니다.

새 환경을 구성할 때 저장소 루트의 PowerShell에서 실행합니다. 기존 `.venv`가 있으면 해당 환경을 사용합니다.

```powershell
cd C:\projects\wcamp_nlp
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu121
.\.venv\Scripts\python.exe -m pip install openai python-dotenv pyarrow
```

`openai`, `python-dotenv`, `pyarrow`는 코드에서 사용하는 추가 의존성입니다. API 키는 환경 변수 `OPENAI_API_KEY` 또는 저장소 루트의 `.env`에 설정합니다.

입력은 `review` 컬럼이 있는 XLSX입니다. 기본 입력 탐색 위치는 `data/review`, `data/fruit`, `data/seafood`, `data/veggie`, `data/meat`의 직하위 파일입니다. 경로의 카테고리 이름으로 facet 설정을 선택하며, 일치하지 않으면 `generic`을 사용합니다.

```powershell
$env:PYTHONUTF8 = "1"
$env:OPENAI_API_KEY = "YOUR_API_KEY"
.\.venv\Scripts\python.exe main.py --files .\data\fruit\product.xlsx
```

`--files`를 생략하면 기본 탐색 파일을 처리합니다. `--resume`은 기존 ABSA·임베딩 캐시를 재사용합니다. 모델·backend·device는 `config.py`, facet 규칙은 `rules/facets.yml`, `rules/facets_by_category.yml`, `rules/thresholds.yml`에서 설정합니다.

## 결과 위치

- `output/YYYYMMDD/<입력 파일명>/`: 군집화 XLSX, 집계 보고서 XLSX, 요약 JSON, 군집 진단 CSV·PNG, HTML 대시보드, `_stable_ids.json`, `meta.json`.
- `output/YYYYMMDD/logs/`: 실행 로그. 같은 날짜 폴더의 `audit/`와 `run_manifest.json`에는 입력 처리 기록과 실행 설정을 저장합니다.
- `output/cache/`: ABSA 캐시. `output/cache/embeddings/`에는 Parquet 임베딩 캐시를 저장합니다.

`--output_dir`로 결과 루트를 바꿀 수 있습니다. 캐시 경로는 별도로 `config.OUTPUT_DIR`와 `config.embed.cache_dir` 설정을 따릅니다.
