# 합성 입력 검증 기록

검증일: 2026-10-04. 기준 Git 커밋: `08a3752898025f4d4a0747af5259ec1dc016c556`. 코드는 Git의 최신 main에서 새 작업 브랜치로 가져왔다. Mac에 동기화되어 있던 참고 ZIP은 수정하지 않았으며 원본 코드의 기준으로 사용하지 않았다.

## 수정 전 재현

`tools/reproduce_legacy_bugs.py`는 `git show`로 **수정 전 실제 소스**를 읽어 캐시 클래스·극성 변환식·집계식·보고서 함수·영구 ID 함수를 실행한다. 무거운 모델을 로드하는 진입점은 실행하지 않는다. 입력은 스크립트에 새로 작성한 합성 문장뿐이다. 재현 로그: [before.txt](../validation/before.txt).

| 결함 | 수정 전 관찰 | 수정 후 검증 |
| --- | --- | --- |
| 극성별 절 ID 재발급 + ID 전용 캐시 | 서로 다른 긍정/부정 문장이 같은 벡터를 받음 | 전체 절 ID 먼저 부여; 문장 digest와 모델 지문을 함께 확인 |
| 캐시 many-to-many 병합 | 같은 ID 입력 2개에서 벡터 4개 반환 | 중복 키 한 번 계산, 입력 행 수·순서 보존 |
| 극성 응답 형식 | 스칼라 `Negative`가 `n`으로 잘려 중립화 | 스칼라/단일 리스트·별칭 정규화, 미정의 숫자 매핑은 명시적 오류 |
| 노이즈 군집 집계 | 군집 하나와 `other`가 군집 둘로 집계 | raw/표시 노이즈 처리 분리, 유효 군집과 노이즈 따로 집계 |
| 보고서 노이즈 합산 | 999번 노이즈가 대표 군집 행에 포함 | 동일 노이즈 판별 함수를 보고서·화면·대표 문장에 적용 |
| 영구 ID 극성 충돌 | 같은 대표 문장의 부정/긍정 군집 ID가 같음 | 극성을 서명에 포함 |
| 영구 ID 순환 충돌 | 999개 군집에 고유 ID 998개만 배정 | 재사용하지 않는 배정; 소진 시 상태 파일을 변경하지 않고 실패 |

이미 가지고 있는 원본 Git 복제본에서 재현하려면 다음처럼 경로를 명시한다. 스크립트는 저장소나 원본 데이터를 다운로드하지 않는다.

```bash
.venv/bin/python tools/reproduce_legacy_bugs.py --legacy-repo /path/to/wcamp_nlp
```

## 수정 후 검증

`bash run_demo.sh`는 `.venv` 생성과 `requirements-demo-lock.txt` 설치부터 실행한다. Python 3.11의 macOS arm64 환경에서 **86개 테스트가 통과**했다. 핵심 회귀 28개, refinement 13개, 영구 ID 5개, 데모·보고서 경계 사례 40개를 포함한다. 실행 결과는 [after.txt](../validation/after.txt), 설치 버전은 [environment.txt](../validation/environment.txt)에 기록한다. 공개 로그에서는 작업 디렉터리의 개인 경로를 `<REPO>`로 치환했다.

테스트는 캐시의 입력 순서·중복·문장 변경·모델 변경·warm 재실행, 빈 입력·전부 긍정·전부 노이즈·단일 문자, ABSA 응답 개수 및 `001`/`NA`/`NULL` ID 보존, 노이즈 집계, 대표 문장 소속, HTML 이스케이프를 확인한다. 별도 프로세스에서 소켓 연결과 `torch`, `openai`, `sentence_transformers`, `pyabsa`, `kss` import를 차단한 상태로 전체 합성 CLI가 성공하는지도 확인한다. 의존성 설치 단계의 인터넷 연결까지 차단한 검증은 아니다.

합성 예제는 실제 수정한 `split_clauses`, `assign_clause_ids`, `normalize_polarity`, `CachingEmbedder`, `cluster_counts`, `extract_representatives`를 호출한다. 원본 모델 대신 명시된 오프라인 분류 규칙과 문자 TF-IDF/DBSCAN을 사용한다. 주요 결과는 리뷰 16개, 절 31개, 긍정 16·중립 2·부정 13, 부정 군집 3개(각 4절), 노이즈 1개다. 점수에 대한 정확도·일반화 성능 주장은 하지 않는다.

## 후속 오류 재현과 수정

이전 수정 커밋 `5ffc89d32ca2f93a801d03c2f0e51889fcffde06`에 남아 있던 refinement 오류 5개를 실제 Git 함수와 합성 벡터로 재현했다. [재현 로그](../validation/refinement-before.txt)는 원군집 1의 하위 군집과 원군집 10의 충돌, 부정 군집 100의 중립 namespace 침범, 생략된 긍정 prefix, 비연속 인덱스의 행 위치 오류, refined ID와 대표 문장 불일치를 기록한다.

부모 ID를 먼저 예약하고 남은 `0..998` 번호에서 하위 군집을 배정한다. row index 대신 행 위치로 벡터를 선택하며, 명시적 prefix와 극성을 검증한다. 용량/namespace 오류는 조용한 fallback으로 숨기지 않고 실패한다. 일반 refinement 실패의 기존 fallback은 유지한다.

`main._finalize_polarity_result`는 최종 `cluster_label`과 `refined_cluster_id`를 일치시키고, 해당 군집에서 대표 문장과 키워드를 다시 계산한다. 기존 parent는 `original_cluster_label`, 적용 여부는 `refinement_applied`로 남긴다. 실제 refine → finalize → 보고서/JSON → 영구 ID 흐름을 합성 벡터·주입된 tokenizer/분할 결정으로 검증했다. 성공/실패가 섞인 극성도 결측 refined ID를 만들지 않으며, 부모 번호가 바뀌어도 같은 대표 문장의 영구 ID가 유지된다.

추가 6개 경계 오류의 [전후 관찰](../validation/edge-cases.txt): 긴 절과 섞인 한 글자 부정 절의 0벡터, 반복 구두점의 가짜 절, 실수형 ID의 대표 문장 누락, null facet의 화면 중단, 기존 HTML의 이스케이프 누락, 노이즈가 상위 20개 군집 중 한 자리를 차지하는 오류를 수정했다. 기본 16개 합성 입력과 샘플 결과는 바꾸지 않았다.

```bash
# 수정 전 refinement 재현: 원본 Git 복제본을 명시
.venv/bin/python tools/reproduce_refinement_bugs.py --legacy-repo /path/to/wcamp_nlp
# 이전 코드의 경계 동작: 지정한 Python 파일만 임시 폴더에 가져오며 데이터는 가져오지 않음
.venv/bin/python tools/check_demo_edges.py --repo /path/to/wcamp_nlp --revision 5ffc89d32ca2f93a801d03c2f0e51889fcffde06
# 현재 코드의 경계 동작
.venv/bin/python tools/check_demo_edges.py
```

## 인터넷 중단 이후 무결성 확인

원격 작업 브랜치의 커밋과 로컬 HEAD가 일치함을 Git 프로토콜과 GitHub 연결 양쪽에서 확인했다. 기존 main은 바뀌지 않았다. 두 Git 저장소의 `fsck`, 독립 번들의 `bundle verify`, 두 가상환경의 `pip check`가 성공했다. 인터넷 중단에 따른 누락이나 저장소 손상은 발견하지 못했다. 이후 변경도 새 커밋의 tree SHA와 원격 ref를 비교해 확인한다.

## 남은 한계

1. 원본 모델 다운로드·외부 API·CUDA·UMAP/HDBSCAN 전체 통합, 대규모 성능·메모리·시간, 과거 결과와의 비교, Windows 실행은 검증하지 않았다. 선택적 불용어 파일이 없어 기본 목록을 쓴다는 경고는 남겨 두었다.
2. 영구 ID는 대표 문장 기반 휴리스틱이다. 서로 다른 같은 극성 군집이 동일 대표 문장 서명을 가지거나 극성당 999개 용량이 소진되면 상태를 저장하지 않고 명시적으로 실패한다. 읽을 수 있지만 내부 벡터가 손상된 캐시도 명시적으로 실패하며 자동 복구를 보증하지 않는다.
3. 규칙 기반 분류는 일반적인 반어·암묵적 불만·모든 부정을 지원하지 않는다. 이 사례들은 학습 모델의 정확도 평가가 아니다.
4. 재연결 뒤에도 브라우저 도구의 관리자 정책 확인이 실패해 화면 캡처와 시각 검사를 수행하지 못했다. HTML 생성·구조·내용·이스케이프 자동 테스트는 통과했다.

이번 실행은 **전체 76.6만 건 분석의 재검증이 아니다.** 원본 리뷰를 읽어 얻은 분석 지표를 합성 결과로 표현하지 않았다. Git 공개 적합성 검사의 원문 구조·패턴 집계는 별도의 감사 기록이다.
