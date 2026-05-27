# HW2 파이프라인 구현 노트

본 문서는 `hw2_pipeline.ipynb`의 설계 결정과 구현 디테일을 정리한 참고용 문서이다.

---

## 1. 확정된 결정 사항

| 항목 | 결정 |
|---|---|
| OpenAI 모델 (챗봇 본체·judge·baseline·simulator) | `gpt-5.4-nano` — 발급된 API 키로 호출 가능한 모델 중 가장 저렴한 것 |
| 종료 턴 수 | 4턴 고정 |
| Confidence 산출 | 옵션 1 (LLM 자가 보고). 옵션 2(logprobs)는 Future Work |
| Prompt 구조 | ipynb 안에 `PROMPTS` 딕셔너리로 모아둠 |
| API 키 입력 | **Universal loader** — Colab Secrets 또는 로컬 `.env` 자동 감지 (셀 4 `try/except`) |
| 검색 결과 개수 | API당 5개 (명세 부록 B 권장 범위) |
| News 백엔드 | **DuckDuckGo News** (`ddgs` 패키지) — NewsAPI 무료 한도(100/24h) 부족으로 교체 |
| 평가 자동화 | 셀 43·51이 `PERSONAS_TO_RUN` 리스트 순회 → 페르소나별 `eval_outputs/{persona}/` 결과 |

---

## 2. 외부 API 사용 메모

### Wikipedia
- 키 불필요. `requests` + 명시적 User-Agent로 MediaWiki REST API 직접 호출.
- `wikipedia` 파이썬 패키지는 User-Agent를 보내지 않아 차단되므로 사용하지 않는다.
- Rate limit 대비: 매 호출 사이 `WIKI_SLEEP=1.0`초 sleep, 429 응답 시 `WIKI_RETRY_AFTER=2.0`초 대기 후 1회 재시도.
- 영문 위키 고정. 한국어 페이지는 검색 품질이 떨어진다.
- 사용 엔드포인트:
  - `https://en.wikipedia.org/w/api.php` (검색)
  - `https://en.wikipedia.org/api/rest_v1/page/summary/` (페이지 요약)

### News (DuckDuckGo News)
- **변경 이력**: 최초 명세는 NewsAPI(newsapi.org)였으나 무료 플랜 100 req/24h가 4페르소나 평가에 부족 → DuckDuckGo News로 교체 (2026-05-27).
- 키 불필요. `ddgs` 파이썬 패키지 사용 (`pip install ddgs`). 사실상 무제한 호출.
- 함수: `search_news()` (cell 20). source_type = `"news"`, 라우팅 키 = `"news"`.
- 반환 필드: `date`, `title`, `body`(snippet), `url`, `image`, `source`. NewsAPI보다 snippet은 짧으나 (~150~200자) 검색 품질·다국적 커버리지 양호.
- **LLM-rewrite fallback**: 1단계가 만든 학술/forensic 쿼리가 뉴스에서 0건이면 LLM에게 짧은 뉴스 친화 쿼리로 재작성 요청 후 1회 재시도. `PROMPTS["rewrite_news_query"]` 사용. ArXiv/Wikipedia에는 영향 없음.

### ArXiv
- 키 불필요. `arxiv` 파이썬 패키지 사용.
- 학술 논문 초록 검색. 과학·기술 관련 주장에서 활용.
- 결과 5개, `SortCriterion.Relevance` 정렬.
- **Rate limit 완화 (2026-05-27)**: `arxiv.Client(delay_seconds=3.5, num_retries=5)` — 공식 가이드라인(3초/회)+마진. 일시 429/503은 자동 재시도, 그래도 실패하면 graceful 빈 결과.

### 검색 쿼리 언어
- 모든 검색 쿼리는 **영어**. 한국어 명제도 1단계 쿼리 추출기가 영어로 변환.
- 이유: News/ArXiv는 영어 매체·논문이 압도적. Wikipedia도 영문판이 정보량 많음.
- 한국어 입력 → 영어 검색 → 한국어 응답 흐름은 자연스럽게 동작.

---

## 3. `respond()` 함수 구조

명세 §2의 시그니처 그대로:

```python
def respond(user_message: str, history: list[dict]) -> dict:
    # 반환 키: reply, sources, classification, evidence_strength, evaluator_state
```

### 첫 턴 vs 이후 턴 분기
- `history`가 비어있으면 첫 턴 → 0단계(분류기) 실행.
- `history`가 있으면 이후 턴 → 첫 턴의 classification을 `history[0]["classification"]`에서 그대로 사용.

### `established_fact`는 즉시 종료
- 첫 턴에 분류가 `established_fact`로 나오면, 1단계는 건너뛰고(원문 그대로 쿼리), 2~4단계만 실행, 5단계는 생략.
- 반환 시 `evaluator_state="close"`, `evidence_strength="strong"`(placeholder).

### 종료 조건 (5단계)
1. 턴 수가 4에 도달 (명세 부록 B에 따라 4턴 고정).
2. 사용자가 명시적 종료 신호 ("그만", "stop", "end", "끝", "알겠어" 등) — LLM이 판정.

---

## 4. LLM 호출 횟수 (명세 §4.1)

| 턴 종류 | 호출 단계 | 횟수 |
|---|---|---|
| 첫 턴 (`established_fact`) | 0, 2, 4 | 3회 |
| 첫 턴 (그 외) | 0, 1, 2, 4, 5 | 5회 |
| 이후 턴 | 1, 2, 4, 5 | 4회 |

`LLMClient.call_count`로 누적 카운트, `usage` 필드로 토큰 누적.
`COST_WARN_THRESHOLD` 도달 시 경고 (현재 9999.0 USD로 설정 — 사실상 비활성).

---

## 5. 5개 응답 모드 (4단계 분기표)

명세 §3.5 표 그대로 구현. `RESPONSE_MODES` 딕셔너리에 각 모드의 system prompt를 둔다.

| 분류 | evidence_strength (이전 턴) | 모드 |
|---|---|---|
| `established_fact` | (해당없음) | `agree` |
| `clearly_false` | (무관) | `strong_rebut` |
| `debatable` | `strong` | `partial_concede` |
| `debatable` | `medium` | `balanced_rebut` |
| `debatable` | `weak` | `strong_rebut` |
| `subjective_opinion` | (무관) | `opinion_rebut` |
| 논점 일탈 시 | (무관) | `redirect` |

분기 로직은 `_select_response_mode(classification, evidence_strength, off_topic)` 함수에 격리.

### 첫 턴의 evidence_strength
- 명세 §3.6: "첫 턴에서는 사용자 주장 자체의 근거 강도를 평가" — 그러나 첫 턴 4단계는 evidence_strength를 입력으로 받지 않음 (분류만 사용).
- 구현: 첫 턴 4단계는 분류만으로 모드 결정 (`debatable`이면 무조건 `balanced_rebut`로 시작). 이후 5단계가 evidence_strength를 평가하고, 다음 턴 4단계가 그걸 사용.

---

## 6. 안전 (Safety, 명세 §6)

- **혐오 주장 거부**: 0단계 프롬프트에 "혐오/차별/폭력 선동 주장은 `hate_speech` 라벨을 출력하라"는 별도 분기. `respond()`가 이 라벨을 받으면 토론 거부 메시지로 즉시 종료.
- **인신공격 금지**: 4단계 모든 응답 모드의 system prompt에 "인신공격, 모욕적 표현 금지" 명시.
- **출처 신뢰도**: 4단계가 인용 시 도메인 표기 (예: `[BBC.com]`, `[Wikipedia]`).
- **검색 결과 없음 처리**: 3단계가 빈 결과를 반환하면 4단계가 "현재 신뢰할 만한 근거를 찾지 못했다"는 응답으로 처리.

---

## 7. User Simulator (명세 §5.3.1)

`simulate_user_turn(claim, history, persona, max_turns)` 함수로 구현.

페르소나 4종 모두 지원 (`believer`, `moderate`, `stubborn`, `evidence_focused`).

평가 실행 루프 `run_evaluation(claim_set, persona, max_turns)`도 같이 구현.

### 7.1 첫 발화 책임 분리
- **첫 턴(원 주장)은 `run_evaluation` 루프가 `claim_item["claim"]`을 그대로 사용자 발화로 사용**한다. simulator를 호출하지 않는다.
- **두 번째 턴부터 `simulate_user_turn`이 후속 발화를 생성**한다. 빈 history로 호출하면 `ValueError`를 던져 잘못된 사용을 차단.
- 이유: 명제 원문은 평가 명제 셋에 정답 라벨과 함께 사전 정의된 정확한 텍스트다. simulator가 이를 재생성하면 미세한 의역으로 분류기 입력이 흔들려 평가 일관성이 떨어진다.

### 7.2 챗봇/시뮬레이터 분리
- 챗봇 본체와 simulator는 별도 `LLMClient` 인스턴스(`llm` vs `simulator_llm`)를 사용한다.
- 두 인스턴스 모두 `gpt-5.4-nano`를 쓰지만, 호출 카운트·토큰·비용이 분리 누적되므로 챗봇 단독 비용을 명확히 측정 가능.

---

## 8. 평가 명제 셋 (`EVAL_CLAIMS`)

명제 20개, 명세 §5.1 배분 유지 (established_fact 2 / clearly_false 6 / debatable 10 / subjective_opinion 2).

각 명제에 `expected_apis` 메타 추가 — 라우터의 API 선택이 의도된 분포와 부합하는지 비교용. **2026-05-27 변경**: `expected_apis` 값 `"newsapi"` → `"news"` 갱신 (DDG News 교체에 따른 라우팅 키 변경).

3개 외부 API 호출이 골고루 일어나도록 명제 표현을 조정:
- ArXiv 적합 명제는 "최근 연구에 따르면", "벤치마크 상에서", "메타분석에 따르면", "실증 연구에 따르면" 등의 학술 신호 포함.
- News 적합 명제는 "최근 ~한 사실"에 대한 해석/평가 형태.
- Wikipedia 적합 명제는 백과사전적 통념·검증된 과학.

---

## 9. 실행 흐름 (Colab / 로컬 공통)

### 9.1 Universal 환경 지원
셀 4는 Colab Secrets와 로컬 `.env` 둘 다 자동 감지 (`try/except`로 분기). 별도 코드 수정 없이 두 환경에서 동작.

### 9.2 패키지 설치
```bash
pip install 'openai==1.*' arxiv requests ddgs
```
(`wikipedia` 패키지는 User-Agent 미설정 문제로 사용하지 않고 직접 HTTP)

### 9.3 키 입력
- **로컬**: `.env` 파일에 `OPENAI_API_KEY=sk-...` 한 줄. (`.env.example` 참고. `.gitignore`에 `.env` 포함.) News는 키 불필요.
- **Colab**: Colab Secrets에 `OPENAI_API_KEY` 등록 → 셀 4가 자동 감지.

### 9.4 셀 실행 순서
1. **셋업** (셀 0~30): 패키지·키·전역설정·LLMClient·Prompts·API점검·0~5단계 함수·EVAL_CLAIMS.
2. **데모(선택, 셀 32~39)**: 단일/멀티턴/시뮬레이터 데모. 비디오 시연 용.
3. **평가**: 셀 41~42 (Dry-run 토글 → 전체/부분) → 셀 43 (4페르소나 자동 eval) → 셀 45 (judge prompts) → 셀 47 (baseline) → 셀 49 (metric 함수) → 셀 51 (페르소나별 metrics 산출).
4. **결과**: `eval_outputs/{persona}/metrics_summary.txt` 4개 생성. 셀 51 끝에 cross-persona 요약 출력.

### 9.5 부분 실행
- 1페르소나 평가: 셀 43에서 `PERSONAS_TO_RUN = ["believer"]`.
- 2명제 dry-run: 셀 42에서 `DRY_RUN_N = 2`.

---

## 10. 주의 사항 / 알려진 한계

- **ArXiv 일시 429/503**: 공식 가이드라인 준수(`delay_seconds=3.5, num_retries=5`)에도 일부 쿼리가 거부됨. 빈 결과 graceful 처리되어 다른 API(Wiki/News)로 보완. 평가 점수에 약간의 비대칭 노이즈 (±0.1~0.2 추정).
- **ArXiv는 STEM 편향**: 인문·사회 주장에는 결과 부실. 라우터 프롬프트에 명시.
- **DDG News 검색 매칭 실패**: 1단계가 만든 학술 phrase가 News에 매칭 안 되는 경우 → `search_news()` 내부 LLM rewrite fallback. rewrite 후에도 0건이면 빈 결과 (graceful).
- **DDG `ddgs` 패키지의 unofficial 성격**: DuckDuckGo가 가끔 봇 차단 trigger 가능. 평가 중 발생 시 잠시 후 재시도. 영구 차단 사례는 본 평가에서 미확인.
- **한국어 입력 처리**: 챗봇 응답은 입력 언어를 따라간다. 검색 쿼리만 영어.
- **logprobs 미사용**: confidence는 LLM 자가 보고 한정. Future Work.
- **Wikipedia rate limit**: 짧은 시간에 많은 쿼리를 보내면 429 발생. sleep + 재시도로 완화하지만, 한 명제당 쿼리 수가 늘면 다시 발생 가능.
- **모델 권한**: 발급된 키는 `gpt-5.4-nano`, `gpt-5.4-mini`만 호출 가능. 다른 모델 사용 시 셀 6의 `MODEL`/`JUDGE_MODEL`/`SIMULATOR_MODEL` 변경.

---

## 11. 평가 셀 구조 (셀 41~51, 2026-05-27 추가)

명세 §9.2의 5종 지표 산출을 노트북 한 파일로 끝낼 수 있도록 셀들을 모아둠.

| 셀 | 역할 | 비고 |
|---|---|---|
| 41 (md) | Dry-run 토글 설명 | |
| 42 (code) | `DRY_RUN_N` 변수 → EVAL_CLAIMS를 N개로 축소 | `None`이면 전체 |
| 43 (code) | **4페르소나 자동 eval 루프** | `PERSONAS_TO_RUN` 리스트 순회. 페르소나별 `eval_outputs/{persona}/log_NN.txt`, `result_NN.txt` 저장. 결과는 `persona_to_results` dict에 보관 (셀 51이 사용). |
| 45 (code) | `JUDGE_PROMPTS` 4종 (persuasion / soundness / persuasiveness / winrate) | 일회성 dict 로딩 |
| 47 (code) | `baseline_llm` + `baseline_response()` + 20개 baseline 응답 생성 | 페르소나 무관. `baseline_replies` dict로 캐시. |
| 49 (code) | 5종 metric 함수 정의 | `judge_llm` 별도 인스턴스. `metric_classification_accuracy/persuasion/soundness/persuasiveness/winrate`. |
| 51 (code) | **페르소나별 metrics 계산 루프** | `PERSONAS_TO_RUN`을 다시 순회. 각 페르소나의 `all_results` (셀 43이 만든 dict에서 꺼냄)로 metrics 계산 후 `eval_outputs/{persona}/metrics_summary.txt` 저장. 끝에 cross-persona 요약 출력. |

### 11.1 `persona_to_results` 인터페이스
셀 43이 `persona_to_results[PERSONA] = all_results` 로 페르소나별 결과 누적. 셀 51이 같은 dict에서 `all_results = persona_to_results[PERSONA]` 로 꺼냄. 셀 43 끝에서 `all_results = persona_to_results[PERSONAS_TO_RUN[-1]]` 로 alias 두어 단일 페르소나 모드의 backward compat 유지.

### 11.2 비용 추적
- 셀 43은 페르소나별로 `llm.cost_usd()`·`simulator_llm.cost_usd()` 델타 출력.
- 셀 51은 페르소나별 metrics_summary.txt 끝에 `judge_llm` 페르소나별 델타 + `baseline_llm` 누적값 기록.
- 4페르소나 전체 실행 시 총 비용 ≈ $3.0 (gpt-5.4-nano 기준).

### 11.3 산출 디렉터리 구조
```
eval_outputs/
└── {persona}/                       # persona ∈ PERSONAS_TO_RUN
    ├── log_01.txt ~ log_20.txt      # 대화 로그 (사람이 읽기 좋은 형식)
    ├── result_01.txt ~ result_20.txt # 메타 (분류·근거강도·sources·API match 등)
    ├── metrics_summary.txt           # AGGREGATE 5종 + per-claim 표 + confusion
    └── metrics_detail_01.txt ~ _20.txt  # judge raw output + rationale 포함
```

페르소나별 디렉터리 분리는 cross-persona 비교를 깔끔하게 함과 동시에, 단일 페르소나 dry-run 시에도 `eval_outputs/believer/` 등 일관된 위치에 산출.
