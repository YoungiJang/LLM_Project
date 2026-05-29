# HW2: Adversarial Fact-Checking Chatbot

사용자 주장에 대해 외부 검색(RAG) 기반으로 실시간 반박을 수행하는 멀티턴 LLM 에이전트.
명세 전체는 [`HW2_pipeline_v2.md`](HW2_pipeline_v2.md), 설계 노트는
[`IMPLEMENTATION_NOTES_v2.md`](IMPLEMENTATION_NOTES_v2.md) 참고.

---

## 1. 사전 준비

### Python 패키지
```bash
pip install 'openai==1.*' arxiv requests ddgs
```

### API 키
**OpenAI 키만 필요**. (뉴스 검색은 DuckDuckGo 사용, 키 불필요)

1. [platform.openai.com](https://platform.openai.com)에서 키 발급
2. 프로젝트 루트에 `.env` 파일 생성 (`.env.example` 참고):
   ```
   OPENAI_API_KEY=sk-proj-...
   ```
3. `.env`는 `.gitignore`에 포함되어 있어 절대 커밋되지 않음.

> Colab 사용자는 Colab Secrets에 `OPENAI_API_KEY` 등록 시 자동 감지.

---

## 2. 실행 방법

### 권장 환경
Jupyter Notebook / Jupyter Lab / VS Code (`.ipynb` 지원). 노트북 `hw2_pipeline_v2_260526.ipynb` 한 파일로 셋업·시연·평가 모두 가능.

### 시나리오 A — 빠른 시연 (~3분, 비디오 데모용)

| 셀 | 역할 |
|---|---|
| 0–11 | 패키지 설치·키 로드·전역 설정·LLMClient·Prompts·API 점검 |
| 13–30 | 0~5단계 함수 정의·EVAL_CLAIMS 로드 |
| 32 | 단일 멀티턴 데모 (1개 명제) |
| 34 | 수동 멀티턴 데모 (Turn 1+2) |
| 36 | User Simulator로 자동 멀티턴 데모 (3개 명제) |
| 38, 39 | 비용·호출 요약 |

위 셀들만 순서대로 실행. 전체 ~3분.

### 시나리오 B — 4페르소나 전체 평가 (~2.5~3시간, 명세 §9.2)

| 셀 | 역할 |
|---|---|
| 0–30 | 셋업 + 함수 정의 + EVAL_CLAIMS |
| (31–39) | **건너뛰기** (데모 셀, 평가에 불필요) |
| 41–42 | Dry-run 토글: `DRY_RUN_N = None`이면 전체 20명제 |
| 43 | **4페르소나 자동 평가** — `PERSONAS_TO_RUN` 리스트 따라 순차 실행, `eval_outputs/{persona}/` 에 로그·메타 저장 |
| 45 | Judge 프롬프트 로드 |
| 47 | Baseline (grounding 없는 ChatGPT) 응답 20개 생성 |
| 49 | 5종 평가 지표 함수 정의 |
| 51 | **페르소나별 metrics 계산** → 각 `eval_outputs/{persona}/metrics_summary.txt` 저장 + cross-persona 비교 출력 |

기본값은 4페르소나 전체 (`PERSONAS_TO_RUN = ["believer", "moderate", "stubborn", "evidence_focused"]`).

### 시나리오 C — 1페르소나 평가 (~40분, 시간 제약 시)

위 시나리오 B와 동일하나 **셀 43에서 `PERSONAS_TO_RUN = ["believer"]`** 로 수정. 다른 페르소나로 바꿀 수도 있음.

### 시나리오 D — 미니 dry-run (~5분, 오류 탐지 용)

위 시나리오 B/C와 동일하나 **셀 42에서 `DRY_RUN_N = 2`** 로 수정. 처음 2개 명제만으로 파이프라인 검증.

---

## 3. 출력 산출물

```
eval_outputs/
├── believer/
│   ├── log_01.txt ~ log_20.txt              # 멀티턴 대화 로그 (사람이 읽기 좋음)
│   ├── result_01.txt ~ result_20.txt        # 메타 (분류·근거강도·sources·API 분포)
│   ├── metrics_summary.txt                  # AGGREGATE 5종 지표 + per-claim 표
│   └── metrics_detail_01.txt ~ _20.txt      # Judge rationale 포함 상세
├── moderate/    (동일 구조)
├── stubborn/    (동일 구조)
└── evidence_focused/    (동일 구조)
```

`metrics_summary.txt` 헤더의 AGGREGATE 섹션이 비디오에 인용할 핵심 결과:
- Classification Accuracy
- Persuasion Score (mean, 0~1)
- Logical Soundness (mean, 1~5)
- Persuasiveness (mean, 1~5)
- Debate Win-rate (vs grounding 없는 baseline ChatGPT)

---

## 4. 예상 시간·비용

| 옵션 | 소요 시간 | API 비용 |
|---|---|---|
| 시나리오 A (시연) | ~3분 | ~$0.05 |
| 시나리오 B (4페르소나) | ~2.5–3시간 | ~$3 |
| 시나리오 C (1페르소나) | ~40분 | ~$0.8 |
| 시나리오 D (dry-run 2명제) | ~5분 | ~$0.1 |

OpenAI `gpt-5.4-nano` 단가 추정 기준. `$20` 한도 안전.

---

## 5. 알려진 한계

- **외부 API 안정성**: ArXiv가 가끔 429/503 반환 (서버 부하). 코드가 5회 재시도 후 빈 결과로 graceful 처리. 평가 시 일부 명제에서 ArXiv 결과 누락 가능.
- **DDG News 검색 매칭**: 1단계가 생성한 학술/포렌식 쿼리가 뉴스에 매칭 안 되는 경우, LLM이 짧은 뉴스 친화 쿼리로 자동 rewrite (`search_news()` 내부).
- **NewsAPI 대신 DDG News 사용**: 명세 §3.4에서 NewsAPI 권장했으나 무료 플랜이 100 req/24h로 평가에 부족 → DuckDuckGo News 백엔드(키 불필요·사실상 무제한)로 교체. 코드 상 routing key는 `"news"`.
- **분류기 confidence 다운그레이드**: 0단계가 confidence=low로 분류한 명제는 안전 차원에서 `debatable`로 다운그레이드 (명세 §3.1). 정치·의학 민감 주제에서 발생.

---

## 6. 파일 구조

```
.
├── README.md                              # 본 파일
├── HW2_pipeline_v2.md                     # 명세 (구현팀 인계용)
├── IMPLEMENTATION_NOTES_v2.md             # 설계 결정·메모
├── hw2_pipeline_v2_260526.ipynb           # 본 노트북 (모든 코드)
├── eval_claims_20_v2.txt                  # 평가 명제 20개 (참고용 텍스트)
├── .env.example                           # API 키 템플릿
├── .gitignore                             # .env, eval_outputs 제외
└── eval_outputs/                          # 평가 산출물 (런타임 생성)
    └── {persona}/...
```

---

## 7. 디버깅·트러블슈팅

| 증상 | 해결 |
|---|---|
| `OPENAI_API_KEY 필요` AssertionError | `.env` 파일 위치·내용 확인 |
| `ModuleNotFoundError: ddgs` | `pip install ddgs` |
| 셀 11 (API 점검) Wikipedia/ArXiv timeout | 네트워크 문제 또는 일시 장애. 재실행 |
| 셀 43 중간 OpenAI rate limit | `MAX_TURNS`을 줄이거나 대기 후 재실행 |
| 셀 51에서 `persona_to_results` not defined | 셀 43이 먼저 실행돼야 함 |
| `MetricsSummary`가 모두 같은 페르소나 이름 | 셀 43과 51의 `PERSONAS_TO_RUN`이 일치하는지 확인 (셀 43 끝에서 alias됨) |

---

## 8. 데모 영상

[데모 영상 보기 (Google Drive)](https://drive.google.com/file/d/1ovDq9H9t7GGRUEXkgfmOq83BsN9_lm0h/view?usp=drive_link)

---

## 9. 명세·과제 정보

- 과제: AI 수업 HW2, 2026년 5월 31일 마감
- 구현팀: 한경찬, 장영인
- 비디오 제작: 박준하, 박소연 (별도)
- 명세 § 5.4·§ 9.2 — 평가 지표 5종 (Classification Accuracy, Persuasion Score, Logical Soundness, Persuasiveness, Debate Win-rate)
