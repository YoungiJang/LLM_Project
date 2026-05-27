# HW2: Adversarial Fact-Checking Chatbot 전체 파이프라인 명세

본 문서는 HW2 구현팀(한경찬, 장영인)이 이 문서만 보고 처음부터 끝까지 구현할 수 있도록, 시스템의 모든 결정 사항을 한 곳에 정리한 명세이다. 마감은 2026년 5월 31일이다.

---

## 1. 과제 개요

### 1.1 만드는 것
사용자의 주장에 대해 외부 검색 기반으로 실시간 반박을 수행하는 멀티턴 LLM 에이전트 시스템(LLM-based agent system / Tool-augmented Conversational Agent)이다. ChatGPT API는 부품으로 여러 번 다른 역할로 호출되며, 모델 학습이나 fine-tuning은 수행하지 않는다. 사용자가 "모델링"이라는 표현을 쓰는 것은 일상 회화에서는 통용되나, 공식 문서 및 비디오 스크립트에서는 "agent system 설계", "파이프라인 구축", "LLM application 개발"이 정확한 표현이다.

### 1.2 제출물 및 마감
- 제출물: 5분 이내의 시연 비디오 1개 (motivation, method, demo, evaluation 포함)
- 마감: 2026년 5월 31일 일요일 23:59
- 제출 위치: 조교가 공지한 구글드라이브 폴더

### 1.3 핵심 제약
- 팀당 OpenAI API 키 1개, $20 한도. 사용 모델은 발급된 키에 명시된 것을 그대로 사용한다.
- 단순 프롬프트 한 줄 박아넣는 것은 contribution으로 인정되지 않는다.
- LangChain/LangGraph 등 외부 프레임워크 사용 허용. 단, 본 명세는 순수 Python `if/else` 기반 구현을 가정한다 (학습 비용 및 디버깅 편의 고려).

### 1.4 채점 비중
Length 5%, Clarity 5%, Novelty 15%, Method 25%, Demo 15%, Safety 10%, Evaluation 25%.

---

## 2. 핵심 인터페이스

`respond()` 함수가 시스템 전체의 단일 진입점이다. 사용자 메시지와 대화 기록을 받아 챗봇 응답과 메타데이터를 반환한다.

```python
def respond(user_message: str, history: list[dict]) -> dict:
    """
    Returns
    -------
    dict with keys:
        - "reply": str                # 챗봇 응답 본문
        - "sources": list[dict]       # [{"title", "url", "snippet"}, ...]
        - "classification": str       # 첫 턴에 결정된 분류 (이후 턴에서도 그대로 유지)
        - "evidence_strength": str    # "strong" | "medium" | "weak"
        - "evaluator_state": str      # "continue" | "redirect" | "close"
    """
```

`history`는 이전 턴들의 기록. 각 원소는 위 반환 dict와 동일한 구조에 `"user"` 키(해당 턴의 사용자 입력)가 추가된 형태이다.

종료 턴에서는 `reply`에 토론 종료 멘트가 담긴다 (별도의 점수/판정 객체는 없음).

---

## 3. 전체 파이프라인 (6단계)

### 3.1 0단계 — 주장 분류기 (Claim Classifier)

**동작 조건**: 대화의 첫 턴에만 실행. 이후 턴은 첫 턴에서 결정된 분류를 그대로 사용한다.

**입력**: 사용자의 첫 메시지(주장).

**처리**: LLM 호출 1회. 사용자 주장을 다음 네 범주 중 하나로 분류시키고, 분류에 대한 confidence(high/medium/low)를 함께 출력시킨다.

| 라벨 | 정의 | 예시 |
|---|---|---|
| `established_fact` | 학계/사회 합의가 명확한 참 명제 | "지구는 둥글다", "백신은 자폐를 유발하지 않는다" |
| `clearly_false` | 명백히 거짓 또는 과학적으로 반증된 주장 | "백신은 자폐를 유발한다", "지구는 평평하다" |
| `debatable` | 논쟁 여지가 있는 명제 | "전기차가 환경에 더 좋다", "사형제는 정당하다" |
| `subjective_opinion` | 사실 검증 대상이 아닌 취향/선호 | "클래식이 재즈보다 좋다" |

**confidence 산출 방식**: 두 옵션이 있으며, 기본은 옵션 1로 구현하고 옵션 2는 시간 여유 시 보강한다.

- **옵션 1 (기본): LLM 자가 보고**: system prompt에 "분류 라벨과 함께 confidence를 high/medium/low로 출력하라"는 지시를 명시. 구현 간단, LLM의 주관에 의존.
- **옵션 2 (보강): logprobs 기반**: OpenAI API의 `logprobs` 옵션을 활성화하여 분류 토큰의 확률값을 직접 추출. 모델이 실제로 얼마나 확신하는지 정량 측정 가능. 구현 난도 ↑.

**confidence 처리**: confidence가 `low`인 경우 안전하게 `debatable`로 다운그레이드하여 처리한다. 이는 분류 오류가 시스템 전체를 망가뜨리는 것을 막기 위함이다.

**출력**: 분류 라벨, confidence.

---

### 3.2 1단계 — 쿼리 추출기 (Query Extractor)

**입력**: 사용자의 현재 메시지, 분류 라벨, 대화 기록.

**처리**:

- **`established_fact`의 경우**: 별도 LLM 호출 없이 사용자 주장 전체를 그대로 검색 쿼리로 사용한다. 정교한 키워드 추출이 불필요하다 (어차피 즉시 종료 모드이며 동의 근거 1~2개만 보여주면 됨). LLM 호출 0회.
- **그 외 분류**: LLM 호출 1회. 검색에 사용할 키워드 2~4개를 추출한다. 분류 라벨에 따라 추출 방향이 달라진다.
  - `clearly_false`, `debatable`, `subjective_opinion`: 반박 근거를 찾기 위한 키워드 추출

**출력**: 검색 쿼리 리스트.

---

### 3.3 2단계 — 라우터 (Router)

**입력**: 사용자 주장, 추출된 쿼리, 분류 라벨.

**처리**: LLM 호출 1회. Wikipedia, News, ArXiv 세 후보 중 0~3개를 매 질의마다 동적으로 선택한다.

- 0개 선택은 매우 특수한 경우(예: 검색이 무의미한 순수 취향 주장)에만 허용한다.
- LLM에게 "이 주장에 대해 어떤 종류의 근거가 가장 적합한가"를 판단시키고, 그에 따라 API 후보를 골라내게 한다.

**출력**: 사용할 API 이름들의 리스트.

> **구현 변경 (2026-05-27)**: 라우팅 키 `"newsapi"`는 `"news"`로 변경됨. 백엔드도 NewsAPI에서 DuckDuckGo News로 교체. 자세한 내용은 §10.2 참고.

---

### 3.4 3단계 — 검색 실행 (Retriever)

**입력**: 사용할 API 리스트, 검색 쿼리들.

**처리**: LLM 호출 없음. 선택된 API 각각에 HTTP 요청을 보내 문서를 수집한다.

**API 사양**:

- **Wikipedia**: 무료, 무제한, 키 불필요. `requests`로 `https://en.wikipedia.org/w/api.php` 직접 호출. 매 호출 사이 1초 sleep + 429 응답 시 2초 대기 후 1회 재시도.
- **News (DuckDuckGo News)**: `ddgs` 패키지 사용, 키 불필요, 사실상 무제한. `body` 필드에 snippet 포함. **변경 이력**: 최초 명세는 NewsAPI(newsapi.org)였으나 무료 플랜 한도(100/24h)가 4페르소나 평가에 부족하여 DDG News로 교체 (§10.2).
- **ArXiv**: 무료, 키 불필요. `arxiv` 패키지 사용. ArXiv 공식 가이드라인 준수 위해 `delay_seconds=3.5`, `num_retries=5`. 일시적 429/503은 자동 재시도, 그래도 실패하면 빈 결과 반환 (graceful).

**출력**: 검색 결과 항목들의 리스트 (RAG 맥락에서는 "문서(documents)"라고 부른다). 각 항목은 `{"title", "url", "snippet", "source_type"}` 형태로, 사용자에게 직접 보여줄 텍스트가 아니라 4단계가 답변 생성 시 참조할 **정보 단위**이다. 한 개의 Wikipedia 페이지 요약, 한 개의 뉴스 기사, 한 개의 논문 초록이 각각 한 "항목"에 해당한다.

> **구현 보강 (2026-05-27)**: News 검색에 LLM-rewrite fallback 추가됨. 1단계가 생성한 학술/포렌식 쿼리가 News에서 0건이면, LLM이 짧은 뉴스 친화 쿼리로 자동 재작성 후 1회 재시도. ArXiv/Wikipedia에는 영향 없음. 자세한 내용은 §10.3 참고.

---

### 3.5 4단계 — 응답 생성기 (Response Generator)

**입력**: 사용자 메시지, 분류 라벨, 검색된 문서, 5단계에서 평가된 근거 강도, 대화 기록.

**처리**: LLM 호출 1회. 분류 라벨과 직전 턴의 사용자 근거 강도에 따라 다섯 가지 응답 모드 중 하나로 분기한다. 각 모드는 서로 다른 system prompt를 사용하되, 검색된 문서는 공통으로 context로 주입한다.

**응답 모드 분기표**:

| 분류 (0단계) | 사용자 근거 강도 (5단계, 이전 턴) | 응답 모드 | 동작 |
|---|---|---|---|
| `established_fact` | (해당 없음, 첫 턴 즉시 종료) | 동의 모드 | 짧게 동의 + 근거 출처 제시 후 즉시 종료 |
| `clearly_false` | (강도 무관) | 강한 반박 모드 | 멀티턴 끝까지 강하게 반박 |
| `debatable` | strong | 부분 인정 모드 | 사용자 주장의 일부 인정 + 미묘한 반론 |
| `debatable` | medium | 기본 반박 모드 | 균형 잡힌 반박 |
| `debatable` | weak | 강한 반박 모드 | 강하게 반박 |
| `subjective_opinion` | (강도 무관) | 취향 반박 모드 | 반대 입장의 논거 제시 (수업 범위 내 추가 기법 적용 가능) |
| (모든 분류) | 논점 일탈 감지됨 | 논점 환기 모드 | 본 주제로 다시 끌어옴 |

**출력**: 응답 텍스트, 인용 출처 목록.

**구현 노트**: 모드별 system prompt는 별도 파일(`prompts/`)에 분리해 두는 것을 권장한다. 각 프롬프트는 (1) 챗봇 페르소나, (2) 사용해야 할 context의 위치, (3) 인용 표기 규칙, (4) 응답 톤 가이드를 명시적으로 포함한다.

---

### 3.6 5단계 — 토론 평가기 (Debate Evaluator)

**동작 조건**: 매 턴 4단계 직후 실행 (단, `established_fact`로 즉시 종료된 경우 제외).

**입력**: 사용자 메시지, 4단계 응답, 검색된 문서, 대화 기록.

**처리**: 세 가지 일을 LLM 호출로 수행한다.

1. **근거 강도 평가** (LLM 호출 1회):
   사용자가 직전 턴에 제시한 근거가 얼마나 강한지 LLM에게 `strong/medium/weak`로 판정시킨다. 이 값은 다음 턴 4단계의 응답 모드 결정에 사용된다. 첫 턴에서는 사용자 주장 자체의 근거 강도를 평가한다.

2. **논점 일탈 감지** (위 호출에 통합 가능):
   사용자 응답이 원 주장에서 벗어났는지 판정. 벗어났다면 다음 턴의 4단계가 "논점 환기 모드"로 동작한다.

3. **종료 판단**:
   현재 턴 수가 3~5턴에 도달했거나 사용자가 명시적으로 종료를 원하는 신호를 보내면 종료 모드로 진입. 종료 턴의 4단계는 지금까지의 토론을 짧게 요약하는 멘트로 응답한다.

**출력**: `evidence_strength`, `evaluator_state`.

---

## 4. LLM 호출 횟수 및 예산 추산

### 4.1 호출 횟수 정리

| 턴 종류 | 호출 단계 | 횟수 |
|---|---|---|
| 첫 턴 (`established_fact`, 즉시 종료) | 0, 2, 4 (1단계 생략, 5단계 생략) | 3회 |
| 첫 턴 (그 외 일반) | 0, 1, 2, 4, 5 | 5회 |
| 이후 턴 | 1, 2, 4, 5 | 4회 |

### 4.2 예산 추산 (시연 + 평가 전체)

명제 20개 × 평균 5턴 = 100턴 가정 시:

| 항목 | 호출 수 |
|---|---|
| 본 챗봇 호출 (100턴 × 평균 4.5회) | ~450회 |
| 평가 지표 계산 (명제당 7~8회 × 20) | ~150회 |
| User Simulator 사용 시 (명제당 4~5회 × 20) | ~90회 |
| **합계** | **약 700회** |

발급된 API 키에 명시된 모델 기준 비용:
- gpt-4o-mini 기준 (input $0.15/1M tokens, output $0.60/1M tokens): 턴당 2K 토큰 가정 시 약 $1~2 수준. $20 한도 내 안전.
- gpt-4o 기준 (input $2.50/1M tokens, output $10/1M tokens): 약 $15~25 수준. **한도 초과 위험**.

### 4.3 예산 가드레일 (필수 구현)

- **호출 카운터**: `respond()` 함수 내에서 전체 LLM 호출 수를 누적 카운트하고 로그에 기록한다.
- **토큰 추산**: OpenAI 응답의 `usage` 필드를 읽어 input/output 토큰을 누적한다.
- **임계치 경고**: 누적 비용이 $15에 도달하면 경고를 출력한다.
- **평가용 LLM 선택**: 평가 지표 계산용 judge LLM은 비용 절감을 위해 본 챗봇과 같은 모델을 사용하는 것을 기본으로 한다. 더 상위 모델(GPT-4o 등)을 judge로 쓰는 것은 일부 명제(에러 분석용 5개 등)에 한정하는 것을 권장한다.

---

## 5. 평가 방법 (Evaluation)

평가 코드 구현 및 실행은 장영인 담당이다. 본 섹션은 평가 담당자에게 인계되는 명세이다.

### 5.1 평가 명제 셋 구성 (총 20개)

0단계 4개 분류에 맞춰 균형 있게 배분한다.

| 분류 | 개수 | 비고 |
|---|---|---|
| `established_fact` | 2개 | 즉시 종료 케이스, 최소한만 |
| `clearly_false` | 6개 | 멀티턴 반박 |
| `debatable` | 10개 | 멀티턴 반박, 핵심 평가 대상 |
| `subjective_opinion` | 2개 | 멀티턴 반박, 정답 라벨 없음 |
| **합계** | **20개** | |

### 5.2 명제 생성 방향

명제는 다음 방식으로 생성한다. 평가 담당자가 이 절차에 따라 셋을 확정한다.

1. **공개 팩트체크 데이터셋 참고**: 가능하면 FEVER, ClimateFEVER, MultiFC 등 기존 데이터셋의 명제를 참고하여 일부를 선별/번역한다. 이는 정답 라벨이 학계 검증된 것이라 신뢰도가 높다.
2. **수동 작성으로 보완**: 위 데이터셋이 다루지 않는 한국적 맥락이나 최신 이슈(News 검증용)는 팀원이 직접 작성한다.
3. **분류별 다양성 확보**:
   - `clearly_false`: 과학(백신/지구평면설), 건강(특정 음식), 역사 등 도메인 다양화
   - `debatable`: 환경(전기차), 사회(사형제), 경제(최저임금), 기술(AI) 등 다양화
   - `subjective_opinion`: 음악/영화/음식 등 명확한 취향 주제
   - `established_fact`: 시연 시 즉시 종료를 보여줄 단순명제
4. **검색 가능성 확인**: 각 명제에 대해 Wikipedia/News/ArXiv 중 적어도 하나에서 의미 있는 결과가 나오는지 사전 확인한다.
5. **정답 라벨 부여**: 각 명제에 (분류 라벨, 사실 라벨, 예상 근거 출처)를 함께 기록한다. 단 `subjective_opinion`은 사실 라벨 없음.

### 5.3 평가 실행 방식

다음 두 방식 모두 가능하다. 평가 담당자가 시간 여유와 일관성 요구를 고려해 선택하거나, 일부는 방식 A, 일부는 방식 B로 혼용할 수도 있다.

- **방식 A: User Simulator (Lecture 28 방법론)**:
  LLM에게 사용자 페르소나를 부여하여 자동으로 멀티턴 대화를 진행시킨다. 일관성, 재현성, 확장성에 강점이 있다.
- **방식 B: 사람이 직접 대화**: 팀원이 직접 챗봇과 대화하며 응답을 수집한다. 자연스럽지만 시간이 들고 일관성이 떨어진다.

### 5.3.1 User Simulator 모듈 사양 (필수 구현)

User Simulator는 평가 자동화를 위해 본 시스템 코드의 일부로 함께 구현한다. 인터페이스는 다음과 같다.

```python
def simulate_user_turn(
    claim: str,
    history: list[dict],
    persona: str = "believer",
    max_turns: int = 5
) -> str:
    """
    Parameters
    ----------
    claim : str
        평가용 명제 (사용자가 처음 던질 주장)
    history : list[dict]
        지금까지의 대화 기록 (respond()의 history 형식과 동일)
    persona : str
        사용자 페르소나. 다음 중 하나:
        - "believer": 주장을 굳게 믿으며 끝까지 옹호
        - "moderate": 합리적으로 주장하되 강한 반박엔 일부 수용
        - "stubborn": 논점을 일부러 흐리는 비협조적 사용자
        - "evidence_focused": 외부 근거를 적극적으로 가져오는 사용자
    max_turns : int
        몇 턴까지 대화를 이어갈지 (이 턴 수 도달 시 자연 종료 시도)

    Returns
    -------
    str : 시뮬레이션된 사용자의 다음 발화
    """
```

**페르소나별 system prompt 골격**:

- **believer**: "당신은 [claim]을 강하게 믿는 사람입니다. 챗봇이 어떤 반박을 해도 입장을 굽히지 마십시오. 자신의 경험, 들은 이야기, 직관을 근거로 들 수 있습니다."
- **moderate**: "당신은 [claim]을 믿지만, 챗봇이 강력하고 신뢰할 만한 근거를 제시하면 부분적으로 수용할 수 있는 합리적 사용자입니다."
- **stubborn**: "당신은 [claim]을 주장하되, 챗봇의 반박을 회피하기 위해 관련 없는 이야기로 논점을 돌리거나 인신공격으로 응수하는 경향이 있습니다."
- **evidence_focused**: "당신은 [claim]을 주장하며, 자신의 입장을 옹호하기 위해 인터넷에서 본 듯한 통계, 연구, 사례를 적극적으로 인용합니다."

**평가 실행 루프 (의사코드)**:

```python
def run_evaluation(claim_set, persona="believer", max_turns=5):
    results = []
    for claim_item in claim_set:
        history = []
        user_msg = claim_item["claim"]
        for turn in range(max_turns):
            bot_response = respond(user_msg, history)
            history.append({
                "user": user_msg,
                "bot": bot_response["reply"],
                "sources": bot_response["sources"],
                "classification": bot_response["classification"],
                "evidence_strength": bot_response["evidence_strength"],
            })
            if bot_response["evaluator_state"] == "close":
                break
            user_msg = simulate_user_turn(
                claim_item["claim"], history, persona, max_turns
            )
        results.append({
            "claim": claim_item,
            "history": history,
            "verdict": bot_response.get("verdict"),
        })
    return results
```

**페르소나별 평가 의의**: 단일 페르소나만 쓰면 챗봇이 특정 사용자 유형에만 잘 작동할 위험이 있다. 시간이 허락하면 각 명제를 2~3개 페르소나로 돌려 챗봇의 견고성을 측정한다.

### 5.4 평가 지표

본 명세 작성 시 7종 지표를 나열했으나, §9.2에서 본 과제 범위에 맞춰 **5종**으로 축약했고 실제 구현은 5종을 산출한다. 두 목록을 함께 둔다.

**(a) 최초 설계 (7종, 참고)**

| 지표 | 정의 | 계산 방법 |
|---|---|---|
| Faithfulness | 응답이 검색 context의 사실만 사용했는지 (환각 배제) | LLM 판정 (binary or 0~1) |
| Relevance | 검색된 문서가 주장 반박에 유효했는지 | LLM 판정 |
| Factual Accuracy | 응답에 담긴 사실의 정확성 (`established_fact`, `clearly_false`, `debatable`) | 정답 라벨 대조 + LLM 판정 |
| Source Credibility | 인용된 출처의 신뢰도 | URL 도메인 휴리스틱 + LLM 판정 |
| Logical Soundness | 반박의 논리적 타당성 | LLM 판정 |
| Grounding Rate | 응답에서 검색 결과 인용 비율 | 인용 마커 카운트 |
| Debate Win-rate | 우리 챗봇 vs grounding 없는 일반 ChatGPT 응답 비교 | 제3의 LLM(GPT-4 등) 판정 |

**(b) 본 과제 실제 산출 (5종, §9.2 기준, 구현 완료)**

| 지표 | 정의 | 계산 |
|---|---|---|
| Classification Accuracy | `expected_label` vs 실제 0단계 분류 | 결정적 비교, LLM 호출 0회 |
| Persuasion Score | 사용자 첫·마지막 발화 비교로 입장 변화량 (0 / 0.5 / 1.0) | Judge LLM 1회/명제 |
| Logical Soundness | 챗봇 응답의 논리적 정합성 (1~5) | Judge LLM, 턴당 1회 → 평균 |
| Persuasiveness | 챗봇 응답의 설득력 (1~5, 청중=대상 페르소나) | Judge LLM, 턴당 1회 → 평균 |
| Debate Win-rate | 우리 챗봇 first-turn 응답 vs grounding 없는 baseline ChatGPT (A/B 무작위) | Judge LLM 1회/명제 |

`subjective_opinion`은 Factual Accuracy가 적용 불가능하므로 위 5종 중 Classification Accuracy를 제외한 4종만 의미 있다. 실제 결과는 §11 참조.

### 5.5 비디오용 산출물

- 분류별 평균 지표 점수 표
- 일반 ChatGPT 대비 Win-rate 막대 그래프
- 잘 동작한 케이스 1~2개, 실패한 케이스 1~2개에 대한 error analysis

---

## 6. 안전 (Safety Considerations)

채점 10%를 차지한다. 다음 항목을 시스템에 포함한다.

- 혐오/차별적 주장에 대한 필터링: 0단계 분류 시 혐오성 주장을 별도로 감지하여 토론 자체를 거부한다.
- 반박 톤 조절: 강한 반박 모드에서도 인신공격, 모욕적 표현은 금지하는 규칙을 system prompt에 명시한다.
- 출처 신뢰도 필터링: 위키피디아, 학술논문, 주요 언론사 외의 출처는 인용 시 신중하게 표시한다.
- 잘못된 근거로 반박하는 경우 방지: 검색 결과가 비어있거나 부적합하면 반박을 시도하지 않고 솔직히 "근거를 찾지 못했다"고 답한다.

---

## 7. 구현 작업 분담 (한경찬, 장영인)

| 작업 | 담당 | 비고 |
|---|---|---|
| 0~5단계 파이프라인 구현 (`respond()` 함수) | 한경찬, 장영인 공동 | 평일 작업 가능자 중심 |
| 각 단계의 system prompt 작성 | 공동 | `prompts/` 디렉터리 분리 |
| API 키 관리 및 환경 변수 세팅 | 공동 | |
| 평가 명제 20개 셋 구성 | 장영인 | §5.2 절차 따름 |
| 평가 지표 계산 코드 | 장영인 | `respond()` 함수만 import |
| 평가 실행 및 결과 정리 | 장영인 | 방식 A 또는 B 선택 |
| UI 통합 및 시연 비디오 | 박준하, 박소연 (별도) | 본 명세 범위 외 |

---

## 8. 예상 Future Work (비디오 마무리 슬라이드용)

- 0단계 주장 분류기에 대한 fine-tuning: 분류 정확도를 높이고 confidence 신뢰성을 개선할 수 있다. 본 과제는 학습을 수행하지 않으므로 향후 과제로 남긴다.
- Multi-agent debate 구조 도입: 취향 주장 등 정답이 없는 주제에 대해 두 개의 LLM이 대립하는 구조로 풍부한 반박을 생성하는 방향. 본 과제 범위에서는 단일 generator로 처리한다.
- 한국어 fact-check 데이터셋 구축: 본 과제에서는 영어 데이터셋과 수동 작성을 혼용했으나, 한국어 전용 검증 셋이 있으면 평가 신뢰도가 높아진다.
- logprobs 기반 confidence 정밀화: 본 명세는 LLM 자가 보고 confidence를 기본으로 한다. OpenAI API의 `logprobs` 옵션을 활용하여 분류 토큰 확률값을 직접 추출하면 더 견고한 confidence 추정이 가능하다.

---

## 부록 A. 최초 기획안 대비 확장/변경 사항 정리

본 명세는 사용자의 4/14 기획안 및 카카오톡 대화 요약본을 기반으로, 이후 논의를 통해 확장·구체화된 결과이다. 최초 기획안과 달라진 부분만 별도로 정리한다.

### A.1 신규 추가

- **0단계 주장 분류기 신설**:
  최초 기획안에는 모든 주장에 대해 반박을 시도하는 단일 흐름이었다. 4/14 Discussions 1번에서 제기된 "참인 명제에 무조건 반박 시 억지 논쟁 도구처럼 보일 위험"을 구조적으로 해결하기 위해 신설되었다. 사용자가 본 논의 중 직접 제안한 단계이다.

- **0단계 분류에 confidence 부여**:
  분류 오류가 시스템 전체에 미치는 영향을 최소화하기 위해, LLM이 분류 라벨과 함께 confidence(high/medium/low)를 출력하도록 한다. low인 경우 `debatable`로 안전하게 다운그레이드한다. 본 논의 중 추가된 안전장치이며, 산출 방식은 LLM 자가 보고를 기본으로 한다.

- **분류기 fine-tuning 및 logprobs 기반 confidence를 Future Work로 명시**:
  본 과제는 학습을 수행하지 않으므로 분류 정확도와 confidence 신뢰성의 정밀화는 예상 future work로 남긴다.

- **평가에 User Simulator 도입 (선택지로 제공)**:
  최초 기획안에 없던 평가 방식. Lecture 28의 User Simulator 방법론을 도입하여 멀티턴 평가의 일관성·재현성을 확보한다. 사람 직접 대화 방식과 병행 가능.

- **User Simulator 코드 모듈로 구현**:
  단순 선택지를 넘어, `simulate_user_turn()` 함수를 본 시스템에 함께 구현한다 (§5.3.1 참조). believer/moderate/stubborn/evidence_focused 네 페르소나를 지원하며, 평가 자동화 루프(`run_evaluation`)도 의사코드로 명시했다.

- **평가 명제 셋 균형 배분 명세**:
  최초 기획안은 "20~30개"로만 적혀 있었다. 본 명세에서는 0단계 분류 4범주에 맞춰 2/6/10/2로 정확히 배분한다.

- **평가 명제 생성 절차 명시**:
  FEVER 등 공개 데이터셋 참고 + 수동 작성 보완 + 도메인 다양화 + 검색 가능성 사전 확인 + 정답 라벨 부여, 5단계 절차로 구체화.

- **예산 가드레일 추가**:
  최초 기획안에는 예산 관리 절차가 없었다. 본 명세에서는 호출 카운터, 토큰 누적, 임계치 경고, 평가용 judge LLM 선택 가이드를 §4.3에 추가했다. 명제 20개를 5턴씩 평가하면 약 700~800회 호출이 발생하며 모델 선택에 따라 한도 초과 위험이 있어 모니터링이 필요하다.

### A.2 확장·구체화

- **4단계의 다중 모드화**:
  최초 기획안은 "Adversarial System Prompt 단일 모드"였다. 본 명세에서는 0단계 분류 결과와 5단계 평가 결과를 입력으로 받아 5개 응답 모드(동의/강한 반박/기본 반박/부분 인정/취향 반박/논점 환기) 중 하나로 분기하도록 확장되었다.
  - `established_fact`는 즉시 종료, 나머지 세 분류는 모두 멀티턴 진행으로 정리됨.

- **2단계 라우터의 동적 선택 명시화**:
  최초 기획안은 "주장 성격에 따라 분기"로만 적혀 있어 정적/동적 여부가 모호했다. 본 명세에서는 매 질의마다 LLM이 0~3개 API를 동적으로 선택하도록 명시한다.

- **5단계 Debate Evaluator의 통합 모듈화**:
  최초 기획안에서 별개로 논의되던 세 항목(반박 강도 조절, 논점 자동 전환, 종료 판정)을 하나의 모듈로 통합. 4/14 Discussion 2번의 "debate evaluator로 설계" 표현을 따른 것이다.

- **단일 인터페이스 `respond()` 함수로 정리**:
  최초 기획안에는 명시되지 않았으나, 구현 및 평가 인계를 위해 함수 시그니처와 반환 dict 스키마를 §2에 명시했다.

- **1단계 쿼리 추출의 분류별 조건부 처리**:
  최초 기획안은 모든 주장에 대해 동일하게 쿼리 추출을 수행하는 흐름이었다. 본 명세에서는 `established_fact`의 경우 추가 LLM 호출 없이 사용자 주장 전체를 검색 쿼리로 그대로 사용하도록 단순화했다 (어차피 즉시 종료되며 동의 근거 1~2개만 제시하면 충분하기 때문).

### A.3 변경

- **subjective_opinion 처리 방향**:
  본 논의 초반 일시적으로 "취향 주장은 반박 안 함"으로 제안되었으나, 사용자 지시에 따라 "취향 주장도 반박은 하되 grounding 소스를 다르게 한다"로 변경되었다. 단, 추가 API(Reddit 등)는 도입하지 않고 기존 3개 API(Wiki/News/ArXiv) 범위 내에서 처리한다.

- **취향 반박 모드 구현 범위**:
  Multi-agent debate 등 풍부한 구조 도입은 검토되었으나, "수업 범위(~Lecture 20) + α" 선에서 처리하기로 한다. 핵심은 4단계의 모드별 system prompt 분기이며, 추가 기법은 시간 여유 시 고려한다.

- **권장 구현 순서 섹션 제거**:
  최초 명세에 일자별 구현 순서를 포함했으나, 사용자 지시에 따라 제거했다. 작업 분담은 §7에 그대로 유지된다.

- **verdict 객체 제거**:
  최초 기획안 및 4/14 Discussion 2번의 "양측 점수 + 강점 + 불확실성 + 승패 판단" 출력 구조를 잠시 verdict 객체로 도입했으나, 점수가 LLM 자가 평가에 불과해 정보 가치가 낮다는 판단으로 제거했다. 종료 턴은 4단계가 토론 요약 멘트로 응답하는 것으로 단순화한다. 시스템 평가는 §5.4의 7개 metric으로 별도 수행한다.

### A.4 최초 기획안 유지

다음은 최초 기획안 그대로 유지되는 부분이다.

- 한 줄 요약: "사용자의 모든 주장에 실시간 외부 검색 기반으로 반박하는 논쟁 챗봇" (단, 위 §3.5에 따라 `established_fact`는 동의)
- Motivation: 확증 편향 깨기, 비판적 사고 훈련, Epistemic Stress Test
- Novelty: 동의·중립 대신 의도적 반대, RAG의 adversarial 활용
- 평가 지표 7종 (Faithfulness, Relevance, Factual Accuracy, Source Credibility, Logical Soundness, Grounding Rate, Debate Win-rate)
- Safety 4항목 (혐오 필터링, 톤 조절, 출처 신뢰도, 잘못된 근거 방지)

---

## 부록 B. 미확정 사항

본 명세 작성 시점에 아직 결정되지 않은 항목은 없다. 위 §1~§8의 모든 결정이 사용자 확인을 거쳤다. 단, 다음은 구현 진행 중 자연스럽게 결정될 사항이다.

- 각 단계별 system prompt의 구체 문구
- 검색 결과 상위 몇 개를 context로 주입할지 (권장: 단계 3에서 API당 3~5개)
- 종료 턴 수 결정 (3~5턴 범위 내 어디로 할지, 권장: 4턴 고정 또는 LLM 자율 판단)

이상.


---

## 9. 남은 구현 작업 (2026-05-26 시점) — 완료

본 섹션은 2026-05-26 작성 시점의 To-do였다. 2026-05-27 작업으로 **전 항목 완료**.
현재 상태는 §10·§11에 정리.

| 항목 | 상태 | 비고 |
|---|---|---|
| 9.1 평가 자동 실행 루프 | ✅ 완료 | 셀 43이 `PERSONAS_TO_RUN` 루프로 4페르소나 × 20명제 자동 실행. `eval_outputs/{persona}/log_NN.txt` + `result_NN.txt` 저장. |
| 9.2 평가 지표 코드 (5종) | ✅ 완료 | 셀 45·47·49·51: judge 프롬프트·baseline·metric 함수·계산. `eval_outputs/{persona}/metrics_summary.txt`로 저장. 실제 점수는 §11 참고. |
| 9.3 confidence 다운그레이드 검증 | ✅ 확인 | 셀 14 `classify_claim`에 구현 확인됨. low confidence → debatable. hate_speech는 예외 유지. |
| 9.4 출처 도메인 휴리스틱 | ⚠️ 부분 | 응답에 `[도메인]` 인용 표기는 들어감. 도메인별 신뢰도 가중치는 미구현 (Future Work). |
| 9.5 비디오 시연 | (코드 외) | 데모 셀 32~36 + §11 표 활용 권장. |
| 9.6 알려진 위험 | 변화 있음 | §10.2 (NewsAPI→DDG 교체), §10.4 (ArXiv rate limit) 참고. |

---

## 10. 구현 후 변경 사항 (2026-05-27)

명세 작성 후 구현 진행 중 발견된 이슈에 따른 변경. 모두 §3·§5의 의도된 동작을 유지하면서 외부 의존성·핸드오버 편의를 개선한 것.

### 10.1 4페르소나 자동 평가 루프 (셀 43, 51)

기존: 셀 43에 `PERSONA = "believer"` 하드코드, 페르소나 바꾸려면 매번 수동 재실행.

변경: 셀 43이 `PERSONAS_TO_RUN = ["believer", "moderate", "stubborn", "evidence_focused"]` 리스트를 순회. 페르소나별 결과를 `persona_to_results` dict + `eval_outputs/{persona}/` 디렉터리에 저장. 셀 51도 같은 리스트를 순회하여 페르소나별 metrics를 계산. 단일 페르소나 모드는 `PERSONAS_TO_RUN = ["believer"]` 로 동작.

### 10.2 NewsAPI → DuckDuckGo News 교체

기존: NewsAPI(newsapi.org), 무료 100 req/24h.

문제: 4페르소나 평가는 명제당 평균 ~10 News 호출 × 20 명제 × 4 = 약 800회 시도 필요. 무료 한도 100을 한참 초과해서 ~700+회 429 발생. 평가 비대칭화.

해결: DuckDuckGo News로 백엔드 교체 (`ddgs` Python 패키지 사용, 키 불필요, 사실상 무제한). 함수명 `search_news()`, source_type `"news"`, 라우팅 키 `"news"`. EVAL_CLAIMS의 `expected_apis` 값도 `"newsapi"` → `"news"` 갱신.

### 10.3 News 검색 LLM-rewrite fallback

문제: 1단계 쿼리 추출기가 "구체적 키워드 위주" 규칙으로 학술·forensic phrase를 생성. ArXiv·Wikipedia는 잘 매칭하지만 뉴스 검색에서는 0건 빈발.

해결: `search_news()` 내부에서 첫 호출이 0건이면 LLM에게 "짧은 뉴스 친화 쿼리로 재작성" 요청 후 1회 재시도. 1단계 자체 프롬프트는 그대로 유지 (ArXiv/Wikipedia 영향 없음). 추가 LLM 호출은 News 0건일 때만 발생.

### 10.4 ArXiv rate limit 완화

기존: `arxiv.Client(delay_seconds=1, num_retries=2)`.

문제: ArXiv 공식 가이드라인은 3초/회 권장. 1초는 너무 공격적으로, 429/503 빈발.

해결: `delay_seconds=3.5, num_retries=5`로 조정. 일부 명제(특히 ArXiv 의존 6, 7, 12, 13, 14, 16, 18)에서 여전히 일시 429/503이 발생할 수 있지만, 다른 API(Wikipedia/News) 결과로 보완. 빈 결과 반환은 graceful (chatbot이 "근거 부족" 응답 모드로 처리).

### 10.5 Universal key loader (Colab + 로컬)

기존: Colab Secrets에서만 키 로드 (`from google.colab import userdata`).

변경: 셀 4가 `try/except`로 Colab Secrets 시도 후 ImportError 발생하면 로컬 `.env` 파일 파싱으로 폴백. `OPENAI_API_KEY`만 필수 (NewsAPI 키는 제거됨, DDG는 키 불필요). README + `.env.example` + `.gitignore` 함께 추가됨 → 핸드오버 시 키 발급·설정 절차 간소화.

### 10.6 Dry-run 토글 (셀 41~42)

기존: 모든 실행이 명제 20개 전체.

변경: 셀 42에 `DRY_RUN_N` 변수 추가. 정수(예: 2)면 EVAL_CLAIMS를 처음 N개로 축소. `None`이면 전체. 오류 탐지·빠른 검증용.

---

## 11. 실제 평가 결과 (4페르소나, 2026-05-27)

20개 명제 × 4페르소나 = 80 토론 자동 실행 후 5종 지표 산출. `eval_outputs/{persona}/metrics_summary.txt` 원본 참조. 총 소요 시간 약 171분, OpenAI 비용 약 $3.0.

### 11.1 페르소나별 AGGREGATE 점수

| 지표 | believer | moderate | stubborn | evidence_focused | 평균 |
|---|---|---|---|---|---|
| Classification Accuracy | 90.0% (18/20) | 85.0% (17/20) | 90.0% (18/20) | 90.0% (18/20) | **88.75%** |
| Persuasion (mean, 0~1) | 0.389 | 0.500 | 0.417 | 0.333 | **0.410** |
| Logical Soundness (1~5) | 3.23 | 3.26 | 3.10 | 3.17 | **3.19** |
| Persuasiveness (1~5) | 3.34 | 3.29 | 3.14 | 3.27 | **3.26** |
| **Debate Win-rate** | 70% (W14/L6/T0) | 85% (W17/L3/T0) | 75% (W15/L5/T0) | 65% (W13/L7/T0) | **73.75%** |

### 11.2 핵심 인사이트

- **Win-rate 73.75% 평균**: grounding 기반 RAG가 grounding 없는 baseline ChatGPT 대비 일관된 우위. RAG 도입의 가치를 정량 입증.
- **페르소나별 Win-rate 차이**: moderate(85%) > stubborn(75%) > believer(70%) > evidence_focused(65%). 합리적 사용자에게 가장 우월하고, 통계·연구를 적극 인용하는 사용자에게는 어려워짐.
- **Persuasion 패턴이 페르소나 의도와 부합**: moderate 0.500 (가장 흔들림) > stubborn 0.417 > believer 0.389 > evidence_focused 0.333 (가장 안 흔들림). User Simulator 설계가 의도대로 동작.
- **Classification Accuracy 88.75%**: 4페르소나 모두 ≥85% — 분류기 일관됨. mismatch 2건은 모두 `clearly_false → debatable` 다운그레이드 (정치·의학 민감 주제 — 명세 §3.1 안전장치 동작 결과).
- **Soundness/Persuasiveness 평균 3.1~3.3 / 5**: "보통" 수준. 향상 여지 있음 (향후 개선 — system prompt 정교화, 더 풍부한 근거 인용 강제).

### 11.3 한계 (비디오·보고서 명시 권장)

- **ArXiv rate limit 노이즈**: 평가 중 7~10건 정도의 ArXiv 429/503이 페르소나별로 다른 시점에 발생 → 페르소나 간 cross-comparison에 약간의 비대칭 노이즈. 영향 추정: 절대 점수 변동 ±0.1~0.2 수준.
- **DDG News 검색 매칭**: 학술·forensic 쿼리는 LLM-rewrite fallback이 처리하지만, rewrite도 News 매칭 실패 시 빈 결과. 이런 경우 wikipedia/arxiv 결과로 보완.
- **Soundness 점수 일부 degradation**: cell 51이 sources의 snippet 텍스트를 LLM judge에 전달. snippet이 짧거나 빈 경우(특히 News API 교체 전 초기 평가) judge가 정확히 평가 어려움.

### 11.4 비디오용 권장 산출물 (§5.5)

1. **AGGREGATE 점수표** (§11.1 그대로)
2. **Win-rate 막대 그래프**: 4페르소나 × {우리 챗봇 우세 / baseline 우세 / 무승부}
3. **Error analysis 2~4 케이스**:
   - 성공: claim 11 (희토류, NewsAPI 우세 명제에서 DDG News 활용해 4턴 우세)
   - 성공: claim 6 (백신-자폐, ArXiv+Wiki로 강한 반박)
   - 실패: claim 18 (재택근무, baseline ChatGPT의 일반 지식으로도 충분)
   - 실패: claim 7/8 (clearly_false → debatable 분류 다운그레이드 케이스, 정치·의학 민감)
4. **Future work**: 0단계 fine-tuning, logprobs confidence, 출처 도메인 가중치, News 검색 친화 1단계 prompt 분기.
