# decision-agent-lab

"LLM이 생성하고, 결정 모델이 판단하고, 코드가 실행한다"는 에이전트 구성을 직접 측정하는 레포입니다.
TypeSafe Jev, OpenAI Decisions API, Upstage Solar Decide, 로컬 Ollama 모델을 같은 인터페이스로 호출하고,
모든 호출의 모델 버전·지연·토큰·비용을 기록합니다. 벤더가 제시한 배수 대신, 직접 측정한 절대값만 씁니다.

## 빠른 시작

```bash
cp .env.example .env   # 키 채우기
uv sync --all-extras
uv run pytest           # 네트워크 없는 단위 테스트
SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt uv run python scripts/smoke.py   # 백엔드별 1회 호출
```

기업망처럼 TLS 가로채기 프록시가 있는 환경에서는 위처럼 시스템 CA 번들을 지정해야 합니다. 코드는 인증서 검증을 끄지 않습니다.

## 두 가지 실험

### 1. 도구 라우팅 (`scripts/bench_toolroute.py`)

사용자 요청과 12개 도구 카탈로그(`data/toolroute/`)를 주고 다음에 호출할 도구 하나를 고르게 하는 과제(요청 40건, 한/영).
결정 모델(Choice)과, 생성 모델이 구조화 출력으로 직접 고르는 방식(`llm_native`)을 비교합니다. 실제 실행은 하지 않는 드라이런입니다.

```bash
SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt uv run python scripts/bench_toolroute.py \
    --routers typesafe:jev-1.13.0,openai_decisions:gpt-6-luna,llm_native:gpt-6-luna --concurrency 2
uv run python scripts/make_toolroute_figure.py
```

서울 1회 측정 요약(`results/summary/toolroute/summary.csv`):

| 라우터 | 라우팅 정확도 | 기권/넘김 정확도 | 오호출률 | p50(ms) | 1k당(USD) |
|---|---|---|---|---|---|
| Jev 1.13 | 1.00 (40/40) | 1.00 | 0% | 165 | 0.030 |
| OpenAI Decisions | 0.90 | 1.00 | 0% | 259 | 0.043 |
| LLM native (gpt-6-luna) | 0.85 | 0.89 | 11% | 1,753 | 0.068 |

### 2. 명령 실행 게이트 (`scripts/bench_gate.py`)

에이전트가 실행하려는 명령·도구 호출을 allow / ask / deny로 분류하는 4단계 게이트(코드 하드 거부 → 읽기 전용 허용목록 →
결정 모델 질문 7개 → 실패 시 보수적 ask). 정답 라벨이 붙은 1,007건으로 측정합니다(커밋된 데이터만으로는 877건,
나머지 130건은 실행 시 내려받음 — 아래 데이터 참고).

```bash
uv run python scripts/fetch_data.py        # jev-harness-lab 130건을 data/cache/로 내려받음(선택)
SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt uv run python scripts/bench_gate.py \
    --sources themsquared,godspede,laya_guard,aitejiu,own
uv run python scripts/analyze_gate.py      # results/summary/gate/*.csv 재생성
uv run python scripts/make_figures.py      # figures/*.png 재생성
```

서울 1회 측정 요약(`results/summary/gate/headline.csv`, 위험=deny 333건·안전=allow 476건 기준):

| 방법 | 위험 통과율 ↓ | 안전 마찰률 ↓ | 1k당(USD) | p50(ms) |
|---|---|---|---|---|
| 코드 규칙만 | 3.6% | 89.9% | 0 | 0 |
| Jev 1.13 | 4.5% | 63.2% | 0.027 | 181 |
| OpenAI Decisions | 4.2% | 52.7% | 0.125 | 329 |
| LLM 판정(gpt-6-luna) | 5.4% | 51.1% | 0.198 | 3,671 |
| Ollama nimble(로컬) | 7.8% | 44.5% | 0 | 454 |
| Upstage solar-decide | 7.8% | 34.7% | 0.315 | 8,256 |

Upstage 비용은 직접 API 정가(백만 토큰당 $0.10) 기준입니다.

## 데이터 출처

| 세트 | 사용 건수 | 라이선스 | 동봉 여부 |
|---|---|---|---|
| themsquared/jev-calibration | 240 | Apache-2.0 | `data/public/`에 동봉 |
| godspede/construct-auto-classifier | 393 | Apache-2.0 | `data/public/`에 동봉 |
| scs0209/laya-guard | 114 (한국어 20 포함) | MIT | `data/public/`에 동봉 |
| Aitejiu/jev-harness-lab | 130 | 미명시 | 동봉 안 함, `scripts/fetch_data.py`가 실행 시 `data/cache/`로만 내려받음 |
| 자체 세트 `data/own/gate_ko_en.jsonl` | 130 (65쌍, 한국어 65) | MIT | 라벨 프로토콜 `LABELING.md`·검수 기록 `REVIEW.md`와 함께 동봉 |

커밋 SHA·sha256·원본 행 수는 `data/public/SOURCES.md`, 라벨 변환·제외 규칙은 `src/dalab/gate/datasets.py`에 있습니다.
외부 세트는 라벨 체계가 달라 allow/ask/deny 세 값으로 맞추었습니다.

## 레포 구조

```
src/dalab/          백엔드 어댑터, 질문 타입, 게이트 정책·분석
scripts/            bench_* (측정), analyze_gate / make_*_figure (표·그림), fetch_data, smoke
config/             models.toml (모델 id·가격·엔드포인트), policy.yaml (게이트 임계값·패턴)
data/               toolroute/, public/(동봉 외부 세트), own/(자체 세트+검수기록)
results/summary/    요약 CSV (원시 로그는 재생성 가능하므로 커밋하지 않음)
figures/            PNG/SVG
tests/              pytest (라이브 호출 테스트는 -m live로 분리)
```

## 정직성 원칙

- 헤드라인 수치는 전부 직접 측정한 값입니다. 외부 인용 수치는 재현하지 않았습니다.
- 모든 호출에 모델 버전·날짜·위치·토큰·비용을 기록합니다(`config/models.toml`의 가격 기준, 2026-10-20 고정).
- 라우팅 40건은 통제된 자체 세트라 편향 가능성이 있고 일부 정답은 논쟁적입니다. 지연·비용은 서울 1회 측정입니다.
- 코드는 어떤 경우에도 TLS 인증서 검증을 끄지 않습니다.

## 라이선스

MIT (`LICENSE`). 동봉한 외부 데이터는 각 `data/public/<세트>/LICENSE`를 따릅니다.
