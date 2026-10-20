# decision-agent-lab

"LLM이 생성하고, 결정 모델이 판단하고, 코드가 실행한다"는 에이전트 구성을 직접 측정하는 레포입니다.
TypeSafe Jev, OpenAI Decisions API, Upstage Solar Decide, 로컬 Ollama 모델을 같은 인터페이스로 호출하고,
모든 호출의 모델 버전·지연·토큰·비용을 기록합니다.

## 빠른 시작

```bash
cp .env.example .env   # 키 채우기
uv sync --all-extras
uv run pytest           # 네트워크 없는 단위 테스트
SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt uv run python scripts/smoke.py   # 백엔드별 1회 호출
```

기업망처럼 TLS 가로채기 프록시가 있는 환경에서는 위처럼 시스템 CA 번들을 지정해야 합니다. 코드는 인증서 검증을 끄지 않습니다.

(실험·결과·그래프 설명은 측정이 끝난 뒤 채웁니다.)
