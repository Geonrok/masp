# MASP AI Agents Common Guide

## 1. Project Overview
- **Name:** Multi-Asset Strategy Platform (MASP)
- **Language:** Python 3.11.9
- **Core Frameworks:** FastAPI (Backend), Streamlit (Frontend/Dashboard)
- **Goal:** Robust, automated trading and backtesting platform.

## 2. Environment Setup
```powershell
# Activate Virtual Environment
.\.venv\Scripts\Activate.ps1

# Run Tests
pytest

# Run Server (example)
uvicorn main:app --reload
streamlit run dashboard.py
```

## 3. P0 Rules (Critical - NEVER VIOLATE)
1.  **No Direct Requests:** Do NOT use `requests` or `aiohttp` directly. ALWAYS use `ConfigApiClient` (or project wrapper).
2.  **Secure Keys:** NEVER import `KeyManager` directly in business logic. Use the injected configuration or service wrapper.
3.  **Safety First:** `MASP_ENABLE_LIVE_TRADING` must default to `0` (False). Ensure Paper Trading is the default mode.
4.  **No Hardcoding:** NEVER hardcode API keys, secrets, or passwords. Use environment variables or the secure vault.

## 4. Code Style
-   **Type Hints:** Mandatory for all function signatures and class attributes.
-   **Docstrings:** Google style docstrings for all public modules, classes, and functions.
-   **Line Length:** Max 100 characters.
-   **Formatting:** Follow PEP 8 (handled by `black`/`ruff`).

## 5. Test Rules
-   **Zero Tolerance:** Maintain 0 failed tests. Fix broken tests immediately.
-   **New Features:** Must include unit tests covering positive and negative cases.
-   **Mocking:** Mock external API calls. Do not hit real endpoints during tests.

## 6. Security Requirements
-   **Encryption:** `KeyManager` must handle encryption/decryption transparently.
-   **Verification:** Use `hmac.compare_digest` for signature verification to prevent timing attacks.
-   **Sanitization:** Log safe data only. Never log raw API keys or tokens.

## 7. Review Policy
-   **Dual Review:** Code changes undergo parallel review by **Codex** and **Gemini**.
-   **Scope:** Security, Performance, Logic, and Style.

## 8. Feedback Application Criteria
-   **P1 (Critical):** Bugs, Security Flaws, Breaking Changes. **MUST FIX.**
-   **P2 (Warning):** Performance issues, Code smells, Edge cases. **STRONGLY RECOMMENDED.**
-   **P3 (Suggestion):** Naming, Comments, Minor refactoring. *Optional.*

## 9. Project Structure
-   `services/`: Core business logic and services.
-   `tests/`: Unit and integration tests (mirrors source structure).
-   `docs/`: Documentation and workflow guides.
-   `scripts/`: Utility scripts for maintenance and automation.

## 10. Commit Message Rules
Format: `<type>: <subject>`

-   `feat`: New feature
-   `fix`: Bug fix
-   `refactor`: Code change that neither fixes a bug nor adds a feature
-   `test`: Adding missing tests or correcting existing tests
-   `docs`: Documentation only changes
-   `chore`: Changes to the build process or auxiliary tools and libraries

---

## 🔄 자동 검수 워크플로우 (필수)

Claude는 모든 작업에서 **직접** bash_tool을 사용하여 Codex/Gemini 검수를 실행합니다.
사용자 개입 없이 자동으로 검수하고, P1 이슈를 해결합니다.

### 검수 실행 (Claude가 직접 수행)

**계획 검수** - 계획 작성 후 Claude가 실행:
```bash
powershell -ExecutionPolicy Bypass -File ./scripts/review-plan.ps1 -Content "계획 내용"
```

**코드 검수** - 코딩 완료 후 Claude가 실행:
```bash
powershell -ExecutionPolicy Bypass -File ./scripts/review-code.ps1
```

### 자동 검수 프로세스

#### Phase 작업 시
1. 계획 작성
2. **Claude가 bash_tool로 review-plan.ps1 실행**
3. 검수 결과 분석 → P1 있으면 수정 후 재검수
4. P1 = 0 확인 후 코딩 진행
5. 코딩 완료
6. **Claude가 bash_tool로 review-code.ps1 실행**
7. 검수 결과 분석 → P1 있으면 수정 후 재검수
8. P1 = 0 확인 후 커밋

#### 단순 작업 시
1. 작업 완료
2. **Claude가 bash_tool로 review-code.ps1 실행**
3. P1 있으면 수정 후 재검수
4. P1 = 0 확인 후 커밋

### 검수 통과 기준
| 등급 | 의미 | 조치 |
|------|------|------|
| P1 (Critical) | 버그, 보안, 크래시 | **필수 수정** (0개 될 때까지 재검수) |
| P2 (Important) | 성능, UX, 안정성 | 권장 수정 |
| P3 (Minor) | 스타일, 명명 | 스킵 가능 |

### 금지 사항
- ❌ 검수 없이 커밋 금지
- ❌ P1 > 0 상태로 커밋 금지
- ❌ 사용자에게 검수 실행 요청 금지 (Claude가 직접 실행)

---

## 🤖 AI 검수 팀 구성

### 검수자 역할

| CLI | 모델 | 역할 | 전문 분야 |
|-----|------|------|-----------|
| **Codex** | GPT-5.2-codex-medium | 백엔드 검수 | 보안, 버그, 에러처리 |
| **Gemini** | Gemini-3-Pro-Preview | 리서치 검수 | 통계, 백테스트, 데이터분석 |
| **OpenCode** | Big Pickle | 성능 검수 | 최적화, 메모리, 지연시간 |
| **Copilot** | Claude Sonnet 4.5 | 최종 검수 | 코드품질, 표준, 승인 |

### 자동 라우팅 규칙

```
libs/strategies/     → Gemini + Codex
libs/backtest/       → Gemini + OpenCode
libs/adapters/       → Codex + OpenCode
libs/risk/           → Codex + Gemini
services/            → Codex + OpenCode
tests/               → Copilot
기타                  → Codex + Copilot (기본)
```

### 검수 프롬프트 위치
- `.ai-review/prompts/codex-review.md`
- `.ai-review/prompts/gemini-review.md`
- `.ai-review/prompts/opencode-review.md`
- `.ai-review/prompts/copilot-review.md`

---

## 📚 컨텍스트 참조 가이드

### 필수 참조 파일 (검수 시)

#### 핵심 전략
- `libs/strategies/kama_tsmom_gate.py` - 메인 전략 (KAMA5/TSMOM90/MA30)
- `libs/strategies/base.py` - 전략 베이스 클래스
- `libs/strategies/indicators.py` - 기술적 지표

#### 거래소 연동
- `libs/adapters/real_upbit_*.py` - 업비트 어댑터
- `libs/adapters/real_bithumb_*.py` - 빗썸 어댑터
- `libs/adapters/real_binance_*.py` - 바이낸스 어댑터

#### 서비스 레이어
- `services/strategy_runner.py` - 전략 실행기 (포지션 동기화 포함)
- `services/automation_scheduler.py` - 자동화 스케줄러
- `services/daily_signal_alert.py` - 텔레그램 알림

#### 리스크 관리
- `libs/risk/drawdown_guard.py` - MDD 관리
- `services/risk_management_service.py` - 리스크 서비스

### 최근 주요 변경사항

<!-- AUTO-UPDATED: Do not edit manually -->
| 날짜 | 커밋 | 변경 내용 |
|------|------|----------|
| 2026-01-26 | ba3973a | 자동화 스케줄러 추가 |
| 2026-01-26 | 6ff0e6b | 포지션 동기화 버그 수정 (BTC Gate 실패 시 매도) |

### 프로젝트 핵심 개념

#### 전략 파라미터
- **KAMA Period**: 5 (Kaufman Adaptive MA)
- **TSMOM Lookback**: 90일 (Time-Series Momentum)
- **Gate MA**: 30일 (BTC 게이트)
- **진입 조건**: (Price > KAMA5 OR Price > Price[90d]) AND BTC > MA30

#### 리스크 한도
- **일간 손실**: -3%
- **주간 손실**: -7%
- **최대 MDD**: -15%

#### 텔레그램 설정
- Bot: @masp_alert_bot
- 일간 시그널: 09:00
- 시장 국면: 09:05
- 리스크 모니터: 매시 정각

---

## 🔄 검수 시스템 사용법

### 자동 라우팅 실행
```powershell
# 변경된 파일 분석 및 검수자 자동 할당
pwsh .ai-review/auto-review.ps1

# 드라이런 (실행 안함)
pwsh .ai-review/auto-review.ps1 -DryRun

# 자동 실행
pwsh .ai-review/auto-review.ps1 -Execute
```

### 검수 결과 수집
```powershell
# 검수 결과 등록
pwsh .ai-review/review-collector.ps1 -Mode collect -Reviewer codex -ReviewContent "..."

# 요약 생성
pwsh .ai-review/review-collector.ps1 -Mode summarize

# 상태 확인
pwsh .ai-review/review-collector.ps1 -Mode status
```

### 검수 결과 파일
- `.ai-review/reviews/{commit}-summary.json` - 통합 요약
- `.ai-review/reviews/{commit}-{reviewer}.md` - 개별 검수 결과
