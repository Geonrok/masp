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

Claude는 모든 작업에서 **직접** Codex/Gemini 검수를 실행합니다.
사용자 개입 없이 자동으로 검수하고, P1 이슈를 해결합니다.

### 검수 대상
| 단계 | 검수 내용 |
|------|----------|
| 계획 | 설계 누락, 보안 이슈, 아키텍처 문제 |
| 코딩 | 버그, 보안 취약점, 베스트 프랙티스 위반 |

### 검수 실행 방법

Claude는 bash_tool을 사용하여 검수 스크립트 실행:

**계획 검수:**
```bash
./scripts/review-plan.ps1 -Content "계획 내용"
```

**코드 검수:**
```bash
./scripts/review-code.ps1
```

### 자동 검수 프로세스

#### Phase 작업 시
1. 계획 작성
2. `review-plan.ps1` 실행하여 계획 검수
3. P1 이슈 있으면 수정 후 재검수 (P1=0 될 때까지)
4. 코딩 진행
5. `review-code.ps1` 실행하여 코드 검수
6. P1 이슈 있으면 수정 후 재검수 (P1=0 될 때까지)
7. 커밋

#### 단순 작업 시
1. 작업 완료
2. `review-code.ps1` 실행
3. P1 이슈 있으면 수정 후 재검수
4. 커밋

### 검수 통과 기준
| 등급 | 의미 | 조치 |
|------|------|------|
| P1 (Critical) | 버그, 보안, 크래시 | **필수 수정** (0개 될 때까지) |
| P2 (Important) | 성능, UX, 안정성 | 권장 수정 |
| P3 (Minor) | 스타일, 명명 | 스킵 가능 |

### 금지 사항
- ❌ 검수 없이 커밋 금지
- ❌ P1 > 0 상태로 커밋 금지
- ❌ 검수 스킵 금지

### 검수 결과 보고 형식
```
## 검수 결과

### Codex
- P1: N개
- P2: N개
- P3: N개
- 상세: ...

### Gemini
- P1: N개
- P2: N개
- P3: N개
- 상세: ...

### 판정: ✅ 통과 / ❌ 재검수 필요
```
