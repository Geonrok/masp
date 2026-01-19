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
