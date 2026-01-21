# MASP Claude Code Workflow Contract (MUST FOLLOW)

You MUST follow the workflow defined in AGENTS.md.
This file is the highest priority instruction for this repository.

## Absolute Rules (P0)

- Read and follow AGENTS.md before doing anything.
- Never violate MASP P0 rules:
  - No direct requests/aiohttp. Use ConfigApiClient/wrapper only.
  - Do not import KeyManager directly in business logic.
  - MASP_ENABLE_LIVE_TRADING must default to 0 (paper trading default).
  - Never hardcode secrets.

## Mandatory Review Workflow (NO EXCEPTIONS)

- After writing a plan, you MUST run:
  ```powershell
  powershell -ExecutionPolicy Bypass -File ./scripts/review-plan.ps1 -Content "<PLAN_TEXT>"
  ```

- After finishing code changes, you MUST run:
  ```powershell
  powershell -ExecutionPolicy Bypass -File ./scripts/review-code.ps1
  ```

- If P1 > 0, you MUST fix issues and re-run review until P1 == 0.
- Never ask the user to run review scripts. You run them yourself.
- Never commit if review was skipped or P1 > 0.

## Review Pass Criteria

| Grade | Meaning | Action |
|-------|---------|--------|
| P1 (Critical) | Bugs, Security, Crashes | **MUST FIX** (re-review until 0) |
| P2 (Important) | Performance, UX, Stability | Recommended fix |
| P3 (Minor) | Style, Naming | Optional |

## Prohibited Actions

- Committing without running review
- Committing with P1 > 0
- Asking user to run review scripts (you must run them yourself)
- Skipping review workflow for any reason

## Output Format Requirement

Every response must end with:

```
WORKFLOW_STATUS: plan_review=[DONE/SKIPPED/NA], code_review=[DONE/SKIPPED/NA], P1=[0/N/unknown], tests=[PASS/FAIL/NA]
```

## Quick Reference

```powershell
# Plan review (after writing plan)
powershell -ExecutionPolicy Bypass -File ./scripts/review-plan.ps1 -Content "plan content here"

# Code review (after code changes)
powershell -ExecutionPolicy Bypass -File ./scripts/review-code.ps1

# Run tests
pytest

# Install git hooks (one-time setup)
powershell -ExecutionPolicy Bypass -File ./scripts/install_git_hooks.ps1
```
