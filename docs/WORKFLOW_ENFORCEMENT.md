# MASP Workflow Enforcement Guide

This document explains how the MASP repository enforces the review workflow defined in AGENTS.md.

## Overview

The MASP repository uses Git hooks to **automatically enforce** the code review workflow. This prevents commits that bypass the mandatory review process or have unresolved P1 issues.

## Workflow Enforcement (Mandatory)

This repo enforces AGENTS.md review workflow via git pre-commit hook.

### One-time Setup

Run this command once after cloning the repository:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/install_git_hooks.ps1
```

This configures Git to use the `.githooks` directory for hooks.

### What Happens on Commit?

1. `pre-commit` hook automatically runs `scripts/review-code.ps1`
2. If it fails (P1 > 0), the commit is **BLOCKED** until issues are fixed
3. You must fix P1 issues and retry the commit

```
┌─────────────────────────────────────────────────────────────┐
│                    Commit Flow                              │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│   git commit -m "message"                                   │
│         │                                                   │
│         ▼                                                   │
│   ┌─────────────────┐                                       │
│   │  pre-commit     │                                       │
│   │  hook triggered │                                       │
│   └────────┬────────┘                                       │
│            │                                                │
│            ▼                                                │
│   ┌─────────────────┐                                       │
│   │ precommit_guard │                                       │
│   │     .ps1        │                                       │
│   └────────┬────────┘                                       │
│            │                                                │
│            ▼                                                │
│   ┌─────────────────┐                                       │
│   │ review-code.ps1 │                                       │
│   │   (AI Review)   │                                       │
│   └────────┬────────┘                                       │
│            │                                                │
│      ┌─────┴─────┐                                          │
│      ▼           ▼                                          │
│   ┌──────┐   ┌──────┐                                       │
│   │ PASS │   │ FAIL │                                       │
│   │(P1=0)│   │(P1>0)│                                       │
│   └──┬───┘   └──┬───┘                                       │
│      │          │                                           │
│      ▼          ▼                                           │
│   Commit     Commit                                         │
│   Allowed    BLOCKED                                        │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## Files Involved

| File | Purpose |
|------|---------|
| `CLAUDE.md` | Project constitution for Claude Code sessions |
| `AGENTS.md` | Master workflow definition |
| `.githooks/pre-commit` | Git hook that triggers on commit |
| `scripts/precommit_guard.ps1` | PowerShell wrapper that runs review |
| `scripts/review-code.ps1` | AI-powered code review script |
| `scripts/install_git_hooks.ps1` | One-time hook installation script |

## Review Priority Levels

| Level | Meaning | Action |
|-------|---------|--------|
| P1 (Critical) | Bugs, Security, Crashes | **MUST FIX** - Commit blocked |
| P2 (Important) | Performance, UX, Stability | Recommended fix |
| P3 (Minor) | Style, Naming | Optional |

## Troubleshooting

### Commit is blocked but I need to force it

**Don't bypass the hook.** Fix the P1 issues first. The hook exists to prevent bugs and security issues from entering the codebase.

### Hook not running

Verify the hooks are installed:

```powershell
git config core.hooksPath
```

Should output: `.githooks`

If not, run the installation script again.

### Review script not found

Ensure you're in the project root and the scripts folder exists:

```powershell
ls scripts/review-code.ps1
```

## For Claude Code Users

When using Claude Code in this repository:

1. Claude reads `CLAUDE.md` at session start
2. Claude must run review scripts directly (not ask user to run them)
3. Claude must fix P1 issues before committing
4. Every Claude response should end with `WORKFLOW_STATUS` line

See `CLAUDE.md` for the complete contract.
