# MASP Multi-AI Code Review System

A parallel code review system using multiple AI reviewers (Codex CLI, Gemini CLI, GitHub Copilot CLI) with caching and JSON logging.

## Features

- **Parallel Execution**: Run multiple AI reviewers simultaneously
- **SHA256 Caching**: Skip duplicate reviews for unchanged code
- **JSON Logging**: Track review history with detailed logs
- **Weighted Consensus**: Configurable voting system for pass/fail decisions
- **P1 Critical Check**: Automatic fail on any security vulnerabilities

## Installation

### Prerequisites

1. **Codex CLI** (OpenAI)
   ```bash
   npm install -g @openai/codex-cli
   ```

2. **Gemini CLI** (Google)
   ```bash
   npm install -g @google/gemini-cli
   ```

3. **GitHub Copilot CLI**
   ```bash
   gh extension install github/gh-copilot
   ```

### Verify Installation

```powershell
codex --version
gemini --version
gh copilot --version
```

## Usage

### Basic Usage

```powershell
# Run code review
.\scripts\review-code.ps1

# Force re-review (ignore cache)
.\scripts\review-code.ps1 -SkipCache

# Verbose output
.\scripts\review-code.ps1 -Verbose

# Custom timeout (seconds)
.\scripts\review-code.ps1 -Timeout 600
```

### What It Reviews

The script automatically reviews:
1. Staged changes (`git diff --staged`)
2. If no staged changes, reviews last commit (`git diff HEAD~1`)

## Configuration

Edit `scripts/review-config.json` to customize:

```json
{
  "reviewers": {
    "codex": { "enabled": true, "weight": 1.0 },
    "gemini": { "enabled": true, "weight": 1.0 },
    "copilot": { "enabled": true, "weight": 0.8 }
  },
  "cache": {
    "enabled": true,
    "maxAgeHours": 24
  },
  "consensus": {
    "passThreshold": 0.5,
    "p1CriticalAny": true
  },
  "timeout": 300
}
```

### Reviewer Weights

- **1.0**: Full weight (Codex, Gemini)
- **0.8**: Reduced weight (Copilot - less precise for code review)

### Consensus Rules

- **passThreshold**: Minimum weighted pass ratio (0.5 = 50%)
- **p1CriticalAny**: Fail if any P1 (critical) issue found

## Severity Levels

| Level | Description | Example |
|-------|-------------|---------|
| P1 | Critical | Security vulnerabilities, data loss, crashes |
| P2 | Major | Bugs, performance issues, missing error handling |
| P3 | Minor | Style, naming, documentation |

## Output Files

### Cache Files

Location: `review-results/.cache/{hash}.json`

- 16-character SHA256 hash of diff content
- Valid for 24 hours (configurable)

### Log Files

Location: `review-results/YYYYMMDD_HHMMSS_{id}.json`

Example:
```json
{
  "id": "a1b2c3d4",
  "timestamp": "2025-01-20T18:00:00.000Z",
  "git": {
    "commit": "abc1234",
    "branch": "master"
  },
  "reviewers": [
    { "name": "Codex", "p1": 0, "p2": 1, "p3": 2 },
    { "name": "Gemini", "p1": 0, "p2": 0, "p3": 1 }
  ],
  "consensus": {
    "passed": true,
    "reason": "Consensus passed (100%)",
    "passRatio": 1.0
  },
  "duration": 5.2,
  "diffHash": "ABC123DEF456...",
  "diffLines": 150
}
```

## Cache Management

### Clear Cache

```powershell
# Clear all cache
Remove-Item review-results/.cache/* -Force

# Clear expired cache (older than 24h)
Get-ChildItem review-results/.cache/*.json |
  Where-Object { $_.LastWriteTime -lt (Get-Date).AddHours(-24) } |
  Remove-Item -Force
```

### View Cache Stats

```powershell
# Count cached reviews
(Get-ChildItem review-results/.cache/*.json).Count

# List recent cache files
Get-ChildItem review-results/.cache/*.json |
  Sort-Object LastWriteTime -Descending |
  Select-Object -First 5
```

## Log Management

### View Recent Logs

```powershell
# List recent review logs
Get-ChildItem review-results/*.json |
  Where-Object { $_.Name -notlike ".*" } |
  Sort-Object LastWriteTime -Descending |
  Select-Object -First 10

# View latest log content
Get-Content (Get-ChildItem review-results/*.json |
  Sort-Object LastWriteTime -Descending |
  Select-Object -First 1)
```

### Analyze Review History

```powershell
# Count passed vs failed reviews
Get-ChildItem review-results/*.json |
  Where-Object { $_.Name -notlike ".*" } |
  ForEach-Object { (Get-Content $_ | ConvertFrom-Json).consensus.passed } |
  Group-Object | Format-Table
```

## Troubleshooting

### "Command not found" Errors

Ensure CLI tools are in your PATH:
```powershell
$env:PATH -split ";" | Where-Object { $_ -like "*npm*" -or $_ -like "*gh*" }
```

### Cache Not Working

1. Check cache directory exists:
   ```powershell
   Test-Path review-results/.cache
   ```

2. Verify cache is enabled in config:
   ```powershell
   (Get-Content scripts/review-config.json | ConvertFrom-Json).cache.enabled
   ```

### Timeout Issues

Increase timeout for large diffs:
```powershell
.\scripts\review-code.ps1 -Timeout 600  # 10 minutes
```

### JSON Parsing Errors

If reviewers return non-JSON output, check:
1. CLI tool version compatibility
2. API authentication status
3. Network connectivity

## Integration with CI/CD

### GitHub Actions Example

```yaml
- name: Run AI Code Review
  shell: pwsh
  run: |
    $result = .\scripts\review-code.ps1
    if (-not $result.consensus.passed) {
      exit 1
    }
```

### Pre-commit Hook (Automated Setup)

MASP now includes automated workflow enforcement via Git hooks.

**One-time setup:**
```powershell
powershell -ExecutionPolicy Bypass -File scripts/install_git_hooks.ps1
```

This configures Git to use `.githooks/pre-commit` which automatically:
1. Runs `scripts/precommit_guard.ps1`
2. Executes `scripts/review-code.ps1`
3. Blocks commit if P1 > 0

See `docs/WORKFLOW_ENFORCEMENT.md` for details.

## Workflow Enforcement Scripts

| Script | Purpose |
|--------|---------|
| `install_git_hooks.ps1` | One-time setup to enable Git hooks |
| `precommit_guard.ps1` | Pre-commit wrapper that runs review |
| `review-code.ps1` | AI-powered code review |
| `review-plan.ps1` | AI-powered plan review |

## License

Part of MASP (Multi-Asset Strategy Platform)
# test
