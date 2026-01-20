<#
.SYNOPSIS
    Code Auto Review (Codex + Gemini Parallel)
.DESCRIPTION
    Reviews code changes using Codex and Gemini in parallel.
    Reviews staged changes or uncommitted changes.
.EXAMPLE
    .\scripts\review-code.ps1
#>

# Get Git diff (staged first, then uncommitted)
$diff = git diff --staged 2>$null
if (-not $diff) {
    $diff = git diff 2>$null
}

if (-not $diff) {
    Write-Host "No changes to review" -ForegroundColor Yellow
    exit 0
}

$prompt = @"
Please review the following code changes.

## Review Criteria
- P1 (Critical): Bugs, security vulnerabilities, crash risks, P0 rule violations
- P2 (Important): Performance issues, lack of error handling, missing tests
- P3 (Minor): Naming, style, comments

## P0 Rules (MASP Project - NEVER VIOLATE)
- No direct requests usage -> Use ConfigApiClient
- No direct KeyManager import
- No hardcoded API keys
- MASP_ENABLE_LIVE_TRADING default value 0 (Paper Trading)

## Output Format
P1: N items
- [P1-1] file:line - Issue description

P2: N items
- [P2-1] file:line - Issue description

P3: N items
- [P3-1] file:line - Issue description

## Code Changes
$diff
"@

Write-Host ""
Write-Host "=== Code Review ===" -ForegroundColor Cyan
Write-Host "Starting Codex + Gemini parallel review..." -ForegroundColor Gray
Write-Host ""

# Parallel execution
$codexJob = Start-Job -ScriptBlock {
    try {
        $result = codex review --uncommitted 2>&1
        return $result
    } catch {
        return "Error: $_"
    }
}

$geminiJob = Start-Job -ScriptBlock {
    param($p)
    try {
        $result = echo $p | gemini 2>&1
        return $result
    } catch {
        return "Error: $_"
    }
} -ArgumentList $prompt

# Progress indicator
Write-Host "Review in progress" -NoNewline -ForegroundColor Gray
while ($codexJob.State -eq "Running" -or $geminiJob.State -eq "Running") {
    Start-Sleep -Seconds 1
    Write-Host "." -NoNewline -ForegroundColor DarkGray
}
Write-Host " Done!" -ForegroundColor Green

# Collect results
$codexResult = Receive-Job $codexJob
$geminiResult = Receive-Job $geminiJob

Remove-Job $codexJob, $geminiJob -Force

# Output results
Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "=== Codex Review ===" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host $codexResult

Write-Host ""
Write-Host "========================================" -ForegroundColor Green
Write-Host "=== Gemini Review ===" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green
Write-Host $geminiResult

Write-Host ""
Write-Host "=== Review Complete ===" -ForegroundColor Cyan
