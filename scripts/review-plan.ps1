<#
.SYNOPSIS
    Plan/Design Auto Review (Codex + Gemini Parallel)
.DESCRIPTION
    Reviews implementation plans using Codex and Gemini in parallel.
    Classifies issues as P1/P2/P3 severity levels.
.PARAMETER Content
    Plan content to review (string or file path)
.EXAMPLE
    .\scripts\review-plan.ps1 -Content "API endpoint addition plan..."
    .\scripts\review-plan.ps1 -Content ".\docs\plan.md"
#>
param(
    [Parameter(Mandatory=$true)]
    [string]$Content
)

# Read file if path provided
if (Test-Path $Content) {
    $Content = Get-Content $Content -Raw -Encoding UTF8
}

$prompt = @"
Please review the following implementation plan.

## Review Criteria
- P1 (Critical): Serious design flaws, security issues, missing essential features
- P2 (Important): Performance concerns, scalability issues, recommended improvements
- P3 (Minor): Naming, documentation, style suggestions

## P0 Rules (MASP Project)
- No direct requests usage -> Use ConfigApiClient
- No direct KeyManager import
- No hardcoded API keys
- MASP_ENABLE_LIVE_TRADING default value 0 (Paper Trading)

## Output Format
P1: N items
- [P1-1] Issue description
- [P1-2] Issue description

P2: N items
- [P2-1] Issue description

P3: N items
- [P3-1] Issue description

## Plan Content
$Content
"@

Write-Host ""
Write-Host "=== Plan Review ===" -ForegroundColor Cyan
Write-Host "Starting Codex + Gemini parallel review..." -ForegroundColor Gray
Write-Host ""

# Parallel execution
$codexJob = Start-Job -ScriptBlock {
    param($p)
    try {
        $result = echo $p | codex chat 2>&1
        return $result
    } catch {
        return "Error: $_"
    }
} -ArgumentList $prompt

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
