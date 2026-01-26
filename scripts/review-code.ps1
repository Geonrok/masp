<#
.SYNOPSIS
    MASP Multi-AI Code Review Script (Codex + Gemini + Copilot + OpenCode)
.DESCRIPTION
    Reviews code changes using four AI CLI tools in parallel:
    - OpenAI Codex CLI (codex review --uncommitted)
    - Google Gemini CLI (gemini "prompt")
    - GitHub Copilot CLI (copilot -p "prompt" --silent --allow-all-tools)
    - OpenCode CLI (opencode run "prompt")

    Features:
    - Parallel execution for speed
    - Auto-fallback: failed reviewers are excluded from consensus
    - Weighted consensus algorithm
    - JSON logging with timestamps
    - P0/P1/P2/P3 severity classification
.EXAMPLE
    .\review-code.ps1
.EXAMPLE
    .\review-code.ps1 -SkipCopilot
.EXAMPLE
    .\review-code.ps1 -SkipOpenCode
.EXAMPLE
    .\review-code.ps1 -GeminiModel "gemini-2.0-flash" -OpenCodeModel "anthropic/claude-sonnet-4-20250514"
.PARAMETER CodexModel
    Model for Codex CLI (e.g., "o3", "o4-mini")
.PARAMETER GeminiModel
    Model for Gemini CLI (e.g., "gemini-2.0-flash", "gemini-2.5-pro")
.PARAMETER CopilotModel
    Model for Copilot CLI (e.g., "gpt-4o", "claude-sonnet-4-20250514")
.PARAMETER OpenCodeModel
    Model for OpenCode CLI in provider/model format (e.g., "anthropic/claude-sonnet-4-20250514", "openai/gpt-4o")
.NOTES
    Requires: codex, gemini, copilot, opencode CLI tools installed and authenticated

    Environment Variables (for global defaults):
    - MASP_CODEX_MODEL    : Default model for Codex
    - MASP_GEMINI_MODEL   : Default model for Gemini
    - MASP_COPILOT_MODEL  : Default model for Copilot
    - MASP_OPENCODE_MODEL : Default model for OpenCode
#>

param(
    [switch]$SkipCopilot,
    [switch]$SkipGemini,
    [switch]$SkipCodex,
    [switch]$SkipOpenCode,
    [switch]$Verbose,
    [string]$CodexModel = $env:MASP_CODEX_MODEL,
    [string]$GeminiModel = $env:MASP_GEMINI_MODEL,
    [string]$CopilotModel = $env:MASP_COPILOT_MODEL,
    [string]$OpenCodeModel = $env:MASP_OPENCODE_MODEL
)

# ============================================================
# Configuration
# ============================================================
$config = @{
    Reviewers = @{
        Codex = @{
            Enabled = -not $SkipCodex
            Weight = 1.0
            Timeout = 120
        }
        Gemini = @{
            Enabled = -not $SkipGemini
            Weight = 1.0
            Timeout = 120
        }
        Copilot = @{
            Enabled = -not $SkipCopilot
            Weight = 1.0
            Timeout = 120
        }
        OpenCode = @{
            Enabled = -not $SkipOpenCode
            Weight = 1.0
            Timeout = 120
        }
    }
    LogDir = "$env:USERPROFILE\.masp\review-logs"
}

# Ensure log directory exists
if (-not (Test-Path $config.LogDir)) {
    New-Item -ItemType Directory -Path $config.LogDir -Force | Out-Null
}

# ============================================================
# Get Git Diff
# ============================================================
$diff = git diff --staged 2>$null
if (-not $diff) {
    $diff = git diff 2>$null
}

if (-not $diff) {
    Write-Host "No changes to review" -ForegroundColor Yellow
    exit 0
}

$diffLineCount = ($diff -split "`n").Count
$projectName = (Get-Item -Path (git rev-parse --show-toplevel 2>$null) -ErrorAction SilentlyContinue).Name
if (-not $projectName) { $projectName = "Unknown" }

# ============================================================
# Review Prompt
# ============================================================
$reviewPrompt = @"
Please review the following code changes.

## Review Criteria
- P1 (Critical): Bugs, security vulnerabilities, crash risks, P0 rule violations
- P2 (Important): Performance issues, lack of error handling, missing tests
- P3 (Minor): Naming, style, comments

## P0 Rules (MASP Project - NEVER VIOLATE)
- No direct requests usage -> Use ConfigApiClient
- No direct KeyManager import
- No hardcoded API keys
- MASP_ENABLE_LIVE_TRADING default value must be 0 (Paper Trading)

## Output Format
Respond with this exact format:

SUMMARY: [One line summary of changes]

P1: [count] items
- [P1-1] file:line - Issue description

P2: [count] items
- [P2-1] file:line - Issue description

P3: [count] items
- [P3-1] file:line - Issue description

VERDICT: [PASS/FAIL] - [reason]

## Code Changes
$diff
"@

# ============================================================
# Display Header
# ============================================================
Write-Host ""
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "  MASP Multi-AI Code Review" -ForegroundColor Cyan
Write-Host "  Project: $projectName" -ForegroundColor Gray
Write-Host "  Changes: $diffLineCount lines" -ForegroundColor Gray
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""

$enabledReviewers = @()
if ($config.Reviewers.Codex.Enabled) { $enabledReviewers += "Codex" }
if ($config.Reviewers.Gemini.Enabled) { $enabledReviewers += "Gemini" }
if ($config.Reviewers.Copilot.Enabled) { $enabledReviewers += "Copilot" }
if ($config.Reviewers.OpenCode.Enabled) { $enabledReviewers += "OpenCode" }

Write-Host "Reviewers: $($enabledReviewers -join ', ')" -ForegroundColor Gray

# Display model info if specified
$modelInfo = @()
if ($CodexModel) { $modelInfo += "Codex:$CodexModel" }
if ($GeminiModel) { $modelInfo += "Gemini:$GeminiModel" }
if ($CopilotModel) { $modelInfo += "Copilot:$CopilotModel" }
if ($OpenCodeModel) { $modelInfo += "OpenCode:$OpenCodeModel" }
if ($modelInfo.Count -gt 0) {
    Write-Host "Models: $($modelInfo -join ', ')" -ForegroundColor Gray
} else {
    Write-Host "Models: (using defaults)" -ForegroundColor DarkGray
}
Write-Host ""

# ============================================================
# Parallel Execution
# ============================================================
$jobs = @{}
$startTime = Get-Date

# Capture PATH for job inheritance
$currentPath = $env:PATH

# Codex Job
if ($config.Reviewers.Codex.Enabled) {
    $jobs.Codex = Start-Job -ScriptBlock {
        param($envPath, $model)
        try {
            $env:PATH = $envPath
            $cmd = "codex review --uncommitted"
            if ($model) { $cmd += " -m $model" }
            $result = Invoke-Expression $cmd 2>&1
            if ($LASTEXITCODE -ne 0 -and -not $result) {
                return "ERROR: Codex failed with exit code $LASTEXITCODE"
            }
            return $result -join "`n"
        } catch {
            return "ERROR: $($_.Exception.Message)"
        }
    } -ArgumentList $currentPath, $CodexModel
}

# Gemini Job
if ($config.Reviewers.Gemini.Enabled) {
    # Save prompt to temp file for Gemini (handles long prompts via stdin)
    $geminiPromptFile = [System.IO.Path]::GetTempFileName()
    $reviewPrompt | Set-Content -Path $geminiPromptFile -Encoding UTF8

    $jobs.Gemini = Start-Job -ScriptBlock {
        param($promptFile, $envPath, $model)
        try {
            # Inherit PATH for npm global binaries
            $env:PATH = $envPath

            # Gemini CLI: pipe prompt via stdin with -p flag
            $modelArg = if ($model) { "-m $model" } else { "" }
            $result = Get-Content -Path $promptFile -Raw | Invoke-Expression "gemini $modelArg -p `"Review the code:`"" 2>&1
            $resultStr = $result -join "`n"

            # Cleanup temp file
            Remove-Item -Path $promptFile -Force -ErrorAction SilentlyContinue

            if ($LASTEXITCODE -ne 0 -and -not $result) {
                return "ERROR: Gemini failed with exit code $LASTEXITCODE"
            }
            return $resultStr
        } catch {
            Remove-Item -Path $promptFile -Force -ErrorAction SilentlyContinue
            return "ERROR: $($_.Exception.Message)"
        }
    } -ArgumentList $geminiPromptFile, $currentPath, $GeminiModel
}

# Copilot Job
if ($config.Reviewers.Copilot.Enabled) {
    $jobs.Copilot = Start-Job -ScriptBlock {
        param($prompt, $envPath, $model)
        try {
            $env:PATH = $envPath

            # Copilot CLI: -p for non-interactive, --silent for clean output
            $modelArg = if ($model) { "-m $model" } else { "" }
            $result = Invoke-Expression "copilot $modelArg -p `"$prompt`" --silent --allow-all-tools" 2>&1
            $resultStr = $result -join "`n"

            # Check for quota exceeded error
            if ($resultStr -match "Quota exceeded|402|no quota") {
                return "QUOTA_EXCEEDED: Copilot Pro quota exhausted"
            }

            if ($LASTEXITCODE -ne 0 -and -not $result) {
                return "ERROR: Copilot failed with exit code $LASTEXITCODE"
            }
            return $resultStr
        } catch {
            return "ERROR: $($_.Exception.Message)"
        }
    } -ArgumentList $reviewPrompt, $currentPath, $CopilotModel
}

# OpenCode Job
if ($config.Reviewers.OpenCode.Enabled) {
    # Save prompt to temp file for OpenCode (handles long prompts better)
    $openCodePromptFile = [System.IO.Path]::GetTempFileName()
    $reviewPrompt | Set-Content -Path $openCodePromptFile -Encoding UTF8

    $jobs.OpenCode = Start-Job -ScriptBlock {
        param($promptFile, $envPath, $model)
        try {
            $env:PATH = $envPath

            # OpenCode CLI: run with file attachment for long prompts
            $modelArg = if ($model) { "-m $model" } else { "" }
            $result = Invoke-Expression "opencode run $modelArg `"Please review this code:`" -f $promptFile" 2>&1
            $resultStr = $result -join "`n"

            # Cleanup temp file
            Remove-Item -Path $promptFile -Force -ErrorAction SilentlyContinue

            if ($LASTEXITCODE -ne 0 -and -not $result) {
                return "ERROR: OpenCode failed with exit code $LASTEXITCODE"
            }
            return $resultStr
        } catch {
            Remove-Item -Path $promptFile -Force -ErrorAction SilentlyContinue
            return "ERROR: $($_.Exception.Message)"
        }
    } -ArgumentList $openCodePromptFile, $currentPath, $OpenCodeModel
}

# Progress Indicator
Write-Host "Review in progress" -NoNewline -ForegroundColor Gray
$timeout = 180  # 3 minutes max
$elapsed = 0

while ($jobs.Values | Where-Object { $_.State -eq "Running" }) {
    Start-Sleep -Seconds 2
    Write-Host "." -NoNewline -ForegroundColor DarkGray
    $elapsed += 2
    
    if ($elapsed -ge $timeout) {
        Write-Host " Timeout!" -ForegroundColor Red
        break
    }
}
Write-Host " Done!" -ForegroundColor Green

$duration = ((Get-Date) - $startTime).TotalSeconds

# ============================================================
# Collect Results
# ============================================================
$results = @{}
$successfulReviewers = @()
$failedReviewers = @()

foreach ($reviewer in $jobs.Keys) {
    $job = $jobs[$reviewer]
    $result = Receive-Job $job -ErrorAction SilentlyContinue
    Remove-Job $job -Force -ErrorAction SilentlyContinue
    
    if ($result -match "^ERROR:|^QUOTA_EXCEEDED:") {
        $failedReviewers += $reviewer
        $results[$reviewer] = @{
            Success = $false
            Output = $result
            Issues = @{ P1 = 0; P2 = 0; P3 = 0 }
            Verdict = "SKIPPED"
        }
    } else {
        $successfulReviewers += $reviewer
        
        # Parse P1/P2/P3 counts
        $p1Count = if ($result -match "P1:\s*(\d+)") { [int]$matches[1] } else { 0 }
        $p2Count = if ($result -match "P2:\s*(\d+)") { [int]$matches[1] } else { 0 }
        $p3Count = if ($result -match "P3:\s*(\d+)") { [int]$matches[1] } else { 0 }
        
        # Parse verdict
        $verdict = "UNKNOWN"
        if ($result -match "VERDICT:\s*(PASS|FAIL)") {
            $verdict = $matches[1]
        } elseif ($p1Count -eq 0) {
            $verdict = "PASS"
        } else {
            $verdict = "FAIL"
        }
        
        $results[$reviewer] = @{
            Success = $true
            Output = $result
            Issues = @{ P1 = $p1Count; P2 = $p2Count; P3 = $p3Count }
            Verdict = $verdict
        }
    }
}

# ============================================================
# Display Results
# ============================================================
foreach ($reviewer in $results.Keys) {
    $r = $results[$reviewer]
    
    if ($r.Success) {
        $color = switch ($reviewer) {
            "Codex" { "Cyan" }
            "Gemini" { "Green" }
            "Copilot" { "Magenta" }
            "OpenCode" { "Yellow" }
            default { "White" }
        }
        
        Write-Host ""
        Write-Host "========================================" -ForegroundColor $color
        Write-Host "  $reviewer Review" -ForegroundColor $color
        Write-Host "  P1: $($r.Issues.P1) | P2: $($r.Issues.P2) | P3: $($r.Issues.P3)" -ForegroundColor Gray
        Write-Host "  Verdict: $($r.Verdict)" -ForegroundColor $(if ($r.Verdict -eq "PASS") { "Green" } else { "Red" })
        Write-Host "========================================" -ForegroundColor $color
        Write-Host $r.Output
    } else {
        Write-Host ""
        Write-Host "========================================" -ForegroundColor DarkGray
        Write-Host "  $reviewer Review - SKIPPED" -ForegroundColor DarkGray
        Write-Host "  Reason: $($r.Output)" -ForegroundColor Yellow
        Write-Host "========================================" -ForegroundColor DarkGray
    }
}

# ============================================================
# Consensus Calculation
# ============================================================
Write-Host ""
Write-Host "============================================================" -ForegroundColor White
Write-Host "  CONSENSUS RESULT" -ForegroundColor White
Write-Host "============================================================" -ForegroundColor White

if ($successfulReviewers.Count -eq 0) {
    Write-Host "  No successful reviews - cannot determine consensus" -ForegroundColor Red
    $consensusVerdict = "ERROR"
    $consensusPassed = $false
} else {
    $passCount = ($results.Values | Where-Object { $_.Success -and $_.Verdict -eq "PASS" }).Count
    $failCount = ($results.Values | Where-Object { $_.Success -and $_.Verdict -eq "FAIL" }).Count
    $totalP1 = ($results.Values | Where-Object { $_.Success } | ForEach-Object { $_.Issues.P1 } | Measure-Object -Sum).Sum
    
    Write-Host "  Successful Reviewers: $($successfulReviewers -join ', ')" -ForegroundColor Gray
    if ($failedReviewers.Count -gt 0) {
        Write-Host "  Skipped Reviewers: $($failedReviewers -join ', ')" -ForegroundColor Yellow
    }
    Write-Host ""
    Write-Host "  PASS votes: $passCount" -ForegroundColor Green
    Write-Host "  FAIL votes: $failCount" -ForegroundColor Red
    Write-Host "  Total P1 issues: $totalP1" -ForegroundColor $(if ($totalP1 -gt 0) { "Red" } else { "Green" })
    
    # Consensus: PASS if majority pass AND no P1 issues
    if ($totalP1 -gt 0) {
        $consensusVerdict = "FAIL"
        $consensusPassed = $false
        Write-Host ""
        Write-Host "  >> CONSENSUS: FAIL (P1 issues found)" -ForegroundColor Red
    } elseif ($passCount -ge $failCount) {
        $consensusVerdict = "PASS"
        $consensusPassed = $true
        Write-Host ""
        Write-Host "  >> CONSENSUS: PASS" -ForegroundColor Green
    } else {
        $consensusVerdict = "FAIL"
        $consensusPassed = $false
        Write-Host ""
        Write-Host "  >> CONSENSUS: FAIL (majority failed)" -ForegroundColor Red
    }
}

Write-Host "============================================================" -ForegroundColor White
Write-Host "  Duration: $([math]::Round($duration, 2))s" -ForegroundColor Gray
Write-Host "============================================================" -ForegroundColor White

# ============================================================
# Save Log
# ============================================================
$logEntry = @{
    timestamp = (Get-Date -Format "o")
    project = $projectName
    diffLines = $diffLineCount
    reviewers = @{
        enabled = $enabledReviewers
        successful = $successfulReviewers
        failed = $failedReviewers
    }
    models = @{
        codex = if ($CodexModel) { $CodexModel } else { "default" }
        gemini = if ($GeminiModel) { $GeminiModel } else { "default" }
        copilot = if ($CopilotModel) { $CopilotModel } else { "default" }
        opencode = if ($OpenCodeModel) { $OpenCodeModel } else { "default" }
    }
    results = @{}
    consensus = @{
        verdict = $consensusVerdict
        passed = $consensusPassed
        passVotes = $passCount
        failVotes = $failCount
        totalP1 = $totalP1
    }
    duration = [math]::Round($duration, 2)
}

foreach ($reviewer in $results.Keys) {
    $logEntry.results[$reviewer] = @{
        success = $results[$reviewer].Success
        issues = $results[$reviewer].Issues
        verdict = $results[$reviewer].Verdict
    }
}

$logFileName = "review_$(Get-Date -Format 'yyyyMMdd_HHmmss')_$($projectName -replace '[^a-zA-Z0-9]', '_').json"
$logPath = Join-Path $config.LogDir $logFileName

try {
    $logEntry | ConvertTo-Json -Depth 10 | Set-Content -Path $logPath -Encoding UTF8
    Write-Host ""
    Write-Host "Log saved: $logPath" -ForegroundColor DarkGray
} catch {
    Write-Host "Failed to save log: $_" -ForegroundColor Yellow
}

# ============================================================
# Exit Code
# ============================================================
if ($consensusPassed) {
    exit 0
} else {
    exit 1
}
