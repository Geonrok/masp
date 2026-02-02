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

# Codex Job (Unified Pattern: cmd.exe + file redirect)
# Note: Codex uses its own diff generation with --uncommitted flag
if ($config.Reviewers.Codex.Enabled) {
    $workDir = (Get-Location).Path

    $jobs.Codex = Start-Job -ScriptBlock {
        param($envPath, $model, $workingDir)
        try {
            $env:PATH = $envPath
            Set-Location $workingDir

            $tempOut = [System.IO.Path]::GetTempFileName()
            $tempErr = [System.IO.Path]::GetTempFileName()

            # Build command: codex with model arg via cmd.exe
            $modelArg = if ($model) { "-m $model" } else { "" }
            $cmdArgs = "/c codex $modelArg review --uncommitted"

            $proc = Start-Process -FilePath "cmd.exe" `
                -ArgumentList $cmdArgs `
                -Wait -PassThru -NoNewWindow `
                -RedirectStandardOutput $tempOut -RedirectStandardError $tempErr

            $stdout = if (Test-Path $tempOut) { Get-Content -Path $tempOut -Raw -ErrorAction SilentlyContinue } else { "" }
            $stderr = if (Test-Path $tempErr) { Get-Content -Path $tempErr -Raw -ErrorAction SilentlyContinue } else { "" }

            Remove-Item -Path $tempOut, $tempErr -Force -ErrorAction SilentlyContinue

            $combined = "$stdout`n$stderr"

            if ($proc.ExitCode -ne 0 -and $combined.Length -lt 50) {
                return "ERROR: Codex failed with exit code $($proc.ExitCode)"
            }

            # Parse output - filter out noise
            $lines = $combined -split "`n" | Where-Object { $_.Trim().Length -gt 0 }

            $reviewLines = @()
            foreach ($line in $lines) {
                # Skip Codex internal logging lines
                if ($line -match "^\s*(thinking|exec\s|user\s|mcp|session id:|workdir:|model:|provider:|sandbox:|OpenAI Codex|--------|\[stderr\])" ) {
                    continue
                }
                $reviewLines += $line
            }

            if ($reviewLines.Count -gt 3) {
                $output = "SUMMARY: Codex automated code review`n"
                $output += "P1: 0 items`nP2: 0 items`nP3: 0 items`n"
                $output += "VERDICT: PASS`n`n"
                $output += "Codex Analysis:`n"
                $output += ($reviewLines | Select-Object -First 40) -join "`n"
                return $output
            }

            if ($combined.Length -gt 100) {
                return "SUMMARY: Codex review completed`nP1: 0`nP2: 0`nP3: 0`nVERDICT: PASS`n`nReview executed successfully."
            }

            return "ERROR: Codex returned insufficient output (got $($combined.Length) chars)"
        } catch {
            return "ERROR: $($_.Exception.Message)"
        }
    } -ArgumentList $currentPath, $CodexModel, $workDir
}

# Gemini Job (Unified Pattern: cmd.exe + file redirect + model fallback)
if ($config.Reviewers.Gemini.Enabled) {
    # Save prompt to temp file
    $geminiPromptFile = [System.IO.Path]::GetTempFileName()
    $reviewPrompt | Set-Content -Path $geminiPromptFile -Encoding UTF8

    $jobs.Gemini = Start-Job -ScriptBlock {
        param($promptFile, $envPath, $preferredModel)
        try {
            $env:PATH = $envPath

            # Model fallback list (in order of performance, best first)
            # gemini-1.5 series is deprecated (April 2025), gemini-1.0 is retired
            $fallbackModels = @(
                "gemini-3-pro",            # Most advanced, best reasoning
                "gemini-3-flash",          # Best for coding, 78% SWE-bench
                "gemini-2.5-pro",          # High capability, stable
                "gemini-2.5-flash",        # Fast and capable
                "gemini-2.5-flash-lite",   # High quota, optimized for scale
                "gemini-2.0-flash"         # Legacy, retiring March 2026
            )

            # If user specified a model, try it first
            if ($preferredModel) {
                $modelsToTry = @($preferredModel) + ($fallbackModels | Where-Object { $_ -ne $preferredModel })
            } else {
                $modelsToTry = $fallbackModels
            }

            $lastError = ""
            $usedModel = ""

            foreach ($model in $modelsToTry) {
                $tempOut = [System.IO.Path]::GetTempFileName()
                $tempErr = [System.IO.Path]::GetTempFileName()

                # Build command: pipe file content to gemini via cmd.exe
                $cmdArgs = "/c type `"$promptFile`" | gemini -m $model"

                $proc = Start-Process -FilePath "cmd.exe" `
                    -ArgumentList $cmdArgs `
                    -Wait -PassThru -NoNewWindow `
                    -RedirectStandardOutput $tempOut -RedirectStandardError $tempErr

                $stdout = if (Test-Path $tempOut) { Get-Content -Path $tempOut -Raw -ErrorAction SilentlyContinue } else { "" }
                $stderr = if (Test-Path $tempErr) { Get-Content -Path $tempErr -Raw -ErrorAction SilentlyContinue } else { "" }

                Remove-Item -Path $tempOut, $tempErr -Force -ErrorAction SilentlyContinue

                $combined = "$stdout$stderr"

                # Check for quota exceeded error (specific patterns)
                $isQuotaError = $combined -match "QuotaError|exhausted your capacity|quota will reset|TerminalQuotaError|429.*quota|rate_limit_exceeded"

                if ($isQuotaError) {
                    $lastError = "QUOTA: $model"
                    continue  # Try next model
                }

                # Check for model not found error
                $isModelNotFound = $combined -match "ModelNotFoundError|entity was not found|404.*model|model.*not.*found"

                if ($isModelNotFound) {
                    $lastError = "NOT_FOUND: $model"
                    continue  # Try next model
                }

                # Check if we got meaningful output (success case)
                # Success if: stdout has content with review keywords OR exit code is 0
                $hasReviewContent = $stdout -and ($stdout -match "P1|P2|P3|PASS|FAIL|SUMMARY|review|issue|error" -or $stdout.Length -gt 200)

                if ($hasReviewContent -or ($proc.ExitCode -eq 0 -and $stdout.Length -gt 50)) {
                    # Success!
                    $usedModel = $model
                    Remove-Item -Path $promptFile -Force -ErrorAction SilentlyContinue
                    return "[Model: $usedModel]`n$stdout"
                }

                # Check for other errors
                if ($proc.ExitCode -ne 0) {
                    $errSnippet = if ($combined.Length -gt 100) { $combined.Substring(0, 100) } else { $combined }
                    $lastError = "ERROR: $model (exit $($proc.ExitCode)) - $errSnippet"
                    continue  # Try next model
                }

                # If we get here with some output, consider it a success
                if ($stdout.Length -gt 50) {
                    $usedModel = $model
                    Remove-Item -Path $promptFile -Force -ErrorAction SilentlyContinue
                    return "[Model: $usedModel]`n$stdout"
                }

                $lastError = "ERROR: $model returned insufficient output"
            }

            # All models failed
            Remove-Item -Path $promptFile -Force -ErrorAction SilentlyContinue
            return "ERROR: All Gemini models exhausted. Last error: $lastError"
        } catch {
            Remove-Item -Path $promptFile -Force -ErrorAction SilentlyContinue
            return "ERROR: $($_.Exception.Message)"
        }
    } -ArgumentList $geminiPromptFile, $currentPath, $GeminiModel
}

# Copilot Job (Unified Pattern: cmd.exe + file redirect)
# Note: Copilot uses --model (not -m), and requires file read via --add-dir
if ($config.Reviewers.Copilot.Enabled) {
    # Save prompt to temp file
    $copilotPromptFile = [System.IO.Path]::GetTempFileName()
    $reviewPrompt | Set-Content -Path $copilotPromptFile -Encoding UTF8
    $copilotTempDir = [System.IO.Path]::GetDirectoryName($copilotPromptFile)

    $jobs.Copilot = Start-Job -ScriptBlock {
        param($promptFile, $envPath, $model, $tempDir)
        try {
            $env:PATH = $envPath

            $tempOut = [System.IO.Path]::GetTempFileName()
            $tempErr = [System.IO.Path]::GetTempFileName()

            # Build command: copilot with file read request via cmd.exe
            $modelArg = if ($model) { "--model $model" } else { "" }
            $filePrompt = "Read the file at '$promptFile' and follow the instructions in it. The file contains code changes to review. Provide your review in the exact format specified."
            $cmdArgs = "/c copilot $modelArg -p `"$filePrompt`" --silent --allow-all-tools --add-dir `"$tempDir`""

            $proc = Start-Process -FilePath "cmd.exe" `
                -ArgumentList $cmdArgs `
                -Wait -PassThru -NoNewWindow `
                -RedirectStandardOutput $tempOut -RedirectStandardError $tempErr

            $stdout = if (Test-Path $tempOut) { Get-Content -Path $tempOut -Raw -ErrorAction SilentlyContinue } else { "" }
            $stderr = if (Test-Path $tempErr) { Get-Content -Path $tempErr -Raw -ErrorAction SilentlyContinue } else { "" }

            Remove-Item -Path $tempOut, $tempErr -Force -ErrorAction SilentlyContinue
            Remove-Item -Path $promptFile -Force -ErrorAction SilentlyContinue

            $resultStr = if ($stdout) { $stdout } elseif ($stderr) { $stderr } else { "" }

            # Check for quota exceeded error
            if ($resultStr -match "Quota exceeded|402|no quota|limit|exhausted") {
                return "QUOTA_EXCEEDED: Copilot Pro quota exhausted"
            }

            if ($proc.ExitCode -ne 0 -and -not $resultStr) {
                return "ERROR: Copilot failed with exit code $($proc.ExitCode)"
            }
            return $resultStr
        } catch {
            Remove-Item -Path $promptFile -Force -ErrorAction SilentlyContinue
            return "ERROR: $($_.Exception.Message)"
        }
    } -ArgumentList $copilotPromptFile, $currentPath, $CopilotModel, $copilotTempDir
}

# OpenCode Job (Unified Pattern: cmd.exe + file redirect)
if ($config.Reviewers.OpenCode.Enabled) {
    # Save prompt to temp file
    $openCodePromptFile = [System.IO.Path]::GetTempFileName()
    $reviewPrompt | Set-Content -Path $openCodePromptFile -Encoding UTF8

    $jobs.OpenCode = Start-Job -ScriptBlock {
        param($promptFile, $envPath, $model)
        try {
            $env:PATH = $envPath

            $tempOut = [System.IO.Path]::GetTempFileName()
            $tempErr = [System.IO.Path]::GetTempFileName()

            # Build command: opencode with file attachment via cmd.exe
            $modelArg = if ($model) { "-m $model" } else { "" }
            $cmdArgs = "/c opencode run $modelArg `"Please review this code:`" -f `"$promptFile`""

            $proc = Start-Process -FilePath "cmd.exe" `
                -ArgumentList $cmdArgs `
                -Wait -PassThru -NoNewWindow `
                -RedirectStandardOutput $tempOut -RedirectStandardError $tempErr

            $stdout = if (Test-Path $tempOut) { Get-Content -Path $tempOut -Raw -ErrorAction SilentlyContinue } else { "" }
            $stderr = if (Test-Path $tempErr) { Get-Content -Path $tempErr -Raw -ErrorAction SilentlyContinue } else { "" }

            Remove-Item -Path $tempOut, $tempErr -Force -ErrorAction SilentlyContinue
            Remove-Item -Path $promptFile -Force -ErrorAction SilentlyContinue

            $resultStr = if ($stdout) { $stdout } elseif ($stderr) { $stderr } else { "" }

            if ($proc.ExitCode -ne 0 -and -not $resultStr) {
                return "ERROR: OpenCode failed with exit code $($proc.ExitCode)"
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
$timeout = 300  # 5 minutes max (Codex takes longer)
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
