<#
.SYNOPSIS
    Runs parallel code reviews using Codex and Gemini.

.DESCRIPTION
    This script executes Codex and Gemini in parallel jobs to review code changes.
    It supports reviewing uncommitted changes, staged changes, or specific files.
    Results are saved to the specified output directory and a summary report is generated.

.PARAMETER Target
    The target for review. Defaults to "--uncommitted".
    Can be "--staged" or a file path.

.PARAMETER OutputDir
    The directory to save review results. Defaults to "./review-results".

.PARAMETER Quiet
    If set, minimizes console output.

.EXAMPLE
    .\scripts\review-parallel.ps1
    .\scripts\review-parallel.ps1 -Target --staged
    .\scripts\review-parallel.ps1 -Target "services/market_data.py" -Quiet
#>

param (
    [string]$Target = "--uncommitted",
    [string]$OutputDir = "./review-results",
    [switch]$Quiet
)

# --- Configuration ---
$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
if (!(Test-Path $OutputDir)) { New-Item -ItemType Directory -Path $OutputDir | Out-Null }

$codexFile = Join-Path $OutputDir "codex_review_$timestamp.md"
$geminiFile = Join-Path $OutputDir "gemini_review_$timestamp.md"
$summaryFile = Join-Path $OutputDir "review_summary_$timestamp.md"

# --- Helper Functions ---
function Write-Color([string]$text, [ConsoleColor]$color = "White") {
    if (-not $Quiet) { Write-Host $text -ForegroundColor $color }
}

function Get-GitDiff([string]$target) {
    if ($target -eq "--uncommitted") {
        return git diff
    } elseif ($target -eq "--staged") {
        return git diff --staged
    } else {
        return git diff $target
    }
}

# --- Main Execution ---
Write-Color "`n=== MASP Parallel Dual Review System ($timestamp) ===" "Cyan"
Write-Color "Target: $Target" "Gray"
Write-Color "Output: $OutputDir" "Gray"

# Define ScriptBlocks for Jobs
$codexScript = {
    param($t, $out)
    try {
        # Execute Codex Review
        $result = codex review $t 2>&1
        $result | Out-File -FilePath $out -Encoding UTF8
        return "Success"
    } catch {
        return "Error: $_ "
    }
}

$geminiScript = {
    param($t, $out)
    try {
        # Get Diff Content
        $diffContent = if ($t -eq "--uncommitted") { git diff } elseif ($t -eq "--staged") { git diff --staged } else { git diff $t }
        
        if ([string]::IsNullOrWhiteSpace($diffContent)) {
            "No changes detected for review." | Out-File -FilePath $out -Encoding UTF8
            return "Skipped (No changes)"
        }

        # Gemini Prompt
        $prompt = @"
You are a senior code reviewer for the MASP project (Python 3.11).
Review the following code diff based on these criteria:

1. **Security:** Check for hardcoded keys, injection risks, insecure imports.
2. **Architecture:** Verify separation of concerns, proper use of services.
3. **Robustness:** Edge cases, error handling, input validation.
4. **Performance:** identifying bottlenecks or inefficient loops.
5. **Best Practices:** PEP 8 compliance, docstrings, typing.

**Output Format:**
Please provide feedback using the following priority levels:
- **[P1] Critical:** Must fix (Bugs, Security).
- **[P2] Warning:** Strongly recommended (Performance, Resilience).
- **[P3] Suggestion:** Optional (Style, Naming).

If the code is good, simply state "LGTM" (Looks Good To Me).

Code Diff:
$diffContent
"@

        # Call Gemini (Assuming 'gemini' CLI tool exists and accepts stdin or prompt arg)
        # Adjust command based on actual environment availability.
        # Here we simulate piping to a generic gemini CLI or echo if not available for demo.
        
        # Note: In a real environment, replace this with the actual CLI call.
        # Example: $prompt | gemini chat > $out
        
        # Checking if 'gemini' command exists, else fail gracefully or use a placeholder
        if (Get-Command "gemini" -ErrorAction SilentlyContinue) {
             $prompt | gemini chat | Out-File -FilePath $out -Encoding UTF8
        } else {
             # Fallback if CLI not installed/aliased (for safety in this script generation)
             "Error: 'gemini' CLI command not found. Please alias it or install it." | Out-File -FilePath $out -Encoding UTF8
             return "Failed (CLI missing)"
        }

        return "Success"
    } catch {
        return "Error: $_ "
    }
}

# Start Jobs
Write-Color "`n[+] Starting Review Agents..." "Green"

$jobCodex = Start-Job -ScriptBlock $codexScript -ArgumentList $Target, $codexFile -Name "CodexReview"
$jobGemini = Start-Job -ScriptBlock $geminiScript -ArgumentList $Target, $geminiFile -Name "GeminiReview"

# Wait for Jobs
while ($jobCodex.State -eq "Running" -or $jobGemini.State -eq "Running") {
    Start-Sleep -Seconds 1
    Write-Host "." -NoNewline -ForegroundColor DarkGray
}
Write-Host ""

# Receive Results
$resCodex = Receive-Job -Job $jobCodex
$resGemini = Receive-Job -Job $jobGemini

# Generate Summary
Write-Color "`n[+] Generating Summary Report..." "Green"

$summaryContent = @"
# Review Summary Report ($timestamp)

## Status
| Agent | Status | Output File |
|-------|--------|-------------|
| Codex | $resCodex | $codexFile |
| Gemini| $resGemini| $geminiFile |

## Codex Summary (Snippet)
$(if (Test-Path $codexFile) { (Get-Content $codexFile | Select-Object -First 20) -join "`n" } else { "N/A" })
... (See full file)

## Gemini Summary (Snippet)
$(if (Test-Path $geminiFile) { (Get-Content $geminiFile | Select-Object -First 20) -join "`n" } else { "N/A" })
... (See full file)

## Next Steps
1. **P1 (Critical):** Fix immediately.
2. **P2 (Warning):** Evaluate and fix if impact is high.
3. **Tests:** Run 'pytest' to ensure no regressions.
4. **Commit:** Only commit if P1s are resolved and tests pass.
"@

$summaryContent | Out-File -FilePath $summaryFile -Encoding UTF8

# Cleanup
Remove-Job -Job $jobCodex
Remove-Job -Job $jobGemini

# Final Output
if ($resCodex -eq "Success" -and $resGemini -eq "Success") {
    Write-Color "Review Completed Successfully!" "Green"
} else {
    Write-Color "Review Completed with Errors." "Red"
}
Write-Color "Summary: $summaryFile" "Cyan"

