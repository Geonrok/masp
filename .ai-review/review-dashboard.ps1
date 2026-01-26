<#
.SYNOPSIS
    AI Review Dashboard - CLI-based review status dashboard.

.DESCRIPTION
    Displays a visual summary of review status for recent commits.

.PARAMETER Commits
    Number of recent commits to show (default: 5)

.PARAMETER Watch
    Continuously update the dashboard

.EXAMPLE
    .\review-dashboard.ps1
    .\review-dashboard.ps1 -Commits 10
    .\review-dashboard.ps1 -Watch
#>

param(
    [int]$Commits = 5,
    [switch]$Watch,
    [switch]$Detailed
)

$ErrorActionPreference = "SilentlyContinue"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ReviewsDir = Join-Path $ScriptDir "reviews"
$ConfigFile = Join-Path $ScriptDir "review-config.json"

function Get-Config {
    if (Test-Path $ConfigFile) {
        return Get-Content $ConfigFile -Raw | ConvertFrom-Json
    }
    return $null
}

function Get-RecentCommits {
    param([int]$Count)

    $commits = git log --oneline -n $Count 2>$null
    $result = @()

    foreach ($line in $commits) {
        if ($line -match '^(\w+)\s+(.+)$') {
            $result += @{
                hash = $Matches[1]
                message = $Matches[2]
            }
        }
    }

    return $result
}

function Get-ReviewSummary {
    param([string]$CommitHash)

    $summaryFile = Join-Path $ReviewsDir "$CommitHash-summary.json"
    if (Test-Path $summaryFile) {
        return Get-Content $summaryFile -Raw | ConvertFrom-Json
    }
    return $null
}

function Get-StatusEmoji {
    param([string]$Status)

    switch ($Status) {
        "PASS" { return "[PASS]" }
        "CHANGES_REQUESTED" { return "[WARN]" }
        "BLOCKED" { return "[FAIL]" }
        "NOT_REVIEWED" { return "[----]" }
        default { return "[????]" }
    }
}

function Get-StatusColor {
    param([string]$Status)

    switch ($Status) {
        "PASS" { return "Green" }
        "CHANGES_REQUESTED" { return "Yellow" }
        "BLOCKED" { return "Red" }
        "NOT_REVIEWED" { return "Gray" }
        default { return "White" }
    }
}

function Show-Dashboard {
    Clear-Host

    $config = Get-Config
    $commits = Get-RecentCommits -Count $Commits

    Write-Host ""
    Write-Host "  ╔══════════════════════════════════════════════════════════════════╗" -ForegroundColor Cyan
    Write-Host "  ║           MASP Code Review Dashboard                             ║" -ForegroundColor Cyan
    Write-Host "  ╚══════════════════════════════════════════════════════════════════╝" -ForegroundColor Cyan
    Write-Host ""

    # Current branch info
    $branch = git branch --show-current 2>$null
    $lastCommit = git log -1 --format="%h %s" 2>$null
    Write-Host "  Branch: $branch" -ForegroundColor White
    Write-Host "  Latest: $lastCommit" -ForegroundColor White
    Write-Host ""

    # Reviewer legend
    Write-Host "  ┌─────────────────────────────────────────────────────────────────┐" -ForegroundColor DarkGray
    Write-Host "  │ Reviewers: " -NoNewline -ForegroundColor DarkGray
    Write-Host "Codex " -NoNewline -ForegroundColor Blue
    Write-Host "│ " -NoNewline -ForegroundColor DarkGray
    Write-Host "Gemini " -NoNewline -ForegroundColor Magenta
    Write-Host "│ " -NoNewline -ForegroundColor DarkGray
    Write-Host "OpenCode " -NoNewline -ForegroundColor Green
    Write-Host "│ " -NoNewline -ForegroundColor DarkGray
    Write-Host "Copilot" -ForegroundColor Yellow
    Write-Host "  └─────────────────────────────────────────────────────────────────┘" -ForegroundColor DarkGray
    Write-Host ""

    # Status legend
    Write-Host "  Status: " -NoNewline -ForegroundColor White
    Write-Host "[PASS] " -NoNewline -ForegroundColor Green
    Write-Host "[WARN] " -NoNewline -ForegroundColor Yellow
    Write-Host "[FAIL] " -NoNewline -ForegroundColor Red
    Write-Host "[----]" -ForegroundColor Gray
    Write-Host ""

    # Header
    Write-Host "  ┌─────────┬──────────────────────────────┬────────┬────────┬────────┬────────┬─────────┐" -ForegroundColor DarkGray
    Write-Host "  │ Commit  │ Message                      │ Codex  │ Gemini │OpenCode│ Copilot│ Overall │" -ForegroundColor DarkGray
    Write-Host "  ├─────────┼──────────────────────────────┼────────┼────────┼────────┼────────┼─────────┤" -ForegroundColor DarkGray

    foreach ($commit in $commits) {
        $summary = Get-ReviewSummary -CommitHash $commit.hash

        $message = $commit.message
        if ($message.Length -gt 28) {
            $message = $message.Substring(0, 25) + "..."
        }
        $message = $message.PadRight(28)

        Write-Host "  │ " -NoNewline -ForegroundColor DarkGray
        Write-Host "$($commit.hash)" -NoNewline -ForegroundColor Cyan
        Write-Host " │ " -NoNewline -ForegroundColor DarkGray
        Write-Host "$message" -NoNewline -ForegroundColor White
        Write-Host " │ " -NoNewline -ForegroundColor DarkGray

        if ($summary) {
            # Codex
            $codexStatus = if ($summary.reviewers.codex) { $summary.reviewers.codex } else { "----" }
            $codexColor = Get-StatusColor -Status $codexStatus
            Write-Host "$(Get-StatusEmoji $codexStatus)" -NoNewline -ForegroundColor $codexColor
            Write-Host " │ " -NoNewline -ForegroundColor DarkGray

            # Gemini
            $geminiStatus = if ($summary.reviewers.gemini) { $summary.reviewers.gemini } else { "----" }
            $geminiColor = Get-StatusColor -Status $geminiStatus
            Write-Host "$(Get-StatusEmoji $geminiStatus)" -NoNewline -ForegroundColor $geminiColor
            Write-Host " │ " -NoNewline -ForegroundColor DarkGray

            # OpenCode
            $opencodeStatus = if ($summary.reviewers.opencode) { $summary.reviewers.opencode } else { "----" }
            $opencodeColor = Get-StatusColor -Status $opencodeStatus
            Write-Host "$(Get-StatusEmoji $opencodeStatus)" -NoNewline -ForegroundColor $opencodeColor
            Write-Host " │ " -NoNewline -ForegroundColor DarkGray

            # Copilot
            $copilotStatus = if ($summary.reviewers.copilot) { $summary.reviewers.copilot } else { "----" }
            $copilotColor = Get-StatusColor -Status $copilotStatus
            Write-Host "$(Get-StatusEmoji $copilotStatus)" -NoNewline -ForegroundColor $copilotColor
            Write-Host " │ " -NoNewline -ForegroundColor DarkGray

            # Overall
            $overallColor = Get-StatusColor -Status $summary.overall_status
            Write-Host " $(Get-StatusEmoji $summary.overall_status) " -NoNewline -ForegroundColor $overallColor
            Write-Host "│" -ForegroundColor DarkGray
        } else {
            Write-Host "[----] │ [----] │ [----] │ [----] │  [----] │" -ForegroundColor Gray
        }

        if ($Detailed -and $summary) {
            Write-Host "  │         │ P0: $($summary.total_issues.p0) │ P1: $($summary.total_issues.p1) │ P2: $($summary.total_issues.p2)" -ForegroundColor DarkGray
            Write-Host "  │         │                              │        │        │        │        │         │" -ForegroundColor DarkGray
        }
    }

    Write-Host "  └─────────┴──────────────────────────────┴────────┴────────┴────────┴────────┴─────────┘" -ForegroundColor DarkGray
    Write-Host ""

    # Summary stats
    $reviewed = $commits | Where-Object { Get-ReviewSummary -CommitHash $_.hash }
    $passed = $reviewed | Where-Object { (Get-ReviewSummary -CommitHash $_.hash).overall_status -eq "PASS" }

    Write-Host "  Summary: " -NoNewline -ForegroundColor White
    Write-Host "$($reviewed.Count)/$($commits.Count) reviewed" -NoNewline -ForegroundColor Cyan
    Write-Host " | " -NoNewline -ForegroundColor DarkGray
    Write-Host "$($passed.Count) passed" -ForegroundColor Green
    Write-Host ""

    # Quick commands
    Write-Host "  Commands:" -ForegroundColor DarkGray
    Write-Host "    pwsh .ai-review/auto-review.ps1           # Run auto review" -ForegroundColor DarkGray
    Write-Host "    pwsh .ai-review/review-collector.ps1 -Mode summarize  # Generate summary" -ForegroundColor DarkGray
    Write-Host ""

    if ($Watch) {
        Write-Host "  [Watching... Press Ctrl+C to exit]" -ForegroundColor DarkGray
        Write-Host ""
    }
}

# Main execution
if ($Watch) {
    while ($true) {
        Show-Dashboard
        Start-Sleep -Seconds 5
    }
} else {
    Show-Dashboard
}
