<#
.SYNOPSIS
    AI Review Collector - Collects and structures review results from multiple CLI reviewers.

.DESCRIPTION
    This script:
    1. Collects review results from each CLI reviewer
    2. Structures them in JSON and Markdown formats
    3. Generates a summary report
    4. Determines overall review status

.PARAMETER Reviewer
    The reviewer CLI name (codex, gemini, opencode, copilot)

.PARAMETER ReviewContent
    The review content from the CLI

.PARAMETER CommitHash
    The git commit hash being reviewed

.PARAMETER Mode
    Operation mode: collect, summarize, status

.EXAMPLE
    .\review-collector.ps1 -Mode collect -Reviewer codex -CommitHash abc123 -ReviewContent "..."
    .\review-collector.ps1 -Mode summarize -CommitHash abc123
    .\review-collector.ps1 -Mode status -CommitHash abc123
#>

param(
    [Parameter(Mandatory=$true)]
    [ValidateSet("collect", "summarize", "status", "init")]
    [string]$Mode,

    [string]$Reviewer,
    [string]$CommitHash,
    [string]$ReviewContent,
    [string]$ReviewFile
)

$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ReviewsDir = Join-Path $ScriptDir "reviews"
$ConfigFile = Join-Path $ScriptDir "review-config.json"

# Ensure reviews directory exists
if (-not (Test-Path $ReviewsDir)) {
    New-Item -ItemType Directory -Path $ReviewsDir -Force | Out-Null
}

function Get-Config {
    if (Test-Path $ConfigFile) {
        return Get-Content $ConfigFile -Raw | ConvertFrom-Json
    }
    return $null
}

function Get-CurrentCommit {
    try {
        return (git rev-parse --short HEAD 2>$null)
    } catch {
        return "unknown"
    }
}

function Initialize-ReviewSession {
    param([string]$Commit)

    $sessionFile = Join-Path $ReviewsDir "$Commit-session.json"
    $session = @{
        commit = $Commit
        started = (Get-Date -Format "yyyy-MM-ddTHH:mm:ss")
        reviews = @{}
        status = "IN_PROGRESS"
    }

    $session | ConvertTo-Json -Depth 10 | Set-Content $sessionFile -Encoding UTF8
    Write-Host "[Review] Session initialized for $Commit"
    return $session
}

function Get-ReviewSession {
    param([string]$Commit)

    $sessionFile = Join-Path $ReviewsDir "$Commit-session.json"
    if (Test-Path $sessionFile) {
        return Get-Content $sessionFile -Raw | ConvertFrom-Json
    }
    return Initialize-ReviewSession -Commit $Commit
}

function Save-ReviewSession {
    param($Session)

    $sessionFile = Join-Path $ReviewsDir "$($Session.commit)-session.json"
    $Session | ConvertTo-Json -Depth 10 | Set-Content $sessionFile -Encoding UTF8
}

function Parse-ReviewContent {
    param([string]$Content)

    $result = @{
        p0_issues = @()
        p1_issues = @()
        p2_suggestions = @()
        status = "PASS"
        raw = $Content
    }

    # Parse P0 issues
    $p0Matches = [regex]::Matches($Content, '(?m)^-\s*\[\s*\]\s*(.+?:\d+.+?)$')
    foreach ($match in $p0Matches) {
        if ($Content.Substring(0, $match.Index) -match 'P0|Blocking') {
            $result.p0_issues += $match.Groups[1].Value
        }
    }

    # Parse P1 issues
    $p1Matches = [regex]::Matches($Content, '(?m)^-\s*\[\s*\]\s*(.+?:\d+.+?)$')
    foreach ($match in $p1Matches) {
        $context = $Content.Substring([Math]::Max(0, $match.Index - 100), [Math]::Min(100, $match.Index))
        if ($context -match 'P1|Important|Required') {
            $result.p1_issues += $match.Groups[1].Value
        }
    }

    # Determine status
    if ($result.p0_issues.Count -gt 0) {
        $result.status = "BLOCKED"
    } elseif ($result.p1_issues.Count -gt 0) {
        $result.status = "CHANGES_REQUESTED"
    } else {
        $result.status = "PASS"
    }

    return $result
}

function Collect-Review {
    param(
        [string]$Reviewer,
        [string]$Commit,
        [string]$Content
    )

    $session = Get-ReviewSession -Commit $Commit
    $parsed = Parse-ReviewContent -Content $Content
    $timestamp = Get-Date -Format "yyyy-MM-ddTHH:mm:ss"

    # Add to session
    $reviewData = @{
        reviewer = $Reviewer
        timestamp = $timestamp
        status = $parsed.status
        p0_count = $parsed.p0_issues.Count
        p1_count = $parsed.p1_issues.Count
        p2_count = $parsed.p2_suggestions.Count
        issues = $parsed
    }

    # PowerShell hashtable to add review
    if ($null -eq $session.reviews) {
        $session.reviews = @{}
    }
    $session.reviews[$Reviewer] = $reviewData

    # Save individual review file
    $reviewFile = Join-Path $ReviewsDir "$Commit-$Reviewer.md"
    @"
# $Reviewer Review

**Commit**: $Commit
**Timestamp**: $timestamp
**Status**: $($parsed.status)

## Issues Summary
- P0 (Blocking): $($parsed.p0_issues.Count)
- P1 (Important): $($parsed.p1_issues.Count)
- P2 (Suggestions): $($parsed.p2_suggestions.Count)

## Full Review

$Content
"@ | Set-Content $reviewFile -Encoding UTF8

    Save-ReviewSession -Session $session

    Write-Host "[Review] Collected $Reviewer review: $($parsed.status)"
    Write-Host "  P0: $($parsed.p0_issues.Count), P1: $($parsed.p1_issues.Count)"

    return $reviewData
}

function Generate-Summary {
    param([string]$Commit)

    $session = Get-ReviewSession -Commit $Commit
    $config = Get-Config

    $overallStatus = "PASS"
    $totalP0 = 0
    $totalP1 = 0
    $totalP2 = 0
    $reviewerStatuses = @{}

    foreach ($reviewer in $session.reviews.PSObject.Properties) {
        $review = $reviewer.Value
        $reviewerStatuses[$reviewer.Name] = $review.status
        $totalP0 += $review.p0_count
        $totalP1 += $review.p1_count
        $totalP2 += $review.p2_count

        if ($review.status -eq "BLOCKED") {
            $overallStatus = "BLOCKED"
        } elseif ($review.status -eq "CHANGES_REQUESTED" -and $overallStatus -ne "BLOCKED") {
            $overallStatus = "CHANGES_REQUESTED"
        }
    }

    $summary = @{
        commit = $Commit
        timestamp = (Get-Date -Format "yyyy-MM-ddTHH:mm:ss")
        overall_status = $overallStatus
        total_issues = @{
            p0 = $totalP0
            p1 = $totalP1
            p2 = $totalP2
        }
        reviewers = $reviewerStatuses
        can_merge = ($overallStatus -eq "PASS")
    }

    # Save summary JSON
    $summaryFile = Join-Path $ReviewsDir "$Commit-summary.json"
    $summary | ConvertTo-Json -Depth 10 | Set-Content $summaryFile -Encoding UTF8

    # Generate summary markdown
    $mdFile = Join-Path $ReviewsDir "$Commit-summary.md"
    $statusEmoji = switch ($overallStatus) {
        "PASS" { "✅" }
        "CHANGES_REQUESTED" { "⚠️" }
        "BLOCKED" { "❌" }
        default { "❓" }
    }

    $reviewerTable = ""
    foreach ($r in $reviewerStatuses.GetEnumerator()) {
        $emoji = switch ($r.Value) {
            "PASS" { "✅" }
            "CHANGES_REQUESTED" { "⚠️" }
            "BLOCKED" { "❌" }
            default { "❓" }
        }
        $reviewerTable += "| $($r.Key) | $emoji $($r.Value) |`n"
    }

@"
# Review Summary

**Commit**: $Commit
**Status**: $statusEmoji $overallStatus
**Can Merge**: $(if ($summary.can_merge) { "Yes" } else { "No" })

## Issue Counts

| Severity | Count |
|----------|-------|
| P0 (Blocking) | $totalP0 |
| P1 (Important) | $totalP1 |
| P2 (Suggestions) | $totalP2 |

## Reviewer Status

| Reviewer | Status |
|----------|--------|
$reviewerTable

## Generated
$((Get-Date -Format "yyyy-MM-dd HH:mm:ss"))
"@ | Set-Content $mdFile -Encoding UTF8

    Write-Host ""
    Write-Host "=========================================="
    Write-Host "  Review Summary: $statusEmoji $overallStatus"
    Write-Host "=========================================="
    Write-Host "  P0: $totalP0 | P1: $totalP1 | P2: $totalP2"
    Write-Host "  Can Merge: $(if ($summary.can_merge) { 'Yes' } else { 'No' })"
    Write-Host ""

    return $summary
}

function Get-ReviewStatus {
    param([string]$Commit)

    $summaryFile = Join-Path $ReviewsDir "$Commit-summary.json"
    if (Test-Path $summaryFile) {
        $summary = Get-Content $summaryFile -Raw | ConvertFrom-Json
        return $summary.overall_status
    }
    return "NOT_REVIEWED"
}

# Main execution
switch ($Mode) {
    "init" {
        $commit = if ($CommitHash) { $CommitHash } else { Get-CurrentCommit }
        Initialize-ReviewSession -Commit $commit
    }
    "collect" {
        if (-not $Reviewer) { throw "Reviewer is required for collect mode" }
        $commit = if ($CommitHash) { $CommitHash } else { Get-CurrentCommit }

        $content = $ReviewContent
        if ($ReviewFile -and (Test-Path $ReviewFile)) {
            $content = Get-Content $ReviewFile -Raw
        }

        if (-not $content) { throw "Review content is required" }

        Collect-Review -Reviewer $Reviewer -Commit $commit -Content $content
    }
    "summarize" {
        $commit = if ($CommitHash) { $CommitHash } else { Get-CurrentCommit }
        Generate-Summary -Commit $commit
    }
    "status" {
        $commit = if ($CommitHash) { $CommitHash } else { Get-CurrentCommit }
        $status = Get-ReviewStatus -Commit $commit
        Write-Host $status
        exit $(if ($status -eq "PASS") { 0 } elseif ($status -eq "NOT_REVIEWED") { 2 } else { 1 })
    }
}
