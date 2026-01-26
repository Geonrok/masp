<#
.SYNOPSIS
    Auto Review Router - Automatically routes code changes to appropriate reviewers.

.DESCRIPTION
    This script:
    1. Analyzes changed files in a commit
    2. Determines which reviewers should review based on file patterns
    3. Generates review commands for each appropriate CLI
    4. Optionally executes reviews automatically

.PARAMETER CommitHash
    The git commit hash to review (default: HEAD)

.PARAMETER DryRun
    If set, only shows what would be done without executing

.PARAMETER Execute
    If set, automatically executes review commands

.EXAMPLE
    .\auto-review.ps1                    # Analyze HEAD, show recommendations
    .\auto-review.ps1 -DryRun            # Show what would be reviewed
    .\auto-review.ps1 -Execute           # Execute all reviews automatically
#>

param(
    [string]$CommitHash,
    [switch]$DryRun,
    [switch]$Execute,
    [switch]$Summary
)

$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ConfigFile = Join-Path $ScriptDir "review-config.json"
$PromptsDir = Join-Path $ScriptDir "prompts"

function Get-Config {
    if (Test-Path $ConfigFile) {
        return Get-Content $ConfigFile -Raw | ConvertFrom-Json
    }
    throw "Config file not found: $ConfigFile"
}

function Get-ChangedFiles {
    param([string]$Commit)

    if ($Commit) {
        $files = git diff-tree --no-commit-id --name-only -r $Commit 2>$null
    } else {
        # Get staged files or last commit
        $files = git diff --cached --name-only 2>$null
        if (-not $files) {
            $files = git diff-tree --no-commit-id --name-only -r HEAD 2>$null
        }
    }

    return $files | Where-Object { $_ -match '\.py$|\.ts$|\.tsx$|\.js$|\.jsx$' }
}

function Get-FileCategory {
    param([string]$FilePath)

    $categories = @()

    # Strategy files
    if ($FilePath -match 'strateg|backtest|indicator') {
        $categories += "strategy"
    }

    # Adapter/Exchange files
    if ($FilePath -match 'adapter|exchange|execution') {
        $categories += "adapter"
    }

    # Risk management
    if ($FilePath -match 'risk|drawdown|guard') {
        $categories += "risk"
    }

    # Service layer
    if ($FilePath -match 'service|runner|scheduler') {
        $categories += "service"
    }

    # Analysis/Research
    if ($FilePath -match 'analysis|research|regime') {
        $categories += "analysis"
    }

    # Tests
    if ($FilePath -match 'test_|_test\.') {
        $categories += "test"
    }

    if ($categories.Count -eq 0) {
        $categories += "general"
    }

    return $categories
}

function Get-ReviewersForFile {
    param(
        [string]$FilePath,
        $Config
    )

    $reviewers = @()
    $categories = Get-FileCategory -FilePath $FilePath

    # Check routing rules
    foreach ($rule in $Config.routing_rules.PSObject.Properties) {
        $pattern = $rule.Name
        if ($pattern -eq "default") { continue }

        if ($FilePath -like "*$pattern*") {
            $reviewers += $rule.Value
        }
    }

    # If no specific match, use default
    if ($reviewers.Count -eq 0) {
        $reviewers = $Config.routing_rules.default
    }

    # Always include final reviewer
    $finalReviewer = $Config.reviewers.PSObject.Properties | Where-Object {
        $_.Value.final_reviewer -eq $true
    } | Select-Object -First 1

    if ($finalReviewer -and $finalReviewer.Name -notin $reviewers) {
        $reviewers += $finalReviewer.Name
    }

    return $reviewers | Select-Object -Unique
}

function Get-ReviewPrompt {
    param(
        [string]$Reviewer,
        $Config
    )

    $promptFile = Join-Path $PromptsDir "$Reviewer-review.md"
    if (Test-Path $promptFile) {
        return Get-Content $promptFile -Raw
    }
    return $null
}

function Format-ReviewPlan {
    param(
        $Plan,
        $Config
    )

    Write-Host ""
    Write-Host "=============================================="
    Write-Host "  MASP Auto Review Plan"
    Write-Host "=============================================="
    Write-Host ""
    Write-Host "Commit: $($Plan.commit)"
    Write-Host "Files Changed: $($Plan.files.Count)"
    Write-Host ""

    Write-Host "Files to Review:"
    foreach ($file in $Plan.files) {
        $categories = (Get-FileCategory -FilePath $file) -join ", "
        Write-Host "  - $file [$categories]"
    }

    Write-Host ""
    Write-Host "Reviewer Assignments:"
    Write-Host ""

    $reviewerFiles = @{}
    foreach ($assignment in $Plan.assignments) {
        if (-not $reviewerFiles[$assignment.reviewer]) {
            $reviewerFiles[$assignment.reviewer] = @()
        }
        $reviewerFiles[$assignment.reviewer] += $assignment.file
    }

    foreach ($reviewer in $reviewerFiles.Keys | Sort-Object) {
        $reviewerConfig = $Config.reviewers.$reviewer
        $emoji = switch ($reviewer) {
            "codex" { "🔵" }
            "gemini" { "🟣" }
            "opencode" { "🟢" }
            "copilot" { "🟠" }
            default { "⚪" }
        }

        Write-Host "$emoji $($reviewerConfig.name) ($($reviewerConfig.model))"
        Write-Host "   Focus: $($reviewerConfig.focus -join ', ')"
        Write-Host "   Files: $($reviewerFiles[$reviewer].Count)"
        foreach ($f in $reviewerFiles[$reviewer]) {
            Write-Host "     - $f"
        }
        Write-Host ""
    }

    Write-Host "----------------------------------------------"
    Write-Host "Review Commands:"
    Write-Host ""

    foreach ($reviewer in $reviewerFiles.Keys | Sort-Object) {
        $fileList = $reviewerFiles[$reviewer] -join " "
        Write-Host "# $reviewer"
        Write-Host "$reviewer review $fileList"
        Write-Host ""
    }
}

function Generate-ReviewCommands {
    param($Plan, $Config)

    $commands = @()

    $reviewerFiles = @{}
    foreach ($assignment in $Plan.assignments) {
        if (-not $reviewerFiles[$assignment.reviewer]) {
            $reviewerFiles[$assignment.reviewer] = @()
        }
        $reviewerFiles[$assignment.reviewer] += $assignment.file
    }

    foreach ($reviewer in $reviewerFiles.Keys | Sort-Object) {
        $fileList = $reviewerFiles[$reviewer] -join " "
        $promptFile = Join-Path $PromptsDir "$reviewer-review.md"

        $commands += @{
            reviewer = $reviewer
            files = $reviewerFiles[$reviewer]
            command = "$reviewer review $fileList"
            prompt_file = $promptFile
        }
    }

    return $commands
}

# Main execution
$config = Get-Config

# Get commit hash
$commit = $CommitHash
if (-not $commit) {
    $commit = git rev-parse --short HEAD 2>$null
}

# Get changed files
$files = Get-ChangedFiles -Commit $commit

if (-not $files -or $files.Count -eq 0) {
    Write-Host "No reviewable files found in commit $commit"
    exit 0
}

# Create review plan
$assignments = @()
foreach ($file in $files) {
    $reviewers = Get-ReviewersForFile -FilePath $file -Config $config
    foreach ($reviewer in $reviewers) {
        $assignments += @{
            file = $file
            reviewer = $reviewer
        }
    }
}

$plan = @{
    commit = $commit
    timestamp = (Get-Date -Format "yyyy-MM-ddTHH:mm:ss")
    files = $files
    assignments = $assignments
}

# Save plan
$planFile = Join-Path $ScriptDir "reviews" "$commit-plan.json"
$plan | ConvertTo-Json -Depth 10 | Set-Content $planFile -Encoding UTF8

if ($Summary) {
    # Just generate summary
    & "$ScriptDir\review-collector.ps1" -Mode summarize -CommitHash $commit
    exit 0
}

# Display plan
Format-ReviewPlan -Plan $plan -Config $config

if ($DryRun) {
    Write-Host "[DryRun] No actions taken"
    exit 0
}

if ($Execute) {
    Write-Host ""
    Write-Host "Executing reviews..."
    Write-Host ""

    # Initialize review session
    & "$ScriptDir\review-collector.ps1" -Mode init -CommitHash $commit

    $commands = Generate-ReviewCommands -Plan $plan -Config $config

    foreach ($cmd in $commands) {
        Write-Host "Executing: $($cmd.reviewer) review..."
        Write-Host "  Command: $($cmd.command)"
        Write-Host "  (Manual execution required - CLI integration pending)"
        Write-Host ""
    }

    Write-Host "Review commands generated. Execute manually or integrate with CI."
}

# Output for scripting
$plan | ConvertTo-Json -Depth 10
