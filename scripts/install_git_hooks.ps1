$ErrorActionPreference = "Stop"

Write-Host "========================================"
Write-Host " MASP Git Hooks Installation"
Write-Host "========================================"
Write-Host ""

# Check if we're in a git repository
$gitDir = git rev-parse --git-dir 2>$null
if (-not $gitDir) {
    Write-Host "[ERROR] Not a git repository. Run this from the MASP project root."
    exit 1
}

# Check if .githooks directory exists
$hooksDir = ".githooks"
if (-not (Test-Path $hooksDir)) {
    Write-Host "[ERROR] .githooks directory not found."
    Write-Host "[ERROR] Please ensure the repository is properly cloned."
    exit 1
}

Write-Host "[hooks] Configuring git hooksPath to .githooks ..."

# Set the hooks path
git config core.hooksPath .githooks

# Verify the setting
$val = git config core.hooksPath
Write-Host "[hooks] core.hooksPath = $val"

Write-Host ""
Write-Host "========================================"
Write-Host " Installation Complete!"
Write-Host "========================================"
Write-Host ""
Write-Host "What happens now:"
Write-Host "  - Every 'git commit' will trigger pre-commit hook"
Write-Host "  - pre-commit runs scripts/review-code.ps1 automatically"
Write-Host "  - If P1 > 0, commit is BLOCKED until issues are fixed"
Write-Host ""
Write-Host "This enforces AGENTS.md workflow compliance."
Write-Host ""

exit 0
