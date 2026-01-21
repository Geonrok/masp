$ErrorActionPreference = "Stop"

Write-Host "[pre-commit] Running mandatory review-code workflow..."
Write-Host "[pre-commit] This enforces AGENTS.md workflow compliance."
Write-Host ""

# Run review-code.ps1 (must return non-zero on failure)
try {
    $scriptPath = Join-Path $PSScriptRoot "review-code.ps1"

    if (-not (Test-Path $scriptPath)) {
        Write-Host "[pre-commit] ERROR: review-code.ps1 not found at $scriptPath"
        exit 1
    }

    & powershell -NoProfile -ExecutionPolicy Bypass -File $scriptPath

    if ($LASTEXITCODE -ne 0) {
        Write-Host ""
        Write-Host "[pre-commit] review-code.ps1 failed with exit code $LASTEXITCODE"
        Write-Host "[pre-commit] Commit BLOCKED. Fix P1 issues and try again."
        exit 1
    }
}
catch {
    Write-Host "[pre-commit] ERROR: $($_.Exception.Message)"
    Write-Host "[pre-commit] Commit BLOCKED due to error."
    exit 1
}

Write-Host ""
Write-Host "[pre-commit] review-code.ps1 passed. Commit allowed."
exit 0
