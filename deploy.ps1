# MASP Quick Deploy Script
# Usage: .\deploy.ps1 -message "Fix: description"

param(
    [string]$message = "Update: $(Get-Date -Format 'yyyy-MM-dd HH:mm')"
)

$ErrorActionPreference = "Stop"

Write-Host "`n=== MASP Deploy Script ===" -ForegroundColor Cyan
Write-Host "Commit message: $message" -ForegroundColor Gray

# Step 1: Git commit and push
Write-Host "`n[1/3] Committing changes..." -ForegroundColor Yellow
git add -A
$status = git status --porcelain
if ($status) {
    git commit -m $message
    Write-Host "Committed successfully" -ForegroundColor Green
} else {
    Write-Host "No changes to commit" -ForegroundColor Gray
}

Write-Host "`n[2/3] Pushing to GitHub..." -ForegroundColor Yellow
git push origin ralph-loop/binance-futures-v1
Write-Host "Pushed successfully" -ForegroundColor Green

# Step 2: Deploy to EC2 via SSM
Write-Host "`n[3/3] Deploying to EC2..." -ForegroundColor Yellow

$commands = @(
    "cd /opt/masp/app",
    "sudo -u masp git pull origin ralph-loop/binance-futures-v1",
    "sudo -u masp docker compose build",
    "sudo -u masp docker compose --profile scheduler up -d --force-recreate"
)

$commandJson = $commands | ConvertTo-Json -Compress

aws ssm send-command `
    --instance-ids "i-0ef28a2261c9eaa41" `
    --document-name "AWS-RunShellScript" `
    --parameters "commands=$commandJson" `
    --comment "$message" `
    --output json | Out-Null

Write-Host "`n=== Deploy command sent! ===" -ForegroundColor Green
Write-Host "`nTo check deployment status, run in SSM session:" -ForegroundColor Cyan
Write-Host "  docker compose ps" -ForegroundColor White
Write-Host "  docker compose logs scheduler --tail=50" -ForegroundColor White
Write-Host ""
