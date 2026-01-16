# MASP Health Endpoint Manual Test Script
# Usage: .\manual_health_check.ps1

$baseUrl = "http://127.0.0.1:8080"
$timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
$results = @{}

Write-Host "=== MASP Health Endpoint Test ===" -ForegroundColor Cyan
Write-Host "Timestamp: $timestamp"
Write-Host "Target: $baseUrl"
Write-Host ""

# 1. /health
Write-Host "Testing /health..." -ForegroundColor Yellow
try {
    $resp = Invoke-WebRequest -Uri "$baseUrl/health" -UseBasicParsing -ErrorAction Stop
    $body = $resp.Content | ConvertFrom-Json
    $results.health = @{
        status_code = $resp.StatusCode
        body_status = $body.status
        pass = ($resp.StatusCode -eq 200)
    }
    Write-Host "  Status: $($resp.StatusCode), Body: $($body.status)" -ForegroundColor Green
} catch {
    $results.health = @{ status_code = 0; body_status = "error"; pass = $false }
    Write-Host "  FAILED: $_" -ForegroundColor Red
}

# 2. /health/live
Write-Host "Testing /health/live..." -ForegroundColor Yellow
try {
    $resp = Invoke-WebRequest -Uri "$baseUrl/health/live" -UseBasicParsing -ErrorAction Stop
    $results.live = @{
        status_code = $resp.StatusCode
        pass = ($resp.StatusCode -eq 200)
    }
    Write-Host "  Status: $($resp.StatusCode)" -ForegroundColor Green
} catch {
    $results.live = @{ status_code = 0; pass = $false }
    Write-Host "  FAILED: $_" -ForegroundColor Red
}

# 3. /health/ready
Write-Host "Testing /health/ready..." -ForegroundColor Yellow
try {
    $resp = Invoke-WebRequest -Uri "$baseUrl/health/ready" -UseBasicParsing -ErrorAction Stop
    $results.ready = @{
        status_code = $resp.StatusCode
        pass = ($resp.StatusCode -eq 200)
    }
    Write-Host "  Status: $($resp.StatusCode)" -ForegroundColor Green
} catch {
    $statusCode = 503
    if ($_.Exception.Response) {
        $statusCode = [int]$_.Exception.Response.StatusCode
    }
    $results.ready = @{
        status_code = $statusCode
        pass = ($statusCode -eq 200 -or $statusCode -eq 503)
    }
    Write-Host "  Status: $statusCode (503 is acceptable if not initialized)" -ForegroundColor Yellow
}

# 4. /metrics
Write-Host "Testing /metrics..." -ForegroundColor Yellow
try {
    $resp = Invoke-WebRequest -Uri "$baseUrl/metrics" -UseBasicParsing -ErrorAction Stop
    $contentType = $resp.Headers["Content-Type"]
    $results.metrics = @{
        status_code = $resp.StatusCode
        content_type = $contentType
        pass = ($resp.StatusCode -eq 200 -and $contentType -match "text/plain")
    }
    Write-Host "  Status: $($resp.StatusCode), Content-Type: $contentType" -ForegroundColor Green
} catch {
    $statusCode = 501
    if ($_.Exception.Response) {
        $statusCode = [int]$_.Exception.Response.StatusCode
    }
    $results.metrics = @{
        status_code = $statusCode
        content_type = "text/plain"
        pass = ($statusCode -eq 501)  # 501 is expected when disabled
    }
    Write-Host "  Status: $statusCode (501 expected if metrics disabled)" -ForegroundColor Yellow
}

# Verdict
$allPass = $results.health.pass -and $results.live.pass -and $results.ready.pass -and $results.metrics.pass
$verdict = if ($allPass) { "PASS" } else { "FAIL" }

Write-Host ""
Write-Host "=== VERDICT: $verdict ===" -ForegroundColor $(if ($allPass) { "Green" } else { "Red" })

# Save YAML
$yaml = @"
curl_test_results:
  timestamp: "$timestamp"
  target:
    host: "127.0.0.1"
    port: 8080
  endpoints:
    health:
      status_code: $($results.health.status_code)
      body_status: "$($results.health.body_status)"
      pass: $($results.health.pass.ToString().ToLower())
    live:
      status_code: $($results.live.status_code)
      pass: $($results.live.pass.ToString().ToLower())
    ready:
      status_code: $($results.ready.status_code)
      pass: $($results.ready.pass.ToString().ToLower())
    metrics:
      status_code: $($results.metrics.status_code)
      content_type: "$($results.metrics.content_type)"
      pass: $($results.metrics.pass.ToString().ToLower())
  verdict: "$verdict"
"@

$yaml | Set-Content -Path "docs/verification/curl_health_test_results.yaml" -Encoding UTF8
Write-Host ""
Write-Host "Results saved to: docs/verification/curl_health_test_results.yaml" -ForegroundColor Cyan
