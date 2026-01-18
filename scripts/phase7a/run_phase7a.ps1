cd "E:\투자\Multi-Asset Strategy Platform"
$env:PYTHONPATH = "."

$ts = Get-Date -Format "yyyyMMdd-HHmm"
$OUT = "outputs\phase7A\$ts"
New-Item -ItemType Directory -Force -Path $OUT | Out-Null
New-Item -ItemType Directory -Force -Path "scripts\phase7a" | Out-Null

Write-Host "========== Phase 7A: 실거래 전환 안전성 검증 ==========" -ForegroundColor Cyan
Write-Host "Evidence 저장 경로: $OUT" -ForegroundColor Yellow

# ========== A1) 환경 확인 ==========
Write-Host "`n[A1] 환경 확인..." -ForegroundColor Green
git rev-parse HEAD | Tee-Object -FilePath "$OUT\A1_git_head.txt"
.\.venv311\Scripts\python --version | Tee-Object -FilePath "$OUT\A1_python_version.txt"

# ========== A2) Kill-Switch 차단 증명 ==========
Write-Host "`n[A2] Kill-Switch 차단 증명..." -ForegroundColor Green

$code = @"
import os
from libs.core.config import Config

print("=== A2-1: Kill-Switch 상태 확인 ===")
try:
    c = Config(asset_class="crypto_spot", strategy_name="kama_tsmom_gate")
    is_active = c.is_kill_switch_active()
    print(f"Config 로드: PASS")
    print(f"Kill-Switch Active: {is_active}")
    print(f"STOP_TRADING env: {os.getenv('STOP_TRADING', 'Not Set')}")
except Exception as e:
    print(f"Config 로드: FAIL - {e}")
"@
Set-Content -Path "scripts\phase7a\A2_killswitch_check.py" -Value $code -Encoding UTF8
.\.venv311\Scripts\python scripts\phase7a\A2_killswitch_check.py | Tee-Object -FilePath "$OUT\A2_killswitch_status.txt"

# A2-2) STOP_TRADING=1 차단 테스트
$env:STOP_TRADING = "1"
$code = @"
import os
print(f"\n=== A2-2: STOP_TRADING=1 차단 테스트 ===")
print(f"STOP_TRADING env: {os.getenv('STOP_TRADING')}")

from services.strategy_runner import StrategyRunner
try:
    r = StrategyRunner(strategy_name="KAMA-TSMOM-Gate", exchange="bithumb", symbols=["BTC/KRW"])
    r.run_once()
    print("결과: FAIL - run_once가 차단되지 않음")
except Exception as e:
    print(f"결과: PASS - 차단됨 ({type(e).__name__}: {str(e)[:100]})")
"@
Set-Content -Path "scripts\phase7a\A2_stop_trading_test.py" -Value $code -Encoding UTF8
.\.venv311\Scripts\python scripts\phase7a\A2_stop_trading_test.py | Tee-Object -FilePath "$OUT\A2_stop_trading.txt"
Remove-Item Env:\STOP_TRADING -ErrorAction SilentlyContinue

# ========== A3) Public API 연결성 테스트 ==========
Write-Host "`n[A3] Public API 연결성 테스트..." -ForegroundColor Green

$code = @"
print("=== A3: Public API 연결성 테스트 (키 불필요) ===")

from libs.adapters.factory import AdapterFactory

tests = [("upbit_spot", "BTC/KRW"), ("bithumb_spot", "BTC/KRW")]

for ex, sym in tests:
    try:
        md = AdapterFactory.create_market_data(ex)
        ticker = md.get_ticker(sym)
        if ticker:
            print(f"{ex} get_ticker({sym}): PASS")
        else:
            print(f"{ex} get_ticker({sym}): FAIL (None 반환)")
    except Exception as e:
        print(f"{ex} get_ticker({sym}): FAIL ({type(e).__name__}: {str(e)[:80]})")
"@
Set-Content -Path "scripts\phase7a\A3_public_connectivity.py" -Value $code -Encoding UTF8
.\.venv311\Scripts\python scripts\phase7a\A3_public_connectivity.py | Tee-Object -FilePath "$OUT\A3_public_connectivity.txt"

# ========== A4) Private 인증 테스트 (get_balance만) ==========
Write-Host "`n[A4] Private 인증 테스트 (get_balance only)..." -ForegroundColor Green

$code = @"
print("=== A4: Private API 인증 테스트 (get_balance만 호출) ===")
print("주의: place_order 호출 절대 금지\n")

from libs.core.config import Config
from libs.adapters.factory import AdapterFactory

cfg = Config(asset_class="crypto_spot", strategy_name="kama_tsmom_gate")

targets = [
    ("upbit_spot", "UPBIT"),
    ("bithumb_spot", "BITHUMB"),
]

for ex, label in targets:
    try:
        exa = AdapterFactory.create_execution(ex, adapter_mode="live", config=cfg)
        bal = exa.get_balance("KRW")
        if bal is not None:
            print(f"{label} get_balance: PASS (인증 성공)")
        else:
            print(f"{label} get_balance: WARN (None 반환, 잔고 0 가능)")
    except Exception as e:
        msg = str(e)[:120]
        if "401" in msg or "unauthorized" in msg.lower() or "signature" in msg.lower():
            print(f"{label} get_balance: FAIL (인증 오류: {msg})")
        else:
            print(f"{label} get_balance: FAIL ({type(e).__name__}: {msg})")
"@
Set-Content -Path "scripts\phase7a\A4_private_auth.py" -Value $code -Encoding UTF8
.\.venv311\Scripts\python scripts\phase7a\A4_private_auth.py | Tee-Object -FilePath "$OUT\A4_private_auth.txt"

# ========== A5) Rate Limit 준수 테스트 ==========
Write-Host "`n[A5] Rate Limit 준수 테스트..." -ForegroundColor Green

$code = @"
import time
print("=== A5: Rate Limit 준수 테스트 (Public API burst) ===")

from libs.adapters.factory import AdapterFactory

tests = [("upbit_spot", "BTC/KRW"), ("bithumb_spot", "BTC/KRW")]

N = 20
SLEEP = 0.12

for ex, sym in tests:
    md = AdapterFactory.create_market_data(ex)
    errors = 0
    rate_limit_hit = False
    t0 = time.time()
    
    for i in range(N):
        try:
            md.get_ticker(sym)
        except Exception as e:
            errors += 1
            msg = str(e).lower()
            if "429" in msg or "too many" in msg or "rate" in msg:
                print(f"{ex} Rate Limit: FAIL (429 감지: {str(e)[:80]})")
                rate_limit_hit = True
                break
        time.sleep(SLEEP)
    
    dt = time.time() - t0
    if not rate_limit_hit:
        if errors == 0:
            print(f"{ex} Rate Limit: PASS (N={N}, sleep={SLEEP}s, elapsed={dt:.2f}s)")
        else:
            print(f"{ex} Rate Limit: WARN (errors={errors}, elapsed={dt:.2f}s)")

print(f"\n=== Rate Limit 정량 분석 ===")
print(f"현재 설정: 0.1s/symbol = 10 symbols/s")
print(f"447개 심볼 예상 처리: {447 * 0.1:.1f}s = {447 * 0.1 / 60:.1f}분")
print(f"Bithumb 제한: 30/s -> 현재 10/s < 30/s (안전)")
print(f"Upbit 제한: 10/s (주문) -> 현재 10/s == 10/s (한계치)")
"@
Set-Content -Path "scripts\phase7a\A5_rate_limit.py" -Value $code -Encoding UTF8
.\.venv311\Scripts\python scripts\phase7a\A5_rate_limit.py | Tee-Object -FilePath "$OUT\A5_rate_limit.txt"

# ========== A6) API 키 보안 점검 ==========
Write-Host "`n[A6] API 키 보안 점검..." -ForegroundColor Green

$code = @"
import os
from dotenv import load_dotenv
load_dotenv(override=False)

print("=== A6-1: API 키 존재 여부 (값 출력 금지) ===")
keys = ["BITHUMB_API_KEY", "BITHUMB_SECRET_KEY", "UPBIT_ACCESS_KEY", "UPBIT_SECRET_KEY"]
for k in keys:
    v = os.getenv(k)
    if not v:
        status = "MISSING"
    elif len(v) < 20:
        status = f"TOO_SHORT ({len(v)} chars)"
    else:
        status = f"SET ({len(v)} chars)"
    print(f"{k}: {status}")
"@
Set-Content -Path "scripts\phase7a\A6_keys_check.py" -Value $code -Encoding UTF8
.\.venv311\Scripts\python scripts\phase7a\A6_keys_check.py | Tee-Object -FilePath "$OUT\A6_keys_status.txt"

# A6-2) .gitignore 확인
Write-Host "[A6-2] .gitignore 확인..." -ForegroundColor Gray
$gitignoreCheck = Select-String -Path ".gitignore" -Pattern "\.env" -Quiet
if ($gitignoreCheck) {
    ".env in .gitignore: PASS" | Tee-Object -FilePath "$OUT\A6_gitignore.txt"
} else {
    ".env in .gitignore: FAIL (보안 위험!)" | Tee-Object -FilePath "$OUT\A6_gitignore.txt"
}

# A6-3) 하드코딩 시크릿 탐지
Write-Host "[A6-3] 하드코딩 시크릿 탐지..." -ForegroundColor Gray
$pattern = '(UPBIT_ACCESS_KEY|UPBIT_SECRET_KEY|BITHUMB_API_KEY|BITHUMB_SECRET_KEY)\s*[:=]\s*[A-Za-z0-9_\-]{20,}'
git grep -n -E $pattern -- "*.py" "*.yaml" "*.json" 2>$null |
  Select-String -NotMatch "getenv|os\.environ|load_dotenv|dotenv" |
  Tee-Object -FilePath "$OUT\A6_hardcoded_secrets.txt"

$hardcoded = Get-Content "$OUT\A6_hardcoded_secrets.txt" -ErrorAction SilentlyContinue
if ([string]::IsNullOrWhiteSpace($hardcoded)) {
    "하드코딩 시크릿: PASS (발견 없음)" | Add-Content "$OUT\A6_hardcoded_secrets.txt"
    Write-Host "하드코딩 시크릿: PASS" -ForegroundColor Green
} else {
    Write-Host "하드코딩 시크릿: FAIL (발견됨!)" -ForegroundColor Red
}

# A6-4) git history pickaxe
Write-Host "[A6-4] git history 유출 흔적..." -ForegroundColor Gray
git log --all -S "UPBIT_ACCESS_KEY=" --pretty=format:"%h %ad %s" --date=short 2>$null | Tee-Object -FilePath "$OUT\A6_gitlog_upbit.txt"
git log --all -S "BITHUMB_API_KEY=" --pretty=format:"%h %ad %s" --date=short 2>$null | Tee-Object -FilePath "$OUT\A6_gitlog_bithumb.txt"

# ========== 결과 요약 출력 ==========
Write-Host "`n========== Phase 7A 결과 요약 ==========" -ForegroundColor Cyan

Write-Host "`n[A2] Kill-Switch:" -ForegroundColor Yellow
Get-Content "$OUT\A2_stop_trading.txt" | Select-Object -Last 2

Write-Host "`n[A3] Public API:" -ForegroundColor Yellow
Get-Content "$OUT\A3_public_connectivity.txt" | Select-Object -Last 2

Write-Host "`n[A4] Private 인증:" -ForegroundColor Yellow
Get-Content "$OUT\A4_private_auth.txt" | Select-Object -Last 4

Write-Host "`n[A5] Rate Limit:" -ForegroundColor Yellow
Get-Content "$OUT\A5_rate_limit.txt" | Select-Object -First 2

Write-Host "`n[A6] 키 보안:" -ForegroundColor Yellow
Get-Content "$OUT\A6_keys_status.txt"

Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host "Evidence 저장 완료: $OUT" -ForegroundColor Green
