cd "E:\투자\Multi-Asset Strategy Platform"
$env:PYTHONPATH = "."

$ts = Get-Date -Format "yyyyMMdd-HHmm"
$OUT = "outputs\phase7A-fix\$ts"
New-Item -ItemType Directory -Force -Path $OUT | Out-Null
New-Item -ItemType Directory -Force -Path "scripts\phase7a" | Out-Null

Write-Host "========== Phase 7A-Fix: NO-GO 이슈 해결 ==========" -ForegroundColor Cyan
Write-Host "Evidence 저장 경로: $OUT" -ForegroundColor Yellow

# ========== A3) Public API 연결성 (get_quote 사용) ==========
Write-Host "`n[A3] Public API 연결성 테스트 (get_quote)..." -ForegroundColor Green

$code = @"
print("=== A3: Public API 연결성 테스트 (get_quote 사용) ===")
import time
from libs.adapters.factory import AdapterFactory

tests = [("upbit_spot", "BTC/KRW"), ("bithumb_spot", "BTC/KRW")]

for ex, sym in tests:
    try:
        md = AdapterFactory.create_market_data(ex)
        quote = md.get_quote(sym)
        if quote and getattr(quote, "last", None):
            print(f"{ex} get_quote({sym}): PASS")
        else:
            print(f"{ex} get_quote({sym}): FAIL (None 또는 last 없음)")
    except Exception as e:
        print(f"{ex} get_quote({sym}): FAIL ({type(e).__name__}: {str(e)[:80]})")
    time.sleep(0.1)
"@
Set-Content -Path "scripts\phase7a\A3_public_connectivity_fix.py" -Value $code -Encoding UTF8
.\.venv311\Scripts\python scripts\phase7a\A3_public_connectivity_fix.py | Tee-Object -FilePath "$OUT\A3_public_connectivity.txt"

# ========== A4a) 가드레일 검증 (Live 차단 확인) ==========
Write-Host "`n[A4a] 가드레일 검증 (Live 모드 차단 확인)..." -ForegroundColor Green

$code = @"
print("=== A4a: 가드레일 검증 (Live 모드 차단 = 정상) ===")
from libs.adapters.factory import AdapterFactory
from libs.core.config import Config

cfg = Config(asset_class="crypto_spot", strategy_name="kama_tsmom_gate")

for ex in ["upbit_spot", "bithumb_spot"]:
    try:
        AdapterFactory.create_execution(ex, adapter_mode="live", config=cfg)
        print(f"{ex} guardrail: FAIL (UNEXPECTED: live adapter created)")
    except RuntimeError as e:
        msg = str(e)
        if "Live trading disabled" in msg:
            print(f"{ex} guardrail: PASS (차단됨 - 정상)")
        else:
            print(f"{ex} guardrail: FAIL (unexpected error: {msg[:80]})")
    except Exception as e:
        print(f"{ex} guardrail: FAIL ({type(e).__name__}: {str(e)[:80]})")
"@
Set-Content -Path "scripts\phase7a\A4a_guardrail_check.py" -Value $code -Encoding UTF8
.\.venv311\Scripts\python scripts\phase7a\A4a_guardrail_check.py | Tee-Object -FilePath "$OUT\A4a_guardrail.txt"

# ========== A4b) Private 인증 (직접 라이브러리) ==========
Write-Host "`n[A4b] Private 인증 테스트 (직접 라이브러리)..." -ForegroundColor Green

$code = @"
print("=== A4b: Private API 인증 (직접 라이브러리, 읽기 전용) ===")
print("주의: place_order 호출 절대 금지\n")

import os
from dotenv import load_dotenv
load_dotenv(override=False)

def mask(s):
    return f"SET(len={len(s)})" if s else "MISSING"

# Upbit 인증
ak = os.getenv("UPBIT_ACCESS_KEY", "")
sk = os.getenv("UPBIT_SECRET_KEY", "")
if ak and sk:
    try:
        import pyupbit
        upbit = pyupbit.Upbit(ak, sk)
        balances = upbit.get_balances()
        if balances is not None:
            print("UPBIT 인증: PASS (private endpoint OK)")
        else:
            print("UPBIT 인증: WARN (balances=None)")
    except Exception as e:
        msg = str(e)[:100]
        print(f"UPBIT 인증: FAIL ({type(e).__name__}: {msg})")
else:
    print(f"UPBIT 인증: SKIP (keys: ak={mask(ak)}, sk={mask(sk)})")

# Bithumb 인증
bk = os.getenv("BITHUMB_API_KEY", "")
bs = os.getenv("BITHUMB_SECRET_KEY", "")
if bk and bs:
    try:
        import pybithumb
        bithumb = pybithumb.Bithumb(bk, bs)
        balance = bithumb.get_balance("BTC")
        if balance is not None:
            print("BITHUMB 인증: PASS (private endpoint OK)")
        else:
            print("BITHUMB 인증: WARN (balance=None)")
    except Exception as e:
        msg = str(e)[:100]
        print(f"BITHUMB 인증: FAIL ({type(e).__name__}: {msg})")
else:
    print(f"BITHUMB 인증: SKIP (keys: bk={mask(bk)}, bs={mask(bs)})")
"@
Set-Content -Path "scripts\phase7a\A4b_private_auth.py" -Value $code -Encoding UTF8
.\.venv311\Scripts\python scripts\phase7a\A4b_private_auth.py | Tee-Object -FilePath "$OUT\A4b_private_auth.txt"

# ========== A5) Rate Limit (get_quote 사용) ==========
Write-Host "`n[A5] Rate Limit 테스트 (get_quote)..." -ForegroundColor Green

$code = @"
import time
print("=== A5: Rate Limit 테스트 (get_quote 사용) ===")

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
            md.get_quote(sym)
        except Exception as e:
            errors += 1
            msg = str(e).lower()
            if "429" in msg or "too many" in msg or "rate" in msg:
                print(f"{ex} Rate Limit: FAIL (429 감지)")
                rate_limit_hit = True
                break
        time.sleep(SLEEP)
    
    dt = time.time() - t0
    if not rate_limit_hit:
        if errors == 0:
            print(f"{ex} Rate Limit: PASS (N={N}, elapsed={dt:.2f}s)")
        else:
            print(f"{ex} Rate Limit: WARN (errors={errors})")
"@
Set-Content -Path "scripts\phase7a\A5_rate_limit_fix.py" -Value $code -Encoding UTF8
.\.venv311\Scripts\python scripts\phase7a\A5_rate_limit_fix.py | Tee-Object -FilePath "$OUT\A5_rate_limit.txt"

# ========== A6) Git History 확인 ==========
Write-Host "`n[A6] Git History 유출 확인..." -ForegroundColor Green

Write-Host "[A6-1] pickaxe 결과 파일명 확인..." -ForegroundColor Gray
git log --all -S "UPBIT_ACCESS_KEY=" --name-only --pretty=format:"COMMIT %h %s" 2>$null | Tee-Object -FilePath "$OUT\A6_pickaxe_files.txt"

Write-Host "[A6-2] 하드코딩 패턴 재확인..." -ForegroundColor Gray
git grep -n "UPBIT_ACCESS_KEY\\s*=" -- "*.py" 2>$null | Tee-Object -FilePath "$OUT\A6_hardcoded_check.txt"

$hardcoded = Get-Content "$OUT\A6_hardcoded_check.txt" -ErrorAction SilentlyContinue
if ([string]::IsNullOrWhiteSpace($hardcoded)) {
    "하드코딩 패턴: PASS (발견 없음 - getenv 참조로 추정)" | Tee-Object -FilePath "$OUT\A6_verdict.txt"
} else {
    "하드코딩 패턴: FAIL (발견됨! 확인 필요)" | Tee-Object -FilePath "$OUT\A6_verdict.txt"
}

Write-Host "`n========== Phase 7A-Fix 결과 요약 ==========" -ForegroundColor Cyan

Write-Host "`n[A3] Public API:" -ForegroundColor Yellow
Get-Content "$OUT\A3_public_connectivity.txt" | Select-Object -Last 3

Write-Host "`n[A4a] 가드레일:" -ForegroundColor Yellow
Get-Content "$OUT\A4a_guardrail.txt" | Select-Object -Last 3

Write-Host "`n[A4b] Private 인증:" -ForegroundColor Yellow
Get-Content "$OUT\A4b_private_auth.txt" | Select-Object -Last 3

Write-Host "`n[A5] Rate Limit:" -ForegroundColor Yellow
Get-Content "$OUT\A5_rate_limit.txt" | Select-Object -Last 3

Write-Host "`n[A6] Git History:" -ForegroundColor Yellow
Get-Content "$OUT\A6_verdict.txt"

Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host "Evidence 저장 완료: $OUT" -ForegroundColor Green
