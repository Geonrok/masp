cd "E:\투자\Multi-Asset Strategy Platform"
$env:PYTHONPATH = "."

# ===== 안전장치: 실거래 금지 강제 =====
if ($env:MASP_ENABLE_LIVE_TRADING -eq "1") {
    Write-Host "FATAL: MASP_ENABLE_LIVE_TRADING=1 (실거래 활성). Phase 7B는 paper only. 중단합니다." -ForegroundColor Red
    exit 1
}
Remove-Item Env:MASP_ENABLE_LIVE_TRADING -ErrorAction SilentlyContinue | Out-Null

$ts  = Get-Date -Format "yyyyMMdd-HHmm"
$OUT = "outputs\phase7B\$ts"
New-Item -ItemType Directory -Force -Path $OUT | Out-Null
New-Item -ItemType Directory -Force -Path "scripts\phase7b" | Out-Null

Write-Host "========== Phase 7B: Upbit Regression 테스트 v7.7 ==========" -ForegroundColor Cyan
Write-Host "Evidence 저장 경로: $OUT" -ForegroundColor Yellow

# ========== B1) 전략 로딩 + MarketData adapter 확인 ==========
Write-Host "`n[B1] 전략 로딩 + MarketData adapter 확인..." -ForegroundColor Green

$code_b1 = @"
import sys
print("=== B1: 전략 로딩 + MarketData adapter 확인 ===")
from services.strategy_runner import StrategyRunner

r = StrategyRunner(strategy_name="KAMA-TSMOM-Gate", exchange="upbit", symbols=["BTC/KRW"])

# 1) Strategy must exist
if r.strategy is None:
    print("Strategy loaded: None")
    print("B1 결과: FAIL (strategy=None)")
    sys.exit(1)

print(f"Strategy loaded: {type(r.strategy).__name__}")

# 2) Runner must have market_data
if not (hasattr(r, 'market_data') and r.market_data):
    print("MarketData adapter: None 또는 미주입")
    print("B1 결과: FAIL (runner.market_data missing)")
    sys.exit(1)

adapter_type = type(r.market_data).__name__
print(f"MarketData adapter: {adapter_type}")

# 3) 기능 테스트
try:
    quote = r.market_data.get_quote("BTC/KRW")
    if quote and hasattr(quote, 'last') and quote.last > 0:
        print(f"기능 테스트: PASS (가격: {quote.last:,.0f} KRW)")
    else:
        print(f"기능 테스트: WARN (quote 구조 이상: {quote})")
except Exception as e:
    print(f"기능 테스트: WARN ({type(e).__name__}: {str(e)[:50]})")

# 4) Strategy 내부 market_data 확인
md_in_strategy = getattr(r.strategy, "market_data", None) or getattr(r.strategy, "_market_data", None)
if md_in_strategy:
    same = (md_in_strategy is r.market_data)
    print(f"Strategy 내부 market_data: {type(md_in_strategy).__name__} (same_object={same})")

print("B1 결과: PASS")
"@
Set-Content -Path "scripts\phase7b\b1_strategy_load.py" -Value $code_b1 -Encoding UTF8
Copy-Item "scripts\phase7b\b1_strategy_load.py" "$OUT\b1_strategy_load.py" -Force
.\.venv311\Scripts\python scripts\phase7b\b1_strategy_load.py 2>&1 | Tee-Object -FilePath "$OUT\B1_strategy_load.txt"
$B1_EXIT = $LASTEXITCODE

# ========== B2) OHLCV 91+ 캔들 확인 ==========
Write-Host "`n[B2] Upbit OHLCV 91+ 캔들 확인..." -ForegroundColor Green

$code_b2 = @"
import sys
print("=== B2: Upbit OHLCV 91+ 캔들 확인 ===")
from libs.adapters.factory import AdapterFactory

md = AdapterFactory.create_market_data("upbit_spot")
ohlcv = md.get_ohlcv("BTC/KRW", "1d", 100)

if ohlcv is None:
    print("OHLCV: None")
    print("B2 결과: FAIL")
    sys.exit(1)

count = len(ohlcv)
print(f"OHLCV candles: {count}")

if count < 91:
    print(f"B2 결과: FAIL ({count} < 91)")
    sys.exit(1)

print(f"First candle: {ohlcv[0]}")
print(f"Last candle:  {ohlcv[-1]}")
print("B2 결과: PASS (91+ 캔들 확보)")
"@
Set-Content -Path "scripts\phase7b\b2_ohlcv.py" -Value $code_b2 -Encoding UTF8
Copy-Item "scripts\phase7b\b2_ohlcv.py" "$OUT\b2_ohlcv.py" -Force
.\.venv311\Scripts\python scripts\phase7b\b2_ohlcv.py 2>&1 | Tee-Object -FilePath "$OUT\B2_ohlcv.txt"
$B2_EXIT = $LASTEXITCODE

# ========== B3) Scheduler --once 실행 (paper) ==========
Write-Host "`n[B3] Upbit scheduler --once (paper 모드)..." -ForegroundColor Green
Write-Host "신호 생성 중..." -ForegroundColor Yellow

.\.venv311\Scripts\python -m services.multi_exchange_scheduler --once --exchange upbit 2>&1 | Tee-Object -FilePath "$OUT\B3_scheduler.txt"
$B3_SCHEDULER_EXIT = $LASTEXITCODE

# ---- B3 결과 분석: Python AST 파싱 ----
$code_b3_analyze = @"
import ast, re, sys

path = sys.argv[1]
txt = open(path, 'r', encoding='utf-8', errors='replace').read()

# 1) Summary Actions 파싱
m = re.search(r"Actions:\s*(\{[^\n\r]*\})", txt)
actions = None
if m:
    try:
        actions = ast.literal_eval(m.group(1))
    except Exception:
        actions = None

# 2) 대안: Result: {...} 파싱
if actions is None:
    m2 = re.search(r"^Result:\s*(\{.*\})\s*$", txt, flags=re.MULTILINE)
    if m2:
        try:
            result = ast.literal_eval(m2.group(1))
            actions = {}
            up = result.get('upbit', {})
            for _, v in up.items():
                a = (v or {}).get('action', 'UNKNOWN')
                actions[a] = actions.get(a, 0) + 1
        except Exception:
            actions = None

if actions is None:
    print("B3 결과: FAIL (could not parse Actions/Result)")
    sys.exit(1)

buy  = int(actions.get('BUY', 0))
sell = int(actions.get('SELL', 0))
hold = int(actions.get('HOLD', 0))
err  = int(actions.get('ERROR', 0))

print(f"BUY={buy} SELL={sell} HOLD={hold} ERROR={err}")

if err != 0:
    print(f"B3 결과: FAIL (ERROR={err})")
    sys.exit(1)

if buy == 0 and sell == 0:
    print(f"B3 결과: PASS_WITH_WARN (ERROR=0 but BUY/SELL=0; 시장상황 가능)")
else:
    print(f"B3 결과: PASS (ERROR=0, BUY={buy}, SELL={sell}, HOLD={hold})")
"@
Set-Content -Path "scripts\phase7b\b3_analyze.py" -Value $code_b3_analyze -Encoding UTF8
Copy-Item "scripts\phase7b\b3_analyze.py" "$OUT\b3_analyze.py" -Force
.\.venv311\Scripts\python scripts\phase7b\b3_analyze.py "$OUT\B3_scheduler.txt" 2>&1 | Tee-Object -FilePath "$OUT\B3_summary.txt"
$B3_EXIT = $LASTEXITCODE

# ========== B4) pytest 전체 실행 ==========
Write-Host "`n[B4] pytest 전체 실행..." -ForegroundColor Green

.\.venv311\Scripts\pytest tests/ -q --tb=short 2>&1 | Tee-Object -FilePath "$OUT\B4_pytest.txt"
$B4_PYTEST_EXIT = $LASTEXITCODE

# pytest 결과 정량 파싱
$pytestOutput = Get-Content "$OUT\B4_pytest.txt" -Raw
$passed = 0; $failed = 0; $skipped = 0
if ($pytestOutput -match "(\d+) passed") { $passed = [int]$matches[1] }
if ($pytestOutput -match "(\d+) failed") { $failed = [int]$matches[1] }
if ($pytestOutput -match "(\d+) skipped") { $skipped = [int]$matches[1] }

Write-Host "pytest: passed=$passed, failed=$failed, skipped=$skipped"

# ========== 최종 판정 ==========
Write-Host "`n========== Phase 7B 최종 판정 ==========" -ForegroundColor Cyan

$B1_PASS = ($B1_EXIT -eq 0)
$B2_PASS = ($B2_EXIT -eq 0)
$B3_PASS = ($B3_EXIT -eq 0)
$B4_PASS = ($failed -eq 0 -and $passed -ge 230)

Write-Host "B1 전략 로딩: $(if($B1_PASS){'PASS'}else{'FAIL'})" -ForegroundColor $(if($B1_PASS){'Green'}else{'Red'})
Write-Host "B2 OHLCV:      $(if($B2_PASS){'PASS'}else{'FAIL'})" -ForegroundColor $(if($B2_PASS){'Green'}else{'Red'})
Write-Host "B3 Scheduler:  $(if($B3_PASS){'PASS'}else{'FAIL'})" -ForegroundColor $(if($B3_PASS){'Green'}else{'Red'})
Write-Host "B4 pytest:     $(if($B4_PASS){'PASS'}else{'FAIL'}) (passed=$passed, failed=$failed)" -ForegroundColor $(if($B4_PASS){'Green'}else{'Red'})

$allPass = $B1_PASS -and $B2_PASS -and $B3_PASS -and $B4_PASS

if ($allPass) {
    Write-Host "`n✅ Phase 7B: GO (모든 테스트 통과)" -ForegroundColor Green
    "Phase 7B 결과: GO`nB1=$B1_PASS B2=$B2_PASS B3=$B3_PASS B4=$B4_PASS (passed=$passed)" | Out-File "$OUT\FINAL_VERDICT.txt" -Encoding UTF8
} else {
    Write-Host "`n❌ Phase 7B: NO-GO (일부 테스트 실패)" -ForegroundColor Red
    "Phase 7B 결과: NO-GO`nB1=$B1_PASS B2=$B2_PASS B3=$B3_PASS B4=$B4_PASS (passed=$passed, failed=$failed)" | Out-File "$OUT\FINAL_VERDICT.txt" -Encoding UTF8
}

Write-Host "`nEvidence 저장 완료: $OUT" -ForegroundColor Yellow
