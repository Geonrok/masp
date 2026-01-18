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
