"""Test Cache Integration in Upbit Adapter"""

from libs.adapters.real_upbit_spot import UpbitSpotMarketData

print("=== Cache Integration Test ===\n")

# Create adapter (with default cache)
adapter = UpbitSpotMarketData()

# First call - should be MISS (API call)
print("[1] First call (Cache MISS):")
q1 = adapter.get_quote("BTC/KRW")
if q1:
    print(f"  BTC/KRW: {q1.last:,.0f} KRW")
else:
    print("  FAIL: No quote")

# Second call - should be HIT (from cache)
print(f"\n[2] Second call (Cache HIT):")
q2 = adapter.get_quote("BTC/KRW")
if q2:
    print(f"  BTC/KRW: {q2.last:,.0f} KRW")
else:
    print("  FAIL: No quote")

# Check cache stats
print(f"\n[3] Cache Statistics:")
stats = adapter._cache.get_stats()
print(f"  Size: {stats['size']}")
print(f"  Hits: {stats['hits']}")
print(f"  Misses: {stats['misses']}")
print(f"  Hit Rate: {stats['hit_rate_pct']:.1f}%")

if stats["hits"] >= 1:
    print("\n✅ Cache Integration Test PASSED")
else:
    print("\n❌ Cache Integration Test FAILED")
