"""
Test Upbit Market Data Adapter

Phase 2A: Real API integration tests
"""

import pytest

from libs.adapters.real_upbit_spot import UpbitSpotExecution, UpbitSpotMarketData


class TestUpbitMarketData:
    """Test Upbit MarketData adapter"""

    @pytest.fixture
    def adapter(self):
        """Create Upbit adapter instance"""
        return UpbitSpotMarketData()

    def test_get_quote_btc_krw(self, adapter):
        """Test BTC/KRW quote retrieval"""
        quote = adapter.get_quote("BTC/KRW")

        assert quote is not None, "Quote should not be None"
        assert quote.symbol == "BTC/KRW"
        assert quote.last > 0, "Last price should be positive"
        assert quote.volume_24h >= 0, "Volume should be non-negative"
        assert quote.timestamp is not None

    def test_get_quote_eth_krw(self, adapter):
        """Test ETH/KRW quote retrieval"""
        quote = adapter.get_quote("ETH/KRW")

        assert quote is not None
        assert quote.symbol == "ETH/KRW"
        assert quote.last > 0

    def test_get_quotes_multiple(self, adapter):
        """Test multiple symbols quote retrieval"""
        symbols = ["BTC/KRW", "ETH/KRW", "SOL/KRW"]
        quotes = adapter.get_quotes(symbols)

        assert len(quotes) == 3, f"Expected 3 quotes, got {len(quotes)}"

        for symbol in symbols:
            assert symbol in quotes, f"Missing quote for {symbol}"
            assert quotes[symbol].last > 0

    def test_is_market_open(self, adapter):
        """Test market open status (should always be True for Upbit)"""
        assert adapter.is_market_open() is True, "Upbit operates 24/7"

    def test_get_orderbook(self, adapter):
        """Test orderbook retrieval"""
        orderbook = adapter.get_orderbook("BTC/KRW")

        assert orderbook is not None
        assert "symbol" in orderbook
        assert orderbook["symbol"] == "BTC/KRW"
        assert "bids" in orderbook
        assert "asks" in orderbook
        assert len(orderbook["bids"]) > 0, "Should have bid levels"
        assert len(orderbook["asks"]) > 0, "Should have ask levels"

        # Check structure of first bid/ask
        first_bid = orderbook["bids"][0]
        assert "price" in first_bid
        assert "size" in first_bid
        assert first_bid["price"] > 0

    def test_symbol_conversion(self, adapter):
        """Test symbol format conversion"""
        # BTC/KRW → KRW-BTC
        assert adapter._convert_symbol("BTC/KRW") == "KRW-BTC"
        assert adapter._convert_symbol("ETH/KRW") == "KRW-ETH"

        # KRW-BTC → BTC/KRW
        assert adapter._revert_symbol("KRW-BTC") == "BTC/KRW"
        assert adapter._revert_symbol("KRW-ETH") == "ETH/KRW"


class TestUpbitExecution:
    """Test Upbit Execution adapter (live guard)"""

    @pytest.fixture
    def adapter(self):
        """Create Upbit execution adapter instance"""
        return UpbitSpotExecution()

    def test_place_order_disabled(self, adapter):
        """Test that place_order raises RuntimeError when live trading disabled"""
        with pytest.raises(RuntimeError) as exc_info:
            adapter.place_order("BTC/KRW", "BUY", 0.001)

        assert "MASP_ENABLE_LIVE_TRADING=1" in str(exc_info.value)

    def test_get_order_status_disabled(self, adapter):
        """Test that get_order_status raises RuntimeError when live trading disabled"""
        with pytest.raises(RuntimeError) as exc_info:
            adapter.get_order_status("test_order_id")

        assert "MASP_ENABLE_LIVE_TRADING=1" in str(exc_info.value)

    def test_cancel_order_disabled(self, adapter):
        """Test that cancel_order raises RuntimeError when live trading disabled"""
        with pytest.raises(RuntimeError) as exc_info:
            adapter.cancel_order("test_order_id")

        assert "MASP_ENABLE_LIVE_TRADING=1" in str(exc_info.value)

    def test_get_balance_disabled(self, adapter):
        """Test that get_balance raises RuntimeError when live trading disabled"""
        with pytest.raises(RuntimeError) as exc_info:
            adapter.get_balance("KRW")

        assert "MASP_ENABLE_LIVE_TRADING=1" in str(exc_info.value)


# Manual test runners (for quick command-line testing)
if __name__ == "__main__":
    print("=== Upbit MarketData Test ===\n")

    adapter = UpbitSpotMarketData()

    # Test 1: Single quote
    print("[Test 1] BTC/KRW single quote:")
    quote = adapter.get_quote("BTC/KRW")
    if quote:
        print(f"  Symbol: {quote.symbol}")
        print(f"  Last: {quote.last:,.0f} KRW")
        print(f"  Volume 24h: {quote.volume_24h:,.2f}")
        print("  ✅ PASS\n")
    else:
        print("  ❌ FAIL: Quote is None\n")

    # Test 2: Multiple quotes
    print("[Test 2] Multiple quotes:")
    quotes = adapter.get_quotes(["BTC/KRW", "ETH/KRW"])
    for symbol, q in quotes.items():
        print(f"  {symbol}: {q.last:,.0f} KRW")
    print(f"  ✅ PASS ({len(quotes)} quotes)\n")

    # Test 3: Orderbook
    print("[Test 3] BTC/KRW orderbook:")
    orderbook = adapter.get_orderbook("BTC/KRW")
    if orderbook:
        print(f"  Bids: {len(orderbook['bids'])} levels")
        print(f"  Asks: {len(orderbook['asks'])} levels")
        print(f"  Best bid: {orderbook['bids'][0]['price']:,.0f} KRW")
        print(f"  Best ask: {orderbook['asks'][0]['price']:,.0f} KRW")
        print("  ✅ PASS\n")
    else:
        print("  ❌ FAIL: Orderbook is None\n")

    # Test 4: Order execution disabled
    print("[Test 4] Order execution disabled:")
    exec_adapter = UpbitSpotExecution()
    try:
        exec_adapter.place_order("BTC/KRW", "BUY", 0.001)
        print("  ❌ FAIL: Should raise RuntimeError\n")
    except RuntimeError as e:
        print(f"  ✅ PASS: {e}\n")
