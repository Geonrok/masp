"""
Test script for eBest (LS증권) adapter.

Usage:
    # Set environment variables first
    export EBEST_APP_KEY=your_app_key
    export EBEST_APP_SECRET=your_app_secret
    export EBEST_ACCOUNT_NO=your_account_number  # For execution tests only

    # Run tests
    python -m pytest tests/test_ebest_adapter.py -v

    # Or run directly
    python tests/test_ebest_adapter.py
"""

import os
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pytest
from dotenv import load_dotenv

load_dotenv()


# Skip all tests if credentials not available or integration tests disabled
EBEST_CREDENTIALS_AVAILABLE = bool(
    os.getenv("EBEST_APP_KEY") and os.getenv("EBEST_APP_SECRET")
)

# Integration tests require explicit opt-in: RUN_EBEST_INTEGRATION=1
RUN_INTEGRATION_TESTS = os.getenv("RUN_EBEST_INTEGRATION") == "1"

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not EBEST_CREDENTIALS_AVAILABLE,
        reason="EBEST_APP_KEY and EBEST_APP_SECRET not set"
    ),
    pytest.mark.skipif(
        not RUN_INTEGRATION_TESTS,
        reason="Integration tests disabled. Set RUN_EBEST_INTEGRATION=1 to run"
    ),
]


class TestEbestMarketData:
    """Test cases for EbestSpotMarketData adapter."""

    @pytest.fixture
    def adapter(self):
        """Create adapter instance."""
        from libs.adapters.real_ebest_spot import EbestSpotMarketData
        return EbestSpotMarketData()

    def test_get_quote(self, adapter):
        """Test getting a quote for Samsung Electronics."""
        quote = adapter.get_quote("005930")  # Samsung Electronics

        assert quote is not None, "Quote should not be None"
        assert quote.symbol == "005930"
        assert quote.last is not None and quote.last > 0, "Price should be positive"
        print(f"\nSamsung (005930) quote: {quote}")

    def test_get_quotes(self, adapter):
        """Test getting multiple quotes."""
        symbols = ["005930", "000660", "035420"]  # Samsung, SK Hynix, NAVER
        quotes = adapter.get_quotes(symbols)

        assert len(quotes) > 0, "Should return at least one quote"
        for symbol, quote in quotes.items():
            print(f"\n{symbol}: {quote}")

    def test_get_ohlcv(self, adapter):
        """Test getting OHLCV data."""
        candles = adapter.get_ohlcv("005930", interval="1d", limit=10)

        assert len(candles) > 0, "Should return at least one candle"

        # Check candle structure
        candle = candles[0]
        assert candle.open > 0
        assert candle.high >= candle.low
        assert candle.close > 0
        assert candle.volume >= 0

        print(f"\nFirst candle: {candle}")
        print(f"Last candle: {candles[-1]}")

    def test_get_all_symbols_kospi(self, adapter):
        """Test getting all KOSPI symbols."""
        symbols = adapter.get_all_symbols("KOSPI")

        assert len(symbols) > 0, "Should return KOSPI symbols"
        assert "005930" in symbols, "Samsung should be in KOSPI"

        print(f"\nLoaded {len(symbols)} KOSPI symbols")
        print(f"First 10: {symbols[:10]}")

    def test_get_all_symbols_kosdaq(self, adapter):
        """Test getting all KOSDAQ symbols."""
        symbols = adapter.get_all_symbols("KOSDAQ")

        assert len(symbols) > 0, "Should return KOSDAQ symbols"

        print(f"\nLoaded {len(symbols)} KOSDAQ symbols")
        print(f"First 10: {symbols[:10]}")

    def test_is_market_open(self, adapter):
        """Test market open check."""
        is_open = adapter.is_market_open()

        # Just check it doesn't error
        print(f"\nMarket open: {is_open}")


class TestEbestExecution:
    """Test cases for EbestSpotExecution adapter.

    WARNING: These tests may place real orders if MASP_ENABLE_LIVE_TRADING=1
    """

    @pytest.fixture
    def adapter(self):
        """Create execution adapter instance."""
        from libs.adapters.real_ebest_execution import EbestSpotExecution
        return EbestSpotExecution()

    @pytest.mark.skipif(
        not os.getenv("EBEST_ACCOUNT_NO"),
        reason="EBEST_ACCOUNT_NO not set"
    )
    def test_get_balance(self, adapter):
        """Test getting account balance."""
        from unittest.mock import patch

        with patch.dict(os.environ, {"MASP_ENABLE_LIVE_TRADING": "1"}):
            balance = adapter.get_balance("KRW")
            print(f"\nKRW Balance: {balance:,.0f}")

    @pytest.mark.skipif(
        not os.getenv("EBEST_ACCOUNT_NO"),
        reason="EBEST_ACCOUNT_NO not set"
    )
    def test_get_all_balances(self, adapter):
        """Test getting all balances including holdings."""
        from unittest.mock import patch

        with patch.dict(os.environ, {"MASP_ENABLE_LIVE_TRADING": "1"}):
            balances = adapter.get_all_balances()

            print(f"\n=== Account Balances ===")
            for b in balances:
                if b.get("currency") == "KRW":
                    print(f"Cash: {b.get('balance', 0):,.0f} KRW (Available: {b.get('available', 0):,.0f})")
                else:
                    print(
                        f"{b.get('name', b.get('currency'))}: "
                        f"{b.get('balance', 0):,.0f} shares @ {b.get('current_price', 0):,.0f} = "
                        f"{b.get('eval_amount', 0):,.0f} KRW (P/L: {b.get('profit_loss', 0):+,.0f})"
                    )


class TestAdapterFactory:
    """Test factory registration."""

    def test_create_market_data(self):
        """Test creating eBest market data adapter via factory."""
        from libs.adapters.factory import AdapterFactory

        adapter = AdapterFactory.create_market_data("ebest_spot")
        assert adapter is not None
        print(f"\nCreated adapter: {type(adapter).__name__}")

    def test_available_exchanges(self):
        """Test that eBest is listed in available exchanges."""
        from libs.adapters.factory import AdapterFactory

        exchanges = AdapterFactory.get_available_exchanges()

        assert "ebest_spot" in exchanges["market_data"]
        assert "ebest" in exchanges["execution"]

        print(f"\nMarket data adapters: {exchanges['market_data']}")
        print(f"Execution adapters: {exchanges['execution']}")


if __name__ == "__main__":
    # Run simple tests when executed directly
    print("=" * 60)
    print("eBest Adapter Test")
    print("=" * 60)

    if not EBEST_CREDENTIALS_AVAILABLE:
        print("\nERROR: EBEST_APP_KEY and EBEST_APP_SECRET not set!")
        print("Please set environment variables:")
        print("  export EBEST_APP_KEY=your_app_key")
        print("  export EBEST_APP_SECRET=your_app_secret")
        sys.exit(1)

    print("\n1. Testing Factory Registration...")
    test_factory = TestAdapterFactory()
    test_factory.test_available_exchanges()

    print("\n2. Testing Market Data Adapter...")
    from libs.adapters.real_ebest_spot import EbestSpotMarketData
    adapter = EbestSpotMarketData()

    print("\n   - Getting Samsung quote...")
    quote = adapter.get_quote("005930")
    if quote:
        print(f"     Samsung (005930): {quote.last:,.0f} KRW")
    else:
        print("     Failed to get quote")

    print("\n   - Getting OHLCV data...")
    candles = adapter.get_ohlcv("005930", interval="1d", limit=5)
    if candles:
        print(f"     Got {len(candles)} candles")
        print(f"     Latest: {candles[-1].timestamp.date()} - Close: {candles[-1].close:,.0f}")
    else:
        print("     Failed to get OHLCV")

    print("\n   - Checking market status...")
    is_open = adapter.is_market_open()
    print(f"     Market open: {is_open}")

    print("\n" + "=" * 60)
    print("Test completed!")
    print("=" * 60)
