from unittest.mock import Mock, patch


def test_load_symbols_pybithumb_success():
    """pybithumb success loads symbols."""
    with patch("libs.adapters.bithumb_public.pybithumb") as mock_pb:
        mock_pb.get_tickers.return_value = ["BTC", "ETH", "XRP"]
        from libs.adapters.bithumb_public import BithumbPublic

        bp = BithumbPublic(cache_ttl=0)
        assert len(bp.symbols) == 3
        assert "BTC/KRW" in bp.symbols


def test_fallback_to_http_api():
    """pybithumb failure triggers HTTP API fallback."""
    with patch("libs.adapters.bithumb_public.pybithumb") as mock_pb:
        mock_pb.get_tickers.side_effect = ImportError()
        with patch("libs.adapters.bithumb_public.requests.get") as mock_get:
            mock_get.return_value = Mock(
                json=Mock(
                    return_value={
                        "status": "0000",
                        "data": {"BTC": {}, "ETH": {}, "date": "123"},
                    }
                )
            )
            from libs.adapters.bithumb_public import BithumbPublic

            bp = BithumbPublic(cache_ttl=0)
            assert "BTC/KRW" in bp.symbols
            assert "date/KRW" not in bp.symbols


def test_fallback_to_default():
    """All API failure falls back to default."""
    with patch("libs.adapters.bithumb_public.pybithumb") as mock_pb:
        mock_pb.get_tickers.side_effect = Exception()
        with patch("libs.adapters.bithumb_public.requests.get") as mock_get:
            mock_get.side_effect = Exception()
            from libs.adapters.bithumb_public import BithumbPublic

            bp = BithumbPublic(cache_ttl=0)
            assert bp.symbols == ["BTC/KRW"]


def test_symbols_format():
    """Symbols end with /KRW."""
    with patch("libs.adapters.bithumb_public.pybithumb") as mock_pb:
        mock_pb.get_tickers.return_value = ["BTC", "ETH"]
        from libs.adapters.bithumb_public import BithumbPublic

        bp = BithumbPublic(cache_ttl=0)
        for symbol in bp.symbols:
            assert symbol.endswith("/KRW")
            assert "/" in symbol


def test_no_duplicates():
    """No duplicates."""
    with patch("libs.adapters.bithumb_public.pybithumb") as mock_pb:
        mock_pb.get_tickers.return_value = ["BTC", "ETH", "BTC"]
        from libs.adapters.bithumb_public import BithumbPublic

        bp = BithumbPublic(cache_ttl=0)
        assert len(bp.symbols) == len(set(bp.symbols))
