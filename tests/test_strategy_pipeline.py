import os
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from services.strategy_runner import StrategyRunner, MIN_ORDER_KRW


def _make_runner(**kwargs):
    strategy = kwargs.pop("strategy", MagicMock())
    execution = kwargs.pop("execution", MagicMock())
    market_data = kwargs.pop("market_data", MagicMock())
    return StrategyRunner(
        strategy_name="ma_crossover_v1",
        exchange="paper",
        symbols=["BTC/KRW"],
        position_size_krw=100000,
        strategy=strategy,
        execution=execution,
        market_data=market_data,
        **kwargs,
    )


def test_kill_switch_raises_error():
    with patch.dict(os.environ, {"STOP_TRADING": "1"}):
        runner = _make_runner()
        with pytest.raises(RuntimeError, match="Kill-Switch"):
            runner.run_once()


def test_buy_uses_krw_not_quantity():
    """BUY 호출 시 amount_krw= 파라미터를 사용하는지 확인"""
    execution = MagicMock()
    execution.place_order.return_value = SimpleNamespace(order_id="order-1", symbol="BTC/KRW")
    runner = _make_runner(execution=execution)
    mock_buy_signal = SimpleNamespace(action="BUY", reason="test")

    result = runner._execute_trade_signal("BTC/KRW", mock_buy_signal, {"trade_price": 50000000})

    # 새로운 인터페이스: amount_krw= 키워드 파라미터 사용
    execution.place_order.assert_called_with(
        "BTC/KRW",
        "BUY",
        order_type="MARKET",
        amount_krw=runner.position_size_krw,  # ✅ amount_krw 사용
    )
    assert result["action"] == "BUY"


def test_dust_sell_skipped():
    execution = MagicMock()
    execution.get_balance.return_value = (MIN_ORDER_KRW - 1000) / 100000000
    runner = _make_runner(execution=execution)
    mock_sell_signal = SimpleNamespace(action="SELL", reason="test")

    result = runner._execute_trade_signal(
        "BTC/KRW",
        mock_sell_signal,
        {"trade_price": 100000000},
    )

    assert result["action"] == "SKIP"
    assert "Dust" in result["reason"]
    execution.place_order.assert_not_called()


def test_gate_veto_blocks_buy():
    execution = MagicMock()
    market_data = MagicMock()
    strategy = MagicMock()
    strategy.check_gate.return_value = True
    strategy.generate_signal.return_value = SimpleNamespace(action="BUY", gate_pass=False)

    runner = _make_runner(strategy=strategy, execution=execution, market_data=market_data)

    result = runner.run_once()

    assert result["BTC/KRW"]["action"] == "BLOCKED"
    market_data.get_quote.assert_not_called()
    execution.place_order.assert_not_called()
