#!/usr/bin/env python3
"""
Binance Futures v12 Long-Only Fear-Based Strategy

Key insight from v11: Short trades in bear markets hurt performance.
Crypto has a long-term bullish bias. This strategy:
- ONLY takes LONG positions
- Enters on Fear in confirmed uptrends
- Exits on Greed or technical stops

Rules:
1. BULL confirmed: Price > EMA200 (rising) AND Price > EMA50
2. Entry: Fear <= 35 (buy fear dips)
3. Exit: Greed >= 70 OR trailing stop OR take profit

Target: Profit Factor > 1.3
"""
from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

DATA_ROOT = Path("E:/data/crypto_ohlcv")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


@dataclass
class BacktestConfig:
    initial_capital: float = 10000.0
    start_date: str = "2020-01-01"
    end_date: Optional[str] = None
    symbols: List[str] = field(default_factory=lambda: ["BTCUSDT", "ETHUSDT"])
    commission_pct: float = 0.001
    slippage_pct: float = 0.0005
    leverage: int = 3
    max_positions: int = 2
    position_size_pct: float = 0.15
    # Strategy parameters
    fear_entry: int = 35     # Enter on Fear
    greed_exit: int = 70     # Exit on Greed
    trend_deviation: float = 0.01  # 1% above EMA200


class DataLoader:
    def __init__(self, data_root: Path = DATA_ROOT):
        self.data_root = data_root

    def load_ohlcv(self, symbol: str, timeframe: str = "4h") -> pd.DataFrame:
        tf_map = {"4h": "binance_futures_4h", "1d": "binance_futures_1d"}
        folder = tf_map.get(timeframe, "binance_futures_4h")
        for file_path in [self.data_root / folder / f"{symbol}.csv",
                          self.data_root / folder / f"{symbol.replace('USDT', '')}.csv"]:
            if file_path.exists():
                df = pd.read_csv(file_path)
                if "datetime" in df.columns:
                    df["datetime"] = pd.to_datetime(df["datetime"])
                    df = df.set_index("datetime")
                elif "timestamp" in df.columns:
                    df["timestamp"] = pd.to_datetime(df["timestamp"])
                    df = df.set_index("timestamp")
                return df[["open", "high", "low", "close", "volume"]].sort_index()
        return pd.DataFrame()

    def load_fear_greed(self) -> pd.DataFrame:
        for filename in ["FEAR_GREED_INDEX_updated.csv", "FEAR_GREED_INDEX.csv"]:
            file_path = self.data_root / filename
            if file_path.exists():
                df = pd.read_csv(file_path)
                if "datetime" in df.columns:
                    df["datetime"] = pd.to_datetime(df["datetime"])
                    df = df.set_index("datetime")
                elif "timestamp" in df.columns:
                    df["timestamp"] = pd.to_datetime(df["timestamp"])
                    df = df.set_index("timestamp")
                if "close" in df.columns:
                    df = df.rename(columns={"close": "value"})
                return df[["value"]] if "value" in df.columns else df
        return pd.DataFrame()

    def load_funding_rate(self, symbol: str = "BTCUSDT") -> pd.DataFrame:
        for filename in [f"{symbol}_funding_full.csv", f"{symbol}_funding.csv"]:
            file_path = self.data_root / "binance_funding_rate" / filename
            if file_path.exists():
                df = pd.read_csv(file_path)
                df["datetime"] = pd.to_datetime(df["datetime"])
                df = df.set_index("datetime")
                return df[["fundingRate"]]
        return pd.DataFrame()


class LongOnlyStrategy:
    """Long-only fear-based strategy."""

    def __init__(self, config: BacktestConfig):
        self.config = config
        self.ema200_period = 200
        self.ema50_period = 50
        self.rsi_period = 14
        self.atr_period = 14

        self.fear_entry = config.fear_entry
        self.greed_exit = config.greed_exit
        self.trend_dev = config.trend_deviation

        # Risk management
        self.stop_atr = 2.5    # Wider stop for trend following
        self.trail_atr = 2.0   # Trailing stop
        self.tp_atr = 4.0      # Target profit

    def calc_ema(self, data: np.ndarray, period: int) -> np.ndarray:
        ema = np.zeros(len(data))
        ema[0] = data[0]
        mult = 2 / (period + 1)
        for i in range(1, len(data)):
            ema[i] = (data[i] - ema[i-1]) * mult + ema[i-1]
        return ema

    def calc_atr(self, df: pd.DataFrame, period: int = 14) -> np.ndarray:
        h, l, c = df["high"].values, df["low"].values, df["close"].values
        tr = np.maximum(h - l, np.maximum(np.abs(h - np.roll(c, 1)), np.abs(l - np.roll(c, 1))))
        tr[0] = h[0] - l[0]
        return pd.Series(tr).rolling(period).mean().values

    def calc_rsi(self, close: np.ndarray, period: int = 14) -> np.ndarray:
        delta = np.diff(close)
        delta = np.insert(delta, 0, 0)
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        avg_gain = pd.Series(gain).rolling(period).mean().values
        avg_loss = pd.Series(loss).rolling(period).mean().values
        rs = avg_gain / np.maximum(avg_loss, 1e-10)
        return 100 - (100 / (1 + rs))

    def is_uptrend(self, price: float, ema50: float, ema200: float, ema200_prev: float) -> bool:
        """Check if we're in a confirmed uptrend."""
        ema200_rising = ema200 > ema200_prev
        price_above_ema200 = price > ema200 * (1 + self.trend_dev)
        ema_aligned = ema50 > ema200  # Golden cross

        return price_above_ema200 and (ema200_rising or ema_aligned)

    def generate_signal(
        self, df: pd.DataFrame, fear_greed: Optional[float], funding: Optional[float]
    ) -> Dict[str, Any]:
        """Generate LONG-only signal."""

        if len(df) < 250:
            return {"signal": "HOLD", "reason": "Insufficient data"}

        close = df["close"].values
        price = close[-1]

        ema200 = self.calc_ema(close, self.ema200_period)
        ema50 = self.calc_ema(close, self.ema50_period)
        atr = self.calc_atr(df, self.atr_period)
        rsi = self.calc_rsi(close, self.rsi_period)

        curr_ema200 = ema200[-1]
        prev_ema200 = ema200[-2]
        curr_ema50 = ema50[-1]
        curr_atr = atr[-1]
        curr_rsi = rsi[-1]

        uptrend = self.is_uptrend(price, curr_ema50, curr_ema200, prev_ema200)

        if fear_greed is None:
            return {"signal": "HOLD", "reason": "No FG data", "uptrend": uptrend}

        signal = "HOLD"
        reason = "NO_SIGNAL"
        pos_mult = 1.0

        # LONG conditions:
        # 1. Confirmed uptrend
        # 2. Fear present
        # 3. RSI not already overbought
        # 4. Funding not extremely high (avoid leverage flush)
        if uptrend:
            is_fear = fear_greed <= self.fear_entry
            rsi_ok = curr_rsi < 70  # Not overbought
            funding_ok = funding is None or funding < 0.002  # Not overcrowded longs

            if is_fear and rsi_ok and funding_ok:
                signal = "LONG"

                # Position sizing by conviction
                if fear_greed <= 15:
                    pos_mult = 1.0
                    reason = f"EXTREME_FEAR({fear_greed:.0f})"
                elif fear_greed <= 25:
                    pos_mult = 0.8
                    reason = f"STRONG_FEAR({fear_greed:.0f})"
                else:
                    pos_mult = 0.6
                    reason = f"FEAR_DIP({fear_greed:.0f})"

                # Boost if RSI also oversold
                if curr_rsi < 35:
                    pos_mult = min(1.0, pos_mult + 0.2)
                    reason += f" RSI_OVERSOLD({curr_rsi:.0f})"

        # Calculate stops
        if signal == "LONG":
            stop_loss = price - self.stop_atr * curr_atr
            take_profit = price + self.tp_atr * curr_atr
        else:
            stop_loss = take_profit = None

        return {
            "signal": signal,
            "reason": reason,
            "uptrend": uptrend,
            "price": price,
            "stop_loss": stop_loss,
            "take_profit": take_profit,
            "atr": curr_atr,
            "rsi": curr_rsi,
            "pos_mult": pos_mult,
            "fear_greed": fear_greed,
        }


class LongOnlyBacktester:
    def __init__(self, config: BacktestConfig):
        self.config = config
        self.loader = DataLoader()
        self.strategy = LongOnlyStrategy(config)
        self.data_4h: Dict[str, pd.DataFrame] = {}
        self.fear_greed: Optional[pd.DataFrame] = None
        self.funding_rates: Dict[str, pd.DataFrame] = {}
        self.equity_curve: List[float] = []
        self.trade_log: List[Dict] = []

    def load_data(self) -> bool:
        logger.info("Loading data...")
        for symbol in self.config.symbols:
            df = self.loader.load_ohlcv(symbol, "4h")
            if len(df) > 100:
                self.data_4h[symbol] = df
                logger.info(f"  {symbol}: {len(df)} candles")
        if not self.data_4h:
            return False

        self.fear_greed = self.loader.load_fear_greed()
        logger.info(f"  Fear/Greed: {len(self.fear_greed)} records")

        for symbol in self.config.symbols:
            fr = self.loader.load_funding_rate(symbol)
            if not fr.empty:
                self.funding_rates[symbol] = fr
        return True

    def get_fg_at(self, ts: pd.Timestamp) -> Optional[float]:
        if self.fear_greed is None or self.fear_greed.empty:
            return None
        date = ts.date()
        mask = self.fear_greed.index.date <= date
        if mask.any():
            return self.fear_greed.loc[self.fear_greed.index[mask][-1], "value"]
        return None

    def get_funding_at(self, symbol: str, ts: pd.Timestamp) -> Optional[float]:
        if symbol not in self.funding_rates:
            return None
        df = self.funding_rates[symbol]
        date = ts.date()
        day_start = pd.Timestamp(date)
        mask = (df.index >= day_start - pd.Timedelta(days=1)) & (df.index < day_start + pd.Timedelta(days=1))
        if mask.any():
            return df.loc[mask, "fundingRate"].sum()
        return None

    def run(self) -> Dict[str, Any]:
        logger.info("=" * 60)
        logger.info("V12 LONG-ONLY FEAR-BASED BACKTEST")
        logger.info("=" * 60)

        start_dt = pd.Timestamp(self.config.start_date)
        end_dt = pd.Timestamp(self.config.end_date) if self.config.end_date else pd.Timestamp.now()

        if "BTCUSDT" not in self.data_4h:
            return {"error": "BTC data required"}

        btc = self.data_4h["BTCUSDT"]
        btc = btc[(btc.index >= start_dt) & (btc.index <= end_dt)]
        timestamps = btc.index.tolist()

        logger.info(f"Period: {timestamps[0]} to {timestamps[-1]}")
        logger.info(f"Fear entry: {self.config.fear_entry}")
        logger.info(f"Greed exit: {self.config.greed_exit}")
        logger.info(f"Trend deviation: {self.config.trend_deviation*100}%")

        capital = self.config.initial_capital
        positions: Dict[str, Dict] = {}
        self.equity_curve = [capital]
        self.trade_log = []

        stats = {"longs": 0, "wins": 0, "losses": 0,
                 "extreme_fear_trades": 0, "strong_fear_trades": 0, "fear_dip_trades": 0}

        for i in range(250, len(timestamps)):
            ts = timestamps[i]
            fg = self.get_fg_at(ts)

            for symbol in self.data_4h.keys():
                df = self.data_4h[symbol]
                df_curr = df[df.index <= ts].tail(300)
                if len(df_curr) < 250:
                    continue

                price = df_curr["close"].iloc[-1]
                funding = self.get_funding_at(symbol, ts)

                # Exit check
                if symbol in positions:
                    pos = positions[symbol]
                    exit_sig, exit_reason = False, ""

                    pos["highest"] = max(pos["highest"], price)
                    trail = pos["highest"] - pos["atr"] * self.strategy.trail_atr
                    pnl_pct = (price - pos["entry_price"]) / pos["entry_price"]

                    # Move stop to breakeven after 2.5% profit
                    if pnl_pct > 0.025:
                        trail = max(trail, pos["entry_price"])

                    if price <= pos["stop_loss"]:
                        exit_sig, exit_reason = True, "stop_loss"
                    elif price <= trail:
                        exit_sig, exit_reason = True, "trailing"
                    elif price >= pos["take_profit"]:
                        exit_sig, exit_reason = True, "take_profit"
                    elif fg and fg >= self.config.greed_exit:
                        exit_sig, exit_reason = True, "greed_exit"

                    if exit_sig:
                        exit_price = price * (1 - self.config.slippage_pct)
                        pnl = (exit_price - pos["entry_price"]) / pos["entry_price"] * pos["leverage"]
                        pnl_usd = pos["size"] * pnl - pos["size"] * self.config.commission_pct * 2
                        capital += pnl_usd

                        self.trade_log.append({
                            "symbol": symbol,
                            "entry_time": pos["entry_time"], "entry_price": pos["entry_price"],
                            "exit_time": ts, "exit_price": exit_price,
                            "pnl_pct": pnl, "pnl_usd": pnl_usd,
                            "exit_reason": exit_reason,
                            "entry_fg": pos.get("entry_fg"),
                        })

                        if pnl_usd > 0:
                            stats["wins"] += 1
                        else:
                            stats["losses"] += 1
                        del positions[symbol]
                        continue

                # Entry check
                if symbol not in positions and len(positions) < self.config.max_positions:
                    sig = self.strategy.generate_signal(df_curr, fg, funding)

                    if sig["signal"] == "LONG":
                        peak = max(self.equity_curve) if self.equity_curve else self.config.initial_capital
                        dd = (capital - peak) / peak if peak > 0 else 0
                        dd_mult = 0.25 if dd < -0.20 else 0.5 if dd < -0.15 else 0.75 if dd < -0.10 else 1.0

                        entry_price = sig["price"] * (1 + self.config.slippage_pct)
                        size = capital * self.config.position_size_pct * sig["pos_mult"] * dd_mult

                        positions[symbol] = {
                            "entry_price": entry_price,
                            "entry_time": ts,
                            "size": size,
                            "leverage": self.config.leverage,
                            "stop_loss": sig["stop_loss"],
                            "take_profit": sig["take_profit"],
                            "atr": sig["atr"],
                            "highest": entry_price,
                            "entry_fg": fg,
                        }

                        stats["longs"] += 1

                        # Track by conviction level
                        if fg and fg <= 15:
                            stats["extreme_fear_trades"] += 1
                        elif fg and fg <= 25:
                            stats["strong_fear_trades"] += 1
                        else:
                            stats["fear_dip_trades"] += 1

                        if stats["longs"] <= 25:
                            logger.info(f"[{ts.strftime('%Y-%m-%d')}] LONG {symbol} @ {entry_price:.2f} - {sig['reason']}")

            # MTM
            pv = capital
            for sym, pos in positions.items():
                if sym in self.data_4h:
                    curr = self.data_4h[sym].loc[self.data_4h[sym].index <= ts]["close"].iloc[-1]
                    unr = (curr - pos["entry_price"]) / pos["entry_price"]
                    pv += pos["size"] * unr * pos["leverage"]
            self.equity_curve.append(pv)

        # Close remaining
        for sym, pos in list(positions.items()):
            if sym in self.data_4h:
                exit_price = self.data_4h[sym].iloc[-1]["close"]
                pnl = (exit_price - pos["entry_price"]) / pos["entry_price"] * pos["leverage"]
                pnl_usd = pos["size"] * pnl - pos["size"] * self.config.commission_pct * 2
                capital += pnl_usd
                self.trade_log.append({
                    "symbol": sym,
                    "entry_time": pos["entry_time"], "entry_price": pos["entry_price"],
                    "exit_time": timestamps[-1], "exit_price": exit_price,
                    "pnl_pct": pnl, "pnl_usd": pnl_usd, "exit_reason": "end",
                })
                if pnl_usd > 0:
                    stats["wins"] += 1
                else:
                    stats["losses"] += 1

        return self._calc_results(stats)

    def _calc_results(self, stats: Dict) -> Dict[str, Any]:
        eq = np.array(self.equity_curve)
        if len(eq) < 2:
            return {"error": "No data"}

        total_ret = (eq[-1] - self.config.initial_capital) / self.config.initial_capital
        n_years = (len(eq) - 1) / (252 * 6)
        ann_ret = (1 + total_ret) ** (1 / max(n_years, 0.1)) - 1
        rets = np.diff(eq) / eq[:-1]
        sharpe = np.mean(rets) / np.std(rets) * np.sqrt(252 * 6) if np.std(rets) > 0 else 0
        peak = np.maximum.accumulate(eq)
        mdd = np.min((eq - peak) / peak)
        total_trades = stats["wins"] + stats["losses"]
        wr = stats["wins"] / total_trades if total_trades > 0 else 0

        if self.trade_log:
            gp = sum(t["pnl_usd"] for t in self.trade_log if t["pnl_usd"] > 0)
            gl = abs(sum(t["pnl_usd"] for t in self.trade_log if t["pnl_usd"] < 0))
            pf = gp / gl if gl > 0 else float("inf")
            avg_win = gp / stats["wins"] if stats["wins"] > 0 else 0
            avg_loss = gl / stats["losses"] if stats["losses"] > 0 else 0
        else:
            pf = avg_win = avg_loss = 0

        exit_reasons = {}
        for t in self.trade_log:
            r = t.get("exit_reason", "unknown")
            if r not in exit_reasons:
                exit_reasons[r] = {"count": 0, "pnl": 0, "wins": 0}
            exit_reasons[r]["count"] += 1
            exit_reasons[r]["pnl"] += t["pnl_usd"]
            if t["pnl_usd"] > 0:
                exit_reasons[r]["wins"] += 1

        return {
            "total_return": total_ret, "annualized_return": ann_ret,
            "sharpe_ratio": sharpe, "max_drawdown": mdd,
            "win_rate": wr, "profit_factor": pf,
            "total_trades": total_trades,
            "wins": stats["wins"], "losses": stats["losses"],
            "extreme_fear_trades": stats["extreme_fear_trades"],
            "strong_fear_trades": stats["strong_fear_trades"],
            "fear_dip_trades": stats["fear_dip_trades"],
            "avg_win": avg_win, "avg_loss": avg_loss,
            "final_capital": eq[-1],
            "exit_reasons": exit_reasons,
        }


def param_sweep(config: BacktestConfig):
    """Sweep fear entry thresholds."""
    results = {}

    for fear in [25, 30, 35, 40]:
        for greed in [65, 70, 75]:
            cfg = BacktestConfig(
                initial_capital=config.initial_capital,
                start_date=config.start_date,
                end_date=config.end_date,
                symbols=config.symbols,
                leverage=config.leverage,
                fear_entry=fear,
                greed_exit=greed,
                trend_deviation=config.trend_deviation,
            )
            bt = LongOnlyBacktester(cfg)
            if bt.load_data():
                res = bt.run()
                if "error" not in res:
                    label = f"F{fear}_G{greed}"
                    results[label] = res
                    logger.info(f"\n{label}: Ret={res['total_return']*100:+.1f}%, "
                               f"MDD={res['max_drawdown']*100:.1f}%, "
                               f"WR={res['win_rate']*100:.1f}%, "
                               f"PF={res['profit_factor']:.2f}, "
                               f"Trades={res['total_trades']}")
    return results


def main():
    parser = argparse.ArgumentParser(description="Long-Only Fear-Based Strategy (v12)")
    parser.add_argument("--symbols", type=str, default="BTCUSDT,ETHUSDT")
    parser.add_argument("--start", type=str, default="2020-01-01")
    parser.add_argument("--end", type=str, default=None)
    parser.add_argument("--capital", type=float, default=10000.0)
    parser.add_argument("--leverage", type=int, default=3)
    parser.add_argument("--fear", type=int, default=35)
    parser.add_argument("--greed", type=int, default=70)
    parser.add_argument("--deviation", type=float, default=0.01)
    parser.add_argument("--sweep", action="store_true")

    args = parser.parse_args()

    config = BacktestConfig(
        initial_capital=args.capital,
        start_date=args.start,
        end_date=args.end,
        symbols=args.symbols.split(","),
        leverage=args.leverage,
        fear_entry=args.fear,
        greed_exit=args.greed,
        trend_deviation=args.deviation,
    )

    logger.info("=" * 60)
    logger.info("V12 LONG-ONLY FEAR-BASED STRATEGY")
    logger.info("=" * 60)

    if args.sweep:
        sweep_results = param_sweep(config)
        logger.info("\n" + "=" * 60)
        logger.info("PARAMETER SWEEP SUMMARY")
        logger.info("=" * 60)

        best = None
        best_pf = 0
        for label, res in sorted(sweep_results.items(), key=lambda x: x[1]["profit_factor"], reverse=True):
            if res["profit_factor"] > best_pf:
                best_pf = res["profit_factor"]
                best = label
            logger.info(f"{label}: Ret={res['total_return']*100:+.1f}%, "
                       f"PF={res['profit_factor']:.2f}, WR={res['win_rate']*100:.1f}%, "
                       f"Trades={res['total_trades']}")

        if best:
            logger.info(f"\nBest: {best} with PF={best_pf:.2f}")
        return 0

    bt = LongOnlyBacktester(config)
    if not bt.load_data():
        return 1

    results = bt.run()

    logger.info("\n" + "=" * 60)
    logger.info("RESULTS")
    logger.info("=" * 60)
    logger.info(f"  Total Return: {results['total_return']*100:.2f}%")
    logger.info(f"  Annualized: {results['annualized_return']*100:.2f}%")
    logger.info(f"  Sharpe: {results['sharpe_ratio']:.2f}")
    logger.info(f"  Max DD: {results['max_drawdown']*100:.2f}%")
    logger.info(f"  Win Rate: {results['win_rate']*100:.1f}%")
    logger.info(f"  Profit Factor: {results['profit_factor']:.2f}")
    logger.info(f"  Trades: {results['total_trades']}")
    logger.info(f"    Extreme Fear: {results['extreme_fear_trades']}")
    logger.info(f"    Strong Fear: {results['strong_fear_trades']}")
    logger.info(f"    Fear Dip: {results['fear_dip_trades']}")
    logger.info(f"  Avg Win: ${results['avg_win']:.2f}, Avg Loss: ${results['avg_loss']:.2f}")
    logger.info(f"  Final: ${results['final_capital']:,.2f}")

    logger.info("\n  Exit Reasons:")
    for reason, data in results.get("exit_reasons", {}).items():
        wr = data['wins'] / data['count'] * 100 if data['count'] > 0 else 0
        logger.info(f"    {reason}: {data['count']} trades, ${data['pnl']:.2f}, WR={wr:.0f}%")

    logger.info("\n" + "=" * 60)
    logger.info("TARGET CHECK")
    logger.info("=" * 60)
    pf_ok = "✓" if results['profit_factor'] >= 1.3 else "✗"
    mdd_ok = "✓" if results['max_drawdown'] >= -0.25 else "✗"
    logger.info(f"  [{pf_ok}] Profit Factor: {results['profit_factor']:.2f} (target >= 1.30)")
    logger.info(f"  [{mdd_ok}] Max DD: {results['max_drawdown']*100:.1f}% (target < 25%)")

    if results['profit_factor'] >= 1.3:
        logger.info("\n  *** TARGET MET - Strategy ready for paper trading validation ***")

    return 0


if __name__ == "__main__":
    sys.exit(main())
