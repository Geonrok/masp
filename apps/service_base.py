"""
Base service template for all asset class services.
"""

import argparse
import time
import traceback
from pathlib import Path
from typing import Optional

from libs.core.config import AssetClass, Config, load_config
from libs.core.event_logger import EventLogger
from libs.core.event_store import EventStore
from libs.core.run_manager import RunContext, RunManager
from libs.core.scheduler import Scheduler
from libs.core.paths import find_repo_root
from libs.strategies.base import Action, StrategyContext
from libs.strategies.loader import load_strategies
from libs.adapters.mock import MockExecutionAdapter, MockMarketDataAdapter

REPO_ROOT = find_repo_root(Path(__file__))
STORAGE_DIR = REPO_ROOT / "storage"
STORAGE_DIR.mkdir(parents=True, exist_ok=True)
STORAGE_PATH = STORAGE_DIR / "local.db"


class StrategyService:
    def __init__(
        self,
        asset_class: AssetClass,
        config_path: Optional[Path] = None,
        config_overrides: Optional[dict] = None,
    ):
        self.asset_class = asset_class
        overrides = config_overrides or {}
        if "storage_dsn" not in overrides:
            overrides["storage_dsn"] = str(STORAGE_PATH)

        self.config = load_config(
            asset_class=asset_class,
            config_path=config_path,
            overrides=overrides,
        )
        
        # Phase 1 준비: real mode 검증 (fail-fast, 항상 호출)
        self.config.validate_real_mode_requirements()
        
        # Phase 1 준비: adapter_mode=mock이면 자동으로 dry_run=True
        if self.config.adapter_mode == "mock":
            self.config.dry_run = True

        self.event_store = EventStore(dsn=self.config.storage_dsn)
        self.run_manager = RunManager(asset_class=asset_class.value)
        self.market_adapter = MockMarketDataAdapter()
        self.execution_adapter = MockExecutionAdapter()
        self.strategies = load_strategies(self.config.enabled_strategies)
        self._start_time = time.time()
        self._heartbeat_run_id = f"{asset_class.value}_heartbeat"

        print(f"[Service] {self.config.effective_service_name}")
        print(f"[Service] Storage: {self.config.storage_dsn}")
        print(f"[Service] Adapter Mode: {self.config.adapter_mode}")
        print(f"[Service] Dry-Run: {self.config.dry_run}")
        if self.config.kill_switch_file:
            print(f"[Service] Kill-Switch: {self.config.kill_switch_file}")

    def _check_kill_switch(self) -> bool:
        """Check if kill switch file exists. Returns True if should terminate."""
        if not self.config.kill_switch_file:
            return False
        
        kill_switch_path = Path(self.config.kill_switch_file)
        if kill_switch_path.exists():
            print(f"[KILL-SWITCH] File detected: {kill_switch_path}")
            print(f"[KILL-SWITCH] Terminating service immediately")
            return True
        return False

    def _execute_run(self) -> None:
        # Phase 1 준비: Kill-switch 체크
        if self._check_kill_switch():
            print("[Service] Kill-switch activated, aborting run")
            return
        
        run_id = self.run_manager.generate_run_id()
        run_ctx = RunContext(run_id=run_id, asset_class=self.asset_class.value)
        run_ctx.start()

        logger = EventLogger(
            asset_class=self.asset_class.value,
            run_id=run_id,
            event_store=self.event_store,
        )

        try:
            # Phase 1 준비: RUN_STARTED에 안전장치 상태 기록
            logger.emit_run_started(
                config_version=self.config.config_version,
                enabled_strategies=[s.strategy_id for s in self.strategies],
                symbols=self.config.symbols,
                extra={
                    "dry_run": self.config.dry_run,
                    "adapter_mode": self.config.adapter_mode,
                    "kill_switch_enabled": (self.config.kill_switch_file is not None),
                },
            )


            market_data = self.market_adapter.get_quotes(self.config.symbols)

            for strategy in self.strategies:
                try:
                    ctx = StrategyContext(
                        config=self.config,
                        run_id=run_id,
                        event_logger=logger,
                        symbols=self.config.symbols,
                        market_data=market_data,
                    )
                    decisions = strategy.execute(ctx)

                    for decision in decisions:
                        logger.emit_signal_decision(
                            strategy_id=strategy.strategy_id,
                            symbol=decision.symbol,
                            action=decision.action.value,
                            notes=decision.notes,
                            metrics=decision.metrics,
                        )
                        run_ctx.increment_decisions()

                        if self.config.paper_mode and decision.action in (Action.BUY, Action.SELL):
                            self._emit_mock_order(logger, strategy.strategy_id, decision, market_data)

                except Exception as e:
                    logger.emit_error(
                        error_message=str(e),
                        error_type=type(e).__name__,
                        strategy_id=strategy.strategy_id,
                        traceback=traceback.format_exc(),
                    )
                    run_ctx.increment_errors()

            run_ctx.finish(success=True)

        except Exception as e:
            logger.emit_error(
                error_message=str(e),
                error_type=type(e).__name__,
                traceback=traceback.format_exc(),
            )
            run_ctx.increment_errors()
            run_ctx.finish(success=False)

        finally:
            logger.emit_run_finished(
                status=run_ctx.status,
                decision_count=run_ctx.decision_count,
                error_count=run_ctx.error_count,
                duration_seconds=run_ctx.duration_seconds,
            )
            print(f"[Service] Run {run_id}: {run_ctx.status}")

    def _emit_mock_order(self, logger, strategy_id, decision, market_data):
        symbol = decision.symbol
        side = decision.action.value
        quote = market_data.get(symbol)
        price = quote.last if quote else 100.0
        quantity = 0.01 if "BTC" in symbol else 1.0

        logger.emit_order_attempt(
            strategy_id=strategy_id,
            symbol=symbol,
            side=side,
            quantity=quantity,
            price=price,
            order_type="MARKET",
        )

        result = self.execution_adapter.place_order(
            symbol=symbol,
            side=side,
            quantity=quantity,
            order_type="MARKET",
        )

        if result.success:
            logger.emit_fill_update(
                strategy_id=strategy_id,
                symbol=symbol,
                side=side,
                filled_quantity=quantity,
                fill_price=result.price or price,
                order_id=result.order_id,
            )

    def _emit_heartbeat(self):
        logger = EventLogger(
            asset_class=self.asset_class.value,
            run_id=self._heartbeat_run_id,
            event_store=self.event_store,
        )
        logger.emit_heartbeat(
            build_version=self.config.build_version,
            config_version=self.config.config_version,
            uptime_seconds=time.time() - self._start_time,
        )

    def run_once(self):
        if self._check_kill_switch():
            return  # Terminate immediately
        self._emit_heartbeat()
        self._execute_run()

    def run_daemon(self):
        scheduler = Scheduler(
            config=self.config,
            run_callback=self._execute_run,
            heartbeat_callback=self._emit_heartbeat,
            heartbeat_interval=30,
        )
        scheduler.run_daemon()


def create_service_main(asset_class: AssetClass):
    def main():
        parser = argparse.ArgumentParser(description=f"{asset_class.value} Service")
        parser.add_argument("--once", action="store_true")
        parser.add_argument("--daemon", action="store_true")
        parser.add_argument("--config", type=str)
        parser.add_argument("--interval", type=int)
        args = parser.parse_args()

        overrides = {}
        if args.interval:
            overrides["schedule"] = {"interval_seconds": args.interval}

        config_path = Path(args.config) if args.config else None
        service = StrategyService(
            asset_class=asset_class,
            config_path=config_path,
            config_overrides=overrides if overrides else None,
        )

        if args.daemon:
            service.run_daemon()
        else:
            service.run_once()

    return main
