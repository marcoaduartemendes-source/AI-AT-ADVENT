import logging
import os
import time
from dataclasses import dataclass
from datetime import datetime

from .coinbase_client import CoinbaseClient
from .strategies.base import Signal, SignalType

logger = logging.getLogger(__name__)

# Hard cap: never exceed this per order regardless of config
HARD_CAP_USD = 20.0
# Minimum order size (Coinbase requirement is typically $1)
MIN_ORDER_USD = 1.0
# Seconds between trades on the same (strategy, product) pair (overridable via env)
COOLDOWN_SECONDS = int(os.environ.get("COOLDOWN_SECONDS", "900"))


@dataclass
class Position:
    product_id: str
    quantity: float
    cost_basis_usd: float
    entry_price: float
    entry_time: datetime
    strategy: str


@dataclass
class TradeRecord:
    timestamp: datetime
    strategy: str
    product_id: str
    side: str
    amount_usd: float
    quantity: float
    price: float
    order_id: str
    pnl_usd: float | None = None
    dry_run: bool = False


class PortfolioManager:
    def __init__(
        self,
        client: CoinbaseClient,
        max_trade_usd: float = HARD_CAP_USD,
        dry_run: bool = True,
        min_confidence: float = 0.6,
        stop_loss_pct: float = 0.02,
        take_profit_pct: float = 0.04,
        max_open_positions: int = 5,
    ):
        self.client = client
        self.max_trade_usd = min(max_trade_usd, HARD_CAP_USD)
        self.dry_run = dry_run
        self.min_confidence = min_confidence
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
        self.max_open_positions = max_open_positions

        # positions[strategy][product_id] = Position
        self.positions: dict[str, dict[str, Position]] = {}
        # last_trade_time[strategy][product_id] = unix timestamp
        self.last_trade_time: dict[str, dict[str, float]] = {}
        self.trade_history: list[TradeRecord] = []
        self._tracker = None  # injected after construction

    def attach_tracker(self, tracker) -> None:
        """Attach a PerformanceTracker so positions survive between runs."""
        self._tracker = tracker
        restored = tracker.load_positions()
        if restored:
            self.positions = restored
            count = sum(len(v) for v in restored.values())
            logger.info(f"Restored {count} open position(s) from database.")

    # ── Internal helpers ─────────────────────────────────────────────────────

    def _get_position(self, strategy: str, product_id: str) -> Position | None:
        return self.positions.get(strategy, {}).get(product_id)

    def _set_position(self, strategy: str, product_id: str, pos: Position | None):
        self.positions.setdefault(strategy, {})
        if pos is None:
            self.positions[strategy].pop(product_id, None)
        else:
            self.positions[strategy][product_id] = pos
        if self._tracker:
            self._tracker.save_positions(self.positions)

    def _in_cooldown(self, strategy: str, product_id: str) -> bool:
        last = self.last_trade_time.get(strategy, {}).get(product_id, 0)
        return (time.time() - last) < COOLDOWN_SECONDS

    def _touch_cooldown(self, strategy: str, product_id: str):
        self.last_trade_time.setdefault(strategy, {})[product_id] = time.time()

    def _open_position_count(self) -> int:
        return sum(len(v) for v in self.positions.values())

    # ── Risk management ──────────────────────────────────────────────────────

    def check_stops(self, current_prices: dict[str, float]) -> list["TradeRecord"]:
        """Force-close any position whose live price hit stop-loss or take-profit.

        current_prices: {product_id: latest_price}
        Returns list of TradeRecords for any forced exits.
        """
        forced_exits: list[TradeRecord] = []
        # Snapshot to avoid mutating dict during iteration
        for strategy, prod_map in list(self.positions.items()):
            for product_id, pos in list(prod_map.items()):
                price = current_prices.get(product_id)
                if price is None or pos.entry_price <= 0:
                    continue
                ret = (price - pos.entry_price) / pos.entry_price
                exit_reason = None
                if ret <= -self.stop_loss_pct:
                    exit_reason = f"STOP-LOSS ({ret * 100:+.2f}%)"
                elif ret >= self.take_profit_pct:
                    exit_reason = f"TAKE-PROFIT ({ret * 100:+.2f}%)"
                if not exit_reason:
                    continue

                logger.warning(
                    f"[{strategy}] Forced exit on {product_id} — {exit_reason} "
                    f"entry=${pos.entry_price:.4f} now=${price:.4f}"
                )
                exit_signal = Signal(
                    strategy_name=strategy,
                    product_id=product_id,
                    signal=SignalType.SELL,
                    confidence=1.0,  # bypass min_confidence
                    price=price,
                    reason=exit_reason,
                    metadata={"forced": True},
                )
                # Bypass cooldown for safety exits
                rec = self._execute_sell(exit_signal, pos)
                if rec:
                    forced_exits.append(rec)
        return forced_exits

    # ── Public API ───────────────────────────────────────────────────────────

    def process_signal(self, signal: Signal) -> TradeRecord | None:
        strategy, product_id = signal.strategy_name, signal.product_id

        if signal.signal == SignalType.HOLD:
            return None

        if signal.confidence < self.min_confidence:
            logger.debug(
                f"[{strategy}] {signal.signal.value} {product_id} skipped — "
                f"confidence {signal.confidence:.2f} < {self.min_confidence}"
            )
            return None

        if self._in_cooldown(strategy, product_id):
            logger.debug(f"[{strategy}] {product_id} in cooldown, skipping")
            return None

        existing = self._get_position(strategy, product_id)

        if signal.signal == SignalType.BUY:
            if existing is not None:
                logger.debug(f"[{strategy}] Already holding {product_id}, skipping BUY")
                return None
            if self._open_position_count() >= self.max_open_positions:
                logger.info(
                    f"[{strategy}] BUY {product_id} skipped — "
                    f"already at max_open_positions={self.max_open_positions}"
                )
                return None
            return self._execute_buy(signal)

        if signal.signal == SignalType.SELL:
            if existing is None:
                logger.debug(f"[{strategy}] No position in {product_id}, skipping SELL")
                return None
            return self._execute_sell(signal, existing)

        return None

    def _execute_buy(self, signal: Signal) -> TradeRecord | None:
        strategy, product_id = signal.strategy_name, signal.product_id
        amount_usd = self.max_trade_usd
        price = signal.price
        quantity = amount_usd / price if price > 0 else 0

        tag = "DRY RUN" if self.dry_run else "LIVE"
        logger.info(
            f"[{strategy}] BUY {product_id}: ${amount_usd:.2f} @ ~${price:.4f} [{tag}]"
        )

        order_id = f"paper_{int(time.time())}"
        if not self.dry_run:
            try:
                result = self.client.create_market_buy(product_id, f"{amount_usd:.2f}")
                order_id = (
                    result.get("order_id")
                    or result.get("success_response", {}).get("order_id", "unknown")
                )
                if order_id and order_id not in ("unknown", ""):
                    time.sleep(1)
                    details = self.client.get_order(order_id)
                    if details.get("average_filled_price"):
                        price = float(details["average_filled_price"])
                    if details.get("filled_size"):
                        quantity = float(details["filled_size"])
            except Exception as exc:
                logger.error(f"[{strategy}] BUY failed for {product_id}: {exc}")
                return None

        position = Position(
            product_id=product_id,
            quantity=quantity,
            cost_basis_usd=amount_usd,
            entry_price=price,
            entry_time=datetime.utcnow(),
            strategy=strategy,
        )
        self._set_position(strategy, product_id, position)
        self._touch_cooldown(strategy, product_id)

        record = TradeRecord(
            timestamp=datetime.utcnow(),
            strategy=strategy,
            product_id=product_id,
            side="BUY",
            amount_usd=amount_usd,
            quantity=quantity,
            price=price,
            order_id=order_id,
            dry_run=self.dry_run,
        )
        self.trade_history.append(record)
        return record

    def _execute_sell(self, signal: Signal, position: Position) -> TradeRecord | None:
        strategy, product_id = signal.strategy_name, signal.product_id
        price = signal.price
        quantity = position.quantity
        amount_usd = quantity * price
        pnl = amount_usd - position.cost_basis_usd

        tag = "DRY RUN" if self.dry_run else "LIVE"
        logger.info(
            f"[{strategy}] SELL {product_id}: {quantity:.8f} @ ~${price:.4f} "
            f"(PnL: ${pnl:+.2f}) [{tag}]"
        )

        order_id = f"paper_{int(time.time())}"
        if not self.dry_run:
            try:
                result = self.client.create_market_sell(product_id, f"{quantity:.8f}")
                order_id = (
                    result.get("order_id")
                    or result.get("success_response", {}).get("order_id", "unknown")
                )
                if order_id and order_id not in ("unknown", ""):
                    time.sleep(1)
                    details = self.client.get_order(order_id)
                    if details.get("average_filled_price"):
                        price = float(details["average_filled_price"])
                        amount_usd = quantity * price
                        pnl = amount_usd - position.cost_basis_usd
            except Exception as exc:
                logger.error(f"[{strategy}] SELL failed for {product_id}: {exc}")
                return None

        self._set_position(strategy, product_id, None)
        self._touch_cooldown(strategy, product_id)

        record = TradeRecord(
            timestamp=datetime.utcnow(),
            strategy=strategy,
            product_id=product_id,
            side="SELL",
            amount_usd=amount_usd,
            quantity=quantity,
            price=price,
            order_id=order_id,
            pnl_usd=pnl,
            dry_run=self.dry_run,
        )
        self.trade_history.append(record)
        return record

    def get_open_positions(self) -> dict:
        out: dict = {}
        for strat, pos_map in self.positions.items():
            for pid, pos in pos_map.items():
                out.setdefault(strat, {})[pid] = {
                    "quantity": pos.quantity,
                    "cost_basis_usd": pos.cost_basis_usd,
                    "entry_price": pos.entry_price,
                    "entry_time": pos.entry_time.isoformat(),
                }
        return out
