"""Risk manager.

Single point of truth for portfolio-level risk. Strategies submit candidate
orders; the manager approves, scales, or rejects each based on:

  • Kill-switch state (drawdown thresholds)
  • Per-position cap (max % of equity per symbol)
  • Per-strategy cap (enforced by allocator + double-checked here)
  • Leverage cap (sum of notional across all positions)
  • Dynamic risk multiplier (auto-deleverage knob)
  • Min trade size (fee viability)

It also persists equity snapshots so the rolling peak (and therefore
drawdown) can be computed across runs.
"""
from __future__ import annotations

import logging
import math
import os
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Dict, Iterable, List, Optional

from .multiplier import DynamicRiskMultiplier, MultiplierState
from .policies import KillSwitchState, RiskConfig

logger = logging.getLogger(__name__)


# ─── Decision types ──────────────────────────────────────────────────────


class Decision(str, Enum):
    APPROVE = "APPROVE"
    SCALE = "SCALE"      # approve but at a different size
    REJECT = "REJECT"


@dataclass
class RiskDecision:
    """Returned by `RiskManager.check_order`."""

    decision: Decision
    approved_notional_usd: float = 0.0   # post-scaling order size in USD
    reason: str = ""
    state: Optional["RiskState"] = None


@dataclass
class RiskState:
    """Snapshot of portfolio risk metrics at a point in time."""

    timestamp: datetime
    equity_usd: float
    peak_equity_usd: float
    drawdown_pct: float
    kill_switch: KillSwitchState
    realized_vol: Optional[float]      # annualized, None until enough samples
    leverage: float                    # gross notional / equity
    multiplier: MultiplierState
    venues_ok: bool = True             # any broker auth/API problems?

    def is_tradeable(self) -> bool:
        return self.kill_switch in (KillSwitchState.NORMAL, KillSwitchState.WARNING)

    def is_closing_only(self) -> bool:
        return self.kill_switch == KillSwitchState.CRITICAL


# ─── Persistence (equity snapshots) ──────────────────────────────────────


class EquitySnapshotDB:
    """Tiny SQLite table for the rolling-peak drawdown calc."""

    def __init__(self, db_path: Optional[str] = None):
        self.db_path = os.path.abspath(
            db_path or os.environ.get("RISK_DB_PATH", "data/risk_state.db")
        )
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        with self._conn() as c:
            c.execute("""
                CREATE TABLE IF NOT EXISTS equity_snapshots (
                    id          INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp   TEXT    NOT NULL,
                    equity_usd  REAL    NOT NULL,
                    note        TEXT
                )
            """)
            c.execute("""
                CREATE TABLE IF NOT EXISTS kill_switch_events (
                    id           INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp    TEXT    NOT NULL,
                    state        TEXT    NOT NULL,
                    drawdown_pct REAL    NOT NULL,
                    note         TEXT
                )
            """)

    @contextmanager
    def _conn(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def record_snapshot(self, equity_usd: float, note: str = "") -> None:
        with self._conn() as c:
            c.execute(
                "INSERT INTO equity_snapshots (timestamp, equity_usd, note) VALUES (?, ?, ?)",
                (datetime.now(timezone.utc).isoformat(), equity_usd, note),
            )

    def peak_equity(self, since: Optional[datetime] = None) -> float:
        with self._conn() as c:
            if since:
                row = c.execute(
                    "SELECT MAX(equity_usd) AS peak FROM equity_snapshots WHERE timestamp >= ?",
                    (since.isoformat(),),
                ).fetchone()
            else:
                row = c.execute(
                    "SELECT MAX(equity_usd) AS peak FROM equity_snapshots"
                ).fetchone()
        return float(row["peak"]) if row and row["peak"] is not None else 0.0

    def recent_returns(self, n: int = 60) -> List[float]:
        """Returns the last `n` snapshot-to-snapshot pct returns."""
        with self._conn() as c:
            rows = c.execute(
                "SELECT equity_usd FROM equity_snapshots ORDER BY id DESC LIMIT ?",
                (n + 1,),
            ).fetchall()
        eq = [float(r["equity_usd"]) for r in rows][::-1]  # ascending
        out = []
        for i in range(1, len(eq)):
            if eq[i - 1] > 0:
                out.append((eq[i] - eq[i - 1]) / eq[i - 1])
        return out

    def record_kill_switch(self, state: KillSwitchState, dd_pct: float, note: str = "") -> None:
        with self._conn() as c:
            c.execute(
                "INSERT INTO kill_switch_events (timestamp, state, drawdown_pct, note) VALUES (?, ?, ?, ?)",
                (datetime.now(timezone.utc).isoformat(), state.value, dd_pct, note),
            )

    def last_kill_switch_event(self) -> Optional[Dict]:
        with self._conn() as c:
            row = c.execute(
                "SELECT * FROM kill_switch_events ORDER BY id DESC LIMIT 1"
            ).fetchone()
        return dict(row) if row else None


# ─── Risk manager ────────────────────────────────────────────────────────


class RiskManager:
    """Orchestrates risk checks for a single trading run."""

    def __init__(
        self,
        brokers: Optional[Dict] = None,    # broker registry; if None, no live equity
        config: Optional[RiskConfig] = None,
        db: Optional[EquitySnapshotDB] = None,
    ):
        self.brokers = brokers or {}
        self.config = config or RiskConfig.from_env()
        self.multiplier = DynamicRiskMultiplier(self.config)
        self.db = db or EquitySnapshotDB()
        self._cached_state: Optional[RiskState] = None
        # Per-cycle broker snapshot cache; refreshed by compute_state()
        self._broker_snapshots: Dict[str, Dict] = {}

    def cached_positions(self, venue: str) -> List:
        """Read-through accessor for the per-cycle position cache.
        Saves redundant get_positions() calls during a single cycle."""
        snap = self._broker_snapshots.get(venue, {})
        return snap.get("positions", [])

    # ── State computation -------------------------------------------------

    def compute_state(self, *, persist: bool = True) -> RiskState:
        """Pull equity from every configured broker, persist snapshot, and
        derive drawdown / vol / multiplier / kill-switch state.

        Performance: each broker's account + positions are fetched once
        and cached in `self._broker_snapshots` for the rest of the cycle
        (so the orchestrator doesn't double-call get_positions on its
        own venue lookups). The cache is cleared by the next
        compute_state() call.
        """
        equity = 0.0
        notional = 0.0
        venues_ok = True
        self._broker_snapshots = {}     # venue → {"account", "positions"}
        for name, adapter in self.brokers.items():
            snap = {}
            try:
                acct = adapter.get_account()
                snap["account"] = acct
                equity += acct.equity_usd
            except Exception as e:
                logger.warning(f"[risk] {name} get_account failed: {e}")
                venues_ok = False
            try:
                positions = adapter.get_positions()
                snap["positions"] = positions
                notional += sum(abs(p.market_price * p.quantity) for p in positions)
            except Exception as e:
                logger.debug(f"[risk] {name} get_positions failed: {e}")
            self._broker_snapshots[name] = snap

        if persist and equity > 0:
            self.db.record_snapshot(equity, note=f"venues={len(self.brokers)}")

        peak = max(self.db.peak_equity(), equity)
        dd_pct = (peak - equity) / peak if peak > 0 else 0.0

        # Realized vol: annualized stdev of recent returns, assuming 5-min cadence
        # (with hourly cron there are ~12 returns/day, ~252 trading days/year)
        rets = self.db.recent_returns(60)
        realized_vol: Optional[float] = None
        if len(rets) >= 10:
            mean = sum(rets) / len(rets)
            var = sum((r - mean) ** 2 for r in rets) / (len(rets) - 1)
            sd = math.sqrt(var)
            # Cron is hourly currently; annualization factor sqrt(24*365)
            realized_vol = sd * math.sqrt(24 * 365)

        leverage = (notional / equity) if equity > 0 else 0.0

        # Pull current VIX from the macro scout's signal-bus output if
        # available; falls back to None gracefully when the bus is empty.
        vix_now = self._latest_vix()

        # Multiplier — auto-adjusts on dd / vol / regime
        mult_state = self.multiplier.compute(
            drawdown_pct=dd_pct,
            realized_vol=realized_vol,
            vix=vix_now,
        )

        ks_state = self.config.state_for_drawdown(dd_pct)
        if ks_state == KillSwitchState.KILL:
            self.db.record_kill_switch(ks_state, dd_pct,
                                        note=f"equity=${equity:.2f} peak=${peak:.2f}")

        state = RiskState(
            timestamp=datetime.now(timezone.utc),
            equity_usd=equity,
            peak_equity_usd=peak,
            drawdown_pct=dd_pct,
            kill_switch=ks_state,
            realized_vol=realized_vol,
            leverage=leverage,
            multiplier=mult_state,
            venues_ok=venues_ok,
        )
        self._cached_state = state
        return state

    # ── Order-level gate -------------------------------------------------

    def check_order(
        self,
        *,
        notional_usd: float,
        symbol: str,
        is_closing: bool = False,
        strategy_name: Optional[str] = None,
        existing_position_usd: float = 0.0,
        state: Optional[RiskState] = None,
        venue: Optional[str] = None,
    ) -> RiskDecision:
        """Approve, scale, or reject a candidate order.

        Args:
            notional_usd: requested order size (always positive USD).
            symbol: instrument identifier.
            is_closing: True if this trade reduces an existing position.
                Closing trades bypass kill-switch CRITICAL but not KILL.
            strategy_name: for logging / per-strategy exposure tracking.
            existing_position_usd: current absolute exposure in this symbol.
            state: pass an already-computed RiskState to avoid re-fetching.
        """
        st = state or self._cached_state or self.compute_state()
        cfg = self.config

        # ─ Kill switch
        if st.kill_switch == KillSwitchState.KILL:
            return RiskDecision(
                Decision.REJECT, 0.0,
                f"KILL switch active (DD {st.drawdown_pct * 100:.1f}%)", st)
        if st.kill_switch == KillSwitchState.CRITICAL and not is_closing:
            return RiskDecision(
                Decision.REJECT, 0.0,
                f"CRITICAL state — closing-only (DD {st.drawdown_pct * 100:.1f}%)", st)

        # ─ Min size
        if notional_usd < cfg.min_trade_usd:
            return RiskDecision(
                Decision.REJECT, 0.0,
                f"order ${notional_usd:.2f} < min ${cfg.min_trade_usd:.2f}", st)

        # ─ Apply dynamic risk multiplier (skip for closing trades)
        approved = notional_usd
        if not is_closing:
            approved = approved * st.multiplier.effective
            if st.multiplier.effective < 0.99:
                logger.info(
                    f"[risk] {strategy_name or '?'} sizing scaled by "
                    f"{st.multiplier.effective:.2f} ({', '.join(st.multiplier.notes) or 'base'})"
                )

        # ─ Per-order ceiling (scales with equity); per-venue override wins
        venue_cap = cfg.cap_for_venue(venue) if venue else cfg.max_trade_usd
        per_order_cap = min(venue_cap,
                            cfg.max_position_pct * st.equity_usd)
        if approved > per_order_cap:
            approved = per_order_cap

        # ─ Per-position cap (incl. existing exposure)
        max_position = cfg.max_position_pct * st.equity_usd
        if not is_closing and existing_position_usd + approved > max_position:
            approved = max(0.0, max_position - existing_position_usd)

        # ─ Leverage cap
        if not is_closing and st.equity_usd > 0:
            current_lev_notional = st.leverage * st.equity_usd
            max_total_notional = cfg.leverage_cap * st.equity_usd
            headroom = max(0.0, max_total_notional - current_lev_notional)
            if approved > headroom:
                approved = headroom

        # ─ Final viability after scaling
        if approved < cfg.min_trade_usd:
            return RiskDecision(
                Decision.REJECT, 0.0,
                f"after scaling, size ${approved:.2f} < min ${cfg.min_trade_usd:.2f}", st)

        if abs(approved - notional_usd) / max(notional_usd, 1e-9) > 0.01:
            return RiskDecision(
                Decision.SCALE, approved,
                f"scaled from ${notional_usd:.2f} to ${approved:.2f}", st)

        return RiskDecision(Decision.APPROVE, approved, "ok", st)

    # ── Macro signal hookup ---------------------------------------------

    def _latest_vix(self) -> Optional[float]:
        """Read the macro scout's most-recent VIX reading from the bus."""
        try:
            from scouts.signal_bus import SignalBus
            bus = SignalBus()
            for row in bus.latest(venue="macro", signal_type="vix_regime", limit=5):
                if row.is_fresh():
                    vix = row.payload.get("vix")
                    if vix is not None:
                        return float(vix)
        except Exception:
            pass
        return None

    # ── Manual controls --------------------------------------------------

    def set_risk_multiplier(self, value: float) -> None:
        """User-set base multiplier. Effective value still depends on
        auto-adjustment factors."""
        self.multiplier.set_base(value)

    def reset_kill_switch(self) -> None:
        """Manual restart after a KILL event. Doesn't reset peak — drawdown
        is computed from the all-time high until equity recovers."""
        self.db.record_kill_switch(
            KillSwitchState.NORMAL, 0.0, note="manual reset"
        )

    # ── Diagnostics ------------------------------------------------------

    def summary_dict(self) -> Dict:
        st = self._cached_state or self.compute_state()
        return {
            "timestamp": st.timestamp.isoformat(),
            "equity_usd": st.equity_usd,
            "peak_equity_usd": st.peak_equity_usd,
            "drawdown_pct": st.drawdown_pct,
            "kill_switch": st.kill_switch.value,
            "realized_vol": st.realized_vol,
            "leverage": st.leverage,
            "multiplier": {
                "base": st.multiplier.base,
                "effective": st.multiplier.effective,
                "notes": st.multiplier.notes,
            },
            "config": {
                "target_vol": self.config.target_portfolio_vol,
                "warning_dd": self.config.warning_dd_pct,
                "critical_dd": self.config.critical_dd_pct,
                "kill_dd": self.config.kill_dd_pct,
                "leverage_cap": self.config.leverage_cap,
            },
        }
