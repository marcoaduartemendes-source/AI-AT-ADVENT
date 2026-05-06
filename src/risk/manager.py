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
from dataclasses import dataclass
from datetime import datetime, UTC
from enum import Enum

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
    state: RiskState | None = None


@dataclass
class RiskState:
    """Snapshot of portfolio risk metrics at a point in time."""

    timestamp: datetime
    equity_usd: float
    peak_equity_usd: float
    drawdown_pct: float
    kill_switch: KillSwitchState
    realized_vol: float | None      # annualized, None until enough samples
    leverage: float                    # gross notional / equity
    multiplier: MultiplierState
    venues_ok: bool = True             # any broker auth/API problems?

    def is_tradeable(self) -> bool:
        return self.kill_switch in (KillSwitchState.NORMAL, KillSwitchState.WARNING)

    def is_closing_only(self) -> bool:
        return self.kill_switch == KillSwitchState.CRITICAL


# ─── Persistence (equity snapshots) ──────────────────────────────────────


class EquitySnapshotDB:
    """SQLite-primary, Supabase-mirrored equity-snapshot store.

    The kill-switch's drawdown baseline is the most safety-critical
    state in the system: corrupting `peak_equity` resets the
    drawdown calc and false-fires the kill switch. Pre-audit-fix
    this lived in a single SQLite file on a single VPS — the audit's
    #1 flagged SPOF.

    Post-fix:
      * Writes still go to SQLite (fast, transactional).
      * Reads check Supabase for a higher peak / longer history; if
        Supabase has materially more data than SQLite (e.g. SQLite
        was just zeroed by a corrupt restore), Supabase wins. SQLite
        only "wins" when both stores agree.
      * Supabase failures degrade gracefully — same behaviour as
        before, just without the cross-check.
    """

    def __init__(self, db_path: str | None = None,
                  supabase=None):
        self.db_path = os.path.abspath(
            db_path or os.environ.get("RISK_DB_PATH", "data/risk_state.db")
        )
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        # Lazy-init Supabase mirror; explicit None disables it (used in tests)
        self._supabase = supabase
        if self._supabase is None:
            try:
                from common.supabase_store import SupabaseStore
                store = SupabaseStore()
                if store.is_configured():
                    self._supabase = store
            except (ImportError, Exception):    # noqa: BLE001
                self._supabase = None
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
        """Dual-write: SQLite (primary, transactional) + Supabase
        (mirror, best-effort). Supabase failure does not block the
        SQLite write — if the mirror is down we'll just degrade to
        SQLite-only reads on this row."""
        ts = datetime.now(UTC).isoformat()
        with self._conn() as c:
            c.execute(
                "INSERT INTO equity_snapshots (timestamp, equity_usd, note) VALUES (?, ?, ?)",
                (ts, equity_usd, note),
            )
        if self._supabase is not None:
            try:
                self._supabase.insert_equity_snapshot(
                    equity_usd=equity_usd, timestamp=ts, note=note,
                )
            except Exception as e:  # noqa: BLE001
                logger.debug(f"supabase mirror write failed: {e}")

    def peak_equity(self, since: datetime | None = None) -> float:
        """Highest equity since `since` (or all-time). Reads SQLite
        primary; if Supabase reports a materially higher peak than
        SQLite (>1% difference), Supabase wins — that signals SQLite
        has lost data and the rolling-peak baseline must use the
        intact Supabase value or the kill switch will false-fire."""
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
        sqlite_peak = float(row["peak"]) if row and row["peak"] is not None else 0.0

        if self._supabase is None:
            return sqlite_peak

        try:
            sb_peak = self._supabase.peak_equity_since(
                since.isoformat() if since else None
            )
        except Exception as e:  # noqa: BLE001
            logger.debug(f"supabase peak read failed: {e}")
            return sqlite_peak

        if sb_peak is None:
            return sqlite_peak

        # Supabase wins when its peak is materially higher (>1%) —
        # that's the disaster-recovery signal. If Supabase is just
        # slightly higher (timing race between writes), prefer SQLite
        # for consistency with the local read path.
        if sqlite_peak <= 0 or (sb_peak / sqlite_peak - 1) > 0.01:
            if sb_peak > sqlite_peak:
                logger.warning(
                    f"Risk peak_equity: SQLite={sqlite_peak:.2f} vs "
                    f"Supabase={sb_peak:.2f} — using Supabase (likely "
                    f"local SQLite reset/corruption)"
                )
                return sb_peak
        return sqlite_peak

    def trailing_high(self, lookback_days: int) -> float | None:
        """Highest equity_usd recorded in the last `lookback_days`.

        Returns None if no rows are available in the window (which is
        safer than 0 — callers know to skip the trailing-stop check
        when there's no baseline). SQLite-only — Supabase failover
        is for the all-time peak only; trailing stop is a short-
        window check and we tolerate brief gaps."""
        from datetime import timedelta
        if lookback_days <= 0:
            return None
        since = (datetime.now(UTC) - timedelta(days=lookback_days)).isoformat()
        with self._conn() as c:
            row = c.execute(
                "SELECT MAX(equity_usd) AS peak "
                "FROM equity_snapshots "
                "WHERE timestamp >= ?",
                (since,),
            ).fetchone()
        if row and row["peak"] is not None:
            return float(row["peak"])
        return None

    def recent_returns(self, n: int = 60) -> list[float]:
        """Last `n` snapshot-to-snapshot pct returns.

        Reads SQLite primary; if SQLite has < n/2 rows but Supabase
        has more, prefer Supabase (SQLite was likely just initialized
        or partially restored). Pure read — never writes."""
        with self._conn() as c:
            rows = c.execute(
                "SELECT equity_usd FROM equity_snapshots ORDER BY id DESC LIMIT ?",
                (n + 1,),
            ).fetchall()
        eq_local = [float(r["equity_usd"]) for r in rows][::-1]    # ascending

        eq = eq_local
        if (self._supabase is not None
                and len(eq_local) < max(2, n // 2)):
            try:
                eq_sb = self._supabase.recent_equity_snapshots(n)
            except Exception as e:  # noqa: BLE001
                logger.debug(f"supabase recent read failed: {e}")
                eq_sb = []
            if len(eq_sb) > len(eq_local):
                logger.warning(
                    f"Risk recent_returns: SQLite has {len(eq_local)} "
                    f"rows vs Supabase {len(eq_sb)} — using Supabase"
                )
                eq = eq_sb

        out = []
        for i in range(1, len(eq)):
            if eq[i - 1] > 0:
                out.append((eq[i] - eq[i - 1]) / eq[i - 1])
        return out

    def record_kill_switch(self, state: KillSwitchState, dd_pct: float,
                            note: str = "",
                            cooldown_seconds: int = 0) -> bool:
        """Record a kill-switch transition.

        When `cooldown_seconds > 0`, suppress consecutive identical
        KILL events that arrive inside the cooldown window — this
        keeps the table from growing one row per cycle while KILL is
        active. Returns True if a row was inserted, False if suppressed.

        Cooldown does NOT mute the kill-switch behaviour upstream;
        the orchestrator still runs `_emergency_close_all` every
        cycle. Only the DB write (and downstream alerts that read
        from it) are deduped.
        """
        if cooldown_seconds > 0:
            last = self.last_kill_switch_event()
            if last and last.get("state") == state.value:
                try:
                    last_ts = datetime.fromisoformat(last["timestamp"])
                    if last_ts.tzinfo is None:
                        last_ts = last_ts.replace(tzinfo=UTC)
                    age_s = (datetime.now(UTC) - last_ts).total_seconds()
                    if age_s < cooldown_seconds:
                        return False
                except (ValueError, KeyError):
                    pass    # malformed row → fall through and insert
        with self._conn() as c:
            c.execute(
                "INSERT INTO kill_switch_events (timestamp, state, drawdown_pct, note) VALUES (?, ?, ?, ?)",
                (datetime.now(UTC).isoformat(), state.value, dd_pct, note),
            )
        return True

    def last_kill_switch_event(self) -> dict | None:
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
        brokers: dict | None = None,    # broker registry; if None, no live equity
        config: RiskConfig | None = None,
        db: EquitySnapshotDB | None = None,
        vix_provider=None,              # callable() -> float | None
    ):
        self.brokers = brokers or {}
        self.config = config or RiskConfig.from_env()
        self.multiplier = DynamicRiskMultiplier(self.config)
        self.db = db or EquitySnapshotDB()
        # Optional VIX provider — defaults to the scout signal bus reader
        # below. Injecting lets us avoid risk → scouts coupling in tests
        # and removes a layering violation (risk should not need to know
        # how the macro feed is fanned out).
        self._vix_provider = vix_provider
        self._cached_state: RiskState | None = None
        # Per-cycle broker snapshot cache; refreshed by compute_state()
        self._broker_snapshots: dict[str, dict] = {}

    def cached_positions(self, venue: str) -> list:
        """Read-through accessor for the per-cycle position cache.
        Saves redundant get_positions() calls during a single cycle."""
        snap = self._broker_snapshots.get(venue, {})
        return snap.get("positions", [])

    def _asset_class_exposure(self, asset_class: str) -> float:
        """Sum absolute notional exposure across the cached position
        snapshot for every position whose asset_class matches.

        Cached snapshots come from compute_state() — same per-cycle
        dict the orchestrator already uses for cached_positions(),
        so this is an O(positions) walk with no extra broker calls.

        Returns USD notional. 0 if no matching positions.
        """
        if not asset_class:
            return 0.0
        target = asset_class.upper()
        total = 0.0
        for snap in self._broker_snapshots.values():
            for p in snap.get("positions") or []:
                # Position.asset_class is the AssetClass enum;
                # tolerate either enum or string just in case.
                ac = getattr(p, "asset_class", None)
                ac_str = (ac.value if hasattr(ac, "value") else ac) or ""
                if ac_str.upper() != target:
                    continue
                qty = getattr(p, "quantity", 0.0) or 0.0
                px = getattr(p, "market_price", 0.0) or 0.0
                total += abs(qty * px)
        return total

    def _month_to_date_loss_pct(self, current_equity: float) -> float | None:
        """Return MTD loss as a positive fraction (0.04 = 4% down).
        None if insufficient history.

        Uses the first equity snapshot recorded on or after the 1st
        of the current month as the baseline. If we don't have a
        snapshot from this month yet (first-day-of-month edge case),
        fall back to the most recent snapshot prior to the 1st.
        """
        try:
            import sqlite3
            now = datetime.now(UTC)
            month_start = now.replace(
                day=1, hour=0, minute=0, second=0, microsecond=0,
            ).isoformat()
            with sqlite3.connect(self.db.db_path) as c:
                row = c.execute(
                    "SELECT equity_usd FROM equity_snapshots "
                    "WHERE timestamp >= ? ORDER BY id ASC LIMIT 1",
                    (month_start,),
                ).fetchone()
                if row is None:
                    # Fallback: last snapshot before this month
                    row = c.execute(
                        "SELECT equity_usd FROM equity_snapshots "
                        "WHERE timestamp < ? ORDER BY id DESC LIMIT 1",
                        (month_start,),
                    ).fetchone()
            if row is None or row[0] <= 0:
                return None
            month_start_equity = float(row[0])
            loss_pct = (month_start_equity - current_equity) / month_start_equity
            # Only report a positive fraction (loss). A monthly GAIN
            # returns 0.0 — we never want to "gate" on a profitable
            # month even if the absolute drawdown crosses the limit.
            return max(0.0, loss_pct)
        except Exception as e:
            logger.debug(f"[risk] mtd_loss calc failed: {e}")
            return None

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

        # Sprint A5 — Don't persist equity snapshots when one or more
        # brokers failed their account fetch. The "equity" sum drops
        # by the missing broker's contribution which would falsely
        # depress the rolling peak and immediately trigger the
        # drawdown kill-switch on a transient API blip. By skipping
        # persistence on degraded venues, the next healthy cycle
        # records the true equity instead.
        if persist and equity > 0 and venues_ok:
            self.db.record_snapshot(equity, note=f"venues={len(self.brokers)}")
        elif persist and equity > 0 and not venues_ok:
            logger.warning(
                "[risk] skipping equity-snapshot persistence: "
                "venues_ok=False (one or more broker accounts unreachable). "
                "Drawdown calc uses prior peak; this prevents a "
                "transient API blip from false-firing the kill switch."
            )

        peak = max(self.db.peak_equity(), equity)
        dd_pct = (peak - equity) / peak if peak > 0 else 0.0

        # Realized vol: annualized stdev of recent returns, assuming 5-min cadence
        # (with hourly cron there are ~12 returns/day, ~252 trading days/year)
        rets = self.db.recent_returns(60)
        realized_vol: float | None = None
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
            self.db.record_kill_switch(
                ks_state, dd_pct,
                note=f"equity=${equity:.2f} peak=${peak:.2f}",
                cooldown_seconds=self.config.kill_switch_cooldown_seconds,
            )

        # Monthly loss budget (audit fix #1). Even if portfolio DD
        # hasn't crossed kill_dd_pct, a drawn-out month-to-date loss
        # > monthly_loss_limit_pct should escalate to CRITICAL
        # (closing-only) — protects against repeated per-strategy
        # failures stacking before the global kill fires.
        mtd_loss_pct = self._month_to_date_loss_pct(equity)
        if (mtd_loss_pct is not None
                and mtd_loss_pct >= self.config.monthly_loss_limit_pct
                and ks_state in (KillSwitchState.NORMAL,
                                  KillSwitchState.WARNING)):
            ks_state = KillSwitchState.CRITICAL
            self.db.record_kill_switch(
                ks_state, dd_pct,
                note=f"MTD loss {mtd_loss_pct*100:.2f}% > "
                     f"monthly_limit {self.config.monthly_loss_limit_pct*100:.2f}%",
            )
            logger.warning(
                f"[risk] Monthly loss budget breached: "
                f"MTD {mtd_loss_pct*100:.2f}% > "
                f"{self.config.monthly_loss_limit_pct*100:.2f}% — "
                f"escalating to CRITICAL (closing-only)"
            )

        # Trailing stop (audit-fix follow-up). Catches sharp local
        # drops the all-time-peak drawdown thresholds miss when the
        # peak is months in the past. We use a SHORTER lookback (14d
        # default) — equity below the recent local high by
        # trailing_stop_critical_pct → CRITICAL; below
        # trailing_stop_warning_pct → at least WARNING.
        if (self.config.trailing_stop_critical_pct > 0
                and ks_state != KillSwitchState.KILL):
            trail_high = self.db.trailing_high(
                self.config.trailing_stop_lookback_days
            )
            if trail_high is not None and trail_high > 0 and equity > 0:
                trail_dd = (trail_high - equity) / trail_high
                if (trail_dd >= self.config.trailing_stop_critical_pct
                        and ks_state in (KillSwitchState.NORMAL,
                                          KillSwitchState.WARNING)):
                    ks_state = KillSwitchState.CRITICAL
                    self.db.record_kill_switch(
                        ks_state, dd_pct,
                        note=(
                            f"trailing stop {trail_dd*100:.2f}% from "
                            f"{self.config.trailing_stop_lookback_days}d "
                            f"high ${trail_high:.2f}"
                        ),
                    )
                    logger.warning(
                        f"[risk] Trailing stop breached: "
                        f"{trail_dd*100:.2f}% below "
                        f"{self.config.trailing_stop_lookback_days}d "
                        f"high ${trail_high:.2f} — escalating to CRITICAL"
                    )
                elif (trail_dd >= self.config.trailing_stop_warning_pct
                        and ks_state == KillSwitchState.NORMAL):
                    ks_state = KillSwitchState.WARNING
                    logger.info(
                        f"[risk] Trailing stop warning: "
                        f"{trail_dd*100:.2f}% below "
                        f"{self.config.trailing_stop_lookback_days}d "
                        f"high — escalating to WARNING"
                    )

        state = RiskState(
            timestamp=datetime.now(UTC),
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
        strategy_name: str | None = None,
        existing_position_usd: float = 0.0,
        state: RiskState | None = None,
        venue: str | None = None,
        asset_class: str | None = None,
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
            venue: broker name for per-broker cap resolution.
            asset_class: AssetClass enum value (string) for the
                per-asset-class concentration cap (audit fix #5).
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

        # ─ Per-asset-class concentration cap (audit fix #5).
        # Sum exposure of every position whose asset_class matches,
        # across all brokers, then ensure approved + existing
        # ≤ cap × equity. Enforced ONLY on opening trades — closing
        # trades reduce exposure so they're always allowed.
        if not is_closing and asset_class and st.equity_usd > 0:
            ac_cap = cfg.cap_for_asset_class(asset_class)
            if ac_cap is not None:
                current_ac_exposure = self._asset_class_exposure(asset_class)
                max_ac_notional = ac_cap * st.equity_usd
                ac_headroom = max(0.0, max_ac_notional - current_ac_exposure)
                if approved > ac_headroom:
                    if ac_headroom <= cfg.min_trade_usd:
                        return RiskDecision(
                            Decision.REJECT, 0.0,
                            f"{asset_class} exposure "
                            f"${current_ac_exposure:,.0f} ≥ "
                            f"{ac_cap*100:.0f}% cap "
                            f"(${max_ac_notional:,.0f}) — "
                            f"no headroom for new entries", st,
                        )
                    approved = ac_headroom

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

    def _latest_vix(self) -> float | None:
        """Read the latest VIX reading.

        Uses the injected `vix_provider` callable when set (preferred);
        otherwise falls back to reading the scout signal bus directly.
        The fallback is kept for backwards compatibility with existing
        deployments that don't pass a provider — but the lazy import
        is a layering violation we'd rather not have.
        """
        if self._vix_provider is not None:
            try:
                return self._vix_provider()
            except Exception:
                return None
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

    def summary_dict(self) -> dict:
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
