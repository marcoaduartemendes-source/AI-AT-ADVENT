"""Risk policies — static configuration + state enums.

`RiskConfig` is loaded once at startup from env vars (so repo vars can tune
the live system without code changes). All thresholds are conservative
defaults aimed at the 15% return / 12% vol / 12% max-DD profile.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from enum import Enum


class KillSwitchState(str, Enum):
    """Top-level portfolio state. Only `NORMAL` permits new opening trades."""

    NORMAL = "NORMAL"        # all systems go
    WARNING = "WARNING"      # at -5% DD or vol spike — auto-deleverage active
    CRITICAL = "CRITICAL"    # at -10% DD — only closing trades allowed
    KILL = "KILL"            # at -15% DD — flat all positions, manual restart


def _envf(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return float(default)


def _envi(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return int(default)


@dataclass(frozen=True)
class RiskConfig:
    """Risk policy configuration. Override any value via the corresponding
    env var (uppercase) at the workflow level."""

    # ── Sizing / vol target ----------------------------------------------
    target_portfolio_vol: float = 0.12
    """Target annualized portfolio vol (12% by default for the 15% return profile)."""

    max_position_pct: float = 0.30
    """No single position > 30% of portfolio equity."""

    max_strategy_pct: float = 0.30
    """No single strategy > 30% of portfolio equity (cross-checked by Allocator)."""

    leverage_cap: float = 2.0
    """Hard upper bound on total notional / equity. Carry sleeves can lever
    inside this cap; the multiplier never pushes past it."""

    # ── Drawdown thresholds ---------------------------------------------
    warning_dd_pct: float = 0.05      # 5% drawdown triggers WARNING
    critical_dd_pct: float = 0.10     # 10% triggers CRITICAL
    kill_dd_pct: float = 0.15         # 15% triggers KILL

    strategy_freeze_dd_pct: float = 0.20
    """Single-strategy drawdown that auto-freezes that strategy."""

    # ── Trailing stop (audit-fix follow-up: portfolio-level safety) ──
    # The drawdown thresholds above measure peak-to-now from the
    # all-time high. That's slow to trigger if the all-time high
    # was set months ago — a short-but-violent drop can hit
    # critical_dd_pct only after weeks of bleed.
    #
    # The trailing stop uses a SHORTER lookback (default 14 days)
    # so a sharp local drop escalates risk state quickly even when
    # the all-time peak is far above. Acts as an early-warning
    # circuit breaker that complements the existing thresholds.
    trailing_stop_lookback_days: int = 14
    trailing_stop_warning_pct: float = 0.04   # 4% from 14-day high
    trailing_stop_critical_pct: float = 0.07  # 7% from 14-day high
    """Trailing-stop window (days) and thresholds. The risk manager
    escalates state when the current equity is more than X% below
    the highest equity in the lookback window. Set _critical_pct=0
    to disable the trailing stop entirely (drawdown thresholds
    still apply)."""

    # ── Monthly loss budget (audit fix #1) ──────────────────────────
    monthly_loss_limit_pct: float = 0.04
    """Hard cap on month-to-date portfolio loss as a fraction of
    month-start equity. When MTD loss exceeds this, every venue
    moves to closing-only (CRITICAL state) until the calendar
    flips. Without this, a strategy can lose 19% twice in a row
    before its per-strategy freeze kicks in — blowing through
    the kill_dd_pct global threshold from a single bad month.

    4% is conservative. Loosen to 0.06–0.08 once the system has
    6+ months of clean live equity history."""

    # ── Per-asset-class concentration cap (audit fix #5) ────────────
    max_asset_class_pct: dict[str, float] = field(default_factory=lambda: {
        # Total notional across ALL strategies cannot exceed these
        # fractions of portfolio equity per asset class. Prevents a
        # situation where 5 ostensibly-different strategies all pile
        # into equity-beta-1 names (SPY, QQQ, NVDA, AAPL) and a
        # single bad SPX day drains the book before per-strategy
        # freezes engage.
        #
        # 2026-06-05: 0.60 was STILL binding pathologically — sum of all
        # ETF/EQUITY strategy target allocations is ~93% (17 strategies
        # competing for a 60% bucket), and droplet cycle data showed 182
        # of 182 rejections in 50 cycles were the ETF cap. After the
        # iteration-priority fix, the top-3 conviction sleeves
        # (risk_parity 22%, tsmom 16%, multifactor 14% = 52%) get filled
        # but everything below position 3 perma-blocks. Raising to 0.90
        # gives the proven core plus the 3x leveraged sleeves room to
        # operate. Safety is preserved by (a) leverage_cap=2.0× gross
        # backstop, (b) the kill-switch's MTD-loss CRITICAL trigger at
        # −4%, (c) the trailing-stop CRITICAL trigger at −7% from
        # 14-day high. EQUITY+ETF share a bucket, so 0.90 of equity
        # gross US-beta is the worst case before the leverage cap binds.
        "EQUITY":           0.90,
        "ETF":              0.90,    # shared bucket with EQUITY
                                     # (same beta exposure)
        "CRYPTO_SPOT":      0.25,
        "CRYPTO_PERP":      0.20,
        "CRYPTO_FUTURE":    0.15,
        "COMMODITY_FUTURE": 0.20,
        "PREDICTION":       0.10,
    })
    """Per-asset-class total exposure cap. Keys are AssetClass enum
    values (uppercased strings). When a strategy's proposal would
    push that asset class above its cap, the proposal is scaled
    down to stay within budget."""

    # ── Vol-spike auto-deleverage ---------------------------------------
    vol_spike_ratio: float = 1.5
    """If realized vol > target_vol * ratio, multiplier deleverages."""

    # ── Multiplier bounds -----------------------------------------------
    multiplier_min: float = 0.5
    multiplier_max: float = 2.0
    multiplier_default: float = 1.0

    # ── Trade gating -----------------------------------------------------
    min_trade_usd: float = 50.0
    """Don't bother sending orders smaller than this — fees would
    dominate. 2026-05-20 raised from 5 → 50: at 10bps round-trip
    + bid-ask spread, a $5 trade pays ~$0.05+ in friction against
    expected P&L of ~$0.10 — net negative on average. $50 floor
    means a 1% move translates to $0.50, comfortably above fee
    drag. Helps fee_discipline grade."""

    max_trade_usd: float = 5000.0
    """Per-order ceiling; prevents one bad signal from blowing up the book.

    2026-08-05: this is a FLAT dollar amount. It does NOT scale with
    equity — an earlier docstring claimed it did, and nothing in
    compute_state() ever touched it. The claim mattered: at $1M equity
    the effective per-order cap is min($5,000, 0.30 x equity) = $5,000,
    so the book physically cannot deploy a large account (24 orders per
    strategy per day x $5k). Raising this via MAX_TRADE_USD_GLOBAL is
    therefore mandatory at scale — and the moment it is raised, the only
    remaining bound is max_position_pct x equity, which is why the ADV
    participation cap below exists."""

    # Per-broker overrides (None = use max_trade_usd above).
    # Used to cap small live tests on one venue (e.g. $50 on Coinbase)
    # without crippling Alpaca paper trading at the same time.
    max_trade_usd_coinbase: float | None = None
    max_trade_usd_alpaca: float | None = None
    max_trade_usd_kalshi: float | None = None

    # ── Liquidity participation cap (2026-08-05 capital-scale review) ──
    # Every other cap in this config is a fraction of EQUITY, so every
    # cap grows with the account while the market does not. At $1M and
    # max_position_pct=0.30 the risk layer would authorise a $300,000
    # order in a name that trades $2M/day — ~15% of a session, which
    # moves the price against itself and makes the fill unlike anything
    # the backtest measured. Backtests here assume infinite liquidity
    # (signal x close price); this cap is what keeps live sizing inside
    # the range where that assumption is roughly true.
    max_adv_participation_pct: float = 0.10
    """Max order notional as a fraction of the symbol's median daily
    dollar volume. 10% is the conventional ceiling above which market
    impact stops being a rounding error. 0 disables the check entirely.
    Closing orders always bypass — getting flat is non-negotiable."""

    adv_lookback_days: int = 20
    """Daily bars used for the median-ADV estimate (~1 trading month)."""

    adv_unknown_max_usd: float = 25_000.0
    """Cap applied when ADV cannot be measured (venue reports no volume,
    broker error, new listing). Deliberately a small positive number
    rather than 0 or infinity: rejecting outright would freeze the book
    on a data hiccup — the exact May-4 failure where a bad broker read
    latched KILL and stopped even the exits — while allowing an
    unmeasured name at full size is the defect this cap exists to close.
    Set 0 to skip unmeasurable symbols instead of clamping them."""

    def cap_for_venue(self, venue: str) -> float:
        per_venue = {
            "coinbase": self.max_trade_usd_coinbase,
            "alpaca":   self.max_trade_usd_alpaca,
            "kalshi":   self.max_trade_usd_kalshi,
        }.get(venue)
        return per_venue if per_venue is not None else self.max_trade_usd

    # ── Cooldown ---------------------------------------------------------
    kill_switch_cooldown_seconds: int = 86400
    """After KILL fires, wait this long before allowing manual restart."""

    # ── Per-strategy daily notional governor ----------------------------
    # Circuit breaker for haywire-but-not-erroring strategies. A bug
    # that emits 60 buy/sell pairs per hour eats fees for the full
    # cycle until DD/MTD/trailing trips. This caps each strategy's
    # daily entry-side notional at a fraction of equity.
    max_strategy_daily_notional_pct: float = 0.50
    """Fraction of equity any one strategy can trade per UTC day before
    new opens are rejected. Closing trades (is_closing=True) bypass."""

    # Audit-fix F5 (2026-05-07): for live-money venues with small per-
    # order caps (e.g. $100 on Coinbase), the notional governor doesn't
    # bite until the strategy has fired ~50 buys. The order-count
    # governor catches a flapping strategy in O(N) instead.
    max_strategy_daily_orders: int = 24
    """Max BUY-side orders per strategy per UTC day. 0 disables.
    24 covers `crypto_funding_carry`'s 8h-cadence × 3 symbols × 4×
    safety multiplier."""

    @classmethod
    def from_env(cls) -> RiskConfig:
        """Build config from env vars. Repo workflow can override any field
        by setting an env var with the corresponding uppercase name."""
        def _opt_envf(name: str) -> float | None:
            v = os.environ.get(name)
            if v is None or v == "":
                return None
            try:
                return float(v)
            except ValueError:
                return None
        return cls(
            target_portfolio_vol=_envf("TARGET_PORTFOLIO_VOL", 0.12),
            max_position_pct=_envf("MAX_POSITION_PCT", 0.30),
            max_strategy_pct=_envf("MAX_STRATEGY_PCT", 0.30),
            leverage_cap=_envf("LEVERAGE_CAP", 2.0),
            warning_dd_pct=_envf("WARNING_DD_PCT", 0.05),
            critical_dd_pct=_envf("CRITICAL_DD_PCT", 0.10),
            kill_dd_pct=_envf("KILL_DD_PCT", 0.15),
            strategy_freeze_dd_pct=_envf("STRATEGY_FREEZE_DD_PCT", 0.20),
            vol_spike_ratio=_envf("VOL_SPIKE_RATIO", 1.5),
            multiplier_min=_envf("MULTIPLIER_MIN", 0.5),
            multiplier_max=_envf("MULTIPLIER_MAX", 2.0),
            multiplier_default=_envf("RISK_MULTIPLIER", 1.0),
            min_trade_usd=_envf("MIN_TRADE_USD", 50.0),
            max_trade_usd=_envf("MAX_TRADE_USD_GLOBAL", 5000.0),
            max_adv_participation_pct=_envf("ADV_PARTICIPATION_PCT", 0.10),
            adv_lookback_days=_envi("ADV_LOOKBACK_DAYS", 20),
            adv_unknown_max_usd=_envf("ADV_UNKNOWN_MAX_USD", 25_000.0),
            max_trade_usd_coinbase=_opt_envf("MAX_TRADE_USD_COINBASE"),
            max_trade_usd_alpaca=_opt_envf("MAX_TRADE_USD_ALPACA"),
            max_trade_usd_kalshi=_opt_envf("MAX_TRADE_USD_KALSHI"),
            kill_switch_cooldown_seconds=_envi("KILL_SWITCH_COOLDOWN_SECONDS", 86400),
            monthly_loss_limit_pct=_envf("MONTHLY_LOSS_LIMIT_PCT", 0.04),
        )

    # Asset classes that share ONE concentration budget because they
    # carry the same underlying risk. EQUITY (single names) and ETF
    # (baskets) are both US equity-beta; enforcing them as two
    # independent 0.90 caps let the book reach 1.80× equity of
    # correlated beta — exactly the pile-in the cap exists to prevent
    # (full-system review 2026-06-11). They now share the "US_BETA"
    # bucket: total EQUITY+ETF exposure is capped at 0.90 combined.
    BUCKET_ALIASES = {"EQUITY": "US_BETA", "ETF": "US_BETA"}

    def bucket_for(self, asset_class: str) -> str:
        """Resolve an asset class to its shared concentration bucket."""
        key = (asset_class or "").upper()
        return self.BUCKET_ALIASES.get(key, key)

    def cap_for_asset_class(self, asset_class: str) -> float | None:
        """Return the max-exposure cap for an asset class, or None
        if no cap is configured (treat as unlimited).

        Resolves through the shared-bucket alias first (EQUITY and ETF
        map to one US_BETA budget), then falls back to the raw class
        key. Lookup is case-insensitive."""
        if not asset_class:
            return None
        bucket = self.bucket_for(asset_class)
        # The cap dict is keyed by raw class (EQUITY/ETF both = 0.90);
        # for aliased buckets, use the min cap of the members so the
        # shared budget is the tightest of the two.
        if bucket == "US_BETA":
            members = [self.max_asset_class_pct.get(k)
                       for k in ("EQUITY", "ETF")]
            members = [m for m in members if m is not None]
            return min(members) if members else None
        return self.max_asset_class_pct.get(bucket)

    def state_for_drawdown(self, dd_pct: float) -> KillSwitchState:
        """Map a drawdown magnitude (0.05 == 5%) to a state."""
        if dd_pct >= self.kill_dd_pct:
            return KillSwitchState.KILL
        if dd_pct >= self.critical_dd_pct:
            return KillSwitchState.CRITICAL
        if dd_pct >= self.warning_dd_pct:
            return KillSwitchState.WARNING
        return KillSwitchState.NORMAL
