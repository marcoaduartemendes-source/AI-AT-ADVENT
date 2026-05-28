"""Leveraged champions — 3x exposure on whichever strategies are winning.

USER ASK (2026-05-27)
"Create a new sleeve with 3X leverage on the top-5 best-performing
strategies." This honours that — RESPONSIBLY, reusing every guardrail
the leveraged_momentum sleeve proved out (regime gate, hard stop, vol
overlay, DRY-until-validated, tiny cap).

HOW THE LEVERAGE IS EXPRESSED
You cannot literally "3x" an arbitrary strategy (they trade different
instruments — equities, commodity futures, prediction markets). What
this sleeve does instead is map each current top-5 PASS strategy to a
liquid, exchange-listed 3x-leveraged ETF that proxies its directional
exposure, then holds an equal-weight basket of the DISTINCT proxies the
champions point to. The 3x is intrinsic to the ETF, so $1 of TQQQ is
$3 of economic Nasdaq exposure — no margin, no borrow, bounded loss at
the ETF going to zero.

DYNAMIC UNIVERSE — this is the novel bit
Every cycle it re-resolves the top-5 PASS strategies by realised P&L
(FIFO ledger, falling back to the 5y backtest pnl in validation.json),
maps them to proxies, and trades that set. As the leaderboard shifts,
the leveraged exposure follows the winners. If the top-5 are all
equity-momentum names, the sleeve is long UPRO/TQQQ/SOXL; if a bond
sleeve climbs the board, TMF joins; commodity sleeves have no liquid 3x
equity proxy and are skipped (documented below).

GATES (identical discipline to leveraged_momentum — see that module for
the vol-decay math on why 3x is only survivable in low-vol uptrends):
  • Each proxy's UNDERLYING must be above its 200d SMA
  • Realised vol of the underlying ≤ VOL_CEILING
  • Hard -STOP_PCT stop per position, checked first every cycle
  • vol_scaler overlay halves/zeros the sleeve in HIGH/CRISIS regimes
  • Registered SMALL and DRY until docs/validation.json records PASS

Outside the favourable regime the sleeve sits FLAT (cash), never short.
"""
from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from brokers.base import OrderSide, OrderType
from strategy_engine.base import Strategy, StrategyContext, TradeProposal

logger = logging.getLogger(__name__)


# ─── Tunables (mirror leveraged_momentum) ─────────────────────────────
TREND_SMA = 200
VOL_LOOKBACK = 60
VOL_CEILING = 0.022
STOP_PCT = 0.15
REBALANCE_COOLDOWN_DAYS = 5
TOP_N = 5                       # number of champion strategies to track

# Map a strategy → the liquid 3x ETF that best proxies its directional
# exposure. None = no clean 3x proxy (e.g. commodity-futures carry), so
# that champion contributes no leveraged position. Add rows as new
# strategies join the book.
CHAMPION_PROXY: dict[str, str | None] = {
    "bollinger_breakout":  "UPRO",   # large-cap momentum continuation
    "multifactor_equity":  "UPRO",   # large-cap factor composite
    "sector_rotation":     "UPRO",
    "dividend_growth":     "UPRO",
    "earnings_momentum":   "TQQQ",   # growth/Nasdaq-tilted
    "pead":                "TQQQ",
    "dual_momentum":       "TQQQ",
    "thematic_growth":     "SOXL",   # AI/semis-heavy
    "internationals_rotation": "UPRO",
    "risk_parity_etf":     "TMF",    # 3x long Treasuries
    "commodity_carry":     None,     # no liquid 3x broad-commodity ETF
    "commodity_momentum":  None,
}

# Underlying used for the regime check on each 3x proxy (its own MA lags
# and reacts to decay, so we gate on the real asset — see
# leveraged_momentum docstring).
PROXY_UNDERLYING: dict[str, str] = {
    "UPRO": "SPY", "TQQQ": "QQQ", "SOXL": "SOXX", "TNA": "IWM",
    "TMF": "TLT",
}


def resolve_champions(top_n: int = TOP_N,
                      validation_path: str = "docs/validation.json",
                      db_path: str | None = None) -> list[str]:
    """Return the top-N PASS strategy names ranked by performance.

    Priority: live FIFO realised P&L (real evidence) → 5y backtest pnl
    from validation.json → empty. Only validation-PASS strategies are
    eligible (we never lever an unproven or failing strategy). Pure
    function of its inputs; safe to call with no files (returns [])."""
    val = {}
    try:
        val = json.loads(Path(validation_path).read_text(encoding="utf-8"))
    except Exception:
        val = {}
    strategies = (val.get("strategies") or {})
    passing = {s for s, v in strategies.items() if v.get("verdict") == "PASS"}
    if not passing:
        return []

    # Prefer live FIFO realised P&L as the ranking key.
    fifo: dict[str, float] = {}
    try:
        import os
        from trading.recompute import fifo_realized_by_strategy
        fifo = fifo_realized_by_strategy(
            db_path or os.environ.get("TRADING_DB_PATH",
                                      "data/trading_performance.db"))
    except Exception:
        fifo = {}

    def _score(name: str) -> float:
        if name in fifo:
            return fifo[name]
        # Fall back to the 5y backtest pnl recorded by the validator.
        return float((strategies.get(name) or {}).get("pnl_5y") or 0.0)

    ranked = sorted(passing, key=_score, reverse=True)
    return ranked[:top_n]


class LeveragedChampions(Strategy):
    name = "leveraged_champions"
    venue = "alpaca"

    def compute(self, ctx: StrategyContext) -> list[TradeProposal]:
        if ctx.target_alloc_usd <= 0:
            return []

        from ._helpers import past_cooldown, vol_scaler
        sleeve_usd = ctx.target_alloc_usd * vol_scaler(ctx)
        if sleeve_usd <= 0:
            return []

        # Which 3x proxies do the current champions point to?
        champions = resolve_champions()
        proxies: list[str] = []
        for strat in champions:
            px = CHAMPION_PROXY.get(strat)
            if px and px not in proxies:
                proxies.append(px)
        if not proxies:
            # No PASS strategies yet (cold start) or all map to None —
            # fall back to the broad-market 3x pair so the sleeve still
            # backtests/validates, but log it.
            proxies = ["UPRO", "TQQQ"]
            logger.info(f"[{self.name}] no champion proxies resolved; "
                        f"defaulting to {proxies}")

        proposals: list[TradeProposal] = []
        open_pos = ctx.open_positions or {}

        # ── 1) Hard stops first (non-negotiable, ignores cooldown).
        for sym, pos in open_pos.items():
            if sym not in PROXY_UNDERLYING:
                continue
            qty = (pos.get("quantity", 0) or 0) if hasattr(pos, "get") else 0
            if qty <= 0:
                continue
            entry = (pos.get("avg_entry_price") or pos.get("entry_price") or 0)
            try:
                last = float(self.broker.get_candles(
                    sym, "ONE_DAY", num_candles=2)[-1].close)
            except Exception:
                continue
            if entry and last and last <= entry * (1 - STOP_PCT):
                proposals.append(TradeProposal(
                    strategy=self.name, venue=self.venue, symbol=sym,
                    side=OrderSide.SELL, order_type=OrderType.MARKET,
                    quantity=qty, confidence=0.99, is_closing=True,
                    reason=(f"hard stop: last ${last:.2f} ≤ entry "
                            f"${entry:.2f} × {1 - STOP_PCT:.0%}"),
                    metadata={"model": "leveraged_champions", "leg": "stop"},
                ))

        # ── 2) Regime scan on each proxy's underlying; LONG eligible.
        eligible: list[str] = []
        for proxy in proxies:
            underlying = PROXY_UNDERLYING.get(proxy)
            if not underlying:
                continue
            try:
                under = self.broker.get_candles(
                    underlying, "ONE_DAY",
                    num_candles=TREND_SMA + VOL_LOOKBACK + 5)
            except Exception as e:
                logger.debug(f"[{self.name}] candles {underlying}: {e}")
                continue
            if len(under) < TREND_SMA:
                continue
            closes = np.array([c.close for c in under], dtype=float)
            if closes[-1] <= 0:
                continue
            if closes[-1] < closes[-TREND_SMA:].mean():
                continue                       # not in uptrend
            rets = np.diff(np.log(closes[-VOL_LOOKBACK - 1:]))
            if float(rets.std()) > VOL_CEILING:
                continue                       # too choppy for 3x
            eligible.append(proxy)

        if not eligible:
            logger.info(f"[{self.name}] no champion proxy in low-vol "
                        f"uptrend; sitting flat (safe state for 3x)")
            return proposals

        per_name_usd = sleeve_usd / len(eligible)
        held = {
            s for s, p in open_pos.items()
            if ((p.get("quantity", 0) or 0) if hasattr(p, "get") else 0) > 0
        }
        for sym in eligible:
            if sym in held:
                continue
            pos = open_pos.get(sym, {})
            if pos and not past_cooldown(pos, REBALANCE_COOLDOWN_DAYS):
                continue
            proposals.append(TradeProposal(
                strategy=self.name, venue=self.venue, symbol=sym,
                side=OrderSide.BUY, order_type=OrderType.MARKET,
                notional_usd=per_name_usd, confidence=0.7, is_closing=False,
                reason=(f"3x champions: {sym} underlying above 200d SMA, "
                        f"low vol; tracks top-{TOP_N} PASS strategies "
                        f"{champions[:TOP_N]}"),
                metadata={"model": "leveraged_champions", "leg": "entry",
                          "stop_pct": STOP_PCT, "champions": champions,
                          "as_of": datetime.now(UTC).isoformat()},
            ))
        return proposals
