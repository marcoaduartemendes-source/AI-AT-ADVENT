"""Thematic-growth strategy — curated 2026 investment themes.

USER ASK
The user asked for a strategy that picks stocks by THEME (their
example: "AI supply-chain component manufacturers") and asked me
to research "the next big themes". This module is that research,
encoded as a strategy.

DESIGN
A theme is a structural, multi-year demand story large enough
that the rising tide lifts a basket of well-positioned names. For
each theme, the strategy:

  1. Filters the basket to names in an uptrend (close > 200d SMA).
  2. Ranks the survivors by 6-month price momentum (a fee-light,
     well-evidenced quality proxy — Jegadeesh-Titman 1993, plus
     it surfaces analyst-revision tailwinds without an FMP call).
  3. Holds the top TOP_PER_THEME names per theme.
  4. Weights across themes by the convictions in THEME_WEIGHT.

WHY MOMENTUM, NOT JUST "BUY THE THEME LIST"
Themes broadcast WHICH SECTOR, but inside any theme there are 10
losers for every winner (think: every consumer-internet darling
of 2021). Within-theme momentum is the cheapest, least-overfit
way to let the market tell us which names are actually winning
the structural wave.

THEME RESEARCH (as of 2026-05)
Ordered by my conviction. Each name is here because it's the
PURE-PLAY public US-listed proxy (or close to it) for the named
structural demand; ADRs are noted.

  1. AI compute / picks-and-shovels  (THEME_WEIGHT = 0.30)
     The capex super-cycle: hyperscaler AI training/inference
     buildout. Demand is largely supply-constrained, not demand-
     constrained, so margins for the picks-and-shovels names
     (foundry, lithography, memory, networking) hold longer
     than pure-AI software. This is the highest-conviction trade
     and the one the user explicitly named ("AI supply chain").
        NVDA — accelerator monopoly (training + inference)
        AMD  — #2 accelerator + EPYC server CPUs
        AVGO — AI ASICs (Google TPU) + networking silicon
        TSM  — sole leading-edge foundry (NVDA/AMD/AAPL/AVGO)
        ASML — sole EUV lithography supplier
        AMAT — depo/etch tools (every fab uses them)
        KLAC — process control (sub-3nm yield-critical)
        LRCX — etch / advanced packaging
        MRVL — custom AI silicon + interconnect
        ARM  — IP licensing into every AI inference SoC
        MU   — HBM memory (the AI bottleneck on every H100/B200)

  2. AI power & cooling              (THEME_WEIGHT = 0.20)
     The non-obvious 2025-26 trade: AI datacenters need ~100GW
     of new US power by 2030, against historical demand growth
     of ~0%/yr. This is a binding physical constraint, not a
     forecast. IPPs with nuclear/long-dated PPAs and electrical
     equipment names (transformers, switchgear, gas turbines) have
     >5y order backlogs. Underweighted by index funds.
        CEG  — largest US nuclear IPP (Microsoft 20y PPA)
        VST  — Texas IPP, nuclear + gas + battery
        NRG  — Texas IPP, gas + retail
        GEV  — GE Vernova (gas turbines, grid, wind)
        ETN  — Eaton: electrical equipment, switchgear
        PWR  — Quanta: utility infrastructure construction
        EATON — same as ETN; keep one symbol

  3. Cybersecurity                   (THEME_WEIGHT = 0.12)
     Secular ~12-15%/yr enterprise spend growth; AI is BOTH
     a new attack surface AND a force-multiplier for defenders.
     Platformisation (Palo Alto / CrowdStrike) wins.
        PANW — broadest platform
        CRWD — endpoint + identity leader
        ZS   — SASE / zero-trust networking
        FTNT — appliances + SD-WAN
        S    — SentinelOne; smaller, AI-native EDR

  4. Defense & reshoring             (THEME_WEIGHT = 0.10)
     Geopolitics (Ukraine continuation, Taiwan tail-risk, Houthis,
     Iran, North Korea) + multi-year DoD budget growth + IRA/CHIPS
     industrial reshoring. Primes have decade-long order books.
        LMT — F-35, missiles, hypersonics
        RTX — Raytheon: missiles, engines
        NOC — bombers, space, submarine systems
        GD  — submarines + IT services
        HII — sole US Navy aircraft-carrier shipbuilder

  5. GLP-1 / obesity                 (THEME_WEIGHT = 0.10)
     Multi-decade TAM expansion (40% US adult obesity → drug-
     addressable). The duopoly is structural — manufacturing scale
     is the moat, not the molecule.
        LLY — Mounjaro / Zepbound; broadest pipeline
        NVO — Novo Nordisk ADR; Ozempic / Wegovy

  6. Robotics & automation           (THEME_WEIGHT = 0.08)
     Labour scarcity + AI-driven step-change in machine vision &
     manipulation. Long cycle, slower than AI compute but with
     similar structural drivers.
        ISRG — surgical robotics (da Vinci installed base moat)
        ROK  — factory automation
        ABBT — Abbott; lab/medtech automation (proxy)

  7. Quantum computing (SPECULATIVE) (THEME_WEIGHT = 0.05)
     Lottery-ticket sleeve. If error-corrected qubits cross a
     useful threshold (the field's "fault-tolerant" milestone),
     the pure-plays are 10-100×. Tiny weight; full loss possible.
        IONQ — trapped-ion architecture
        RGTI — Rigetti; superconducting

A name appearing in multiple themes (none currently do, but if
added e.g. NVDA in AI compute + quantum) is held once with the
HIGHER of the two theme weights — no double-counting.

LIVE-PROMOTION GATE
DRY at 2.5% target. Stays paper until validation harness PASSes.
"""
from __future__ import annotations

import logging

import numpy as np

from brokers.base import OrderSide, OrderType
from strategy_engine.base import Strategy, StrategyContext, TradeProposal

logger = logging.getLogger(__name__)


# ─── Theme definitions ────────────────────────────────────────────────

THEMES: dict[str, list[str]] = {
    "ai_compute":         ["NVDA", "AMD", "AVGO", "TSM", "ASML",
                            "AMAT", "KLAC", "LRCX", "MRVL", "ARM", "MU"],
    "ai_power":           ["CEG", "VST", "NRG", "GEV", "ETN", "PWR"],
    "cybersecurity":      ["PANW", "CRWD", "ZS", "FTNT", "S"],
    "defense":            ["LMT", "RTX", "NOC", "GD", "HII"],
    "obesity_glp1":       ["LLY", "NVO"],
    "robotics":           ["ISRG", "ROK", "ABBT"],
    "quantum_spec":       ["IONQ", "RGTI"],
    # ── 2026-05-20 user-requested expansion: more thematic surface
    # area for alpha. Each added theme has a concrete structural
    # driver, not just a buzzword:
    "clean_energy":       ["NEE", "FSLR", "ENPH", "RUN", "ICLN", "BEP"],
    # Reshoring / CHIPS Act / IRA — overlap with ai_power & defense
    # but specifically captures the industrial-cap-ex story.
    "reshoring_chips":    ["TXN", "INTC", "ON", "ENTG", "ROP", "ETN"],
    # Lithium / EV supply chain — secular EV growth + grid storage
    # demand. ALB is dominant Western lithium; LIT is the basket ETF.
    "lithium_ev_supply":  ["ALB", "LIT", "TSLA", "RIVN", "PCRFY"],
    # Space economy — launch + satcom + defense overlap (RKLB, ASTS,
    # LMT, BA). Small, speculative, high-beta.
    "space_economy":      ["RKLB", "ASTS", "LMT", "BA", "IRDM"],
    # Biotech innovation — gene/cell therapy + GLP-1 distinct names.
    # Long-cycle, binary-event-heavy; small weight reflects that.
    "biotech_innovation": ["REGN", "VRTX", "MRNA", "BNTX", "ALNY"],
    # Water — undercovered but structural (population + climate);
    # AWK is the largest pure-play US water utility, XYL the
    # tech/infrastructure exposure.
    "water":              ["AWK", "XYL", "PHO", "FIW"],
}

# Cross-theme weights — re-normalised after the expansion. Higher-
# conviction themes (ai_compute, ai_power) keep their dominance.
# Weights sum to ~1.0; small residual is the cash buffer.
THEME_WEIGHT: dict[str, float] = {
    "ai_compute":         0.22,   # was 0.30; still largest
    "ai_power":           0.16,   # was 0.20
    "cybersecurity":      0.10,
    "defense":            0.08,
    "obesity_glp1":       0.08,
    "robotics":           0.06,
    "quantum_spec":       0.04,
    "clean_energy":       0.06,
    "reshoring_chips":    0.06,
    "lithium_ev_supply":  0.04,
    "space_economy":      0.03,
    "biotech_innovation": 0.04,
    "water":              0.03,
}

MOM_LOOKBACK = 126            # ~6 months trading days
TREND_SMA = 200               # eligibility: above 200d SMA
TOP_PER_THEME = 2             # hold top-2 momentum names per theme
REBALANCE_COOLDOWN_DAYS = 14  # quarterly-ish churn; keep fees low


class ThematicGrowth(Strategy):
    name = "thematic_growth"
    venue = "alpaca"

    def compute(self, ctx: StrategyContext) -> list[TradeProposal]:
        if ctx.target_alloc_usd <= 0:
            return []

        from ._helpers import past_cooldown, vol_scaler
        sleeve_usd = ctx.target_alloc_usd * vol_scaler(ctx)
        if sleeve_usd <= 0:
            return []

        # Per-theme picks: rank by 6m momentum among names above
        # 200d SMA, keep TOP_PER_THEME. Skip theme entirely if 0
        # eligible names (don't force-allocate to broken themes).
        picks: dict[str, list[tuple[str, float]]] = {}  # theme -> [(sym, mom)]
        for theme, names in THEMES.items():
            scored = self._theme_momentum(names)
            if not scored:
                continue
            picks[theme] = scored[:TOP_PER_THEME]

        if not picks:
            logger.info(f"[{self.name}] no theme has eligible names; "
                        f"sitting out this cycle")
            return []

        # Per-name USD allocation:
        #   sleeve · theme_weight · (1/picks_in_theme)
        # The cross-theme weights sum to ~0.95; the residual is a
        # small cash buffer — intentional, not a bug.
        target_alloc: dict[str, float] = {}
        for theme, lst in picks.items():
            tw = THEME_WEIGHT.get(theme, 0.0)
            if tw <= 0 or not lst:
                continue
            per = sleeve_usd * tw / len(lst)
            for sym, _mom in lst:
                # If a symbol appears in multiple themes, keep the
                # larger allocation (no double-count).
                target_alloc[sym] = max(target_alloc.get(sym, 0.0), per)

        target_set = set(target_alloc)
        proposals: list[TradeProposal] = []
        open_pos = ctx.open_positions or {}

        # ── Exits — held names that are NOT in any current theme pick.
        for sym, pos in open_pos.items():
            qty = (pos.get("quantity", 0) or 0) if hasattr(pos, "get") else 0
            if qty <= 0:
                continue
            if sym in target_set:
                continue           # still picked
            if not past_cooldown(pos, REBALANCE_COOLDOWN_DAYS):
                continue           # min-hold respected
            # Only exit if WE put it on — i.e. it's in our universe.
            # Avoid stomping on another strategy's position.
            in_universe = any(sym in v for v in THEMES.values())
            if not in_universe:
                continue
            proposals.append(TradeProposal(
                strategy=self.name, venue=self.venue, symbol=sym,
                side=OrderSide.SELL, order_type=OrderType.MARKET,
                quantity=qty, confidence=0.85, is_closing=True,
                reason=f"no longer top-{TOP_PER_THEME} in its theme",
                metadata={"model": "thematic_growth", "leg": "exit"},
            ))

        # ── Entries — target names we don't already hold.
        held = {
            s for s, p in open_pos.items()
            if ((p.get("quantity", 0) or 0) if hasattr(p, "get") else 0) > 0
        }
        for sym, usd in target_alloc.items():
            if sym in held:
                continue
            if usd <= 0:
                continue
            # Which theme attributed this pick? (first match wins.)
            sym_theme = next((t for t, lst in picks.items()
                              if any(s == sym for s, _ in lst)), "?")
            proposals.append(TradeProposal(
                strategy=self.name, venue=self.venue, symbol=sym,
                side=OrderSide.BUY, order_type=OrderType.MARKET,
                notional_usd=usd, confidence=0.7, is_closing=False,
                reason=(f"top-{TOP_PER_THEME} 6m momentum in theme "
                        f"'{sym_theme}' (weight "
                        f"{THEME_WEIGHT.get(sym_theme,0):.0%})"),
                metadata={"model": "thematic_growth", "leg": "entry",
                          "theme": sym_theme},
            ))
        return proposals

    # ── Helpers ──────────────────────────────────────────────────────

    def _theme_momentum(self, symbols: list[str]) -> list[tuple[str, float]]:
        """Return [(sym, 6m return)] sorted desc, restricted to names
        trading above their 200d SMA. Names with insufficient history
        or fetch errors are silently dropped."""
        need = max(TREND_SMA, MOM_LOOKBACK) + 5
        out: list[tuple[str, float]] = []
        for sym in symbols:
            try:
                candles = self.broker.get_candles(
                    sym, "ONE_DAY", num_candles=need + 10)
            except Exception as e:
                logger.debug(f"[{self.name}] candles {sym}: {e}")
                continue
            if len(candles) < need:
                continue
            closes = np.array([c.close for c in candles], dtype=float)
            if closes[-1] <= 0:
                continue
            if closes[-1] < closes[-TREND_SMA:].mean():
                continue                # not in uptrend
            p0 = closes[-MOM_LOOKBACK]
            if p0 <= 0:
                continue
            mom = (closes[-1] / p0) - 1.0
            out.append((sym, float(mom)))
        out.sort(key=lambda x: x[1], reverse=True)
        return out
