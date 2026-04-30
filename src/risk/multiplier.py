"""Dynamic risk multiplier.

Single 0.5x-2.0x knob that scales sizing across the entire portfolio.
Manual base value comes from `RISK_MULTIPLIER` env var; the live multiplier
auto-adjusts down on stress signals (drawdown, vol spike, regime).

Effective multiplier = base × min(adjustments)

So if any one signal says "deleverage", the multiplier deleverages — even
if the others are clear. Conservative by design: re-leverage requires all
signals to clear.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from .policies import KillSwitchState, RiskConfig


@dataclass
class MultiplierState:
    """What the multiplier is right now and why."""

    base: float                  # user-set base (default 1.0)
    effective: float             # final multiplier applied to sizing
    drawdown_factor: float = 1.0 # auto-de-lev factor from current DD
    vol_factor: float = 1.0      # auto-de-lev factor from realized vol
    regime_factor: float = 1.0   # external regime override (VIX etc.)
    notes: List[str] = field(default_factory=list)


class DynamicRiskMultiplier:
    """Computes the effective risk multiplier given live state."""

    def __init__(self, config: RiskConfig):
        self.config = config
        self._base: float = config.multiplier_default

    # ── User control ------------------------------------------------------

    @property
    def base(self) -> float:
        return self._base

    def set_base(self, value: float) -> None:
        """Manual override — clamped to [multiplier_min, multiplier_max]."""
        v = max(self.config.multiplier_min, min(self.config.multiplier_max, float(value)))
        self._base = v

    # ── Auto-adjustment factors ------------------------------------------

    def _drawdown_factor(self, dd_pct: float) -> tuple[float, str]:
        """Step-down based on current portfolio drawdown."""
        if dd_pct >= self.config.kill_dd_pct:
            return 0.0, f"KILL ({dd_pct * 100:.1f}% DD)"
        if dd_pct >= self.config.critical_dd_pct:
            return 0.3, f"CRITICAL ({dd_pct * 100:.1f}% DD)"
        if dd_pct >= self.config.warning_dd_pct:
            return 0.6, f"WARNING ({dd_pct * 100:.1f}% DD)"
        return 1.0, ""

    def _vol_factor(self, realized_vol: Optional[float]) -> tuple[float, str]:
        """Step-down when realized vol exceeds target by `vol_spike_ratio`."""
        if realized_vol is None or realized_vol <= 0:
            return 1.0, ""
        ratio = realized_vol / self.config.target_portfolio_vol
        if ratio > self.config.vol_spike_ratio:
            # Linear de-lev: target vol / realized vol
            f = max(0.4, self.config.target_portfolio_vol / realized_vol)
            return f, f"vol_spike ({realized_vol * 100:.1f}%/{self.config.target_portfolio_vol * 100:.1f}%)"
        return 1.0, ""

    def _regime_factor(self, vix: Optional[float]) -> tuple[float, str]:
        """External regime override. Coarse VIX-based scaling."""
        if vix is None:
            return 1.0, ""
        if vix > 35:
            return 0.5, f"VIX={vix:.1f} (extreme)"
        if vix > 25:
            return 0.75, f"VIX={vix:.1f} (elevated)"
        return 1.0, ""

    # ── Compute -----------------------------------------------------------

    def compute(
        self,
        drawdown_pct: float,
        realized_vol: Optional[float] = None,
        vix: Optional[float] = None,
    ) -> MultiplierState:
        dd_f, dd_note = self._drawdown_factor(drawdown_pct)
        vol_f, vol_note = self._vol_factor(realized_vol)
        rg_f, rg_note = self._regime_factor(vix)

        # Take the most conservative auto-adjustment
        auto_factor = min(dd_f, vol_f, rg_f)
        effective = self._base * auto_factor
        # Clamp
        effective = max(0.0, min(self.config.multiplier_max, effective))

        notes = [n for n in (dd_note, vol_note, rg_note) if n]
        return MultiplierState(
            base=self._base,
            effective=effective,
            drawdown_factor=dd_f,
            vol_factor=vol_f,
            regime_factor=rg_f,
            notes=notes,
        )
