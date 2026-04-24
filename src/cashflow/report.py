"""Build styled HTML and plain-text cash flow forecast reports."""
from datetime import datetime
from typing import List

from .forecaster import WeekForecast, LOW_BALANCE_THRESHOLD


def _fmt(amount: float, sign: bool = True) -> str:
    if sign:
        return f"+${amount:,.2f}" if amount >= 0 else f"-${abs(amount):,.2f}"
    return f"${abs(amount):,.2f}"


def _clr(amount: float, positive_green: bool = True) -> str:
    green, red = "#2f9e44", "#e03131"
    if positive_green:
        return green if amount >= 0 else red
    return red if amount >= 0 else green  # inverted (expenses: higher = worse)


_CARD = """
<div style="flex:1;min-width:150px;background:{bg};border-radius:12px;padding:18px 20px">
  <div style="font-size:11px;color:{fg};font-weight:700;text-transform:uppercase;letter-spacing:.06em">{label}</div>
  <div style="font-size:22px;font-weight:800;color:{fg};margin-top:8px">{value}</div>
</div>"""


def _card(label: str, value: str, fg: str, bg: str) -> str:
    return _CARD.format(label=label, value=value, fg=fg, bg=bg)


def build_report_html(weeks: List[WeekForecast], sources: List[str]) -> str:
    today = datetime.now().strftime("%B %d, %Y")
    sources_str = ", ".join(s.title() for s in sources) if sources else "No sources connected"

    total_in = sum(w.projected_income for w in weeks)
    total_out = sum(w.projected_expenses for w in weeks)
    total_net = total_in + total_out
    end_bal = weeks[-1].closing_balance if weeks else 0.0

    # ── Summary cards ────────────────────────────────────────────
    cards = "".join([
        _card("Total Inflows",  f"+${total_in:,.2f}",  "#2f9e44", "#f0fdf4"),
        _card("Total Outflows", f"-${abs(total_out):,.2f}", "#e03131", "#fff5f5"),
        _card("Net (13 wks)",   _fmt(total_net),  _clr(total_net),
              "#f0fdf4" if total_net >= 0 else "#fff5f5"),
        _card("Ending Balance", f"${end_bal:,.2f}", _clr(end_bal), "#f8f9fa"),
    ])

    # ── Low-balance warnings ──────────────────────────────────────
    low = [w for w in weeks if w.closing_balance < LOW_BALANCE_THRESHOLD]
    warnings = ""
    if low:
        items_html = "".join(
            f"<li>Week {w.week_number} "
            f"({w.week_start.strftime('%b %d')} – {w.week_end.strftime('%b %d')}): "
            f"<strong>${w.closing_balance:,.2f}</strong></li>"
            for w in low
        )
        warnings = f"""
    <div style="margin:24px 0;padding:16px 20px;background:#fff5f5;
                border-left:4px solid #e03131;border-radius:4px">
      <strong style="color:#e03131">⚠️ Low Balance Alert</strong>
      <ul style="margin:8px 0 0;padding-left:20px;color:#c92a2a">{items_html}</ul>
    </div>"""

    # ── Weekly table rows ─────────────────────────────────────────
    rows = ""
    for w in weeks:
        net_clr = _clr(w.projected_net)
        bal_clr = _clr(w.closing_balance)
        rows += f"""
      <tr>
        <td style="padding:10px 12px;border-bottom:1px solid #e2e8f0;font-weight:600;white-space:nowrap">
          Wk {w.week_number}
          <div style="font-size:11px;color:#718096;font-weight:400">
            {w.week_start.strftime('%b %d')} – {w.week_end.strftime('%b %d')}
          </div>
        </td>
        <td style="padding:10px 12px;border-bottom:1px solid #e2e8f0;color:#2f9e44;
                   text-align:right;white-space:nowrap">
          +${w.projected_income:,.2f}
        </td>
        <td style="padding:10px 12px;border-bottom:1px solid #e2e8f0;color:#e03131;
                   text-align:right;white-space:nowrap">
          -${abs(w.projected_expenses):,.2f}
        </td>
        <td style="padding:10px 12px;border-bottom:1px solid #e2e8f0;font-weight:700;
                   color:{net_clr};text-align:right;white-space:nowrap">
          {_fmt(w.projected_net)}
        </td>
        <td style="padding:10px 12px;border-bottom:1px solid #e2e8f0;font-weight:800;
                   color:{bal_clr};text-align:right;white-space:nowrap">
          ${w.closing_balance:,.2f}
        </td>
      </tr>"""

    th = ("padding:10px 12px;text-align:{align};font-size:11px;color:#718096;"
          "font-weight:700;text-transform:uppercase;letter-spacing:.05em;"
          "border-bottom:2px solid #e2e8f0;background:#f7f8fc")

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>13-Week Cash Flow Forecast</title>
</head>
<body style="margin:0;padding:24px 16px;background:#f0f4f8;
             font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;color:#2d3748">
<div style="max-width:820px;margin:0 auto">
  <div style="background:#fff;border-radius:16px;padding:40px 44px;
              box-shadow:0 4px 24px rgba(0,0,0,.08)">

    <h1 style="margin:0 0 4px;font-size:26px;font-weight:800;color:#1a202c">
      13-Week Cash Flow Forecast
    </h1>
    <p style="margin:0 0 4px;font-size:14px;color:#718096">Generated {today}</p>
    <p style="margin:0 0 28px;font-size:13px;color:#a0aec0">Sources: {sources_str}</p>

    <div style="display:flex;gap:16px;flex-wrap:wrap;margin-bottom:28px">{cards}</div>

    {warnings}

    <table width="100%" cellpadding="0" cellspacing="0" style="border-collapse:collapse;font-size:14px">
      <thead>
        <tr>
          <th style="{th.format(align='left')}">Week</th>
          <th style="{th.format(align='right')}">Inflows</th>
          <th style="{th.format(align='right')}">Outflows</th>
          <th style="{th.format(align='right')}">Net</th>
          <th style="{th.format(align='right')}">Balance</th>
        </tr>
      </thead>
      <tbody>{rows}
      </tbody>
    </table>

    <div style="margin-top:32px;padding:16px 20px;background:#f8f9fa;
                border-radius:8px;font-size:13px;color:#718096;line-height:1.6">
      <strong>Methodology:</strong> Recurring items (subscriptions, payroll, loans) are
      detected from your transaction history and projected to their next expected dates.
      Variable spend categories (groceries, dining, etc.) use an 8-week rolling average.
      Set <code>CURRENT_BALANCE</code> to anchor the opening balance accurately.
    </div>

    <div style="margin-top:24px;text-align:center;font-size:12px;color:#a0aec0">
      Cash Flow Forecast &middot; AI-AT-ADVENT &middot; Powered by Claude
    </div>
  </div>
</div>
</body>
</html>"""


def build_report_text(weeks: List[WeekForecast]) -> str:
    lines = ["13-WEEK CASH FLOW FORECAST", "=" * 56, ""]
    for w in weeks:
        lines.append(
            f"Week {w.week_number:>2}  "
            f"{w.week_start.strftime('%b %d')} – {w.week_end.strftime('%b %d')}"
        )
        lines.append(f"  Inflows:  +${w.projected_income:>11,.2f}")
        lines.append(f"  Outflows: -${abs(w.projected_expenses):>11,.2f}")
        lines.append(f"  Net:       ${w.projected_net:>+11,.2f}")
        lines.append(f"  Balance:   ${w.closing_balance:>11,.2f}")
        lines.append("")
    total_net = sum(w.projected_net for w in weeks)
    lines += [
        "-" * 56,
        f"  13-week net: ${total_net:>+,.2f}",
        f"  Ending balance: ${weeks[-1].closing_balance:>,.2f}" if weeks else "",
    ]
    return "\n".join(lines)
