"""Hedge fund 13F scout — learn from the best.

USER MANDATE (2026-05-20)
"Research the best-performing hedge funds in the world on a daily
basis and learn from them what strategies work and don't work and
what other thematics to take into consideration."

WHAT IT DOES
Polls SEC EDGAR daily for new 13F-HR filings from the top
quant/multi-strategy alpha funds. 13F filings disclose long
positions for funds with >$100M AUM, filed quarterly with a 45-day
delay. The signal value isn't real-time — it's a structural read
on what the smartest, most-resourced players are accumulating
and exiting.

FUNDS TRACKED (the empirically highest-alpha generators)
  Renaissance Technologies     (Medallion is private; their public
                                feeders still file, signal IS noisy)
  Two Sigma Investments
  Citadel Advisors
  Millennium Management
  Bridgewater Associates
  AQR Capital Management
  D.E. Shaw

OUTPUT
  • Signal bus rows of type `hedge_fund_13f` with the new filing
    accession number, link, and filer.
  • A consolidated docs/hedge_fund_13f.json the dashboard can
    render — most-recent filings list.

The actual holdings extraction is a follow-up (the SEC's 13F
infoTable XML needs proper parsing). This scout's MVP: surface
when a new filing drops, link to it, so the strategic-review
agent (and Claude on the next run) can read the actual table.
"""
from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from urllib.error import URLError
from urllib.request import Request, urlopen

from .base import ScoutAgent, ScoutSignal

logger = logging.getLogger(__name__)


# Central Index Keys (10-digit, leading zeros required for EDGAR URLs).
# Verified against SEC EDGAR full-text search 2026-05-20.
FUNDS = {
    "0001037389": "Renaissance Technologies",
    "0001179392": "Two Sigma Investments",
    "0001423053": "Citadel Advisors",
    "0001273087": "Millennium Management",
    "0001350694": "Bridgewater Associates",
    "0001167557": "AQR Capital Management",
    "0001009207": "D.E. Shaw",
}

# SEC requires a real User-Agent with contact info — without this,
# requests are silently 403'd.
_UA = ("ai-at-advent research bot "
        "(contact: marcoaduartemendes@gmail.com)")


def _fetch(url: str, timeout: int = 12) -> dict | None:
    try:
        req = Request(url, headers={"User-Agent": _UA,
                                       "Accept": "application/json"})
        with urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read())
    except (URLError, json.JSONDecodeError, TimeoutError) as e:
        logger.warning(f"13F fetch {url[:80]}: {e}")
        return None


def _latest_13f(cik10: str) -> dict | None:
    """Return {accession, filed_date, form, primary_doc_url} for the
    most recent 13F-HR filing from this fund, or None."""
    data = _fetch(f"https://data.sec.gov/submissions/CIK{cik10}.json")
    if not data:
        return None
    recent = (data.get("filings") or {}).get("recent") or {}
    forms = recent.get("form") or []
    accs = recent.get("accessionNumber") or []
    dates = recent.get("filingDate") or []
    primary = recent.get("primaryDocument") or []
    for i, f in enumerate(forms):
        if f in ("13F-HR", "13F-HR/A"):
            acc = accs[i] if i < len(accs) else None
            if not acc:
                continue
            acc_nodash = acc.replace("-", "")
            doc = primary[i] if i < len(primary) else ""
            return {
                "accession": acc,
                "filed_date": dates[i] if i < len(dates) else "",
                "form": f,
                "primary_doc_url": (
                    f"https://www.sec.gov/Archives/edgar/data/"
                    f"{int(cik10)}/{acc_nodash}/{doc}"
                ),
                "filing_index_url": (
                    f"https://www.sec.gov/cgi-bin/browse-edgar?"
                    f"action=getcompany&CIK={cik10}&type=13F"
                ),
            }
    return None


class HedgeFund13FScout(ScoutAgent):
    """Daily check for new 13F filings from top alpha funds."""

    name = "hedge_fund_13f"

    def scan(self) -> list[ScoutSignal]:
        signals: list[ScoutSignal] = []
        snapshot: list[dict] = []
        for cik, fund in FUNDS.items():
            info = _latest_13f(cik)
            if not info:
                continue
            row = {"fund": fund, "cik": cik, **info}
            snapshot.append(row)
            # Publish one signal per fund per cycle. TTL 7d — 13F
            # filings are quarterly so older-than-week is stale.
            signals.append(ScoutSignal(
                venue="research", signal_type="hedge_fund_13f",
                payload=row, ttl_seconds=7 * 86_400,
            ))

        # Sort newest filing first, dump a consolidated dashboard JSON.
        snapshot.sort(key=lambda r: r.get("filed_date", ""), reverse=True)
        out = {
            "as_of": datetime.now(UTC).isoformat(),
            "n_funds": len(FUNDS),
            "n_filings_seen": len(snapshot),
            "filings": snapshot,
            "method_notes": (
                "13F-HR filings disclose long US positions for funds "
                ">$100M AUM, filed 45 days after quarter-end. Holdings "
                "extraction is a follow-up; this scout surfaces filing "
                "events so the analyst (or LLM) can read the table."
            ),
        }
        try:
            p = Path("docs/hedge_fund_13f.json")
            p.parent.mkdir(parents=True, exist_ok=True)
            tmp = p.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(out, indent=2), encoding="utf-8")
            tmp.replace(p)
            logger.info(f"hedge_fund_13f: {len(snapshot)}/{len(FUNDS)} "
                         f"funds with 13F filings")
        except Exception as e:
            logger.warning(f"hedge_fund_13f: dashboard write failed: {e}")
        return signals
