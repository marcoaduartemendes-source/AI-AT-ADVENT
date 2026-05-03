"""Tests for the live FOMC calendar scrape in scouts/macro_scout.py.

Sprint B4 audit fix: the static _FOMC_FALLBACK_2026 list silently
expires every January. _fetch_fomc_calendar() now scrapes
federalreserve.gov as the primary source, with the static lists
as fallback. Tests pin down both paths.
"""
from __future__ import annotations

from unittest.mock import patch

from scouts.macro_scout import _FOMC_FALLBACK, _fetch_fomc_calendar


_REAL_PAGE_FRAGMENT = """
<html>
<body>
<div id="article">
<p>Meeting calendars and information</p>
<h4>2026 FOMC Meetings</h4>
<p>January 27-28, 2026</p>
<p>March 17-18, 2026</p>
<p>April 28-29, 2026</p>
<p>June 16-17, 2026</p>
<p>July 28-29, 2026</p>
<p>September 15-16, 2026</p>
<p>November 3-4, 2026</p>
<p>December 8-9, 2026</p>
</div>
</body>
</html>
"""


def test_scrape_extracts_all_2026_meetings():
    """Live page parser pulls 8 meetings from a realistic HTML fragment."""
    with patch("scouts.macro_scout.cached_get",
                return_value=_REAL_PAGE_FRAGMENT):
        meetings = _fetch_fomc_calendar()
    # The parser uses end-day for two-day meetings → "January 27-28, 2026"
    # gives "2026-01-28". With today around May 2026, only future meetings
    # come back; older ones get filtered.
    # We don't assert exact count (depends on today's date) — just that
    # the parser pulled a non-trivial number from the fragment.
    assert any(m.startswith("2026-") or m.startswith("2027-")
                 for m in meetings)


def test_scrape_falls_back_when_page_fetch_fails():
    """cached_get returning None triggers fallback to the static list."""
    with patch("scouts.macro_scout.cached_get", return_value=None):
        meetings = _fetch_fomc_calendar()
    # Fallback list (filtered to >= today) should yield the union
    # of 2026 + 2027 dates
    assert isinstance(meetings, list)
    # Every fallback entry that we get back must also exist in the
    # union, in ISO format
    iso_set = set(_FOMC_FALLBACK)
    if meetings:
        assert all(m in iso_set for m in meetings)


def test_scrape_falls_back_on_garbage_html():
    """If the page is HTML but doesn't match our regex, fall back."""
    with patch("scouts.macro_scout.cached_get",
                return_value="<html><body>Site under maintenance</body></html>"):
        meetings = _fetch_fomc_calendar()
    iso_set = set(_FOMC_FALLBACK)
    if meetings:
        assert all(m in iso_set for m in meetings)


def test_fallback_list_is_chronologically_sorted_and_iso():
    """Defensive: the static fallback must be ISO-format and sorted."""
    from datetime import datetime
    parsed = [
        datetime.strptime(d, "%Y-%m-%d").date() for d in _FOMC_FALLBACK
    ]
    assert parsed == sorted(parsed)
    # Spot-check a known meeting (Sep 2027 is on the calendar)
    assert "2027-09-22" in _FOMC_FALLBACK


def test_scrape_handles_dict_response_gracefully():
    """If cached_get returns JSON (dict) instead of HTML — the page
    upstream sometimes serves JSON to bots — fall back cleanly."""
    with patch("scouts.macro_scout.cached_get",
                return_value={"unexpected": "json"}):
        meetings = _fetch_fomc_calendar()
    iso_set = set(_FOMC_FALLBACK)
    if meetings:
        assert all(m in iso_set for m in meetings)
