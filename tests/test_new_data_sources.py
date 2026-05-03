"""Tests for the 4 new data wrappers shipped alongside scout integrations:

  • CMEFedWatchClient        — Fed Funds futures-implied probabilities
  • PolymarketClient         — second prediction-market venue
  • NewsRSSClient            — multi-source headline aggregator
  • BinanceClient            — public futures + spot data

Each wrapper has the same shape: `is_configured()`, disk cache, error-
tolerant fetch. We hit no network — every test stubs requests.get or
provides a hardcoded XML/JSON fixture.
"""
from __future__ import annotations

import json
from datetime import date, UTC
from unittest.mock import MagicMock, patch

from backtests.data.binance import (
    BinanceClient,
    coinbase_to_binance,
)
from backtests.data.cme_fedwatch import (
    CMEFedWatchClient,
    _parse_fedwatch,
)
from backtests.data.news_rss import (
    NewsRSSClient,
    _parse_feed,
)
from backtests.data.polymarket import (
    PolymarketClient,
    _meaningful_tokens,
)


# ─── CME FedWatch ─────────────────────────────────────────────────────


def test_cme_fedwatch_always_configured(tmp_path):
    c = CMEFedWatchClient(cache_dir=str(tmp_path))
    assert c.is_configured() is True


def test_cme_parse_fedwatch_handles_top_level_meetings():
    payload = {
        "meetings": [
            {
                "meetingDate": "2026-06-17",
                "probabilities": [
                    {"lo": 425, "hi": 450, "probability": 0.62},
                    {"lo": 400, "hi": 425, "probability": 0.30},
                ],
            },
            {
                "date": "2026-07-29",
                "rates": [
                    # Test percent-form normalization
                    {"lower": 400, "upper": 425, "p": 55.0},
                ],
            },
        ],
    }
    rows = _parse_fedwatch(payload)
    assert len(rows) == 2
    # Sorted oldest first
    assert rows[0].meeting_date == date(2026, 6, 17)
    # First row: 2 prob entries
    assert len(rows[0].target_rate_probs) == 2
    # Probability normalized to 0-1
    assert rows[1].target_rate_probs[0][2] == 0.55


def test_cme_parse_handles_garbage():
    """Malformed CME response shouldn't crash — return empty list."""
    assert _parse_fedwatch({}) == []
    assert _parse_fedwatch({"meetings": "not a list"}) == []
    assert _parse_fedwatch({"meetings": [None, "string", 42]}) == []
    # Missing required fields
    assert _parse_fedwatch({"meetings": [{"meetingDate": "2026-06-17"}]}) == []


def test_cme_uses_disk_cache(tmp_path):
    """Two calls within TTL should hit the disk cache, not requests."""
    c = CMEFedWatchClient(cache_dir=str(tmp_path))
    payload = {"meetings": []}
    # Pre-seed the cache file directly
    cache_path = c._cache_path()
    cache_path.write_text(json.dumps(payload), encoding="utf-8")
    # Should NOT hit the network
    with patch("backtests.data.cme_fedwatch.requests.get") as mock_get:
        result = c.upcoming_meetings()
        assert mock_get.call_count == 0
    assert result == []


def test_cme_fetch_failure_returns_empty(tmp_path):
    c = CMEFedWatchClient(cache_dir=str(tmp_path))
    with patch("backtests.data.cme_fedwatch.requests.get") as mock_get:
        mock_get.return_value.status_code = 403
        mock_get.return_value.text = "forbidden"
        result = c.upcoming_meetings(cache_ttl_seconds=0)
    assert result == []


# ─── Polymarket ───────────────────────────────────────────────────────


def test_polymarket_always_configured(tmp_path):
    assert PolymarketClient(cache_dir=str(tmp_path)).is_configured() is True


def test_polymarket_meaningful_tokens_filters_stopwords():
    tokens = _meaningful_tokens("Will the Fed cut rates in June?")
    # "the", "in" are stopwords; ≤3 chars filtered
    assert "fed" in tokens
    assert "cut" in tokens
    assert "rates" in tokens
    assert "june" in tokens
    assert "the" not in tokens
    assert "in" not in tokens


def test_polymarket_active_markets_parses_response(tmp_path):
    c = PolymarketClient(cache_dir=str(tmp_path))
    fake_response = MagicMock()
    fake_response.status_code = 200
    fake_response.json.return_value = [
        {
            "id": "12345",
            "question": "Will Bitcoin close above $100k on Dec 31?",
            "lastTradePrice": 0.45,
            "volume24hr": 50000,
            "liquidity": 100000,
            "category": "Crypto",
            "endDate": "2026-12-31T23:59:59Z",
        },
        {
            # Probability stored as percent
            "id": "67890",
            "question": "Will SP500 hit 6000 by end of 2026?",
            "lastTradePrice": 75.0,    # 75% form
            "volume24hr": 200,         # below threshold
            "category": "Markets",
        },
    ]
    with patch("backtests.data.polymarket.requests.get",
                 return_value=fake_response):
        markets = c.active_markets(min_volume_usd=1000)
    # Only the high-volume market kept
    assert len(markets) == 1
    assert markets[0].market_id == "12345"
    assert 0 <= markets[0].yes_price <= 1
    assert markets[0].volume_24h_usd == 50000


def test_polymarket_find_kalshi_match_returns_best_overlap(tmp_path):
    c = PolymarketClient(cache_dir=str(tmp_path))
    from backtests.data.polymarket import PolymarketContract
    pm_universe = [
        PolymarketContract(
            market_id="A", question="Will Fed cut rates in June?",
            end_date=None, yes_price=0.55, no_price=0.45,
            volume_24h_usd=10000, liquidity_usd=20000,
            category="Politics", raw={},
        ),
        PolymarketContract(
            market_id="B", question="Will Bitcoin top $100k?",
            end_date=None, yes_price=0.40, no_price=0.60,
            volume_24h_usd=50000, liquidity_usd=80000,
            category="Crypto", raw={},
        ),
    ]
    match = c.find_kalshi_match(
        "Will the Federal Reserve cut rates in June 2026?",
        polymarket_markets=pm_universe,
    )
    assert match is not None
    assert match.market_id == "A"


def test_polymarket_find_kalshi_match_no_overlap(tmp_path):
    c = PolymarketClient(cache_dir=str(tmp_path))
    from backtests.data.polymarket import PolymarketContract
    pm_universe = [
        PolymarketContract(
            market_id="A", question="Super Bowl winner",
            end_date=None, yes_price=0.5, no_price=0.5,
            volume_24h_usd=10000, liquidity_usd=20000,
            category="Sports", raw={},
        ),
    ]
    match = c.find_kalshi_match(
        "Will Fed cut rates in June?", polymarket_markets=pm_universe,
    )
    assert match is None


# ─── News RSS ─────────────────────────────────────────────────────────


def test_news_rss_always_configured(tmp_path):
    assert NewsRSSClient(cache_dir=str(tmp_path)).is_configured() is True


def test_news_parse_rss_2_0_basic():
    body = """<?xml version="1.0"?>
    <rss version="2.0"><channel>
      <title>Test feed</title>
      <item>
        <title>Apple beats earnings on iPhone strength</title>
        <link>https://example.com/a</link>
        <pubDate>Mon, 03 May 2026 10:00:00 +0000</pubDate>
        <description>AAPL beat. Strong iPhone numbers.</description>
      </item>
      <item>
        <title>Tesla cuts prices ahead of Q3</title>
        <link>https://example.com/t</link>
        <pubDate>Mon, 03 May 2026 11:00:00 +0000</pubDate>
        <description>TSLA price cut.</description>
      </item>
    </channel></rss>
    """
    items = _parse_feed("test", body, max_items=10)
    assert len(items) == 2
    assert items[0].title.startswith("Apple")
    assert items[0].url == "https://example.com/a"
    assert items[0].published_at is not None


def test_news_parse_garbage_xml_returns_empty():
    assert _parse_feed("test", "not xml at all", 10) == []
    assert _parse_feed("test", "", 10) == []


def test_news_search_tickers_matches_with_word_boundary(tmp_path):
    """Ticker must appear delimited — "AAPL" should NOT match "AAPLE"."""
    c = NewsRSSClient(cache_dir=str(tmp_path))
    from backtests.data.news_rss import NewsItem
    from datetime import datetime
    now = datetime.now(UTC)
    items = [
        NewsItem(source="test", title="AAPL beats Q1 earnings",
                  url="u1", published_at=now, summary="...", raw_id="1"),
        NewsItem(source="test", title="AAPLE Foundation announces grant",
                  url="u2", published_at=now, summary="...", raw_id="2"),
        NewsItem(source="test", title="$TSLA halted briefly",
                  url="u3", published_at=now, summary="...", raw_id="3"),
    ]
    result = c.search_tickers(["AAPL", "TSLA"], items=items)
    assert "AAPL" in result and len(result["AAPL"]) == 1
    assert result["AAPL"][0].url == "u1"   # AAPLE excluded
    assert "TSLA" in result and len(result["TSLA"]) == 1


def test_news_search_tickers_filters_by_age(tmp_path):
    c = NewsRSSClient(cache_dir=str(tmp_path))
    from backtests.data.news_rss import NewsItem
    from datetime import datetime, timedelta
    now = datetime.now(UTC)
    items = [
        NewsItem(source="test", title="AAPL recent",
                  url="u1", published_at=now - timedelta(hours=2),
                  summary="...", raw_id="1"),
        NewsItem(source="test", title="AAPL ancient",
                  url="u2", published_at=now - timedelta(hours=48),
                  summary="...", raw_id="2"),
    ]
    result = c.search_tickers(["AAPL"], items=items, within_hours=24)
    assert len(result["AAPL"]) == 1
    assert result["AAPL"][0].url == "u1"


# ─── Binance ──────────────────────────────────────────────────────────


def test_binance_always_configured(tmp_path):
    assert BinanceClient(cache_dir=str(tmp_path)).is_configured() is True


def test_binance_symbol_mapping():
    assert coinbase_to_binance("BTC-USD") == "BTCUSDT"
    assert coinbase_to_binance("ETH-USD") == "ETHUSDT"
    # Unknown symbol falls back to suffix substitution
    assert coinbase_to_binance("XYZ-USD") == "XYZUSDT"
    assert coinbase_to_binance("RANDOM") == "RANDOM"


def test_binance_funding_history_parses_response(tmp_path):
    c = BinanceClient(cache_dir=str(tmp_path))
    fake_response = MagicMock()
    fake_response.status_code = 200
    fake_response.json.return_value = [
        {"symbol": "BTCUSDT", "fundingTime": 1746000000000,
         "fundingRate": "0.0001"},
        {"symbol": "BTCUSDT", "fundingTime": 1746028800000,
         "fundingRate": "0.0002"},
    ]
    with patch("backtests.data.binance.requests.get",
                return_value=fake_response):
        rows = c.funding_history("BTC-USD")
    assert len(rows) == 2
    assert rows[0].timestamp.timestamp() < rows[1].timestamp.timestamp()
    assert rows[0].funding_rate == 0.0001


def test_binance_consensus_agree(tmp_path):
    """Two-venue agreement: both APRs >1% and within 1% → agree."""
    c = BinanceClient(cache_dir=str(tmp_path))
    fake_resp = MagicMock()
    fake_resp.status_code = 200
    # 0.005% per 8h × 1095 = 5.475% APR
    fake_resp.json.return_value = [
        {"symbol": "BTCUSDT", "fundingTime": 1746000000000,
         "fundingRate": "0.00005"}
    ]
    with patch("backtests.data.binance.requests.get",
                return_value=fake_resp):
        consensus, agree = c.consensus_funding_apr(
            "BTC-USD", bybit_apr=5.0,
        )
    assert agree is True
    # Average ~5.24%
    assert 4.5 < consensus < 6.0


def test_binance_consensus_disagree_when_only_one_elevated(tmp_path):
    c = BinanceClient(cache_dir=str(tmp_path))
    fake_resp = MagicMock()
    fake_resp.status_code = 200
    # Binance flat at 0.5% APR; Bybit elevated at 6%
    fake_resp.json.return_value = [
        {"symbol": "BTCUSDT", "fundingTime": 1746000000000,
         "fundingRate": "0.0000046"}     # ≈ 0.5% APR
    ]
    with patch("backtests.data.binance.requests.get",
                return_value=fake_resp):
        _, agree = c.consensus_funding_apr("BTC-USD", bybit_apr=6.0)
    assert agree is False


def test_binance_429_skips_gracefully(tmp_path):
    """429 rate-limit on Binance shouldn't crash; returns empty list."""
    c = BinanceClient(cache_dir=str(tmp_path), timeout_seconds=1)
    rate_limited = MagicMock()
    rate_limited.status_code = 429
    rate_limited.text = "rate limit"
    with patch("backtests.data.binance.requests.get",
                return_value=rate_limited):
        with patch("time.sleep"):    # don't actually sleep in test
            rows = c.funding_history("BTC-USD")
    assert rows == []
