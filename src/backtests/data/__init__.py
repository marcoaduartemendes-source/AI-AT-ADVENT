"""Historical-data fetchers for backtests.

Each module wraps one external data source. They all return data in
the same shape (typed dataclasses) so backtests can mix sources
without caring where the bytes came from.

Currently:
  - polygon.py : Polygon.io for equity earnings, financials, daily bars,
                 crypto aggregates. Requires POLYGON_API_KEY env var.
"""
