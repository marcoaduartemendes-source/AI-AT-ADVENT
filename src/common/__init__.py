"""Shared utilities used by scouts + strategies + adapters."""
from .http_cache import cached_get, clear_cache

__all__ = ["cached_get", "clear_cache"]
