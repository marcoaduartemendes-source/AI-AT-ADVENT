import hashlib
import hmac
import json
import time
import uuid
from typing import Dict, List, Optional

import requests


class CoinbaseClient:
    """Coinbase Advanced Trade API client using HMAC authentication."""

    BASE_URL = "https://api.coinbase.com"

    def __init__(self, api_key: str, api_secret: str):
        self.api_key = api_key
        self.api_secret = api_secret.strip()
        self.session = requests.Session()

    def _sign(self, method: str, path: str, body: str = "") -> Dict[str, str]:
        timestamp = str(int(time.time()))
        message = f"{timestamp}{method.upper()}{path}{body}"
        sig = hmac.new(
            self.api_secret.encode("utf-8"),
            message.encode("utf-8"),
            digestmod=hashlib.sha256,
        ).hexdigest()
        return {
            "CB-ACCESS-KEY": self.api_key,
            "CB-ACCESS-SIGN": sig,
            "CB-ACCESS-TIMESTAMP": timestamp,
            "Content-Type": "application/json",
        }

    def _get(self, path: str, params: Optional[Dict] = None) -> Dict:
        headers = self._sign("GET", path)
        resp = self.session.get(
            f"{self.BASE_URL}{path}", headers=headers, params=params, timeout=15
        )
        resp.raise_for_status()
        return resp.json()

    def _post(self, path: str, body: Dict) -> Dict:
        body_str = json.dumps(body)
        headers = self._sign("POST", path, body_str)
        resp = self.session.post(
            f"{self.BASE_URL}{path}", headers=headers, data=body_str, timeout=15
        )
        resp.raise_for_status()
        return resp.json()

    def get_accounts(self) -> List[Dict]:
        data = self._get("/api/v3/brokerage/accounts")
        return data.get("accounts", [])

    def get_candles(
        self, product_id: str, granularity: str, start: int, end: int
    ) -> List[Dict]:
        path = f"/api/v3/brokerage/products/{product_id}/candles"
        data = self._get(path, {"start": start, "end": end, "granularity": granularity})
        return data.get("candles", [])

    def get_product(self, product_id: str) -> Dict:
        return self._get(f"/api/v3/brokerage/products/{product_id}")

    def get_best_bid_ask(self, product_ids: List[str]) -> Dict:
        return self._get("/api/v3/brokerage/best_bid_ask", {"product_ids": product_ids})

    def create_market_buy(self, product_id: str, quote_size: str) -> Dict:
        """Buy spending exactly `quote_size` USD."""
        body = {
            "client_order_id": str(uuid.uuid4()),
            "product_id": product_id,
            "side": "BUY",
            "order_configuration": {"market_market_ioc": {"quote_size": quote_size}},
        }
        return self._post("/api/v3/brokerage/orders", body)

    def create_market_sell(self, product_id: str, base_size: str) -> Dict:
        """Sell `base_size` units of the base asset."""
        body = {
            "client_order_id": str(uuid.uuid4()),
            "product_id": product_id,
            "side": "SELL",
            "order_configuration": {"market_market_ioc": {"base_size": base_size}},
        }
        return self._post("/api/v3/brokerage/orders", body)

    def get_order(self, order_id: str) -> Dict:
        return self._get(f"/api/v3/brokerage/orders/historical/{order_id}")
