import hashlib
import hmac
import json
import secrets
import time
import uuid

import jwt
import requests


class CoinbaseClient:
    """Coinbase Advanced Trade API client.

    Auto-detects auth mode from the key format:

    • CDP / JWT  — key starts with 'organizations/…/apiKeys/…'
                   secret is an EC P-256 PEM private key
                   (current default for new Coinbase keys)

    • Legacy HMAC — any other key format
                   secret is a base64 string
    """

    BASE_URL = "https://api.coinbase.com"
    HOST = "api.coinbase.com"

    def __init__(self, api_key: str = "", api_secret: str = ""):
        self.api_key = api_key.strip()
        self.session = requests.Session()

        if not self.api_key:
            # Public-only mode: no auth, only public endpoints work
            self.auth_mode = "public"
        elif self.api_key.startswith("organizations/"):
            self.auth_mode = "jwt"
            # python-dotenv does not unescape — convert literal "\n" → real newline
            self.private_key = api_secret.replace("\\n", "\n").strip()
            if "BEGIN" not in self.private_key:
                raise ValueError(
                    "COINBASE_API_SECRET must be a PEM private key "
                    "(starts with '-----BEGIN EC PRIVATE KEY-----')"
                )
        else:
            self.auth_mode = "hmac"
            self.api_secret = api_secret.strip()

    # ── Auth ──────────────────────────────────────────────────────────────────

    def _build_jwt(self, method: str, path: str) -> str:
        """Build a short-lived JWT for one request."""
        uri = f"{method.upper()} {self.HOST}{path}"
        now = int(time.time())
        payload = {
            "sub": self.api_key,
            "iss": "cdp",
            "nbf": now,
            "exp": now + 120,
            "aud": ["retail_rest_api_proxy"],
            "uri": uri,
        }
        return jwt.encode(
            payload,
            self.private_key,
            algorithm="ES256",
            headers={"kid": self.api_key, "nonce": secrets.token_hex(10)},
        )

    def _headers(self, method: str, path: str, body: str = "") -> dict[str, str]:
        if self.auth_mode == "jwt":
            token = self._build_jwt(method, path)
            return {
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json",
            }
        # Legacy HMAC
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

    # ── HTTP helpers ─────────────────────────────────────────────────────────

    def _get(self, path: str, params: dict | None = None) -> dict:
        resp = self.session.get(
            f"{self.BASE_URL}{path}",
            headers=self._headers("GET", path),
            params=params,
            timeout=15,
        )
        resp.raise_for_status()
        return resp.json()

    def _post(self, path: str, body: dict) -> dict:
        body_str = json.dumps(body)
        resp = self.session.post(
            f"{self.BASE_URL}{path}",
            headers=self._headers("POST", path, body_str),
            data=body_str,
            timeout=15,
        )
        resp.raise_for_status()
        return resp.json()

    # ── Endpoints ────────────────────────────────────────────────────────────

    def get_accounts(self) -> list[dict]:
        data = self._get("/api/v3/brokerage/accounts")
        return data.get("accounts", [])

    def get_candles(
        self, product_id: str, granularity: str, start: int, end: int
    ) -> list[dict]:
        # Use public market endpoint (no auth) when running unauthenticated
        if self.auth_mode == "public":
            path = f"/api/v3/brokerage/market/products/{product_id}/candles"
            resp = self.session.get(
                f"{self.BASE_URL}{path}",
                params={"start": start, "end": end, "granularity": granularity},
                timeout=15,
            )
            resp.raise_for_status()
            return resp.json().get("candles", [])
        path = f"/api/v3/brokerage/products/{product_id}/candles"
        data = self._get(path, {"start": start, "end": end, "granularity": granularity})
        return data.get("candles", [])

    def get_product(self, product_id: str) -> dict:
        return self._get(f"/api/v3/brokerage/products/{product_id}")

    def get_best_bid_ask(self, product_ids: list[str]) -> dict:
        return self._get("/api/v3/brokerage/best_bid_ask", {"product_ids": product_ids})

    def create_market_buy(self, product_id: str, quote_size: str) -> dict:
        """Buy spending exactly `quote_size` USD."""
        body = {
            "client_order_id": str(uuid.uuid4()),
            "product_id": product_id,
            "side": "BUY",
            "order_configuration": {"market_market_ioc": {"quote_size": quote_size}},
        }
        return self._post("/api/v3/brokerage/orders", body)

    def create_market_sell(self, product_id: str, base_size: str) -> dict:
        """Sell `base_size` units of the base asset."""
        body = {
            "client_order_id": str(uuid.uuid4()),
            "product_id": product_id,
            "side": "SELL",
            "order_configuration": {"market_market_ioc": {"base_size": base_size}},
        }
        return self._post("/api/v3/brokerage/orders", body)

    def get_order(self, order_id: str) -> dict:
        return self._get(f"/api/v3/brokerage/orders/historical/{order_id}")
