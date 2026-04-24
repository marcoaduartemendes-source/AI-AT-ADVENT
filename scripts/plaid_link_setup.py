#!/usr/bin/env python3
"""
One-time interactive setup to link your Chase account via Plaid.

Run this locally (not in CI):
    python scripts/plaid_link_setup.py

It will:
  1. Create a Plaid Link token.
  2. Open a browser so you can authenticate with Chase.
  3. Exchange the public token for a permanent access token.
  4. Print the PLAID_ACCESS_TOKEN you should save to your .env
     and GitHub Actions secrets.

Prerequisites:
  - Sign up at https://dashboard.plaid.com  (free development account)
  - Copy PLAID_CLIENT_ID and PLAID_SECRET into your .env
  - pip install plaid-python python-dotenv
"""
import os
import sys
import webbrowser
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import parse_qs, urlparse

from dotenv import load_dotenv

load_dotenv()

PLAID_CLIENT_ID = os.getenv("PLAID_CLIENT_ID", "")
PLAID_SECRET = os.getenv("PLAID_SECRET", "")
PLAID_ENV = os.getenv("PLAID_ENV", "production")
PORT = 8765

_state: dict = {"link_token": "", "public_token": ""}


class _LinkHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        qs = parse_qs(urlparse(self.path).query)
        if "public_token" in qs:
            _state["public_token"] = qs["public_token"][0]
            self.send_response(200)
            self.end_headers()
            self.wfile.write(
                b"<html><body style='font-family:sans-serif;padding:40px'>"
                b"<h2 style='color:#2f9e44'>&#10003; Chase linked successfully!</h2>"
                b"<p>You can close this tab and return to the terminal.</p>"
                b"</body></html>"
            )
        else:
            # Serve the Plaid Link UI
            html = f"""<!DOCTYPE html>
<html><head>
  <script src="https://cdn.plaid.com/link/v2/stable/link-initialize.js"></script>
</head>
<body style="font-family:sans-serif;padding:40px">
  <p>Opening Plaid Link&hellip; (allow pop-ups if nothing happens)</p>
  <button onclick="handler.open()" style="padding:12px 24px;font-size:16px;
    background:#667eea;color:#fff;border:none;border-radius:8px;cursor:pointer">
    Connect Chase Account
  </button>
  <script>
    var handler = Plaid.create({{
      token: '{_state["link_token"]}',
      onSuccess: function(public_token, metadata) {{
        window.location.href = '/?public_token=' + public_token;
      }},
      onExit: function(err, metadata) {{
        document.body.innerHTML = '<p style="color:red">Cancelled or error: ' +
          JSON.stringify(err) + '</p>';
      }},
    }});
    handler.open();
  </script>
</body></html>"""
            self.send_response(200)
            self.end_headers()
            self.wfile.write(html.encode())

    def log_message(self, *_):
        pass  # suppress access logs


def main():
    if not PLAID_CLIENT_ID or not PLAID_SECRET:
        print("ERROR: Set PLAID_CLIENT_ID and PLAID_SECRET in your .env file first.")
        print("  Sign up at https://dashboard.plaid.com (free development tier)")
        sys.exit(1)

    try:
        import plaid
        from plaid.api import plaid_api
        from plaid.model.country_code import CountryCode
        from plaid.model.item_public_token_exchange_request import ItemPublicTokenExchangeRequest
        from plaid.model.link_token_create_request import LinkTokenCreateRequest
        from plaid.model.link_token_create_request_user import LinkTokenCreateRequestUser
        from plaid.model.products import Products
    except ImportError:
        print("ERROR: Run  pip install plaid-python  first.")
        sys.exit(1)

    env_map = {
        "sandbox": plaid.Environment.Sandbox,
        "development": plaid.Environment.Development,
        "production": plaid.Environment.Production,
    }
    cfg = plaid.Configuration(
        host=env_map.get(PLAID_ENV, plaid.Environment.Production),
        api_key={"clientId": PLAID_CLIENT_ID, "secret": PLAID_SECRET},
    )
    client = plaid_api.PlaidApi(plaid.ApiClient(cfg))

    print(f"Creating Plaid Link token (env={PLAID_ENV})…")
    resp = client.link_token_create(LinkTokenCreateRequest(
        user=LinkTokenCreateRequestUser(client_user_id="cash-flow-local"),
        client_name="Cash Flow Forecast",
        products=[Products("transactions")],
        country_codes=[CountryCode("US")],
        language="en",
    ))
    _state["link_token"] = resp["link_token"]
    print("Link token created.")

    server = HTTPServer(("localhost", PORT), _LinkHandler)
    url = f"http://localhost:{PORT}"
    print(f"\nOpening {url} in your browser…")
    print("Complete the Chase login in the browser, then return here.")
    webbrowser.open(url)

    while not _state["public_token"]:
        server.handle_request()

    print("\nExchanging public token for permanent access token…")
    ex = client.item_public_token_exchange(
        ItemPublicTokenExchangeRequest(public_token=_state["public_token"])
    )
    access_token = ex["access_token"]
    item_id = ex["item_id"]

    print(f"\n{'=' * 60}")
    print("SUCCESS!  Add these to your .env and GitHub Actions secrets:")
    print(f"\n  PLAID_ACCESS_TOKEN={access_token}")
    print(f"  # Item ID (keep for reference): {item_id}")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()
