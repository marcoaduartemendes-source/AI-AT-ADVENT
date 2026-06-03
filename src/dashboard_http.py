"""Basic-auth static file server for the live dashboard.

Why a 50-line custom server instead of `python -m http.server`:
  The stock module serves `/opt/ai-at-advent/docs` to the world with no
  authentication, no token, no TLS. Open port 8080 publicly and your
  full trade ledger, equity history, kill-switch state, and per-strategy
  P&L become world-readable to anyone who can reach the IP — a
  retirement-critical-money disaster waiting to happen.

This wraps stdlib SimpleHTTPRequestHandler with HTTP Basic auth (RFC
7617). Credentials come from /etc/aaa.env via systemd:
    DASHBOARD_USER=...
    DASHBOARD_PASS=...
Both MUST be set; the process refuses to start without them so we
can't accidentally serve unauthenticated.

NOT a substitute for TLS: HTTP Basic sends the password
base64-encoded over plain HTTP, sniffable on a hostile network. The
right setup for a real-money dashboard is also one of:
  • DigitalOcean cloud firewall restricting :8080 to your IP/32
  • SSH port-forward (`ssh -L 8080:localhost:8080 root@<box>`), which
    gives TLS-equivalent security and lets you keep the firewall closed
This auth is the third layer of defence: even if the firewall is wide
open, the page can't be read without the password.
"""
from __future__ import annotations

import base64
import hmac
import http.server
import logging
import os
import socketserver
import sys

logger = logging.getLogger("dashboard-http")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")


def _expected_header() -> str:
    user = os.environ.get("DASHBOARD_USER", "").strip()
    pwd = os.environ.get("DASHBOARD_PASS", "").strip()
    if not user or not pwd:
        sys.stderr.write(
            "FATAL: DASHBOARD_USER and DASHBOARD_PASS must be set in "
            "/etc/aaa.env before this server will start. Generate a "
            "strong password (e.g. `openssl rand -base64 24`) and add "
            "both vars.\n"
        )
        sys.exit(2)
    return "Basic " + base64.b64encode(f"{user}:{pwd}".encode()).decode()


EXPECTED = _expected_header()
REALM = os.environ.get("DASHBOARD_REALM", "ai-at-advent")
ROOT = os.environ.get("DASHBOARD_ROOT", "/opt/ai-at-advent/docs")
PORT = int(os.environ.get("DASHBOARD_PORT", "8080"))


class _Handler(http.server.SimpleHTTPRequestHandler):
    def _authed(self) -> bool:
        got = self.headers.get("Authorization", "")
        # Constant-time compare so brute-forcers can't time-of-check the
        # username/password byte-by-byte.
        if hmac.compare_digest(got, EXPECTED):
            return True
        self.send_response(401)
        self.send_header("WWW-Authenticate", f'Basic realm="{REALM}"')
        self.send_header("Content-Length", "0")
        self.end_headers()
        return False

    def do_GET(self):     # noqa: N802 — stdlib name
        if self._authed():
            super().do_GET()

    def do_HEAD(self):    # noqa: N802
        if self._authed():
            super().do_HEAD()

    # Squash the noisy default access log — one line per static asset
    # would drown out anything useful in journalctl.
    def log_message(self, fmt, *args):
        pass


class _Server(socketserver.ThreadingMixIn, http.server.HTTPServer):
    """Threaded so one slow/broken client can't block the whole dashboard.

    2026-06-03: the single-threaded TCPServer was wedging after random
    ConnectionResetError events from health-checks/scanners/timed-out
    browsers — nginx then hit it with 504 and the service appeared "up"
    but unresponsive. Threading isolates each request; daemon_threads
    means we don't leak threads on shutdown.
    """
    daemon_threads = True
    allow_reuse_address = True

    def handle_error(self, request, client_address):
        # Don't crash the server on a client connection blip. Suppress
        # ConnectionResetError noise; log anything else.
        exc = sys.exc_info()[1]
        if isinstance(exc, (ConnectionResetError, BrokenPipeError, TimeoutError)):
            return
        logger.warning(f"request error from {client_address}: {exc!r}")


def main() -> int:
    os.chdir(ROOT)
    bind = os.environ.get("DASHBOARD_BIND", "0.0.0.0").strip()
    with _Server((bind, PORT), _Handler) as srv:
        # Optional TLS — wrap the socket if DASHBOARD_CERT + DASHBOARD_KEY
        # are set. Lets the dashboard listen on 443 with HTTPS so it works
        # from networks that block 8080 (common on home/mobile networks).
        cert = os.environ.get("DASHBOARD_CERT", "").strip()
        key = os.environ.get("DASHBOARD_KEY", "").strip()
        if cert and key:
            import ssl
            ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
            ctx.load_cert_chain(certfile=cert, keyfile=key)
            srv.socket = ctx.wrap_socket(srv.socket, server_side=True)
            scheme = "https"
        else:
            scheme = "http"
        logger.info(f"dashboard-http: serving {ROOT} on {scheme}://{bind}:{PORT} "
                    f"(Basic auth required, realm={REALM!r})")
        try:
            srv.serve_forever()
        except KeyboardInterrupt:
            pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
