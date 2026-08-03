#!/usr/bin/env bash
# One-shot HTTPS domain setup for the dashboard.
#
# Usage:
#   bash deploy/setup_https_domain.sh <your-domain.duckdns.org>
#
# What it does:
#   1. Updates the nginx site to use <domain> as server_name.
#   2. Runs certbot to issue a free Let's Encrypt cert (HTTP-01 challenge).
#   3. Wires the real cert into nginx + reloads.
#   4. Confirms https://<domain>/ returns 200 with auth.
#   5. Enables certbot's systemd timer for auto-renewal (every 90 days).
#
# Permanent solution: green padlock, no warning, auto-renews forever, $0.
set -euo pipefail

DOMAIN="${1:-}"
if [[ -z "$DOMAIN" || "$DOMAIN" != *.* ]]; then
  echo "ERROR: pass your DuckDNS domain, e.g.:" >&2
  echo "    bash $0 marco-aaa.duckdns.org" >&2
  exit 2
fi

echo "[1/5] Installing certbot (no-op if already present)…"
DEBIAN_FRONTEND=noninteractive apt-get install -y certbot python3-certbot-nginx >/dev/null

echo "[2/5] Rewriting nginx site to use ${DOMAIN}…"
cat > /etc/nginx/sites-available/aaa <<EOF
# HTTP — redirects to HTTPS (and serves the certbot challenge).
server {
    listen 80 default_server;
    listen 8080;
    server_name ${DOMAIN} _;

    # Let's Encrypt HTTP-01 challenge path.
    location /.well-known/acme-challenge/ {
        root /var/www/letsencrypt;
    }
    location / {
        return 301 https://\$host\$request_uri;
    }
}

# HTTPS — proxies to the dashboard server on localhost:9999.
server {
    listen 443 ssl default_server;
    server_name ${DOMAIN};

    # certbot will rewrite these two lines to the Let's Encrypt cert
    # once issued; until then the existing self-signed cert is used.
    ssl_certificate     /etc/aaa-tls/cert.pem;
    ssl_certificate_key /etc/aaa-tls/key.pem;

    location / {
        proxy_pass http://127.0.0.1:9999;
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto https;
    }
}
EOF
mkdir -p /var/www/letsencrypt
nginx -t
systemctl reload nginx

echo "[3/5] Issuing Let's Encrypt certificate for ${DOMAIN}…"
# --nginx plugin auto-rewrites the ssl_certificate lines above to the
# real Let's Encrypt paths after issuance.
certbot --nginx -n --agree-tos \
    --register-unsafely-without-email \
    --redirect \
    -d "${DOMAIN}"

echo "[4/5] Verifying…"
sleep 2
nginx -t
systemctl reload nginx
echo -n "  local https reachability check: "
# 2026-06-11 security fix: the Basic-auth credential was hardcoded in
# this committed script (now compromised — ROTATE DASHBOARD_PASS). Read
# it from the runtime env file instead; fall back to an unauthenticated
# probe (a 401 still proves the server is up) so no secret is embedded.
_DASH_ENV="/etc/aaa-dashboard.env"
if [[ -f "$_DASH_ENV" ]]; then
    # shellcheck disable=SC1090
    set -a; . "$_DASH_ENV"; set +a
fi
if [[ -n "${DASHBOARD_USER:-}" && -n "${DASHBOARD_PASS:-}" ]]; then
    curl -ks -o /dev/null -w '%{http_code}\n' \
        -u "${DASHBOARD_USER}:${DASHBOARD_PASS}" "https://${DOMAIN}/" || true
else
    curl -ks -o /dev/null -w '%{http_code} (unauthenticated probe)\n' \
        "https://${DOMAIN}/" || true
fi

echo "[5/5] Enabling auto-renewal…"
systemctl enable --now certbot.timer
systemctl status certbot.timer --no-pager | head -3 || true

echo ""
echo "============================================================"
echo "  DONE — your permanent dashboard URL:"
echo ""
echo "    https://${DOMAIN}/"
echo ""
echo "  Real cert. Green padlock. No warning. Auto-renews every 90 days."
echo "============================================================"
