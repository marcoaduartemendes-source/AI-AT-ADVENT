#!/usr/bin/env bash
# AI-AT-ADVENT one-shot installer.
#
# Run on a fresh Ubuntu 24.04 VPS as root. Idempotent — safe to re-run.
# Sets up Python 3.12, clones the repo, drops systemd unit files,
# creates the env-file placeholder, but DOES NOT start timers
# (you must fill in /etc/aaa.env first).
#
# Usage on your laptop:
#     scp deploy/install.sh root@<IP>:/root/install.sh
#     ssh root@<IP> 'bash /root/install.sh'

set -euo pipefail

REPO_URL="${REPO_URL:-https://github.com/marcoaduartemendes-source/AI-AT-ADVENT.git}"
REPO_BRANCH="${REPO_BRANCH:-claude/daily-ai-news-digest-vcaC1}"
INSTALL_DIR="/opt/ai-at-advent"
ENV_FILE="/etc/aaa.env"
SERVICE_USER="aaa"

echo "════════════════════════════════════════════════════════════════"
echo "  AI-AT-ADVENT installer"
echo "  Repo:    $REPO_URL"
echo "  Branch:  $REPO_BRANCH"
echo "  Path:    $INSTALL_DIR"
echo "════════════════════════════════════════════════════════════════"

if [[ $EUID -ne 0 ]]; then
    echo "ERROR: Run as root (sudo bash install.sh)" >&2
    exit 1
fi

# ─── 1. System packages ───────────────────────────────────────────────
echo "[1/6] Updating apt + installing prerequisites"
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install -y -qq \
    git curl ca-certificates \
    python3 python3-venv python3-pip python3-dev \
    build-essential \
    sqlite3 \
    >/dev/null

# Pin Python to 3.12 explicitly if 3.11/3.10 is the system default
if ! python3 --version | grep -qE "Python 3\.1[2-9]"; then
    echo "  installing Python 3.12 from deadsnakes"
    apt-get install -y -qq software-properties-common >/dev/null
    add-apt-repository -y ppa:deadsnakes/ppa
    apt-get update -qq
    apt-get install -y -qq python3.12 python3.12-venv python3.12-dev >/dev/null
    PY=python3.12
else
    PY=python3
fi
echo "  python: $($PY --version)"

# ─── 2. Service user ──────────────────────────────────────────────────
if ! id -u "$SERVICE_USER" >/dev/null 2>&1; then
    echo "[2/6] Creating service user '$SERVICE_USER'"
    useradd --system --create-home --shell /usr/sbin/nologin "$SERVICE_USER"
else
    echo "[2/6] User '$SERVICE_USER' already exists"
fi

# ─── 3. Clone or update the repo ──────────────────────────────────────
# Add safe.directory before any git ops because if a previous partial
# run left the dir owned by `$SERVICE_USER`, root's `git` will refuse
# with "dubious ownership". Idempotent.
git config --global --add safe.directory "$INSTALL_DIR" 2>/dev/null || true
if [[ -d "$INSTALL_DIR/.git" ]]; then
    echo "[3/6] Updating existing checkout at $INSTALL_DIR"
    git -C "$INSTALL_DIR" fetch --all --quiet
    git -C "$INSTALL_DIR" checkout "$REPO_BRANCH" --quiet
    git -C "$INSTALL_DIR" reset --hard "origin/$REPO_BRANCH" --quiet
else
    echo "[3/6] Cloning repo to $INSTALL_DIR"
    git clone --branch "$REPO_BRANCH" --quiet "$REPO_URL" "$INSTALL_DIR"
fi
chown -R "$SERVICE_USER:$SERVICE_USER" "$INSTALL_DIR"

# ─── 4. Python venv + dependencies ────────────────────────────────────
echo "[4/6] Building Python venv at $INSTALL_DIR/.venv"
sudo -u "$SERVICE_USER" $PY -m venv "$INSTALL_DIR/.venv"
sudo -u "$SERVICE_USER" "$INSTALL_DIR/.venv/bin/pip" install --quiet --upgrade pip wheel
sudo -u "$SERVICE_USER" "$INSTALL_DIR/.venv/bin/pip" install --quiet -r "$INSTALL_DIR/requirements.txt"

# ─── 5. Environment file ──────────────────────────────────────────────
if [[ ! -f "$ENV_FILE" ]]; then
    echo "[5/6] Creating placeholder $ENV_FILE — YOU MUST FILL THIS IN"
    cp "$INSTALL_DIR/deploy/aaa.env.example" "$ENV_FILE"
    # 640 (not 600) so the systemd unit running as $SERVICE_USER can
    # read it via `bash -c '. /etc/aaa.env'`. Previously install.sh
    # set 600 but deploy_vps.yml set 640 — the 600 path silently
    # broke first-run because the unit couldn't read its own secrets.
    chown root:"$SERVICE_USER" "$ENV_FILE"
    chmod 640 "$ENV_FILE"
else
    echo "[5/6] $ENV_FILE already exists — leaving alone"
fi

# ─── 6. Systemd units ─────────────────────────────────────────────────
echo "[6/6] Installing systemd units"
for unit in orchestrator scouts dashboard; do
    cp "$INSTALL_DIR/deploy/systemd/$unit.service" "/etc/systemd/system/$unit.service"
    cp "$INSTALL_DIR/deploy/systemd/$unit.timer"   "/etc/systemd/system/$unit.timer"
done
systemctl daemon-reload

# ─── Done ─────────────────────────────────────────────────────────────
cat <<'BANNER'

════════════════════════════════════════════════════════════════════
  Installation complete. Next steps:

  1. Edit /etc/aaa.env and fill in every secret.
     (Same names as your GitHub Actions secrets:
      COINBASE_API_KEY, ALPACA_API_KEY_ID, FMP_API_KEY,
      PUSHOVER_USER_KEY, PUSHOVER_APP_TOKEN, etc.)

       nano /etc/aaa.env

  2. Start the timers:

       systemctl enable --now orchestrator.timer scouts.timer dashboard.timer

  3. Verify:

       systemctl list-timers --all
       journalctl -u orchestrator.service -f

  4. After 24h of clean runs, disable the GitHub Actions cron in
     .github/workflows/orchestrator.yml etc to avoid double-runs.

════════════════════════════════════════════════════════════════════
BANNER
