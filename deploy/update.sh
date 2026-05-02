#!/usr/bin/env bash
# Pull the latest commit and refresh dependencies.
#
# Run on the VPS as root, e.g. via SSH:
#     ssh root@<IP> 'bash /opt/ai-at-advent/deploy/update.sh'
#
# Or wire it to a cron / webhook listener for push-on-merge deploys.

set -euo pipefail

INSTALL_DIR="${INSTALL_DIR:-/opt/ai-at-advent}"
SERVICE_USER="${SERVICE_USER:-aaa}"

if [[ $EUID -ne 0 ]]; then
    echo "ERROR: Run as root" >&2
    exit 1
fi

cd "$INSTALL_DIR"

# Ensure runtime dirs exist (systemd ProtectSystem=strict makes
# everything outside ReadWritePaths read-only, so /opt/.../data
# must exist before the service starts or it fails with
# "Failed to set up mount namespacing").
install -d -o "$SERVICE_USER" -g "$SERVICE_USER" \
    "$INSTALL_DIR/data" "$INSTALL_DIR/docs"

echo "[1/4] Fetching latest"
sudo -u "$SERVICE_USER" git fetch --all --quiet

CURRENT="$(sudo -u "$SERVICE_USER" git rev-parse HEAD)"
sudo -u "$SERVICE_USER" git reset --hard "origin/$(git rev-parse --abbrev-ref HEAD)" --quiet
NEW="$(sudo -u "$SERVICE_USER" git rev-parse HEAD)"

if [[ "$CURRENT" == "$NEW" ]]; then
    echo "  already at $CURRENT — nothing to do"
    exit 0
fi
echo "  $CURRENT → $NEW"

echo "[2/4] Refreshing Python deps (only if requirements.txt changed)"
if sudo -u "$SERVICE_USER" git diff "$CURRENT" "$NEW" --name-only | grep -q "^requirements.txt$"; then
    sudo -u "$SERVICE_USER" "$INSTALL_DIR/.venv/bin/pip" install --quiet -r requirements.txt
else
    echo "  requirements unchanged"
fi

echo "[3/4] Refreshing systemd unit files (if changed)"
for unit in orchestrator scouts dashboard; do
    src="$INSTALL_DIR/deploy/systemd/$unit.service"
    dst="/etc/systemd/system/$unit.service"
    if ! cmp -s "$src" "$dst"; then
        cp "$src" "$dst"
        cp "$INSTALL_DIR/deploy/systemd/$unit.timer" "/etc/systemd/system/$unit.timer"
        echo "  updated $unit"
    fi
done
systemctl daemon-reload

echo "[4/4] Deploy complete. Next timer tick will run the new code."
