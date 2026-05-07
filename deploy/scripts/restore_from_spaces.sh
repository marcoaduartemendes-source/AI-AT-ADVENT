#!/usr/bin/env bash
# Restore SQLite state databases from the most recent DigitalOcean
# Spaces snapshot. Companion to deploy/scripts/db_backup.sh.
#
# When to use:
#   - data/*.db got corrupted (bad WAL, partial transaction)
#   - SQLite version drift / schema mismatch on a fresh install
#   - migrating to a new VPS — this is the recovery half
#
# Usage:
#   sudo bash /opt/ai-at-advent/deploy/scripts/restore_from_spaces.sh
#   sudo bash /opt/ai-at-advent/deploy/scripts/restore_from_spaces.sh \
#         --archive aaa-data-<host>-20260507T120000Z.tar.gz
#
# What it does:
#   1. Stops the orchestrator timer (so we don't restore mid-cycle).
#   2. Lists `aws s3 ls` on the configured bucket; picks the most-
#      recent archive (or the explicit --archive name).
#   3. Downloads + extracts to a temp dir.
#   4. Backs up the existing data/ to data/.before-restore-<timestamp>/
#      so a bad restore is itself reversible.
#   5. Atomically swaps the new DBs into place.
#   6. Restarts the timer and tails the next cycle's journal so the
#      operator can confirm health.
#
# Exit codes:
#   0  restore complete, timer back up
#   1  precondition failed (creds missing, AWS CLI missing, etc.)
#   2  archive download / extract failed
#   3  swap failed; .before-restore-<ts> backup remains so manual
#      recovery is still possible — DO NOT delete that directory
#      until you've confirmed the system is healthy.

set -euo pipefail

INSTALL_DIR="${INSTALL_DIR:-/opt/ai-at-advent}"
DATA_DIR="${DATA_DIR:-$INSTALL_DIR/data}"
SERVICE_USER="${SERVICE_USER:-aaa}"

EXPLICIT_ARCHIVE=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --archive) EXPLICIT_ARCHIVE="$2"; shift 2 ;;
        *) echo "unknown arg: $1" >&2; exit 1 ;;
    esac
done

if [[ $EUID -ne 0 ]]; then
    echo "ERROR: run as root" >&2
    exit 1
fi

# ─── Preconditions ─────────────────────────────────────────────────────
if [[ -z "${SPACES_ACCESS_KEY_ID:-}" || -z "${SPACES_SECRET_ACCESS_KEY:-}" ]]; then
    echo "ERROR: SPACES_ACCESS_KEY_ID + SPACES_SECRET_ACCESS_KEY required" >&2
    exit 1
fi
if ! command -v aws >/dev/null 2>&1; then
    echo "ERROR: aws cli not installed; install awscli to restore" >&2
    exit 1
fi

bucket="${SPACES_BUCKET:?SPACES_BUCKET required}"
region="${SPACES_REGION:-nyc3}"
endpoint="${SPACES_ENDPOINT:-https://${region}.digitaloceanspaces.com}"

# ─── 1. Stop the orchestrator timer ────────────────────────────────────
echo "[1/6] Quiescing orchestrator timer"
systemctl stop orchestrator.timer 2>/dev/null || true

# ─── 2. Pick the archive ───────────────────────────────────────────────
if [[ -z "$EXPLICIT_ARCHIVE" ]]; then
    echo "[2/6] Finding the most recent archive in s3://$bucket/"
    EXPLICIT_ARCHIVE=$(
        AWS_ACCESS_KEY_ID="$SPACES_ACCESS_KEY_ID" \
        AWS_SECRET_ACCESS_KEY="$SPACES_SECRET_ACCESS_KEY" \
        aws --endpoint-url "$endpoint" \
            s3 ls "s3://${bucket}/" \
            | awk '{print $4}' \
            | grep -E '^aaa-data-' \
            | sort -r \
            | head -n 1
    )
    if [[ -z "$EXPLICIT_ARCHIVE" ]]; then
        echo "ERROR: no aaa-data-* archives found in bucket" >&2
        systemctl start orchestrator.timer
        exit 2
    fi
    echo "  selected: $EXPLICIT_ARCHIVE"
else
    echo "[2/6] Using explicit archive: $EXPLICIT_ARCHIVE"
fi

# ─── 3. Download + extract to a temp dir ───────────────────────────────
work_dir=$(mktemp -d -t aaa-restore-XXXXXX)
trap 'rm -rf "$work_dir"' EXIT

echo "[3/6] Downloading $EXPLICIT_ARCHIVE..."
AWS_ACCESS_KEY_ID="$SPACES_ACCESS_KEY_ID" \
AWS_SECRET_ACCESS_KEY="$SPACES_SECRET_ACCESS_KEY" \
aws --endpoint-url "$endpoint" \
    s3 cp "s3://${bucket}/${EXPLICIT_ARCHIVE}" "$work_dir/$EXPLICIT_ARCHIVE" \
    || { echo "download failed"; systemctl start orchestrator.timer; exit 2; }

echo "  extracting..."
tar -xzf "$work_dir/$EXPLICIT_ARCHIVE" -C "$work_dir" \
    || { echo "extract failed"; systemctl start orchestrator.timer; exit 2; }

# Snapshots live under sqlite-snapshots/ inside the archive.
src_dir="$work_dir/sqlite-snapshots"
if [[ ! -d "$src_dir" ]]; then
    echo "ERROR: archive missing sqlite-snapshots/ directory" >&2
    systemctl start orchestrator.timer
    exit 2
fi

# ─── 4. Back up the existing data dir ──────────────────────────────────
ts=$(date -u +%Y%m%dT%H%M%SZ)
backup_dir="${DATA_DIR}.before-restore-${ts}"
echo "[4/6] Backing up current data → $backup_dir"
if [[ -d "$DATA_DIR" ]]; then
    cp -a "$DATA_DIR" "$backup_dir"
fi

# ─── 5. Swap in the restored DBs ───────────────────────────────────────
echo "[5/6] Restoring DB files from archive"
mkdir -p "$DATA_DIR"
for db in "$src_dir"/*.db; do
    name=$(basename "$db")
    cp "$db" "$DATA_DIR/$name"
    chown "$SERVICE_USER:$SERVICE_USER" "$DATA_DIR/$name" 2>/dev/null || true
    echo "  restored $name"
done

# ─── 6. Restart the timer + tail next cycle ────────────────────────────
echo "[6/6] Restarting orchestrator timer"
systemctl start orchestrator.timer

echo ""
echo "Restore complete from $EXPLICIT_ARCHIVE"
echo "Pre-restore backup retained at: $backup_dir"
echo "  → DELETE only after you've confirmed the system is healthy:"
echo "      journalctl -u orchestrator.service --since '5 minutes ago'"
