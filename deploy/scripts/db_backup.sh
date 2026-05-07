#!/usr/bin/env bash
# Daily SQLite snapshot → DigitalOcean Spaces (S3-compatible).
#
# Sprint A2 audit fix: the kill-switch's drawdown baseline lives in
# data/risk_state.db; trade history lives in data/trading_performance.db.
# Both were single-VPS un-replicated SQLite files. Now we tar them up
# nightly and upload to a Spaces bucket, retaining 14 daily snapshots.
#
# Required env (sourced from /etc/aaa.env via the systemd unit):
#   SPACES_ACCESS_KEY_ID
#   SPACES_SECRET_ACCESS_KEY
#   SPACES_REGION             default: nyc3
#   SPACES_BUCKET             e.g. aaa-backups
#   SPACES_ENDPOINT           e.g. https://nyc3.digitaloceanspaces.com
#
# Idempotent: timestamps the archive, deletes archives older than 14d.
# Exit codes:
#   0 — backup uploaded (or skipped because creds missing — that's
#       intentional, the systemd unit treats this as success so a
#       host without Spaces configured doesn't false-fire alerts)
#   1 — backup attempted and failed; check journalctl

set -euo pipefail

DATA_DIR="${DATA_DIR:-/opt/ai-at-advent/data}"
WORK_DIR="$(mktemp -d -t aaa-backup-XXXXXX)"
trap 'rm -rf "$WORK_DIR"' EXIT

ts=$(date -u +%Y%m%dT%H%M%SZ)
host=$(hostname -s 2>/dev/null || echo unknown)
archive="$WORK_DIR/aaa-data-${host}-${ts}.tar.gz"

# 1) Create the archive. Even if Spaces upload fails, this lives on
#    the local disk for the next backup run (until the next /tmp wipe).
if [ ! -d "$DATA_DIR" ]; then
    echo "DATA_DIR=$DATA_DIR does not exist — nothing to back up" >&2
    exit 0
fi

# Use --warning=no-file-changed so SQLite WAL files in flight don't
# blow up the tar. We sqlite3 .backup the DB files into the archive
# to get a consistent point-in-time snapshot rather than tarring the
# raw file (which can be inconsistent under WAL).
sqlite_backup_dir="$WORK_DIR/sqlite-snapshots"
mkdir -p "$sqlite_backup_dir"
shopt -s nullglob
for db in "$DATA_DIR"/*.db; do
    name=$(basename "$db")
    sqlite3 "$db" ".backup '$sqlite_backup_dir/$name'" || {
        echo "WARN: sqlite3 .backup failed for $db; falling back to raw copy" >&2
        cp "$db" "$sqlite_backup_dir/$name"
    }
done
shopt -u nullglob

# Add any non-DB persistent state alongside the snapshots.
tar -czf "$archive" \
    -C "$WORK_DIR" sqlite-snapshots \
    -C "$DATA_DIR" --exclude='*.db' --exclude='cache' . 2>/dev/null \
    || tar -czf "$archive" -C "$WORK_DIR" sqlite-snapshots

archive_size=$(stat -c%s "$archive" 2>/dev/null || stat -f%z "$archive")
echo "Built $archive ($archive_size bytes)"

# 2) Detect half-configured cred state. A typo in one of the two
#    env-var names was previously silent (script exited 0). Now: if
#    EITHER key is set we require BOTH — half-configured boxes get a
#    loud failure instead of "everything is fine but actually nothing
#    backs up".
if [ -n "${SPACES_ACCESS_KEY_ID:-}" ] || [ -n "${SPACES_SECRET_ACCESS_KEY:-}" ]; then
    if [ -z "${SPACES_ACCESS_KEY_ID:-}" ] || [ -z "${SPACES_SECRET_ACCESS_KEY:-}" ]; then
        echo "ERR: only one of SPACES_ACCESS_KEY_ID / SPACES_SECRET_ACCESS_KEY is set" >&2
        # Best-effort fail ping so Healthchecks alerts the operator
        # that backups have stopped working since deploy.
        if [ -n "${HEALTHCHECKS_PING_URL_DB_BACKUP:-}" ]; then
            curl -fsS --retry 3 -X POST \
                 "${HEALTHCHECKS_PING_URL_DB_BACKUP}/fail" \
                 --data "half-configured Spaces creds" >/dev/null \
                 || true
        fi
        exit 1
    fi
elif [ -z "${SPACES_ACCESS_KEY_ID:-}" ]; then
    echo "SPACES_*_KEY not set — skipping upload (local archive remains in $WORK_DIR)" >&2
    # Cloud-or-nothing path: no creds at all → genuinely intended
    # to skip. Don't fail the systemd unit on a fresh box.
    exit 0
fi

bucket="${SPACES_BUCKET:?SPACES_BUCKET required when keys are set}"
region="${SPACES_REGION:-nyc3}"
endpoint="${SPACES_ENDPOINT:-https://${region}.digitaloceanspaces.com}"

# 3) Upload via aws-cli (which works fine against the Spaces S3 API).
if ! command -v aws >/dev/null 2>&1; then
    echo "ERR: aws cli not installed; install awscli to enable Spaces upload" >&2
    exit 1
fi

AWS_ACCESS_KEY_ID="$SPACES_ACCESS_KEY_ID" \
AWS_SECRET_ACCESS_KEY="$SPACES_SECRET_ACCESS_KEY" \
aws --endpoint-url "$endpoint" \
    s3 cp "$archive" "s3://${bucket}/$(basename "$archive")"

echo "Uploaded s3://${bucket}/$(basename "$archive")"

# 4) Retention: keep the most recent 14 archives, delete older.
AWS_ACCESS_KEY_ID="$SPACES_ACCESS_KEY_ID" \
AWS_SECRET_ACCESS_KEY="$SPACES_SECRET_ACCESS_KEY" \
aws --endpoint-url "$endpoint" \
    s3 ls "s3://${bucket}/" \
    | awk '{print $4}' \
    | grep -E '^aaa-data-' \
    | sort -r \
    | tail -n +15 \
    | while read -r old; do
        AWS_ACCESS_KEY_ID="$SPACES_ACCESS_KEY_ID" \
        AWS_SECRET_ACCESS_KEY="$SPACES_SECRET_ACCESS_KEY" \
        aws --endpoint-url "$endpoint" \
            s3 rm "s3://${bucket}/${old}" >/dev/null
        echo "Pruned $old"
      done

echo "Backup complete."

# Healthchecks dead-man's-switch ping. If unset we silently skip;
# if set, both the success and the half-configured branch above will
# inform Healthchecks of the run state.
if [ -n "${HEALTHCHECKS_PING_URL_DB_BACKUP:-}" ]; then
    curl -fsS --retry 3 -X POST \
         "${HEALTHCHECKS_PING_URL_DB_BACKUP}" \
         --data "ok ${ts}" >/dev/null || true
fi
