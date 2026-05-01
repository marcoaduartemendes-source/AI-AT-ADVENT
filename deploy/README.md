# Hetzner VPS Deployment

This directory holds everything you need to migrate the trading
system off GitHub Actions cron and onto a real always-on Linux box.
The current GH-Actions setup works, but it has three problems that
get worse as we scale up:

1. **State lives in GH Actions cache** — fragile, gets evicted, no
   point-in-time recovery. We've already had silent data loss when
   the cache reset.
2. **Cron jitter is 0–60 seconds** on shared GH runners. Fine for
   5-min cycles, painful for 1-min decision-making.
3. **Secrets are GH-managed** — no programmatic rotation, no
   per-environment scoping, no audit log of access.

After this deploy:
- Orchestrator runs every 5 min via `systemd` timer with sub-second
  jitter, logs to `journalctl`, restarts on crash.
- Scouts, dashboard, strategic review run on their own systemd
  timers (no cross-contention).
- Same `git pull` triggers a deploy via a tiny `webhook` listener
  (planned next iteration).
- Same code paths still work in CI for the test suite — nothing here
  is a Linux-specific fork.

## File layout

| File | Purpose |
|---|---|
| `install.sh` | One-shot installer. Run as root over SSH after you have a fresh VPS. Installs Python, clones the repo, sets up systemd, starts timers. |
| `systemd/orchestrator.service` | Runs `run_orchestrator.py --once` on each timer trigger. Pulls env from `/etc/aaa.env`. |
| `systemd/orchestrator.timer` | Fires every 5 min. |
| `systemd/scouts.service` + `scouts.timer` | Scout sweep, every 30 min. |
| `systemd/dashboard.service` + `dashboard.timer` | Dashboard rebuild, every 15 min. |
| `aaa.env.example` | Template for `/etc/aaa.env`. Hand-edit on the server with your secrets. |
| `update.sh` | Pulls latest from main + restarts. Run on push (or via the webhook listener). |

## Setup walkthrough

### 1. Provision the box (Hetzner, 5 min)

- https://console.hetzner.cloud → Add Server
- **Location:** Falkenstein DE (close to Supabase EU; cheapest)
- **Image:** Ubuntu 24.04
- **Type:** CX22 (€4.51/mo) — 2 vCPU, 4 GB RAM, 40 GB disk. Plenty.
- **SSH Keys:** add your public key (so password isn't needed)
- **Name:** `aaa-prod`
- **Create & buy.**

After ~15 sec the IPv4 shows up in the dashboard.

### 2. Run the installer

From your laptop:

```bash
ssh root@<IP>  # confirm SSH works first

# Then back on your laptop, copy the installer over:
scp deploy/install.sh root@<IP>:/root/install.sh

# SSH in and run it:
ssh root@<IP> 'bash /root/install.sh'
```

The installer:
- Updates apt, installs Python 3.12, git, curl
- Clones the repo to `/opt/ai-at-advent`
- Creates a system user `aaa` (the trading bot doesn't run as root)
- Drops systemd unit files in `/etc/systemd/system/`
- Creates a placeholder `/etc/aaa.env` and **stops** before starting
  timers — you fill in secrets first

### 3. Fill in secrets on the server

```bash
ssh root@<IP>
nano /etc/aaa.env  # paste each secret, save with Ctrl+O, exit Ctrl+X
```

Same env vars as your GH Actions workflow has now (Coinbase, Alpaca,
Kalshi, FMP, Pushover, etc). Copy from
https://github.com/marcoaduartemendes-source/AI-AT-ADVENT/settings/secrets/actions

### 4. Start the timers

```bash
systemctl daemon-reload
systemctl enable --now orchestrator.timer scouts.timer dashboard.timer
systemctl status orchestrator.timer
journalctl -u orchestrator.service -f  # tail logs
```

### 5. Disable GitHub Actions cron (optional, after 1 week of clean runs)

Edit `.github/workflows/orchestrator.yml`:
- Comment out the `schedule:` block (keep `workflow_dispatch:` for manual triggers)
- Same for `scouts.yml` and `dashboard.yml`

GH Actions stays as the **test runner** + **dashboard build** until
Phase 3.x finishes (real DB, real monitoring). The bot's actual
trading decisions move to the VPS first.

## Watch live

```bash
# Tail orchestrator logs
journalctl -u orchestrator.service -f

# What's running now / next?
systemctl list-timers --all

# Disk usage
du -sh /opt/ai-at-advent/data
```

## Rollback

If anything goes wrong:

```bash
systemctl stop orchestrator.timer scouts.timer dashboard.timer
```

GH Actions cron is still active until you disable it, so trading
keeps going there.
