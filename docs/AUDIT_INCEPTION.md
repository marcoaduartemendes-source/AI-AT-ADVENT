# Full Audit — Every Dollar Since Inception

**Prepared:** 2026-06-11 · **Method:** reconstruction from droplet-authored
telemetry committed to git (`docs/*.json` snapshots), kill-switch event log,
and `--status` output. No estimates — each figure cites its source.

---

## 1. The money trail

| Date | Equity | Source |
|---|---:|---|
| ~April (inception) | ~$100,000.00 | paper book funding |
| pre-May 4 | $101,820.41 | peak recorded in `kill_switch_events` |
| **May 4, 13:00 UTC** | **$0.00 (read)** | broker API failure → KILL latched |
| May 29 – 30 | $99,610.86 | `benchmark.json`, 10 identical snapshots over 20h |
| all-time peak | $102,892.69 | `--status`, June 5 |
| June 5, 19:41 UTC | $98,166.89 | `--status`, June 5 |

**Net since inception: −$1,833.11 (−1.8%).**
**SPY over the comparable window: +4.5%. Underperformance ≈ 6.3 points.**

---

## 2. The critical finding: the money was not traded away

Total realized P&L recorded across the entire visible ledger: **$0.50.**

Breakdown of the last 50 recorded trades (May 18–20, the most recent real
activity):

| Fill status | Count |
|---|---:|
| CANCELED | 38 (76%) |
| FILLED | 10 (20%) |
| PENDING | 2 (4%) |

- Rows carrying any P&L at all: **5**
- Sum of that P&L: **$0.50**
- 47 of the 50 came from `low_vol_anomaly`, `rsi_mean_reversion`, and
  `pairs_trading` — all three deleted on 2026-05-22 for validation FAIL.
- Only 3 came from a strategy that still exists (`multifactor_equity`).

**Conclusion: the −$1,833 is not trading losses. It is mark-to-market drift
on positions that were opened and then never managed**, because the engine
stopped running. The book didn't lose a fight; it was left in the ring alone.

---

## 3. Why: the engine was not running

From the June 5 cycle ring-buffer (50 cycles spanning May 17 → June 5):

| Measure | Value |
|---|---:|
| Window covered | 19.1 days |
| Cycles expected at 5-min cadence | ~5,514 |
| Cycles actually recorded | 50 |
| **Share of schedule executed** | **0.9%** |

Order flow in that window: **536 proposed → 128 submitted (24%)**, with
**110 errors**, of which the dominant cause was
`[crypto_xsmom] execution failed: Coinbase USD wallet too low: $0.00`.

Root cause of the freeze (verified in the event log): on **May 4** a broker
`get_account()` failure returned **$0.00 equity**. The risk engine read that
as a **100% drawdown**, latched KILL, and — because a latched KILL also
rejected *closing* orders — the book could neither trade nor exit. It stayed
that way until the manual reset on **June 5**.

---

## 4. The accounting failure

This is the part that matters most, and the operator's complaint is correct.

On May 29–30 the system graded itself:

| Axis | Score | Stated reason |
|---|---:|---|
| `alpha_track` | **10.0 / 10** | "live 30d Sharpe +3.39 across 50 trades" |
| `alpha_vs_spy` | 0.0 / 10 | "bot 30d −1.7% vs SPY +4.5% → excess −6.2pp" |
| **Overall** | **5.01 / 10** | — |

The "50 trades" behind that perfect 10/10 were the 50 rows audited in §2:
**38 canceled, 2 pending, 10 filled, 5 with any P&L, $0.50 total.**

The system computed a Sharpe ratio over five data points totalling fifty
cents and scored itself perfect on alpha — while the one honest axis
(`alpha_vs_spy`) correctly reported zero. Averaging them produced a
**passing 5.01/10 for a book that had earned $0.50 and was losing to SPY by
6.2 points.**

Three distinct defects, all confirmed:

1. **No minimum-sample gate.** Sharpe over 5 observations is noise, not
   signal, and was reported as fact.
2. **Canceled and pending orders were counted as "trades."** 76% of the
   denominator never executed.
3. **Gross, not net.** Coinbase/Kalshi fees were never subtracted anywhere
   in the pipeline (fixed 2026-06-11, unverified live).

---

## 5. What I could NOT determine from available data

These require the droplet and are the gaps in this audit:

| Unknown | Command to resolve |
|---|---|
| Full trade ledger (I see only the last-50 window) | `sqlite3 data/trading_performance.db "SELECT COUNT(*), MIN(timestamp), MAX(timestamp) FROM trades;"` |
| Realized P&L, all time, net | `sqlite3 data/trading_performance.db "SELECT strategy, COUNT(*), ROUND(SUM(COALESCE(pnl_usd,0)),2) FROM trades WHERE fill_status='FILLED' GROUP BY strategy ORDER BY 3;"` |
| Current open positions + cost basis | `sudo -u aaa bash -c 'set -a; . /etc/aaa.env; set +a; .venv/bin/python src/run_orchestrator.py --status'` |
| Whether GH Actions wrote a second, conflicting trade history | `sqlite3 data/trading_performance.db "SELECT COUNT(*) FROM trades WHERE timestamp BETWEEN '2026-05-05' AND '2026-06-03';"` |
| Broker's own statement (the only true ground truth) | Alpaca dashboard → Account → Activity |

---

## 6. Verdict

The system did not lose money by trading badly. **It lost money by not
trading at all**, while a broken metric told it that it was excellent.

Three failures compounded:

1. **A safety mechanism that could not be escaped** — a data glitch was
   read as a catastrophic loss, and the resulting halt also blocked exits.
2. **No detection** — nothing alarmed for a month of silence.
3. **A scoreboard that rewarded noise** — a 10/10 on fifty cents.

Any rebuild that does not fix the *measurement* first will produce the same
outcome with different code.
