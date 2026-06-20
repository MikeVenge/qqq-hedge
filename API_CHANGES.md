# QQQ Hedge API — What Changed (caller migration guide)

Audience: anyone calling the QQQ hedge service (`qqq-hedge-production.up.railway.app`)
over **MCP** (`/mcp`) or the **REST API** (`/api/hedge`). For the full current
reference see [`README.md`](README.md); the model logic is in
[`HEDGING_LOGIC.md`](HEDGING_LOGIC.md).

> TL;DR — three things to know:
> 1. **The signal flipped from a SHORT hedge ratio to a LONG deployment + cash.** Output keys and sign changed (BREAKING).
> 2. **The MCP tool signature changed** from `(from_date, to_date)` to `(date, vt, book_id)`.
> 3. **New async REST API** (`POST /api/hedge` → poll `GET /api/hedge/{job_id}`), and a **`book_id`** option that sizes off a Mango portfolio's volatility.

---

## 1. BREAKING — output semantics flipped (short ratio → long deployment)

**Before:** the signal returned a **short hedge ratio** (negative) — "how much QQQ to short" as a tail-risk overlay.

```json
// OLD output
{ "date": "2026-06-01", "position": -0.05, "position_pct": "5.0% short",
  "close": 742.74, "sma200": 617.66, "drawdown": 0.0 }
```

**Now:** it returns a **gross long deployment** in `[0, 1.5]` plus a cash sleeve —
"how much of the book to be invested," with the rest in cash (negative cash =
financed leverage). It is **long-only + cash; never short.**

```json
// NEW output
{ "as_of_date": "2026-06-01", "requested_date": "2026-06-01",
  "vt": 15.0, "target_vol": 0.15,
  "exposure": 0.932, "exposure_pct": "93.2% invested",
  "cash": 0.068, "cash_pct": "6.8% cash",
  "gate": 1.0, "regime": "risk-on",
  "w_vol": 0.932, "leverage_capped": false,
  "rv20": 0.161, "rv20_pct": "16.1%",
  "close": 742.74, "sma100": 633.47, "sma200": 617.66,
  "vol_source": "qqq" }
```

### Field mapping (old → new)
| Old | New | Note |
|---|---|---|
| `position` (negative short) | `exposure` (0–1.5 long) | **sign flipped + meaning changed**; `cash = 1 − exposure` |
| `position_pct` ("X% short") | `exposure_pct` ("X% invested") + `cash_pct` | |
| `date` | `as_of_date` (+ `requested_date`) | |
| `drawdown`, `sma200` (only) | `gate`, `regime`, `w_vol`, `rv20`, `sma100`, `sma200` | new two-layer model |
| — | `vol_source`, `leverage_capped`, `target_vol`, `vt` | new |

If you previously consumed `position` as a short notional, you must now invert
the framing: deploy `exposure` of the book long and hold `cash` in cash.

---

## 2. BREAKING — MCP tool signature changed

**`qqq_hedge_signal`**
- **Before:** `qqq_hedge_signal(from_date=None, to_date=None)` — single-date *or* a
  date-range mode that returned `{from_date, to_date, n_trading_days, daily:[...]}`.
- **Now:** `qqq_hedge_signal(date=None, vt=15, book_id=None)` — single as-of date
  only. **The range/`daily[]` mode was removed.** To get a series, call per date.
  - `date` — ISO `YYYY-MM-DD`; omit for latest; non-trading day rolls back.
  - `vt` — vol-target level (any positive number; `15` → 15% target, `23` → 23%).
  - `book_id` — optional (see §4).

**`qqq_hedge_backtest`**
- **Before:** `(start, end, base_hedge, dd_tier_1, dd_tier_2, dd_tier_3)`.
- **Now:** `(start, end, vt=15, leverage_cap=1.5, fed_funds_rate=0.0)`. Returns
  vol-target stats (`mean_w_vol`, `pct_at_leverage_cap`, Calmar, …) and
  `recent_exposures` (was `recent_positions`). `book_id` is **not** supported here.

---

## 3. NEW — async REST API (no MCP client needed)

Plain HTTP, submit-and-poll:

```
POST /api/hedge        body: { "date"?: "YYYY-MM-DD", "vt"?: 15, "book_id"?: int }
  → 202 { job_id, status:"pending", poll_url }

GET  /api/hedge/{job_id}
  → 202 { status:"pending"|"running" }              # keep polling
  → 200 { status:"done",  result:{ …same object as §1… } }
  → 200 { status:"error", error:"…" }
  → 404 { error:"unknown job_id" }
```

`vt`/`date`/`book_id` may also be query params. Invalid `vt` or `book_id` → `400`.
The `result` object is identical to the MCP tool's return value.

---

## 4. NEW — `book_id`: size off a Mango portfolio's volatility

Pass a Mango trading-book ID and the inverse-vol scalar uses the **book's 30-day
realized portfolio volatility** instead of QQQ's. The **SMA regime gate still
uses QQQ** — only the magnitude (vol) layer changes.

```bash
curl -X POST .../api/hedge -d '{"vt":30,"book_id":132}'   # → 202, then poll
```

Extra fields appear in `result` when `book_id` is used:
```json
"vol_source": "portfolio",        // "qqq" when book_id omitted
"portfolio_vol": 0.4977, "portfolio_vol_pct": "49.8%",
"book_id": 132, "book_name": "NAES", "n_constituents": 20
```
When sourced from a book, `rv20` mirrors `portfolio_vol`, so the invariant
`w_vol == min(target_vol / rv20, 1.5)` still holds.

**Caller notes:**
- **Slower** — the server fetches each constituent's prices (N+1 Alpha Vantage
  calls; ~50s for ~20 names). Use the **async REST** path and poll; don't expect
  an instant MCP response for large books. Books are capped at 60 constituents.
- Long/short books use **signed** weights (shorts subtract) → true net-portfolio
  vol. Options are dropped and weights renormalized.
- Uses **current** holdings applied to the trailing 30 days (current-portfolio
  realized vol). Historical/backtest with a book is not supported yet.
- Errors (unknown/empty book, prices unavailable) come back as
  `status:"error"` with a message — same shape as any other job error.

### Worked example: `{"vt":15}` → `{"vt":30,"book_id":132}`

Same endpoint, two requests. Left = QQQ vol (before); right = book 132 ("NAES")
portfolio vol at VT30 (now). Real values, as-of 2026-06-16.

```
# BEFORE                                  # NOW
POST /api/hedge {"vt":15}                 POST /api/hedge {"vt":30,"book_id":132}
```
| field | `{"vt":15}` | `{"vt":30,"book_id":132}` |
|---|---|---|
| `vol_source` | `qqq` | `portfolio` |
| vol used (`rv20`) | 0.2918 (QQQ 29.2%) | 0.4977 (NAES `portfolio_vol` 49.8%) |
| `target_vol` | 0.15 | 0.30 |
| `w_vol` = min(target/vol, 1.5) | 0.514 | 0.6027 |
| `gate` (QQQ, unchanged) | 1.0 | 1.0 |
| **`exposure`** | **51.4% invested** | **60.3% invested** |
| `cash` | 48.6% cash | 39.7% cash |
| extra fields | — | `book_id`, `book_name`, `n_constituents`, `portfolio_vol` |
| latency | ~2s | ~50s (N+1 price fetches → poll) |

MCP equivalent: `qqq_hedge_signal(vt=15)` → `qqq_hedge_signal(vt=30, book_id=132)`.

---

## 5. Behavior changes (same interface, different numbers)

- **Symmetric 3-day SMA-gate confirmation:** the regime gate (1.0/0.5/0.0) only
  changes after price holds the new side of an SMA for 3 consecutive closes,
  both down and up. Reduces whipsaw; the gate you read is the *confirmed* state.
- **`rv20` is same-day, log-return vol:** the 20-day realized vol includes the
  most recent close (not lagged) and uses log returns. (A prior build lagged it
  one day, which understated vol right after a move.)

These change the values you get for a given date but not the request/response shape.

---

## Quick migration checklist
- [ ] Read `exposure`/`cash` instead of `position`; flip the sign/meaning (long, not short).
- [ ] Switch the MCP call to `qqq_hedge_signal(date, vt[, book_id])`; drop range mode (loop dates if you need a series).
- [ ] If you used `from_date`/`to_date`, replace with per-date `date` calls.
- [ ] Optionally adopt the REST API for non-MCP clients (submit → poll).
- [ ] Optionally pass `book_id` to size off a portfolio; handle the slower async path and the extra `vol_source`/`portfolio_vol` fields.
