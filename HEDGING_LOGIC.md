# QQQ Hedge — Signal Logic Specification

A complete, self-contained description of the QQQ vol-target hedge **logic** for
independent review. (For HTTP/MCP usage see `README.md`; this document is about
the math and design only.) Implementation lives in
[`lib/qqq_hedge.py`](lib/qqq_hedge.py).

---

## 1. Purpose

Given a **date** and a **volatility-target level (`vt`)**, output the **gross long
deployment** for a long-QQQ / tech book: what fraction of capital to hold in QQQ,
with the remainder in a cash sleeve. It is **long-only + cash** — it never shorts.

```
exposure ∈ [0.0, 1.5]      # fraction of book invested in QQQ (1.0 = fully invested)
cash     = 1 − exposure    # remainder; negative cash = financed leverage
```

The signal is the product of two independent layers — a **direction gate** and a
**volatility scalar**:

```
exposure(t) = gate(t) × w_vol(t)
```

---

## 2. Inputs & data

- **Price series:** QQQ daily **adjusted close** from Alpha Vantage
  (`TIME_SERIES_DAILY_ADJUSTED`, field `"5. adjusted close"`), loaded from
  2019-01-01 (`lib/data.py::load_ohlcv_alphavantage`).
- **`vt`** (vol-target level): any positive number. `target_vol = vt / 100`
  (`vt=15` → 0.15, `vt=23` → 0.23). Default `15`.
- **`date`** (as-of): rolls back to the most recent trading day ≤ the requested
  date. Omitted ⇒ latest trading day.

**Look-ahead:** every indicator at date *t* uses only closes ≤ *t*. The as-of-*t*
signal is what you would hold *after* the close of *t*.

---

## 3. Layer 1 — SMA regime gate (direction)

Two simple moving averages of the close:

```
sma_fast = SMA(close, 100)      # min 100 obs
sma_slow = SMA(close, 200)      # min 200 obs
```

Raw gate from how many SMAs the price is above:

| Condition | gate |
|---|---|
| close > SMA100 **and** close > SMA200 | **1.0** (risk-on) |
| close > exactly one | **0.5** (half) |
| close < both | **0.0** (cash) |

### 3a. Symmetric N-day confirmation (debounce)  ⚠️ *not in the source document*

The gate does **not** flip the instant price crosses an SMA. Each SMA carries an
"effective above/below" state that flips **only after price holds the new side
for `confirm_days` (default 3) consecutive closes — in BOTH directions** (down
*and* up). Applied **per SMA**, so SMA100 can confirm-break (1.0→0.5) before
SMA200 (0.5→0.0), and each re-risks only after 3 confirmed closes back above.

Algorithm (`QQQVolTargetSignal._debounced_above`), per SMA:

```
raw_above[t] = close[t] > sma[t]                  # boolean
run_len[t]   = length of the current consecutive run of raw_above
candidate[t] = raw_above[t]  if run_len[t] >= confirm_days  else  NaN
debounced    = candidate.ffill()                  # carry last confirmed state
# cold start: seed the first valid day with its raw reading
```

Then `gate = 0.5 × (debounced_above_fast + debounced_above_slow)` mapped to
{0.0, 0.5, 1.0}. `confirm_days = 1` reproduces the instantaneous gate (the
source document's behavior).

**Empirical effect** (QQQ 2019–2026, VT15): gate state transitions drop from
**84 → 28** (≈ −66% whipsaw) vs the instantaneous gate.

---

## 4. Layer 2 — inverse-volatility scalar (magnitude)

```
rv20(t) = stdev( log_returns[t-19 .. t] ) × sqrt(252)      # annualized
log_returns[t] = ln( close[t] / close[t-1] )
# sample std (ddof=1); 20-day window; INCLUDES the most recent close (through t)

w_vol(t) = min( target_vol / rv20(t),  leverage_cap )      # leverage_cap = 1.50
```

- Uses **log returns** for the vol estimate (time-additive; standard for RV).
- `rv20` is **through date t** (the latest close is included) — it is *not*
  lagged. (See Design Decisions: a prior version wrongly lagged it one day.)
- `w_vol` is the deployment the vol layer alone would choose: 1.0 when realized
  vol equals the target; >1 (lever) when calmer; <1 (de-risk) when more volatile;
  capped at 1.5×.

Worked values (gate assumed 1.0):

| `vt` | rv20 | `w_vol` = min(vt/100 / rv20, 1.5) | exposure |
|---|---|---|---|
| 10 | 8% | min(1.25, 1.5) = 1.25 | 125% (lever) |
| 10 | 30% | 0.333 | 33.3% |
| 15 | 10% | 1.50 (boundary) | 150% (cap) |
| 15 | 8% | min(1.875, 1.5) = 1.50 | 150% (cap) |

---

## 5. Combination

```
exposure(t) = gate(t) × w_vol(t)          # 0.0 .. 1.5
cash(t)     = 1 − exposure(t)             # < 0 ⇒ financed leverage
leverage_capped = (target_vol / rv20) >= 1.50
```

Notes:
- **`gate = 0` overrides everything** → exposure 0 (100% cash) regardless of
  `w_vol`. The two layers can also both adjust at once
  (e.g. gate 0.5 × w_vol 0.56 = 0.28).
- No additional clamp: exposure is naturally bounded by `gate ≤ 1` and
  `w_vol ≤ 1.5`, so `exposure ∈ [0, 1.5]`.

---

## 6. Default configuration (`VolTargetConfig`)

| Param | Default | Meaning |
|---|---|---|
| `target_vol` | 0.15 | full-deployment vol (`vt`/100) |
| `vol_window` | 20 | realized-vol window (trading days) |
| `annualization` | 252 | √252 vol scaling |
| `leverage_cap` | 1.50 | cap on `w_vol` |
| `sma_fast` / `sma_slow` | 100 / 200 | gate SMAs |
| `gate_both/one/none` | 1.0 / 0.5 / 0.0 | gate levels |
| `confirm_days` | 3 | symmetric gate confirmation |

**Warm-up / NaN:** needs ≥200 sessions (SMA200) and ≥20 log returns (rv20).
Earlier rows yield NaN exposure; the signal returns an error
(`"Insufficient history…"`). Dates before data start return
`"No QQQ trading day on or before …"`.

---

## 7. Backtest methodology (`QQQVolTargetSignal.backtest`)

```
exposure(t)        from info through t (close, SMA, rv20 all ≤ t)
portfolio_ret(t+1) = exposure(t) · r_simple(t+1)  +  (1 − exposure(t)) · (ff/252)
buyhold_ret(t+1)   = r_simple(t+1)
```

- **Single position lag** (`exposure.shift(1)`) is the *only* look-ahead control:
  a position sized at the close of *t* earns the *t→t+1* return. (This is why the
  vol itself is **not** lagged — lagging both would double-lag.)
- **P&L uses simple returns** (`pct_change`) for correct arithmetic compounding;
  only the **vol estimate** uses log returns.
- **Cash sleeve** earns annual `fed_funds_rate` (**default 0.0** — understates the
  real cash return; a knob, not wired to live FFR).
- **No transaction costs / no turnover penalty / no leverage financing cost.**

Stats produced: annualized return & vol, Sharpe, max drawdown (on the cumulative
curve), Calmar, total return, mean exposure, mean `w_vol`, % time in cash, %
levered, % at the leverage cap.

**Reference results** (QQQ, VT15, 2020-01 → 2026-06, `fed_funds_rate=0`):
Sharpe ≈ 1.08, maxDD ≈ −12.9% (vs buy-and-hold Sharpe ≈ 0.92, maxDD ≈ −35%),
mean `w_vol` ≈ **0.81** (matches the source document's stated figure).

---

## 8. Output object (`hedge_parameters`)

```json
{
  "as_of_date": "2026-06-05", "requested_date": "2026-06-05",
  "vt": 15.0, "target_vol": 0.15,
  "exposure": 0.6318, "exposure_pct": "63.2% invested",
  "cash": 0.3682, "cash_pct": "36.8% cash",
  "gate": 1.0, "regime": "risk-on",
  "w_vol": 0.6318, "leverage_capped": false,
  "rv20": 0.2374, "rv20_pct": "23.7%",
  "close": 705.06, "sma100": 637.88, "sma200": 620.86
}
```
Invariant: `exposure == round(gate × w_vol, 4)` and `w_vol == min(target_vol / rv20, 1.5)`.

---

## 9. Design decisions & deviations from the source document

The source spec ("Two-layered hedge mechanism") defines: a same-day SMA100/200
gate (100/50/0%) and `w_vol = min(σ_target / σ_realized(t−1), 1.5)` on 20-day
realized vol, cash at Fed Funds. This implementation **deviates** as follows:

1. **Symmetric 3-day gate confirmation** — *added* (not in the doc). The doc's
   gate is instantaneous. Rationale: cut whipsaw. **Open question:** does the
   symmetric up-side confirmation delay re-entry too much after a sharp recovery?
2. **`rv20` is same-day (through t), not `σ(t−1)`.** The doc's `(t−1)` is a
   backtest no-look-ahead convention; here it's handled by the position lag, so
   the live as-of-t signal uses vol through *t* (includes the latest move).
3. **Log returns for vol** (doc unspecified; "20d realized vol"). Backtest P&L
   still uses simple returns.
4. **Cash return default 0.0** (doc says Fed Funds). Configurable via
   `fed_funds_rate`, not connected to a live rate.
5. **Adjusted close** is used for SMAs *and* vol (dividend-adjusted).

---

## 10. Points worth scrutinizing (for review)

- **Gate binarity:** `gate = 0` forces 0% exposure regardless of vol — a hard
  cliff. Is a softer floor (e.g. min 5–10%) preferable?
- **Confirmation symmetry:** is 3-day confirmation on *re-entry* (up-side) helping
  or hurting? Asymmetric (fast de-risk, slow re-risk) was considered and rejected.
- **Vol estimator:** 20-day rolling std, ddof=1, log returns, √252. No EWMA,
  no overnight/intraday split, no vol-of-vol. Single window — no blend.
- **Leverage:** cap 1.5× rarely binds at low `vt`; **financing cost of leverage
  is not modeled**, and cash earns 0 by default — both bias backtest returns.
- **Costs:** no transaction costs or turnover modeling; the gate + vol scalar
  change the position daily, so real-world drag is understated.
- **Live timing:** as-of-t uses the close of *t*; in practice you trade at the
  next open — a one-bar implementation slip not captured in the backtest.
- **Data:** adjusted vs raw close affects long-window SMAs over dividend history.
- **Calmar in the source doc (2.392 for VT10)** could not be reproduced and looks
  internally inconsistent with its own −14.18% max drawdown; treat doc
  performance figures with caution (the *mechanics* reconcile; the headline
  ratios may not).

---

## 11. Code map

| Function | Role |
|---|---|
| `VolTargetConfig` | parameters; `from_vt(vt)` builds `target_vol = vt/100` |
| `compute_exposure_indicators` | SMA100/200 + rv20 (log, same-day, annualized) |
| `QQQVolTargetSignal._debounced_above` | per-SMA symmetric N-day confirmation |
| `QQQVolTargetSignal.from_series` | vectorized gate × w_vol → exposure DataFrame |
| `QQQVolTargetSignal.compute` | single-bar (instantaneous, no confirmation) helper |
| `QQQVolTargetSignal.backtest` | position-lagged P&L vs buy-and-hold + stats |
| `hedge_parameters` | builds the as-of output object |

Tests: `tests/test_qqq_hedge.py` (pure-lib unit tests, incl. the debounce and
log-vol regression locks).
