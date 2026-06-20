# EWMA Volatility — Procedure & Integration Notes

Handoff doc for computing **EWMA (exponentially-weighted moving average) realized
volatility** for the book portfolio-vol path, and how it would slot into this
service. Companion to [`HEDGING_LOGIC.md`](HEDGING_LOGIC.md) (overall model) and
[`README.md`](README.md) (API).

> **Status: analysis / proposal — NOT in production yet.** The live signal
> computes portfolio vol as a **30-day equal-weight** rolling std
> (`lib/portfolio_vol.py::portfolio_realized_vol_asof`). EWMA is documented here
> as the recommended alternative + an implementation plan. Nothing below is wired
> into the deployed code until someone implements §6.

---

## 1. Why EWMA (the problem it fixes)

The current vol estimator is an **equal-weight trailing 30-day** std of the
portfolio's daily log returns: every day in the window gets weight `1/30`, then
drops to `0` the moment it ages out. That produces a **"cliff" / "ghosting"
artifact**: vol steps *down* sharply ~30 days after a shock, on a quiet day,
purely because the big return left the window.

Real example (book 132 "NAES", equal-weight constituents):

| as-of | equal-weight 30d vol | what changed |
|---|---|---|
| 2025-05-16 | **70%** (peak ~81%) | window still holds the April-2025 tariff-shock days |
| 2025-05-23 | **36%** | the −13% / −9.2% / +15.7% / −8.5% April days exited the window |
| 2025-06-06 | **30%** | fully past the shock |

Nothing was actually that volatile on May 16 (the May 19–Jun 30 returns annualize
to ~25%); the 81% was the window "remembering" April. The hedge would then **ramp
exposure up in a cliff ~30 days after a crash** for no contemporaneous reason.

**EWMA cures this** by weighting recent returns more and decaying old ones
geometrically — old shocks fade smoothly instead of dropping off a ledge. On the
same dates, EWMA(λ=0.94) reads 57% → 50% → 42% (a smooth glide, no cliff).

---

## 2. The procedure (exact)

Inputs: the portfolio's **daily log returns** `r_t = ln(NAV_t / NAV_{t-1})`,
where `NAV` is the book value index from
`lib/portfolio_vol.py::portfolio_value_series` (equal-weight, cash-excluded,
Alpha Vantage adjusted close) — i.e. the **same return series** the equal-weight
vol already uses.

```
EWMA variance recursion (RiskMetrics, zero-mean):
    σ²_t = λ · σ²_{t-1} + (1 − λ) · r_t²

annualized vol:
    vol_t = sqrt( σ²_t × 252 )
```

- `λ` ∈ (0,1) is the **decay factor** (persistence). `1 − λ` is the weight on
  today's squared return.
- The squared return from `k` days ago carries weight `(1 − λ)·λ^k` — geometric
  decay, never an abrupt drop.
- **Seeding:** with `adjust=False` (recursive form) the series is seeded
  `σ²_0 = r_0²`. Acceptable here because we have a long warm-up before the
  query window; an alternative is to seed with the sample variance of the first
  N returns. Document whichever is chosen.
- **Annualization:** ×√252, identical to the equal-weight path.

### Reference quantities for a given λ
- **Half-life** (days for a shock's weight to halve): `ln(0.5) / ln(λ)`
- **Center of mass** (effective memory): `λ / (1 − λ)`
- **Weight on today:** `1 − λ`

| λ | weight on today | half-life | character |
|---|---|---|---|
| 0.80 | 20% | ~3.1 d | fast, twitchy, hugs spot vol |
| **0.94** | 6% | **~11.2 d** | **RiskMetrics daily default — balanced** |
| 0.97 | 3% | ~22.8 d | smooth, slow, long memory |
| 0.99 | 1% | ~68.9 d | very smooth, very laggy |

A 30-day equal-weight window has center of mass ~15 d, so **λ≈0.94 has a similar
*level* but no cliff** — that's why it's the natural default replacement.

---

## 3. Compute it (copy-paste)

**Python (matches the analysis):**
```python
import numpy as np
from lib.data import load_ohlcv_alphavantage
from lib.mango import resolve_book_constituents
from lib.portfolio_vol import portfolio_value_series

book  = resolve_book_constituents(132)                       # equal-weight, cash excluded
panel = load_ohlcv_alphavantage(book["symbols"], start="2019-01-01")
nav   = portfolio_value_series(panel["returns"], book["weights"])
r     = np.log(nav).diff().dropna()                          # daily log returns

lam = 0.94
ewma_var = r.pow(2).ewm(alpha=1 - lam, adjust=False).mean()  # σ²_t recursion
ewma_vol = np.sqrt(ewma_var * 252)                           # annualized
# scalar "as of latest":
print(float(ewma_vol.iloc[-1]))
```
`adjust=False` is the RiskMetrics recursion `σ²_t = λσ²_{t-1} + (1−λ)r_t²`.

**Regenerate the comparison plot** (equal-weight vs EWMA 0.94 vs 0.80):
```
python3 plot_ewma_vol.py 132        # -> ewma_vol.png (committed)
```

**JS (used in the analysis charts — verified to match pandas):**
```js
function ewVol(r, lam){                 // r = array of daily log returns
  let v = r[0]*r[0], out = [Math.sqrt(v*252)];
  for (let i=1;i<r.length;i++){ v = lam*v + (1-lam)*r[i]*r[i]; out.push(Math.sqrt(v*252)); }
  return out;                           // annualized vol series (fraction)
}
```

---

## 4. Verified results (book 132, ~Apr 2025 – Jun 2026, 301 vol days)

| date | equal-weight 30d | EWMA 0.94 | EWMA 0.80 |
|---|---|---|---|
| 2025-05-16 | 70.1% | 57.5% | (highest/earliest spike) |
| 2025-05-23 | 36.3% (cliff) | 50.0% | lower (forgot April faster) |
| 2025-06-06 | 30.2% | 42.4% | — |
| 2026-06-18 (latest) | 54.8% | 59.3% | — |

- Largest daily book returns (the shock that drove the cliff): 2025-04-09
  **+15.7%**, 2025-04-03 **−13.0%**, 2025-04-04 −9.2%, 2025-04-10 −8.5%.
- λ=0.80 reacts first and decays fastest (twitchy); λ=0.94 is smooth; equal-weight
  cliffs. Lower λ ⇒ more reactive but higher turnover; higher λ ⇒ smoother but
  laggier.

---

## 5. Behavioral trade-off for the hedge

- **Equal-weight (current):** stable between shocks, but **cliff re-risk ~30 days
  after a crash** (exposure jumps when the shock ages out, not when risk falls).
- **EWMA:** vol (and therefore `w_vol = min(VT/vol, cap)` and exposure) responds
  **smoothly and contemporaneously**; lower turnover than a low-λ but no ghost step.
- Choosing λ trades **responsiveness vs stability/turnover**. RiskMetrics 0.94 is
  the conventional daily default.

---

## 6. Integration plan (when implementing)

Mirror the existing **`weighting`** option (which already threads equal/gross
through the stack) — add a `vol_method` (+ `lambda`) the same way.

1. **`lib/portfolio_vol.py`** — add EWMA to `portfolio_realized_vol_asof`:
   - new params `vol_method: str = "rolling" | "ewma"` and `lam: float = 0.94`
     (or `half_life`, converting `λ = 0.5**(1/half_life)`).
   - `"rolling"` = current `std(log-returns[-window:], ddof=1)×√252`.
   - `"ewma"` = `sqrt( (r².ewm(alpha=1-λ, adjust=False).mean()).iloc[as_of] × 252 )`.
   - keep the same return dict + add `vol_method`, and `lambda`/`half_life`.
   - guards: still need adequate history; EWMA needs a warm-up (use the full
     loaded history, not just `window`).
2. **`mcp_server.py`** — thread params through exactly like `weighting`:
   `_compute_hedge_signal(date, vt, book_id, weighting, vol_method, lam)` →
   `portfolio_realized_vol_asof(...)`; add to `qqq_hedge_signal(...)`, parse in
   `submit_hedge_job` (validate `vol_method ∈ {rolling,ewma}`, `0<λ<1`), pass via
   `_process_hedge_job`, include in `job["request"]`.
3. **Output** — surface `vol_method` and `lambda`/`half_life` in the `result`
   (next to `weighting`, `portfolio_vol`), so callers see how vol was estimated.
4. **README / HEDGING_LOGIC** — document the new option + default.

### Decisions to make first
- **Default estimator:** keep `rolling` (no behavior change) and make `ewma`
  opt-in, OR switch the default to `ewma(0.94)` (changes live numbers — coordinate,
  it will shift exposures, e.g. book 132 latest 54.8% → 59.3%).
- **λ vs half-life** as the user-facing knob (half-life is more intuitive).
- **Seeding** (`r_0²` vs initial sample variance) and **window arg** meaning under EWMA.
- **Same window length?** EWMA replaces the 30d window entirely; `window` only
  matters for `rolling`.

---

## 7. Verification when implemented
- Unit test: `ewma` path equals the pandas one-liner (`adjust=False`) to ~1e-12;
  λ→half-life math; λ=1 edge / λ bounds rejected.
- Cross-check the JS chart formula vs pandas (already verified equal).
- Live: `qqq_hedge_signal(book_id=132, vol_method="ewma", lambda=0.94)` returns
  `vol_method:"ewma"`, a smooth `portfolio_vol`, and reconciles
  `w_vol == min(vt/100 / portfolio_vol, 1.5)`.

## Key facts to carry forward
- Return series = **daily LOG returns of the equal-weight, cash-excluded book NAV**
  (Alpha Vantage adjusted close) — same input as the current vol.
- EWMA recursion `σ²_t = λσ²_{t-1} + (1−λ)r_t²`, annualize ×√252.
- **λ=0.94 ⇒ half-life ~11 days** (RiskMetrics daily standard); lower λ = faster.
- It exists to remove the **equal-weight 30d cliff** (shock aging out of the window).
- Production today = equal-weight 30d; EWMA is **not yet wired in**.
