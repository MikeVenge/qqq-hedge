#!/usr/bin/env python3
"""
Plot EWMA vs equal-weight realized portfolio volatility for a Mango book.

Reproduces the analysis in EWMA_VOL.md: the 30-day equal-weight vol "cliffs"
when a shock ages out of the window, while EWMA vols decay smoothly. Saves a PNG.

Usage:
    python3 plot_ewma_vol.py [BOOK_ID] [--out ewma_vol.png]
Defaults: BOOK_ID=132 (NAES). Requires ALPHAVANTAGE_API_KEY (read from .env).
"""

import os
import sys
from pathlib import Path

# Load .env (ALPHAVANTAGE_API_KEY) without a hard dotenv dependency.
_env = Path(__file__).resolve().parent / ".env"
if _env.exists():
    for _line in _env.read_text().splitlines():
        _line = _line.strip()
        if _line and not _line.startswith("#") and "=" in _line:
            _k, _v = _line.split("=", 1)
            os.environ.setdefault(_k.strip(), _v.strip())

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from lib.data import load_ohlcv_alphavantage
from lib.mango import resolve_book_constituents
from lib.portfolio_vol import portfolio_value_series

WINDOW = 30
ANN = 252.0
LAMBDAS = [0.94, 0.80]   # EWMA decay factors to overlay


def ewma_vol(r, lam):
    """Annualized EWMA vol series: sigma^2_t = lam*sigma^2_{t-1} + (1-lam)*r_t^2."""
    return np.sqrt(r.pow(2).ewm(alpha=1 - lam, adjust=False).mean() * ANN)


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    book_id = int(args[0]) if args else 132
    out = "ewma_vol.png"
    if "--out" in sys.argv:
        out = sys.argv[sys.argv.index("--out") + 1]

    book = resolve_book_constituents(book_id)
    if "error" in book:
        sys.exit(f"book {book_id}: {book['error']}")
    panel = load_ohlcv_alphavantage(book["symbols"], start="2019-01-01")
    if panel is None:
        sys.exit("could not load constituent prices from Alpha Vantage")

    nav = portfolio_value_series(panel["returns"], book["weights"])
    r = np.log(nav).diff().dropna()                       # daily log returns
    eq = (r.rolling(WINDOW).std(ddof=1) * np.sqrt(ANN)).dropna()
    start = eq.index[0]                                   # align display to eq's first valid day

    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.plot(eq.index, eq.values * 100, color="#1D9E75", lw=1.6,
            label=f"{WINDOW}d equal-weight (current)")
    styles = {0.94: ("#534AB7", (0, (6, 4))), 0.80: ("#BA7517", (0, (2, 3)))}
    for lam in LAMBDAS:
        ev = ewma_vol(r, lam).loc[start:]
        hl = np.log(0.5) / np.log(lam)
        color, dash = styles.get(lam, ("#888780", "--"))
        ax.plot(ev.index, ev.values * 100, color=color, lw=1.5, linestyle=dash,
                label=f"EWMA λ={lam:.2f} (half-life {hl:.1f}d)")

    med = float(eq.median() * 100)
    ax.axhline(med, color="#888780", lw=1, ls=":", label=f"equal-weight median {med:.0f}%")

    ax.set_title(f"Realized portfolio vol: EWMA vs equal-weight — "
                 f"book {book_id} ({book.get('book_name')}), {book['n_constituents']} names "
                 f"(cash excl. {book.get('dropped_cash')})", fontsize=11)
    ax.set_ylabel("annualized 30d realized vol (%)")
    ax.set_ylim(0, max(90, eq.max() * 100 * 1.05))
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper right", fontsize=9, framealpha=0.9)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    print(f"saved {out}  | days={len(eq)} {eq.index[0].date()}..{eq.index[-1].date()}")
    print(f"latest: equal-weight {eq.iloc[-1]*100:.1f}%  "
          + "  ".join(f"EWMA{lam:.2f} {float(ewma_vol(r,lam).iloc[-1])*100:.1f}%" for lam in LAMBDAS))


if __name__ == "__main__":
    main()
