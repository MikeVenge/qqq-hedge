"""
Factor grammar: constrained vocabulary for autonomous factor construction.

Defines primitives (raw data fields), operators (transforms), and a
FactorExpression class that can be parsed from string, serialized, and
evaluated on a stock-date DataFrame panel.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Primitives – raw columns expected in the data panel
# ---------------------------------------------------------------------------

PRIMITIVES = {
    "return": "Daily stock return (pct_change of adjusted close)",
    "price": "Adjusted close price",
    "volume": "Daily trading volume (shares)",
    "market_return": "Benchmark (XLF ETF) daily return",
    "turnover": "Daily turnover ratio (volume / shares outstanding proxy)",
    "volume_ratio": "Volume relative to 20-day average (volume / rolling_mean(volume, 20))",
    "realized_vol": "20-day realized volatility of returns",
    "price_to_ma": "Price / 20-day moving average price",
    "market_vol": "20-day realized volatility of market returns",
    # Technical signal primitives
    "sma_50": "50-day simple moving average of price",
    "sma_100": "100-day simple moving average of price",
    "sma_200": "200-day simple moving average of price",
    "price_to_sma50": "Price / 50-day SMA (trend position)",
    "price_to_sma100": "Price / 100-day SMA (intermediate trend)",
    "price_to_sma200": "Price / 200-day SMA (long-term trend)",
    "sma50_to_sma200": "50-day SMA / 200-day SMA (golden/death cross signal)",
    "rsi_14": "14-day RSI (mean-reversion signal, 0-100)",
    "return_10d": "Cumulative 10-day return (momentum)",
    "return_20d": "Cumulative 20-day return (momentum)",
    "return_60d": "Cumulative 60-day return (quarterly momentum)",
    "beta_60d": "60-day rolling beta to market return",
    "corr_market_60d": "60-day rolling correlation with market return",
    "high_low_range": "Daily (high-low)/close range proxy from realized vol",
    "distance_from_52w_high": "(Price - 52-week high) / 52-week high",
}

# ---------------------------------------------------------------------------
# Operators – transforms that can be applied to primitives or sub-expressions
# ---------------------------------------------------------------------------

OPERATORS: dict[str, dict[str, Any]] = {
    # Time-series (applied per asset)
    "lag": {"args": ["x", "n"], "desc": "Lag x by n periods"},
    "rolling_mean": {"args": ["x", "n"], "desc": "Rolling mean of x over n periods"},
    "rolling_std": {"args": ["x", "n"], "desc": "Rolling std of x over n periods"},
    "rolling_max": {"args": ["x", "n"], "desc": "Rolling max of x over n periods"},
    "rolling_min": {"args": ["x", "n"], "desc": "Rolling min of x over n periods"},
    "rolling_corr": {"args": ["x", "y", "n"], "desc": "Rolling correlation between x and y over n periods"},
    "diff": {"args": ["x", "n"], "desc": "x - lag(x, n)"},
    "ewma": {"args": ["x", "span"], "desc": "Exponential weighted moving average"},
    "rolling_sum": {"args": ["x", "n"], "desc": "Rolling sum of x over n periods"},
    # Cross-sectional (applied per date)
    "rank": {"args": ["x"], "desc": "Cross-sectional rank (0-1)"},
    "zscore": {"args": ["x"], "desc": "Cross-sectional z-score"},
    # Arithmetic
    "ratio": {"args": ["x", "y"], "desc": "x / y (safe division)"},
    "log": {"args": ["x"], "desc": "Natural log of abs(x)"},
    "abs": {"args": ["x"], "desc": "Absolute value"},
    "sign": {"args": ["x"], "desc": "Sign function (-1, 0, 1)"},
    "add": {"args": ["x", "y"], "desc": "x + y"},
    "sub": {"args": ["x", "y"], "desc": "x - y"},
    "mul": {"args": ["x", "y"], "desc": "x * y"},
    "neg": {"args": ["x"], "desc": "Negate x"},
    "clip": {"args": ["x", "n", "span"], "desc": "Clip x between -n and +span (winsorize)"},
    "max_op": {"args": ["x", "y"], "desc": "Element-wise max(x, y)"},
    "min_op": {"args": ["x", "y"], "desc": "Element-wise min(x, y)"},
}

MAX_EXPRESSION_DEPTH = 4


# ---------------------------------------------------------------------------
# FactorExpression – AST node for a factor formula
# ---------------------------------------------------------------------------

@dataclass
class FactorExpression:
    """A single node in the factor expression tree.

    Leaf nodes have op=None and name set to a primitive name.
    Interior nodes have op set to an operator name and children populated.
    """

    op: str | None = None          # None for primitives
    name: str | None = None        # primitive name (leaf only)
    children: list[FactorExpression] = field(default_factory=list)
    params: dict[str, int | float] = field(default_factory=dict)  # e.g. {"n": 20}

    # -- Serialization -------------------------------------------------------

    def to_string(self) -> str:
        """Serialize to a human-readable formula string."""
        if self.op is None:
            return self.name or ""
        child_strs = [c.to_string() for c in self.children]
        param_strs = [str(v) for v in self.params.values()]
        all_args = child_strs + param_strs
        return f"{self.op}({', '.join(all_args)})"

    def depth(self) -> int:
        if not self.children:
            return 0
        return 1 + max(c.depth() for c in self.children)

    def __repr__(self) -> str:
        return self.to_string()

    # -- Evaluation ----------------------------------------------------------

    def evaluate(self, panel: pd.DataFrame) -> pd.DataFrame:
        """Evaluate this expression on a stock-date panel.

        panel: DataFrame with MultiIndex (date, symbol) or columns = symbols
               and the data columns named after primitives.
               For wide format: index=dates, columns=symbols for a single
               primitive.

        For our pipeline, *panel* is a dict-like namespace where each key
        is a primitive name mapping to a (dates x symbols) DataFrame.
        We therefore accept ``panel`` as a dict[str, DataFrame].
        """
        return _eval_node(self, panel)


# ---------------------------------------------------------------------------
# Parsing – convert string formula to FactorExpression
# ---------------------------------------------------------------------------

_TOKEN_RE = re.compile(r"(\w+\.?\w*|[(),])")


def parse_expression(formula: str) -> FactorExpression:
    """Parse a formula string like ``rank(ratio(rolling_mean(volume, 5), price))``
    into a FactorExpression tree.

    Raises ValueError on syntax errors or depth violations.
    """
    tokens = _TOKEN_RE.findall(formula.strip())
    expr, pos = _parse_tokens(tokens, 0)
    if pos < len(tokens):
        raise ValueError(f"Unexpected token at position {pos}: {tokens[pos]}")
    if expr.depth() > MAX_EXPRESSION_DEPTH:
        raise ValueError(
            f"Expression depth {expr.depth()} exceeds max {MAX_EXPRESSION_DEPTH}"
        )
    return expr


def _parse_tokens(tokens: list[str], pos: int) -> tuple[FactorExpression, int]:
    if pos >= len(tokens):
        raise ValueError("Unexpected end of expression")

    token = tokens[pos]

    # Check if next token is '(' → it's a function call
    if pos + 1 < len(tokens) and tokens[pos + 1] == "(":
        op_name = token
        if op_name not in OPERATORS:
            raise ValueError(f"Unknown operator: {op_name}")
        pos += 2  # skip name and '('
        children: list[FactorExpression] = []
        params: dict[str, int | float] = {}

        op_info = OPERATORS[op_name]
        arg_names = op_info["args"]

        arg_idx = 0
        while pos < len(tokens) and tokens[pos] != ")":
            if tokens[pos] == ",":
                pos += 1
                continue
            # Determine if this argument should be a numeric param
            if arg_idx < len(arg_names) and arg_names[arg_idx] in ("n", "span"):
                # Numeric parameter
                try:
                    val = float(tokens[pos])
                    if val == int(val):
                        val = int(val)
                    params[arg_names[arg_idx]] = val
                    pos += 1
                except ValueError:
                    # Could be a sub-expression
                    child, pos = _parse_tokens(tokens, pos)
                    children.append(child)
            else:
                child, pos = _parse_tokens(tokens, pos)
                children.append(child)
            arg_idx += 1

        if pos < len(tokens) and tokens[pos] == ")":
            pos += 1

        return FactorExpression(op=op_name, children=children, params=params), pos

    # It's a primitive or numeric literal
    if token in PRIMITIVES:
        return FactorExpression(name=token), pos + 1

    # Try numeric
    try:
        float(token)
        return FactorExpression(name=token), pos + 1
    except ValueError:
        pass

    raise ValueError(f"Unknown token: {token}")


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

def _eval_node(
    node: FactorExpression,
    data: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """Recursively evaluate a FactorExpression node.

    data: dict mapping primitive names to (dates x symbols) DataFrames.
    """
    if node.op is None:
        # Leaf: primitive reference
        name = node.name
        if name in data:
            return data[name].copy()
        # Numeric constant
        try:
            val = float(name)
            ref = next(iter(data.values()))
            return pd.DataFrame(val, index=ref.index, columns=ref.columns)
        except (ValueError, StopIteration):
            raise ValueError(f"Unknown primitive: {name}")

    # Evaluate children first
    child_vals = [_eval_node(c, data) for c in node.children]
    n = node.params.get("n")
    span = node.params.get("span")

    op = node.op

    # --- Time-series operators (per column / asset) ---
    if op == "lag":
        return child_vals[0].shift(n or 1)
    if op == "rolling_mean":
        return child_vals[0].rolling(window=n or 20, min_periods=1).mean()
    if op == "rolling_std":
        return child_vals[0].rolling(window=n or 20, min_periods=2).std()
    if op == "rolling_max":
        return child_vals[0].rolling(window=n or 20, min_periods=1).max()
    if op == "rolling_min":
        return child_vals[0].rolling(window=n or 20, min_periods=1).min()
    if op == "rolling_corr":
        return child_vals[0].rolling(window=n or 60, min_periods=20).corr(child_vals[1])
    if op == "rolling_sum":
        return child_vals[0].rolling(window=n or 10, min_periods=1).sum()
    if op == "diff":
        return child_vals[0].diff(periods=n or 1)
    if op == "ewma":
        return child_vals[0].ewm(span=span or 20, min_periods=1).mean()

    # --- Cross-sectional operators (per row / date) ---
    if op == "rank":
        return child_vals[0].rank(axis=1, pct=True)
    if op == "zscore":
        row_mean = child_vals[0].mean(axis=1)
        row_std = child_vals[0].std(axis=1).replace(0, np.nan)
        return child_vals[0].sub(row_mean, axis=0).div(row_std, axis=0)

    # --- Arithmetic ---
    if op == "ratio":
        denom = child_vals[1].replace(0, np.nan)
        return child_vals[0] / denom
    if op == "log":
        return np.log(child_vals[0].abs().replace(0, np.nan))
    if op == "abs":
        return child_vals[0].abs()
    if op == "sign":
        return np.sign(child_vals[0])
    if op == "add":
        return child_vals[0] + child_vals[1]
    if op == "sub":
        return child_vals[0] - child_vals[1]
    if op == "mul":
        return child_vals[0] * child_vals[1]
    if op == "neg":
        return -child_vals[0]
    if op == "clip":
        lo = -(n or 3.0)
        hi = span or 3.0
        return child_vals[0].clip(lo, hi)
    if op == "max_op":
        return pd.DataFrame(
            np.maximum(child_vals[0].values, child_vals[1].values),
            index=child_vals[0].index, columns=child_vals[0].columns,
        )
    if op == "min_op":
        return pd.DataFrame(
            np.minimum(child_vals[0].values, child_vals[1].values),
            index=child_vals[0].index, columns=child_vals[0].columns,
        )

    raise ValueError(f"Unimplemented operator: {op}")


# ---------------------------------------------------------------------------
# Grammar description for LLM prompt
# ---------------------------------------------------------------------------

def grammar_description() -> str:
    """Return a structured text description of the grammar for use in LLM prompts."""
    lines = ["## Factor Grammar\n"]
    lines.append("### Primitives (raw data fields):")
    for name, desc in PRIMITIVES.items():
        lines.append(f"  - `{name}`: {desc}")
    lines.append("\n### Operators:")
    for name, info in OPERATORS.items():
        args = ", ".join(info["args"])
        lines.append(f"  - `{name}({args})`: {info['desc']}")
    lines.append(f"\n### Constraints:")
    lines.append(f"  - Maximum expression depth: {MAX_EXPRESSION_DEPTH}")
    lines.append("  - All time-series ops use only past/present data (no look-ahead)")
    lines.append("  - Cross-sectional ops (rank, zscore) applied per-date")
    return "\n".join(lines)
