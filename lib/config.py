"""
Central default settings for the factor discovery pipeline.
Users can override via:
  1. Optional config file: place config.json (or settings.json) in project root.
  2. Function arguments when calling run() from code or CLI.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

# -----------------------------------------------------------------------------
# Default settings
# -----------------------------------------------------------------------------

DEFAULT_FACTOR_DISCOVERY = {
    "llm_backend": "anthropic",  # "moonshot" or "anthropic"
    "llm_model": None,  # None => use backend default
    "max_rounds": 10,
    "is_end": "2023-12-31",
    "fwd_horizon": 10,  # forward return horizon in days
    "gate_t_ic_promote": 2.0,
    "gate_sharpe_promote": 1.0,
    "gate_t_ic_retire": 1.0,
    "max_expression_depth": 3,
    "aggregation_method": "linear",  # "linear" or "lgbm"
    "benchmark": "XLF",
    "out_dir": None,  # None => "output"
}

DEFAULTS = {
    "factor_discovery": DEFAULT_FACTOR_DISCOVERY,
}


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _config_path(root: Path | None = None) -> Path | None:
    root = root or _project_root()
    for name in ("config.json", "settings.json"):
        p = root / name
        if p.exists():
            return p
    return None


def load_user_config(root: Path | None = None) -> dict[str, Any]:
    """Load optional user overrides from config.json or settings.json in project root."""
    path = _config_path(root)
    if path is None:
        return {}
    try:
        with path.open(encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def get_settings(
    app: str,
    config_path: Path | None = None,
    root: Path | None = None,
    **overrides: Any,
) -> dict[str, Any]:
    """Return settings for an app: defaults + optional config file + overrides."""
    app = app.strip().lower()
    if app not in DEFAULTS:
        raise ValueError(f"Unknown app: {app}. Use one of {list(DEFAULTS)}")
    base = dict(DEFAULTS[app])
    root = root or _project_root()
    if config_path is not None:
        if config_path.exists():
            try:
                with config_path.open(encoding="utf-8") as f:
                    data = json.load(f)
                if isinstance(data, dict) and app in data and isinstance(data[app], dict):
                    for k, v in data[app].items():
                        if k in base:
                            base[k] = v
            except Exception:
                pass
    else:
        user = load_user_config(root)
        if app in user and isinstance(user[app], dict):
            for k, v in user[app].items():
                if k in base:
                    base[k] = v
    for k, v in overrides.items():
        if k in base:
            base[k] = v
    return base
