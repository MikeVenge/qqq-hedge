"""
Factor aggregation: combine multiple promoted factors into a composite score.

Supports two methods from the paper:
  - Linear: equal-weighted z-score average (Blitz et al., 2023 approach)
  - LightGBM: non-linear tree-based aggregation for interaction effects
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def aggregate_linear(factor_panels: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Equal-weighted z-score average of all factor panels.

    Args:
        factor_panels: dict mapping factor name to (dates x symbols) DataFrame
            of preprocessed factor scores.

    Returns:
        (dates x symbols) DataFrame of composite scores.
    """
    if not factor_panels:
        raise ValueError("No factors to aggregate")

    # Stack all factors and compute mean per (date, symbol)
    all_scores = []
    for name, panel in factor_panels.items():
        # Z-score each factor cross-sectionally per date
        row_mean = panel.mean(axis=1)
        row_std = panel.std(axis=1).replace(0, np.nan)
        zscored = panel.sub(row_mean, axis=0).div(row_std, axis=0)
        all_scores.append(zscored)

    # Average across factors
    stacked = np.stack([df.values for df in all_scores], axis=0)
    composite = np.nanmean(stacked, axis=0)

    ref = next(iter(factor_panels.values()))
    return pd.DataFrame(composite, index=ref.index, columns=ref.columns)


def aggregate_lgbm(
    factor_panels: dict[str, pd.DataFrame],
    forward_returns: pd.DataFrame,
    train_end: str,
    n_estimators: int = 200,
    learning_rate: float = 0.05,
    max_depth: int = 4,
) -> pd.DataFrame:
    """Non-linear aggregation via LightGBM.

    Trains on data up to train_end, predicts composite scores for full panel.

    Args:
        factor_panels: dict mapping factor name to (dates x symbols) DataFrame.
        forward_returns: (dates x symbols) DataFrame of next-period returns.
        train_end: last date for training (ISO format string).
        n_estimators: number of boosting rounds.
        learning_rate: LightGBM learning rate.
        max_depth: max tree depth.

    Returns:
        (dates x symbols) DataFrame of composite scores.
    """
    import lightgbm as lgb

    factor_names = sorted(factor_panels.keys())
    ref = next(iter(factor_panels.values()))
    dates = ref.index
    symbols = ref.columns

    # Build flat training dataset: (date, symbol) rows, factor columns
    rows = []
    targets = []
    date_labels = []

    for dt in dates:
        for sym in symbols:
            feat = []
            valid = True
            for fn in factor_names:
                val = factor_panels[fn].loc[dt, sym] if sym in factor_panels[fn].columns else np.nan
                if np.isnan(val):
                    valid = False
                    break
                feat.append(val)
            if not valid:
                continue
            if dt in forward_returns.index and sym in forward_returns.columns:
                ret = forward_returns.loc[dt, sym]
                if np.isnan(ret):
                    continue
                rows.append(feat)
                targets.append(ret)
                date_labels.append(dt)

    if not rows:
        raise ValueError("No valid training samples")

    X = np.array(rows)
    y = np.array(targets)
    date_arr = pd.DatetimeIndex(date_labels)

    # Split train/test
    train_mask = date_arr <= pd.Timestamp(train_end)
    X_train, y_train = X[train_mask], y[train_mask]

    if len(X_train) < 100:
        raise ValueError(f"Insufficient training samples: {len(X_train)}")

    # Train LightGBM
    model = lgb.LGBMRegressor(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbose=-1,
    )
    model.fit(X_train, y_train)

    # Predict composite scores for entire panel
    composite = pd.DataFrame(np.nan, index=dates, columns=symbols)
    for dt in dates:
        for sym in symbols:
            feat = []
            valid = True
            for fn in factor_names:
                val = factor_panels[fn].loc[dt, sym] if sym in factor_panels[fn].columns else np.nan
                if np.isnan(val):
                    valid = False
                    break
                feat.append(val)
            if valid:
                pred = model.predict(np.array([feat]))[0]
                composite.loc[dt, sym] = pred

    return composite
