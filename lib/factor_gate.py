"""
Factor promotion gates: rule-based selection for the agentic discovery loop.

Implements the paper's Eq. 3.10 gatekeeping logic:
  - Promote: factor passes both statistical and economic thresholds
  - Retire: factor clearly fails statistical bar
  - Hold: borderline case for further review
"""

from __future__ import annotations

from dataclasses import dataclass

from lib.factor_evaluator import FactorMetrics


@dataclass
class GateThresholds:
    """Configurable thresholds for the promotion gate.

    Defaults follow Harvey et al. (2016) multiple-testing adjustment:
    t_ic_promote >= 3.0 to control false discovery rate.
    """

    t_ic_promote: float = 3.0   # Harvey et al. (2016) threshold
    sharpe_promote: float = 1.0  # Minimum long-short Sharpe to promote
    t_ic_retire: float = 1.0    # Below this t-stat → retire
    max_turnover: float = 0.6   # Max daily 2-sided turnover (fraction of portfolio)

    @classmethod
    def from_config(cls, config: dict) -> GateThresholds:
        return cls(
            t_ic_promote=config.get("gate_t_ic_promote", 3.0),
            sharpe_promote=config.get("gate_sharpe_promote", 1.0),
            t_ic_retire=config.get("gate_t_ic_retire", 1.0),
            max_turnover=config.get("gate_max_turnover", 0.6),
        )


def gate_decision(
    metrics: FactorMetrics,
    thresholds: GateThresholds | None = None,
    existing_factors: dict | None = None,
) -> str:
    """Apply the gatekeeping logic to a set of factor metrics.

    Returns:
        "promote" - factor passes all bars, add to library
        "retire"  - factor clearly fails, discard
        "hold"    - borderline, keep for further review
    """
    if thresholds is None:
        thresholds = GateThresholds()

    t_ic = metrics.ic_t_stat
    sharpe = metrics.ls_sharpe
    turnover = getattr(metrics, "turnover", 0.0)

    # Statistical + economic gate
    if t_ic >= thresholds.t_ic_promote and sharpe >= thresholds.sharpe_promote:
        # Turnover gate
        if turnover > thresholds.max_turnover:
            return "hold"
        # Redundancy gate: check correlation with existing promoted factors
        if existing_factors:
            max_corr = getattr(metrics, "max_corr_with_existing", 0.0)
            if max_corr > 0.7:
                return "hold"
        return "promote"
    if t_ic < thresholds.t_ic_retire:
        return "retire"
    return "hold"


def gate_reason(
    metrics: FactorMetrics,
    decision: str,
    thresholds: GateThresholds | None = None,
) -> str:
    """Human-readable explanation of the gate decision."""
    if thresholds is None:
        thresholds = GateThresholds()

    turnover = getattr(metrics, "turnover", 0.0)
    max_corr = getattr(metrics, "max_corr_with_existing", 0.0)

    if decision == "promote":
        return (
            f"Promoted: t_IC={metrics.ic_t_stat:.2f} >= {thresholds.t_ic_promote} "
            f"AND Sharpe={metrics.ls_sharpe:.2f} >= {thresholds.sharpe_promote}"
        )
    if decision == "retire":
        return (
            f"Retired: t_IC={metrics.ic_t_stat:.2f} < {thresholds.t_ic_retire} "
            f"(below minimum statistical threshold)"
        )
    # Hold reasons
    reasons = []
    if metrics.ic_t_stat < thresholds.t_ic_promote:
        reasons.append(f"t_IC={metrics.ic_t_stat:.2f} < {thresholds.t_ic_promote}")
    if metrics.ls_sharpe < thresholds.sharpe_promote:
        reasons.append(f"Sharpe={metrics.ls_sharpe:.2f} < {thresholds.sharpe_promote}")
    if turnover > thresholds.max_turnover:
        reasons.append(f"turnover={turnover:.2f} > {thresholds.max_turnover}")
    if max_corr > 0.7:
        reasons.append(f"redundant (corr={max_corr:.2f} with existing factor)")
    return f"Hold: {'; '.join(reasons)}" if reasons else "Hold: borderline"
