"""Causal filtering and leader-conditioned rollout evaluation tools."""

from .rollout import (
    RolloutMetrics,
    RolloutResult,
    integrate_constant_acceleration,
    leader_conditioned_rollout,
    summarize_rollout,
)
from .causal import (
    CausalFilterResult,
    PredictiveMetrics,
    causal_acceleration_filter,
    normal_mixture_interval,
    summarize_causal_predictions,
)
from .baselines import (
    AccelerationCalibrationData,
    IDMFitResult,
    RolloutCalibrationData,
    calibrate_idm_acceleration,
    calibrate_idm_gap,
)

__all__ = [
    "RolloutMetrics",
    "RolloutResult",
    "integrate_constant_acceleration",
    "leader_conditioned_rollout",
    "summarize_rollout",
    "CausalFilterResult",
    "PredictiveMetrics",
    "causal_acceleration_filter",
    "normal_mixture_interval",
    "summarize_causal_predictions",
    "AccelerationCalibrationData",
    "IDMFitResult",
    "RolloutCalibrationData",
    "calibrate_idm_acceleration",
    "calibrate_idm_gap",
]
