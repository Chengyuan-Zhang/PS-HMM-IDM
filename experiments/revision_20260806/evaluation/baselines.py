"""Fair classical IDM baselines for prediction and rollout comparisons."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence, Tuple

import numpy as np
from scipy.optimize import least_squares

from ..model import CorrectedIDM
from .rollout import leader_conditioned_rollout


IDM_PARAMETER_NAMES = (
    "desired_speed",
    "minimum_gap",
    "time_headway",
    "max_acceleration",
    "comfortable_deceleration",
)
DEFAULT_INITIAL_PARAMETERS = np.array([33.0, 2.0, 1.6, 1.5, 1.67])
DEFAULT_LOWER_BOUNDS = np.array([1.0, 0.01, 0.05, 0.05, 0.05])
DEFAULT_UPPER_BOUNDS = np.array([60.0, 20.0, 8.0, 8.0, 10.0])


@dataclass(frozen=True)
class AccelerationCalibrationData:
    """One training sequence for acceleration-domain IDM calibration."""

    speed_mps: np.ndarray
    gap_m: np.ndarray
    relative_speed_mps: np.ndarray
    acceleration_mps2: np.ndarray
    trajectory_id: str

    def __post_init__(self) -> None:
        arrays = [
            np.asarray(self.speed_mps, dtype=float),
            np.asarray(self.gap_m, dtype=float),
            np.asarray(self.relative_speed_mps, dtype=float),
            np.asarray(self.acceleration_mps2, dtype=float),
        ]
        if not arrays[0].ndim == 1 or not all(array.shape == arrays[0].shape for array in arrays):
            raise ValueError("all acceleration-calibration arrays must be equal-length vectors")
        if len(arrays[0]) == 0 or not all(np.all(np.isfinite(array)) for array in arrays):
            raise ValueError("calibration arrays must be non-empty and finite")
        if np.any(arrays[0] < 0.0) or np.any(arrays[1] <= 0.0):
            raise ValueError("observed speeds must be non-negative and gaps strictly positive")


@dataclass(frozen=True)
class RolloutCalibrationData:
    """One leader-conditioned sequence for gap-domain IDM calibration."""

    leader_position_m: np.ndarray
    leader_speed_mps: np.ndarray
    follower_length_m: np.ndarray
    follower_position_m: np.ndarray
    follower_speed_mps: np.ndarray
    observed_gap_m: np.ndarray
    dt_s: float
    trajectory_id: str

    def __post_init__(self) -> None:
        arrays = [
            np.asarray(self.leader_position_m, dtype=float),
            np.asarray(self.leader_speed_mps, dtype=float),
            np.asarray(self.follower_length_m, dtype=float),
            np.asarray(self.follower_position_m, dtype=float),
            np.asarray(self.follower_speed_mps, dtype=float),
            np.asarray(self.observed_gap_m, dtype=float),
        ]
        if not arrays[0].ndim == 1 or not all(array.shape == arrays[0].shape for array in arrays):
            raise ValueError("all rollout-calibration arrays must be equal-length vectors")
        if len(arrays[0]) < 2 or not all(np.all(np.isfinite(array)) for array in arrays):
            raise ValueError("rollout-calibration arrays must have at least two finite points")
        if self.dt_s <= 0.0 or np.any(arrays[2] <= 0.0):
            raise ValueError("dt_s and follower lengths must be positive")


@dataclass(frozen=True)
class IDMFitResult:
    """Optimization result with the information needed for audit and scoring."""

    parameters: np.ndarray
    residual_standard_deviation_mps2: float
    calibration_domain: str
    rmse: float
    mae: float
    num_observations: int
    success: bool
    message: str
    num_function_evaluations: int
    active_mask: np.ndarray

    @property
    def model(self) -> CorrectedIDM:
        return idm_from_array(self.parameters)


def idm_from_array(parameters: np.ndarray) -> CorrectedIDM:
    """Create a corrected IDM from the canonical five-parameter vector."""

    values = np.asarray(parameters, dtype=float)
    if values.shape != (5,):
        raise ValueError("IDM parameter vector must have shape (5,)")
    return CorrectedIDM(*values.tolist())


def _validate_optimizer_inputs(
    initial_parameters: np.ndarray,
    bounds: Tuple[np.ndarray, np.ndarray],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    initial = np.asarray(initial_parameters, dtype=float)
    lower = np.asarray(bounds[0], dtype=float)
    upper = np.asarray(bounds[1], dtype=float)
    if initial.shape != lower.shape or lower.shape != upper.shape or initial.shape != (5,):
        raise ValueError("initial parameters and both bounds must have shape (5,)")
    if np.any(lower <= 0.0) or np.any(upper <= lower):
        raise ValueError("IDM bounds must be positive and strictly ordered")
    if np.any(initial < lower) or np.any(initial > upper):
        raise ValueError("initial parameters must lie within bounds")
    return initial, lower, upper


def acceleration_residuals(
    parameters: np.ndarray,
    sequences: Sequence[AccelerationCalibrationData],
) -> np.ndarray:
    """Concatenate measured-minus-IDM acceleration residuals."""

    if not sequences:
        raise ValueError("at least one acceleration sequence is required")
    model = idm_from_array(parameters)
    return np.concatenate(
        [
            np.asarray(sequence.acceleration_mps2, dtype=float)
            - np.asarray(
                model.acceleration(
                    sequence.speed_mps,
                    sequence.gap_m,
                    sequence.relative_speed_mps,
                ),
                dtype=float,
            )
            for sequence in sequences
        ]
    )


def rollout_gap_residuals(
    parameters: np.ndarray,
    sequences: Sequence[RolloutCalibrationData],
) -> np.ndarray:
    """Concatenate measured-minus-simulated gap residuals.

    Non-positive simulated gaps are clipped *only inside the explicit failure
    handling of this calibration objective*. Collisions remain visible through
    large gap residuals and are assessed separately during validation.
    """

    if not sequences:
        raise ValueError("at least one rollout sequence is required")
    model = idm_from_array(parameters)
    residuals = []
    for sequence in sequences:
        rollout = leader_conditioned_rollout(
            leader_position_m=sequence.leader_position_m,
            leader_speed_mps=sequence.leader_speed_mps,
            follower_length_m=sequence.follower_length_m,
            initial_follower_position_m=float(sequence.follower_position_m[0]),
            initial_follower_speed_mps=float(sequence.follower_speed_mps[0]),
            dt_s=float(sequence.dt_s),
            acceleration_policy=lambda _step, speed, gap, relative_speed, _rng: float(
                model.acceleration(
                    speed,
                    gap,
                    relative_speed,
                    clip_nonpositive_gap=True,
                )
            ),
            seed=0,
            stop_on_collision=False,
        )
        residuals.append(
            np.asarray(sequence.observed_gap_m, dtype=float) - rollout.gap
        )
    combined = np.concatenate(residuals)
    if not np.all(np.isfinite(combined)):
        raise FloatingPointError("rollout objective produced non-finite residuals")
    return combined


def _fit(
    residual_function,
    *,
    initial_parameters: np.ndarray,
    bounds: Tuple[np.ndarray, np.ndarray],
    calibration_domain: str,
    max_nfev: int,
) -> IDMFitResult:
    initial, lower, upper = _validate_optimizer_inputs(initial_parameters, bounds)
    result = least_squares(
        residual_function,
        x0=initial,
        bounds=(lower, upper),
        method="trf",
        max_nfev=max_nfev,
        x_scale="jac",
    )
    residuals = np.asarray(residual_function(result.x), dtype=float)
    rmse = float(np.sqrt(np.mean(residuals**2)))
    residual_scale = max(rmse, np.sqrt(np.finfo(float).eps))
    return IDMFitResult(
        parameters=np.asarray(result.x, dtype=float),
        residual_standard_deviation_mps2=residual_scale,
        calibration_domain=calibration_domain,
        rmse=rmse,
        mae=float(np.mean(np.abs(residuals))),
        num_observations=int(len(residuals)),
        success=bool(result.success),
        message=str(result.message),
        num_function_evaluations=int(result.nfev),
        active_mask=np.asarray(result.active_mask, dtype=int),
    )


def calibrate_idm_acceleration(
    sequences: Sequence[AccelerationCalibrationData],
    *,
    initial_parameters: np.ndarray = DEFAULT_INITIAL_PARAMETERS,
    bounds: Tuple[np.ndarray, np.ndarray] = (DEFAULT_LOWER_BOUNDS, DEFAULT_UPPER_BOUNDS),
    max_nfev: int = 2000,
) -> IDMFitResult:
    """Fit pooled or unpooled IDM parameters using acceleration residuals."""

    return _fit(
        lambda parameters: acceleration_residuals(parameters, sequences),
        initial_parameters=initial_parameters,
        bounds=bounds,
        calibration_domain="acceleration",
        max_nfev=max_nfev,
    )


def calibrate_idm_gap(
    sequences: Sequence[RolloutCalibrationData],
    *,
    initial_parameters: np.ndarray = DEFAULT_INITIAL_PARAMETERS,
    bounds: Tuple[np.ndarray, np.ndarray] = (DEFAULT_LOWER_BOUNDS, DEFAULT_UPPER_BOUNDS),
    max_nfev: int = 2000,
) -> IDMFitResult:
    """Fit a trajectory-simulation baseline by minimizing net-gap residuals."""

    return _fit(
        lambda parameters: rollout_gap_residuals(parameters, sequences),
        initial_parameters=initial_parameters,
        bounds=bounds,
        calibration_domain="gap",
        max_nfev=max_nfev,
    )
