"""Scientific target and emissions for the legacy-faithful reconstruction."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Mapping, Sequence, Tuple

import numpy as np
from numpy.typing import ArrayLike, NDArray

from experiments.revision_20260806.model import JointStateSpace


FloatArray = NDArray[np.float64]
COVARIATE_ORDER: Tuple[str, str, str] = ("speed", "gap", "relative_speed")
IDM_PARAMETER_ORDER: Tuple[str, str, str, str, str] = (
    "desired_speed",
    "minimum_gap",
    "time_headway",
    "maximum_acceleration",
    "comfortable_deceleration",
)
LOG_TWO_PI = float(np.log(2.0 * np.pi))


def _finite_float_array(value: ArrayLike, name: str) -> FloatArray:
    array = np.asarray(value, dtype=float)
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values.")
    return array


@dataclass(frozen=True)
class ContextStandardizer:
    """Affine transform for the fixed physical order ``[v, s, delta_v]``."""

    mean: FloatArray
    scale: FloatArray

    def __post_init__(self) -> None:
        mean = _finite_float_array(self.mean, "context mean")
        scale = _finite_float_array(self.scale, "context scale")
        if mean.shape != (3,) or scale.shape != (3,):
            raise ValueError("Context mean and scale must have shape (3,).")
        if np.any(scale <= 0.0):
            raise ValueError("Every context scale must be strictly positive.")
        object.__setattr__(self, "mean", mean.copy())
        object.__setattr__(self, "scale", scale.copy())

    @classmethod
    def fit(cls, physical_contexts: Sequence[ArrayLike]) -> "ContextStandardizer":
        contexts = tuple(
            _finite_float_array(context, "physical context")
            for context in physical_contexts
        )
        if not contexts or any(
            context.ndim != 2
            or context.shape[0] == 0
            or context.shape[1] != 3
            for context in contexts
        ):
            raise ValueError(
                "Each physical context must have shape (positive_time, 3)."
            )
        stacked = np.concatenate(contexts, axis=0)
        scale = stacked.std(axis=0)
        if np.any(scale <= 1.0e-12):
            raise ValueError("A context covariate has effectively zero variance.")
        return cls(stacked.mean(axis=0), scale)

    def transform(self, physical_context: ArrayLike) -> FloatArray:
        context = _finite_float_array(physical_context, "physical context")
        if context.ndim != 2 or context.shape[1] != 3:
            raise ValueError("physical_context must have shape (time, 3).")
        return np.asarray((context - self.mean) / self.scale, dtype=float)

    def inverse_transform(self, standardized_context: ArrayLike) -> FloatArray:
        context = _finite_float_array(standardized_context, "standardized context")
        if context.ndim != 2 or context.shape[1] != 3:
            raise ValueError("standardized_context must have shape (time, 3).")
        return np.asarray(context * self.scale + self.mean, dtype=float)


@dataclass(frozen=True)
class StoryTrajectory:
    """One independent follower trajectory in SI and standardized units."""

    acceleration: FloatArray
    physical_context: FloatArray
    standardized_context: FloatArray
    trajectory_id: str
    minimum_allowed_speed_mps: float = 0.0

    def __post_init__(self) -> None:
        acceleration = _finite_float_array(self.acceleration, "acceleration")
        physical = _finite_float_array(self.physical_context, "physical context")
        standardized = _finite_float_array(
            self.standardized_context, "standardized context"
        )
        if acceleration.ndim != 1 or acceleration.size == 0:
            raise ValueError("acceleration must be a non-empty vector.")
        expected = (acceleration.size, 3)
        if physical.shape != expected or standardized.shape != expected:
            raise ValueError(f"Context arrays must have shape {expected}.")
        if not np.isfinite(self.minimum_allowed_speed_mps):
            raise ValueError("minimum_allowed_speed_mps must be finite.")
        if np.any(physical[:, 0] < self.minimum_allowed_speed_mps - 1.0e-12):
            raise ValueError(
                "Follower speed is below the explicitly permitted observation bound."
            )
        if np.any(physical[:, 1] <= 0.0):
            raise ValueError("Net gap must be strictly positive.")
        if not self.trajectory_id:
            raise ValueError("trajectory_id must be non-empty.")
        object.__setattr__(self, "acceleration", acceleration.copy())
        object.__setattr__(self, "physical_context", physical.copy())
        object.__setattr__(self, "standardized_context", standardized.copy())

    @classmethod
    def from_physical(
        cls,
        acceleration: ArrayLike,
        physical_context: ArrayLike,
        standardizer: ContextStandardizer,
        trajectory_id: str,
        *,
        minimum_allowed_speed_mps: float = 0.0,
    ) -> "StoryTrajectory":
        physical = np.asarray(physical_context, dtype=float)
        return cls(
            acceleration=np.asarray(acceleration, dtype=float),
            physical_context=physical,
            standardized_context=standardizer.transform(physical),
            trajectory_id=trajectory_id,
            minimum_allowed_speed_mps=minimum_allowed_speed_mps,
        )


def original_idm_acceleration(
    parameters: ArrayLike,
    physical_context: ArrayLike,
    *,
    acceleration_exponent: float = 4.0,
    minimum_allowed_speed_mps: float = 0.0,
) -> FloatArray:
    """Evaluate the historical untruncated IDM used by the legacy code.

    The context columns are speed, net gap, and follower-minus-leader relative
    speed.  In particular, this target intentionally uses

    ``s_star = s0 + v*T + v*delta_v/(2*sqrt(a*b))``

    without the later ``max(0, ...)`` modification.  This is a frozen model
    choice for the same-configuration reconstruction, not a claim that the
    historical variant is preferable in general.
    """

    theta = _finite_float_array(parameters, "IDM parameters")
    context = _finite_float_array(physical_context, "physical context")
    if theta.shape != (5,):
        raise ValueError("IDM parameters must have shape (5,).")
    if context.ndim != 2 or context.shape[1] != 3:
        raise ValueError("physical_context must have shape (time, 3).")
    if np.any(theta <= 0.0) or not np.isfinite(acceleration_exponent):
        raise ValueError("IDM parameters must be positive and exponent finite.")
    if acceleration_exponent <= 0.0:
        raise ValueError("acceleration_exponent must be positive.")
    speed, gap, relative_speed = context.T
    if not np.isfinite(minimum_allowed_speed_mps):
        raise ValueError("minimum_allowed_speed_mps must be finite.")
    if np.any(speed < minimum_allowed_speed_mps - 1.0e-12) or np.any(gap <= 0.0):
        raise ValueError(
            "IDM observations violate the explicit speed bound or positive-gap support."
        )
    desired_speed, minimum_gap, headway, maximum_acceleration, deceleration = theta
    desired_gap = (
        minimum_gap
        + speed * headway
        + speed
        * relative_speed
        / (2.0 * np.sqrt(maximum_acceleration * deceleration))
    )
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        acceleration = maximum_acceleration * (
            1.0
            - (speed / desired_speed) ** acceleration_exponent
            - (desired_gap / gap) ** 2
        )
    if not np.all(np.isfinite(acceleration)):
        raise FloatingPointError("Historical IDM acceleration is not finite.")
    return np.asarray(acceleration, dtype=float)


def normal_log_density(
    observation: ArrayLike, mean: ArrayLike, variance: float
) -> FloatArray:
    if not np.isfinite(variance) or variance <= 0.0:
        raise ValueError("Normal variance must be finite and positive.")
    residual = np.asarray(observation, dtype=float) - np.asarray(mean, dtype=float)
    with np.errstate(over="ignore", invalid="ignore"):
        result = -0.5 * (
            LOG_TWO_PI + np.log(variance) + residual * residual / variance
        )
    return np.asarray(result, dtype=float)


def stable_cholesky(
    covariance: ArrayLike,
    *,
    initial_jitter: float = 1.0e-10,
    max_attempts: int = 8,
) -> Tuple[FloatArray, FloatArray]:
    matrix = _finite_float_array(covariance, "covariance")
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("covariance must be square.")
    symmetric = 0.5 * (matrix + matrix.T)
    identity = np.eye(matrix.shape[0])
    jitter = 0.0
    for _ in range(max_attempts):
        adjusted = symmetric + jitter * identity
        try:
            return np.linalg.cholesky(adjusted), adjusted
        except np.linalg.LinAlgError:
            jitter = initial_jitter if jitter == 0.0 else 10.0 * jitter
    raise ValueError("covariance is not numerically positive definite.")


def multivariate_normal_log_density(
    observations: ArrayLike,
    mean: ArrayLike,
    covariance: ArrayLike,
) -> FloatArray:
    x = _finite_float_array(observations, "scenario observations")
    location = _finite_float_array(mean, "scenario mean")
    if x.ndim != 2 or location.shape != (x.shape[1],):
        raise ValueError("Scenario observations and mean are incompatible.")
    factor, _ = stable_cholesky(covariance)
    centered = x - location
    solved = np.linalg.solve(factor, centered.T)
    quadratic = np.sum(solved * solved, axis=0)
    log_determinant = 2.0 * np.log(np.diag(factor)).sum()
    return np.asarray(
        -0.5 * (x.shape[1] * LOG_TWO_PI + log_determinant + quadratic),
        dtype=float,
    )


def emission_log_probabilities(
    trajectory: StoryTrajectory,
    state_space: JointStateSpace,
    idm_parameters: ArrayLike,
    residual_variance: ArrayLike,
    scenario_mean: ArrayLike,
    scenario_covariance: ArrayLike,
) -> FloatArray:
    """Return ``log p(a_t, x_t | b_t, s_t)`` over all joint states."""

    parameters = _finite_float_array(idm_parameters, "IDM parameters")
    variance = _finite_float_array(residual_variance, "residual variance")
    means = _finite_float_array(scenario_mean, "scenario mean")
    covariances = _finite_float_array(scenario_covariance, "scenario covariance")
    if parameters.shape != (state_space.num_behavior_states, 5):
        raise ValueError("IDM parameter shape does not match behavior states.")
    if variance.shape != (state_space.num_behavior_states,) or np.any(variance <= 0.0):
        raise ValueError("Residual variance shape/support is invalid.")
    if means.shape != (state_space.num_scenario_states, 3):
        raise ValueError("Scenario mean shape does not match scenario states.")
    if covariances.shape != (state_space.num_scenario_states, 3, 3):
        raise ValueError("Scenario covariance shape does not match scenario states.")

    time_count = trajectory.acceleration.size
    behavior_log = np.empty((time_count, state_space.num_behavior_states))
    for behavior in range(state_space.num_behavior_states):
        predicted = original_idm_acceleration(
            parameters[behavior],
            trajectory.physical_context,
            minimum_allowed_speed_mps=trajectory.minimum_allowed_speed_mps,
        )
        behavior_log[:, behavior] = normal_log_density(
            trajectory.acceleration, predicted, float(variance[behavior])
        )

    scenario_log = np.empty((time_count, state_space.num_scenario_states))
    for scenario in range(state_space.num_scenario_states):
        scenario_log[:, scenario] = multivariate_normal_log_density(
            trajectory.standardized_context,
            means[scenario],
            covariances[scenario],
        )

    joint = np.empty((time_count, state_space.num_joint_states))
    for behavior in range(state_space.num_behavior_states):
        for scenario in range(state_space.num_scenario_states):
            joint_state = state_space.encode(behavior, scenario)
            joint[:, joint_state] = (
                behavior_log[:, behavior] + scenario_log[:, scenario]
            )
    if np.any(np.isnan(joint)) or np.any(np.isposinf(joint)):
        raise ValueError("Emission log probabilities contain NaN or +inf.")
    return joint


def emission_log_probability_sequences(
    trajectories: Sequence[StoryTrajectory],
    state_space: JointStateSpace,
    idm_parameters: ArrayLike,
    residual_variance: ArrayLike,
    scenario_mean: ArrayLike,
    scenario_covariance: ArrayLike,
) -> Tuple[FloatArray, ...]:
    if not trajectories:
        raise ValueError("At least one trajectory is required.")
    return tuple(
        emission_log_probabilities(
            trajectory,
            state_space,
            idm_parameters,
            residual_variance,
            scenario_mean,
            scenario_covariance,
        )
        for trajectory in trajectories
    )


def scientific_target_fingerprint(
    target_payload: Mapping[str, Any],
    *,
    cohort_sha256: str,
) -> str:
    """Hash the model target and cohort, excluding incidental run metadata."""

    if len(cohort_sha256) != 64 or any(
        character not in "0123456789abcdef" for character in cohort_sha256.lower()
    ):
        raise ValueError("cohort_sha256 must be a hexadecimal SHA-256 digest.")
    payload = {
        "cohort_sha256": cohort_sha256.lower(),
        "scientific_target": target_payload,
    }
    canonical = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()
