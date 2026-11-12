"""Behavior and observed-context emission likelihoods."""

from __future__ import annotations

from typing import Sequence, Tuple

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ..model import CorrectedIDM, JointStateSpace
from .structures import SamplerState, Trajectory


FloatArray = NDArray[np.float64]
LOG_TWO_PI = float(np.log(2.0 * np.pi))


def idm_acceleration(parameters: ArrayLike, physical_context: ArrayLike) -> FloatArray:
    """Evaluate the corrected IDM in SI units for one parameter vector."""

    theta = np.asarray(parameters, dtype=float)
    context = np.asarray(physical_context, dtype=float)
    if theta.shape != (5,):
        raise ValueError("IDM parameters must have shape (5,).")
    if context.ndim != 2 or context.shape[1] != 3:
        raise ValueError("physical_context must have shape (time, 3).")
    model = CorrectedIDM(*theta)
    return np.asarray(
        model.acceleration(context[:, 0], context[:, 1], context[:, 2]),
        dtype=float,
    )


def normal_log_density(
    observation: ArrayLike, mean: ArrayLike, variance: float
) -> FloatArray:
    if not np.isfinite(variance) or variance <= 0.0:
        raise ValueError("Normal variance must be finite and positive.")
    residual = np.asarray(observation, dtype=float) - np.asarray(mean, dtype=float)
    with np.errstate(over="ignore", invalid="ignore"):
        result = -0.5 * (LOG_TWO_PI + np.log(variance) + residual * residual / variance)
    return np.asarray(result, dtype=float)


def stable_cholesky(
    covariance: ArrayLike,
    *,
    initial_jitter: float = 1.0e-10,
    max_attempts: int = 8,
) -> Tuple[FloatArray, FloatArray]:
    """Return a Cholesky factor and the minimally jittered symmetric matrix."""

    matrix = np.asarray(covariance, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("covariance must be square.")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("covariance must be finite.")
    symmetric = 0.5 * (matrix + matrix.T)
    jitter = 0.0
    identity = np.eye(matrix.shape[0])
    for _ in range(max_attempts):
        adjusted = symmetric + jitter * identity
        try:
            return np.linalg.cholesky(adjusted), adjusted
        except np.linalg.LinAlgError:
            jitter = initial_jitter if jitter == 0.0 else 10.0 * jitter
    raise ValueError("covariance is not numerically positive definite.")


def multivariate_normal_log_density(
    observations: ArrayLike, mean: ArrayLike, covariance: ArrayLike
) -> FloatArray:
    x = np.asarray(observations, dtype=float)
    location = np.asarray(mean, dtype=float)
    if x.ndim != 2 or location.shape != (x.shape[1],):
        raise ValueError("Observation and mean dimensions are incompatible.")
    factor, _ = stable_cholesky(covariance)
    centered = x - location
    solved = np.linalg.solve(factor, centered.T)
    quadratic = np.sum(solved * solved, axis=0)
    log_determinant = 2.0 * np.log(np.diag(factor)).sum()
    return -0.5 * (x.shape[1] * LOG_TWO_PI + log_determinant + quadratic)


def emission_log_probabilities(
    trajectory: Trajectory,
    state: SamplerState,
    state_space: JointStateSpace,
) -> FloatArray:
    """Return ``log p(a_t, x_t | b_t, s_t)`` for every joint state."""

    time_count = trajectory.acceleration.size
    behavior_log = np.empty((time_count, state_space.num_behavior_states))
    for behavior in range(state_space.num_behavior_states):
        predicted = idm_acceleration(
            state.idm_parameters[behavior], trajectory.physical_context
        )
        behavior_log[:, behavior] = normal_log_density(
            trajectory.acceleration,
            predicted,
            float(state.residual_variance[behavior]),
        )

    scenario_log = np.empty((time_count, state_space.num_scenario_states))
    for scenario in range(state_space.num_scenario_states):
        scenario_log[:, scenario] = multivariate_normal_log_density(
            trajectory.standardized_context,
            state.scenario_mean[scenario],
            state.scenario_covariance[scenario],
        )

    joint = np.empty((time_count, state_space.num_joint_states))
    for behavior in range(state_space.num_behavior_states):
        for scenario in range(state_space.num_scenario_states):
            encoded = state_space.encode(behavior, scenario)
            joint[:, encoded] = behavior_log[:, behavior] + scenario_log[:, scenario]
    if np.any(np.isnan(joint)) or np.any(np.isposinf(joint)):
        raise ValueError("Emission log probabilities contain NaN or +inf.")
    return joint


def emission_log_probability_sequences(
    trajectories: Sequence[Trajectory],
    state: SamplerState,
    state_space: JointStateSpace,
) -> Tuple[FloatArray, ...]:
    return tuple(
        emission_log_probabilities(trajectory, state, state_space)
        for trajectory in trajectories
    )

