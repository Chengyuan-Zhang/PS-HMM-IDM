"""Fixed-metric block proposals for the five log-IDM parameters.

This module deliberately does not select or adapt a kernel inside a running
chain.  A caller can construct one metric from a chain's initial latent paths,
residual variances, contexts, and IDM parameters, then reuse that metric in a
fixed symmetric random-walk Metropolis kernel.  Keeping the metric fixed is
what makes the proposal-density terms cancel from the Metropolis ratio.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence, Tuple

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ..model import JointStateSpace
from .emissions import idm_acceleration, normal_log_density, stable_cholesky
from .structures import LogIDMPrior, Trajectory


FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]
JacobianMethod = Literal["analytic", "finite_difference"]


@dataclass(frozen=True)
class IDMBlockMetric:
    """Per-behavior fixed Gaussian metric in five-dimensional log-IDM space.

    ``precision_eigenvalues_raw`` diagnose the unmodified local precision.
    ``precision_eigenvalues`` and all covariance fields describe the stable
    metric actually used by the proposal.  The eigenvalue floor is only active
    when the raw local precision is too ill-conditioned for reliable inversion.
    """

    expansion_log_parameters: FloatArray
    precision: FloatArray
    covariance: FloatArray
    cholesky: FloatArray
    precision_eigenvalues_raw: FloatArray
    precision_eigenvalues: FloatArray
    covariance_eigenvalues: FloatArray
    raw_condition_number: FloatArray
    condition_number: FloatArray
    num_observations: IntArray
    likelihood_rank: IntArray
    used_prior_only: NDArray[np.bool_]
    regularized: NDArray[np.bool_]
    jacobian_method: JacobianMethod
    finite_difference_step: float
    relative_eigenvalue_floor: float


@dataclass(frozen=True)
class IDMBlockMHUpdate:
    """Result of one fixed-metric block proposal for every behavior state."""

    parameters: FloatArray
    accepted: NDArray[np.bool_]


def log_idm_acceleration_jacobian(
    log_parameters: ArrayLike,
    physical_context: ArrayLike,
    *,
    method: JacobianMethod = "analytic",
    finite_difference_step: float = 1.0e-5,
) -> FloatArray:
    """Differentiate corrected-IDM acceleration with respect to log parameters.

    At the ``max(0, dynamic_gap)`` kink the analytic derivative follows the
    inactive branch, matching the identifiability audit.  Finite differences
    are available as an independent numerical check away from that kink.
    """

    eta, context = _validate_log_parameters_and_context(
        log_parameters, physical_context
    )
    if method == "analytic":
        return _analytic_log_jacobian(eta, context)
    if method == "finite_difference":
        if (
            not np.isfinite(finite_difference_step)
            or finite_difference_step <= 0.0
        ):
            raise ValueError(
                "finite_difference_step must be finite and positive."
            )
        return _finite_difference_log_jacobian(
            eta, context, float(finite_difference_step)
        )
    raise ValueError(f"Unknown Jacobian method: {method!r}.")


def build_fixed_idm_block_metric(
    expansion_parameters: ArrayLike,
    residual_variance: ArrayLike,
    trajectories: Sequence[Trajectory],
    paths: Sequence[IntArray],
    state_space: JointStateSpace,
    prior: LogIDMPrior,
    *,
    jacobian_method: JacobianMethod = "analytic",
    finite_difference_step: float = 1.0e-5,
    relative_eigenvalue_floor: float = 1.0e-12,
) -> IDMBlockMetric:
    """Build one stable local posterior covariance per behavior state.

    For Jacobian ``J_b`` and observation variance ``sigma_b^2``, the raw
    precision is

    ``J_b.T @ J_b / sigma_b^2 + diag(1 / prior_sd^2)``.

    A behavior with no assigned observations uses the Gaussian log-prior
    covariance exactly.  This is a local proposal geometry, not a Laplace
    approximation used to change the posterior target.
    """

    parameters, variance, trajectory_tuple, path_tuple = _validate_kernel_inputs(
        expansion_parameters,
        residual_variance,
        trajectories,
        paths,
        state_space,
        prior,
    )
    if (
        not np.isfinite(relative_eigenvalue_floor)
        or relative_eigenvalue_floor <= 0.0
        or relative_eigenvalue_floor >= 1.0
    ):
        raise ValueError("relative_eigenvalue_floor must lie strictly in (0, 1).")

    behavior_count = state_space.num_behavior_states
    dimension = 5
    expansion_log = np.log(parameters)
    prior_precision_diagonal = 1.0 / prior.log_standard_deviation**2
    prior_precision = np.diag(prior_precision_diagonal)
    prior_covariance = np.diag(prior.log_standard_deviation**2)

    precision = np.empty((behavior_count, dimension, dimension), dtype=float)
    covariance = np.empty_like(precision)
    cholesky = np.empty_like(precision)
    eigenvalues_raw = np.empty((behavior_count, dimension), dtype=float)
    eigenvalues = np.empty_like(eigenvalues_raw)
    covariance_eigenvalues = np.empty_like(eigenvalues_raw)
    raw_condition = np.empty(behavior_count, dtype=float)
    condition = np.empty(behavior_count, dtype=float)
    observation_count = np.empty(behavior_count, dtype=np.int64)
    likelihood_rank = np.empty(behavior_count, dtype=np.int64)
    prior_only = np.empty(behavior_count, dtype=bool)
    regularized = np.empty(behavior_count, dtype=bool)

    for behavior in range(behavior_count):
        context, _ = _behavior_observations(
            trajectory_tuple, path_tuple, state_space, behavior
        )
        observation_count[behavior] = context.shape[0]
        prior_only[behavior] = context.shape[0] == 0

        if prior_only[behavior]:
            raw = np.sort(prior_precision_diagonal)
            stabilized = raw.copy()
            behavior_precision = prior_precision.copy()
            behavior_covariance = prior_covariance.copy()
            behavior_cholesky = np.diag(prior.log_standard_deviation)
            likelihood_rank[behavior] = 0
            regularized[behavior] = False
        else:
            jacobian = log_idm_acceleration_jacobian(
                expansion_log[behavior],
                context,
                method=jacobian_method,
                finite_difference_step=finite_difference_step,
            )
            whitened = jacobian / np.sqrt(variance[behavior])
            likelihood_rank[behavior] = int(np.linalg.matrix_rank(whitened))
            behavior_precision = whitened.T @ whitened + prior_precision
            behavior_precision = 0.5 * (
                behavior_precision + behavior_precision.T
            )
            raw, eigenvectors = np.linalg.eigh(behavior_precision)
            if not np.all(np.isfinite(raw)) or raw[-1] <= 0.0:
                raise FloatingPointError(
                    "The local IDM block precision has invalid eigenvalues."
                )
            floor = max(
                np.finfo(float).tiny,
                float(relative_eigenvalue_floor) * float(raw[-1]),
            )
            stabilized = np.maximum(raw, floor)
            regularized[behavior] = bool(np.any(stabilized != raw))
            behavior_precision = (
                eigenvectors * stabilized[None, :]
            ) @ eigenvectors.T
            behavior_precision = 0.5 * (
                behavior_precision + behavior_precision.T
            )
            behavior_covariance = (
                eigenvectors * (1.0 / stabilized)[None, :]
            ) @ eigenvectors.T
            behavior_covariance = 0.5 * (
                behavior_covariance + behavior_covariance.T
            )
            behavior_cholesky, behavior_covariance = stable_cholesky(
                behavior_covariance
            )

        behavior_covariance_eigenvalues = np.linalg.eigvalsh(
            behavior_covariance
        )
        if np.any(behavior_covariance_eigenvalues <= 0.0):
            raise FloatingPointError(
                "The fixed IDM block covariance is not positive definite."
            )
        precision[behavior] = behavior_precision
        covariance[behavior] = behavior_covariance
        cholesky[behavior] = behavior_cholesky
        eigenvalues_raw[behavior] = raw
        eigenvalues[behavior] = stabilized
        covariance_eigenvalues[behavior] = behavior_covariance_eigenvalues
        raw_condition[behavior] = _condition_number(raw)
        condition[behavior] = _condition_number(stabilized)

    return IDMBlockMetric(
        expansion_log_parameters=expansion_log.copy(),
        precision=precision,
        covariance=covariance,
        cholesky=cholesky,
        precision_eigenvalues_raw=eigenvalues_raw,
        precision_eigenvalues=eigenvalues,
        covariance_eigenvalues=covariance_eigenvalues,
        raw_condition_number=raw_condition,
        condition_number=condition,
        num_observations=observation_count,
        likelihood_rank=likelihood_rank,
        used_prior_only=prior_only,
        regularized=regularized,
        jacobian_method=jacobian_method,
        finite_difference_step=float(finite_difference_step),
        relative_eigenvalue_floor=float(relative_eigenvalue_floor),
    )


def sample_idm_parameters_block_mh(
    current_parameters: ArrayLike,
    residual_variance: ArrayLike,
    trajectories: Sequence[Trajectory],
    paths: Sequence[IntArray],
    state_space: JointStateSpace,
    prior: LogIDMPrior,
    metric: IDMBlockMetric,
    proposal_scale: ArrayLike,
    rng: np.random.Generator,
) -> IDMBlockMHUpdate:
    """Apply fixed symmetric Gaussian block random walks in log-IDM space.

    Each behavior receives exactly one five-dimensional proposal.  Bounds are
    hard truncations of the log-space posterior: an out-of-bounds proposal is
    rejected before drawing a Metropolis uniform.  The fixed covariance makes
    the Gaussian proposal symmetric, and the target is already defined in log
    space, so neither proposal-density nor change-of-variables terms appear.
    """

    current, variance, trajectory_tuple, path_tuple = _validate_kernel_inputs(
        current_parameters,
        residual_variance,
        trajectories,
        paths,
        state_space,
        prior,
    )
    behavior_count = state_space.num_behavior_states
    scale = np.asarray(proposal_scale, dtype=float)
    if scale.shape != (behavior_count,):
        raise ValueError(f"proposal_scale must have shape ({behavior_count},).")
    if not np.all(np.isfinite(scale)) or np.any(scale <= 0.0):
        raise ValueError("proposal_scale must be finite and strictly positive.")
    block_cholesky = np.asarray(metric.cholesky, dtype=float)
    if block_cholesky.shape != (behavior_count, 5, 5):
        raise ValueError(
            "metric.cholesky is incompatible with the behavior-state count."
        )
    if not np.all(np.isfinite(block_cholesky)):
        raise ValueError("metric.cholesky must be finite.")
    if np.any(np.diagonal(block_cholesky, axis1=1, axis2=2) <= 0.0):
        raise ValueError("metric.cholesky must have a positive diagonal.")

    updated = current.copy()
    accepted = np.zeros(behavior_count, dtype=bool)
    log_lower = np.log(prior.lower)
    log_upper = np.log(prior.upper)

    for behavior in range(behavior_count):
        context, observed = _behavior_observations(
            trajectory_tuple, path_tuple, state_space, behavior
        )
        current_log = np.log(updated[behavior])
        proposed_log = current_log + scale[behavior] * (
            block_cholesky[behavior] @ rng.standard_normal(5)
        )
        if np.any(proposed_log < log_lower) or np.any(proposed_log > log_upper):
            continue

        current_target = _log_idm_target(
            current_log, context, observed, float(variance[behavior]), prior
        )
        proposed_target = _log_idm_target(
            proposed_log, context, observed, float(variance[behavior]), prior
        )
        log_acceptance = proposed_target - current_target
        # ``+inf`` is a valid probability-one improvement and ``-inf`` a
        # probability-zero proposal.  Only NaN lacks a defined MH decision.
        if not np.isnan(log_acceptance) and np.log(rng.random()) < min(
            0.0, float(log_acceptance)
        ):
            updated[behavior] = np.exp(proposed_log)
            accepted[behavior] = True

    return IDMBlockMHUpdate(updated, accepted)


def _validate_kernel_inputs(
    parameters: ArrayLike,
    residual_variance: ArrayLike,
    trajectories: Sequence[Trajectory],
    paths: Sequence[IntArray],
    state_space: JointStateSpace,
    prior: LogIDMPrior,
) -> Tuple[FloatArray, FloatArray, Tuple[Trajectory, ...], Tuple[IntArray, ...]]:
    parameter_array = np.asarray(parameters, dtype=float)
    variance = np.asarray(residual_variance, dtype=float)
    expected = (state_space.num_behavior_states, 5)
    if parameter_array.shape != expected or variance.shape != (expected[0],):
        raise ValueError("IDM parameter or residual-variance shape is invalid.")
    if (
        not np.all(np.isfinite(parameter_array))
        or np.any(parameter_array <= 0.0)
        or not np.all(np.isfinite(variance))
        or np.any(variance <= 0.0)
    ):
        raise ValueError(
            "IDM parameters and residual variances must be finite and positive."
        )
    if np.any(parameter_array < prior.lower) or np.any(parameter_array > prior.upper):
        raise ValueError("Current IDM parameters violate the configured bounds.")

    trajectory_tuple = tuple(trajectories)
    path_tuple = tuple(np.asarray(path, dtype=np.int64) for path in paths)
    if not trajectory_tuple or len(trajectory_tuple) != len(path_tuple):
        raise ValueError(
            "Each non-empty trajectory collection needs exactly one path."
        )
    for trajectory, path in zip(trajectory_tuple, path_tuple):
        if path.shape != trajectory.acceleration.shape:
            raise ValueError("Every latent path must match its trajectory length.")
        state_space.decode_array(path)
    return parameter_array, variance, trajectory_tuple, path_tuple


def _behavior_observations(
    trajectories: Tuple[Trajectory, ...],
    paths: Tuple[IntArray, ...],
    state_space: JointStateSpace,
    behavior: int,
) -> Tuple[FloatArray, FloatArray]:
    contexts = []
    accelerations = []
    for trajectory, path in zip(trajectories, paths):
        behavior_path = state_space.decode_array(path)[0]
        selected = behavior_path == behavior
        contexts.append(trajectory.physical_context[selected])
        accelerations.append(trajectory.acceleration[selected])
    return np.concatenate(contexts, axis=0), np.concatenate(accelerations, axis=0)


def _validate_log_parameters_and_context(
    log_parameters: ArrayLike, physical_context: ArrayLike
) -> Tuple[FloatArray, FloatArray]:
    eta = np.asarray(log_parameters, dtype=float)
    context = np.asarray(physical_context, dtype=float)
    if eta.shape != (5,) or not np.all(np.isfinite(eta)):
        raise ValueError("log_parameters must be a finite vector with shape (5,).")
    if context.ndim != 2 or context.shape[1] != 3:
        raise ValueError("physical_context must have shape (observations, 3).")
    if not np.all(np.isfinite(context)):
        raise ValueError("physical_context must be finite.")
    if np.any(context[:, 0] < 0.0) or np.any(context[:, 1] <= 0.0):
        raise ValueError("Speeds must be non-negative and gaps strictly positive.")
    return eta, context


def _analytic_log_jacobian(
    log_parameters: FloatArray, context: FloatArray
) -> FloatArray:
    desired_speed, minimum_gap, time_headway, acceleration, deceleration = np.exp(
        log_parameters
    )
    speed = context[:, 0]
    gap = context[:, 1]
    relative_speed = context[:, 2]
    free_road = (speed / desired_speed) ** 4.0
    interaction = speed * relative_speed / (
        2.0 * np.sqrt(acceleration * deceleration)
    )
    dynamic = speed * time_headway + interaction
    active = dynamic > 0.0
    desired_gap = minimum_gap + np.maximum(0.0, dynamic)
    inverse_gap_squared = 1.0 / gap**2
    interaction_ratio = desired_gap**2 * inverse_gap_squared
    bracket = 1.0 - free_road - interaction_ratio

    derivative = np.empty((context.shape[0], 5), dtype=float)
    derivative[:, 0] = 4.0 * acceleration * free_road
    derivative[:, 1] = (
        -2.0 * acceleration * desired_gap * minimum_gap * inverse_gap_squared
    )
    desired_gap_time = np.where(active, speed * time_headway, 0.0)
    derivative[:, 2] = (
        -2.0 * acceleration * desired_gap * desired_gap_time * inverse_gap_squared
    )
    desired_gap_acceleration = np.where(active, -0.5 * interaction, 0.0)
    derivative[:, 3] = acceleration * bracket - (
        2.0
        * acceleration
        * desired_gap
        * desired_gap_acceleration
        * inverse_gap_squared
    )
    desired_gap_deceleration = np.where(active, -0.5 * interaction, 0.0)
    derivative[:, 4] = -(
        2.0
        * acceleration
        * desired_gap
        * desired_gap_deceleration
        * inverse_gap_squared
    )
    return derivative


def _finite_difference_log_jacobian(
    log_parameters: FloatArray, context: FloatArray, step: float
) -> FloatArray:
    jacobian = np.empty((context.shape[0], 5), dtype=float)
    for parameter_index in range(5):
        perturbation = np.zeros(5)
        perturbation[parameter_index] = step
        upper = idm_acceleration(np.exp(log_parameters + perturbation), context)
        lower = idm_acceleration(np.exp(log_parameters - perturbation), context)
        jacobian[:, parameter_index] = (upper - lower) / (2.0 * step)
    return jacobian


def _log_idm_prior(log_parameters: FloatArray, prior: LogIDMPrior) -> float:
    standardized = (
        log_parameters - prior.log_mean
    ) / prior.log_standard_deviation
    return float(
        -0.5 * np.sum(standardized * standardized)
        - np.log(prior.log_standard_deviation).sum()
        - 0.5 * log_parameters.size * np.log(2.0 * np.pi)
    )


def _log_idm_target(
    log_parameters: FloatArray,
    context: FloatArray,
    observed: FloatArray,
    variance: float,
    prior: LogIDMPrior,
) -> float:
    if observed.size:
        predicted = idm_acceleration(np.exp(log_parameters), context)
        log_likelihood = float(
            normal_log_density(observed, predicted, variance).sum()
        )
    else:
        log_likelihood = 0.0
    return log_likelihood + _log_idm_prior(log_parameters, prior)


def _condition_number(eigenvalues: FloatArray) -> float:
    if eigenvalues.size == 0 or eigenvalues[0] <= 0.0:
        return float("inf")
    return float(eigenvalues[-1] / eigenvalues[0])
