"""Conjugate and Metropolis-within-Gibbs emission updates."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.stats import invwishart, wishart

from ..model import JointStateSpace
from .emissions import idm_acceleration, normal_log_density, stable_cholesky
from .structures import (
    InverseGammaPrior,
    LogIDMPrior,
    LogNormalWishartPrior,
    NormalInverseWishartPrior,
    Trajectory,
)


FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


@dataclass(frozen=True)
class NormalInverseWishartPosterior:
    mean: FloatArray
    mean_strength: float
    degrees_of_freedom: float
    scale: FloatArray
    sample_size: int


@dataclass(frozen=True)
class LogNormalWishartPosterior:
    """Posterior parameters for a precision Normal-Wishart hierarchy."""

    log_mean_location: FloatArray
    mean_strength: float
    degrees_of_freedom: float
    precision_scale: FloatArray
    sample_size: int


@dataclass(frozen=True)
class IDMMHUpdate:
    parameters: FloatArray
    accepted: NDArray[np.bool_]


def _validate_assignments(
    trajectories: Sequence[Trajectory],
    paths: Sequence[IntArray],
    state_space: JointStateSpace,
) -> Tuple[Tuple[Trajectory, ...], Tuple[IntArray, ...]]:
    trajectory_tuple = tuple(trajectories)
    path_tuple = tuple(np.asarray(path, dtype=np.int64) for path in paths)
    if not trajectory_tuple or len(trajectory_tuple) != len(path_tuple):
        raise ValueError("Each non-empty trajectory collection needs exactly one path.")
    for trajectory, path in zip(trajectory_tuple, path_tuple):
        if path.shape != trajectory.acceleration.shape:
            raise ValueError("Every latent path must match its trajectory length.")
        state_space.decode_array(path)
    return trajectory_tuple, path_tuple


def normal_inverse_wishart_posterior(
    prior: NormalInverseWishartPrior, observations: ArrayLike
) -> NormalInverseWishartPosterior:
    """Compute the stable NIW posterior, including the empty-state case."""

    x = np.asarray(observations, dtype=float)
    dimension = prior.mean.size
    if x.ndim != 2 or x.shape[1] != dimension:
        raise ValueError(f"observations must have shape (n, {dimension}).")
    if not np.all(np.isfinite(x)):
        raise ValueError("NIW observations must be finite.")
    sample_size = x.shape[0]
    if sample_size == 0:
        return NormalInverseWishartPosterior(
            prior.mean.copy(),
            prior.mean_strength,
            prior.degrees_of_freedom,
            prior.scale.copy(),
            0,
        )

    sample_mean = x.mean(axis=0)
    centered = x - sample_mean
    scatter = centered.T @ centered
    mean_strength = prior.mean_strength + sample_size
    mean = (
        prior.mean_strength * prior.mean + sample_size * sample_mean
    ) / mean_strength
    mean_difference = sample_mean - prior.mean
    scale = (
        prior.scale
        + scatter
        + (prior.mean_strength * sample_size / mean_strength)
        * np.outer(mean_difference, mean_difference)
    )
    scale = 0.5 * (scale + scale.T)
    stable_cholesky(scale)
    return NormalInverseWishartPosterior(
        mean,
        mean_strength,
        prior.degrees_of_freedom + sample_size,
        scale,
        sample_size,
    )


def sample_normal_inverse_wishart(
    posterior: NormalInverseWishartPosterior,
    rng: np.random.Generator,
) -> Tuple[FloatArray, FloatArray]:
    covariance = np.asarray(
        invwishart.rvs(
            df=posterior.degrees_of_freedom,
            scale=posterior.scale,
            random_state=rng,
        ),
        dtype=float,
    )
    dimension = posterior.mean.size
    covariance = covariance.reshape(dimension, dimension)
    _, covariance = stable_cholesky(covariance)
    mean = rng.multivariate_normal(
        posterior.mean, covariance / posterior.mean_strength
    )
    if not np.all(np.isfinite(mean)):
        raise FloatingPointError("NIW mean draw is not finite.")
    return np.asarray(mean, dtype=float), covariance


def log_normal_wishart_posterior(
    prior: LogNormalWishartPrior, log_parameters: ArrayLike
) -> LogNormalWishartPosterior:
    """Compute the manuscript precision-Wishart posterior exactly.

    ``prior.precision_scale`` is ``W0`` in
    ``Lambda ~ Wishart(nu0, W0)``.  Consequently the posterior scale is the
    inverse of ``W0^-1 + scatter + mean correction``; it is not that sum
    itself.  Hard IDM bounds are a support indicator on the joint hierarchy,
    so they are constant in this conditional update once ``eta`` is fixed.
    """

    values = np.asarray(log_parameters, dtype=float)
    dimension = prior.log_mean_location.size
    if values.ndim != 2 or values.shape[1] != dimension:
        raise ValueError(
            f"log_parameters must have shape (states, {dimension})."
        )
    if not np.all(np.isfinite(values)):
        raise ValueError("Hierarchy log parameters must be finite.")
    sample_size = int(values.shape[0])
    if sample_size == 0:
        return LogNormalWishartPosterior(
            prior.log_mean_location.copy(),
            prior.mean_strength,
            prior.degrees_of_freedom,
            prior.precision_scale.copy(),
            0,
        )

    sample_mean = values.mean(axis=0)
    centered = values - sample_mean
    scatter = centered.T @ centered
    mean_strength = prior.mean_strength + sample_size
    posterior_mean = (
        prior.mean_strength * prior.log_mean_location
        + sample_size * sample_mean
    ) / mean_strength
    mean_difference = sample_mean - prior.log_mean_location
    prior_inverse_scale = np.linalg.solve(
        prior.precision_scale, np.eye(dimension)
    )
    posterior_inverse_scale = (
        prior_inverse_scale
        + scatter
        + (prior.mean_strength * sample_size / mean_strength)
        * np.outer(mean_difference, mean_difference)
    )
    posterior_inverse_scale = 0.5 * (
        posterior_inverse_scale + posterior_inverse_scale.T
    )
    _, posterior_inverse_scale = stable_cholesky(posterior_inverse_scale)
    posterior_scale = np.linalg.solve(
        posterior_inverse_scale, np.eye(dimension)
    )
    posterior_scale = 0.5 * (posterior_scale + posterior_scale.T)
    _, posterior_scale = stable_cholesky(posterior_scale)
    return LogNormalWishartPosterior(
        np.asarray(posterior_mean, dtype=float),
        float(mean_strength),
        float(prior.degrees_of_freedom + sample_size),
        np.asarray(posterior_scale, dtype=float),
        sample_size,
    )


def sample_log_normal_wishart(
    posterior: LogNormalWishartPosterior,
    rng: np.random.Generator,
) -> Tuple[FloatArray, FloatArray]:
    """Draw ``(m, Lambda)`` using SciPy's Wishart scale convention."""

    precision = np.asarray(
        wishart.rvs(
            df=posterior.degrees_of_freedom,
            scale=posterior.precision_scale,
            random_state=rng,
        ),
        dtype=float,
    )
    dimension = posterior.log_mean_location.size
    precision = precision.reshape(dimension, dimension)
    precision = 0.5 * (precision + precision.T)
    _, precision = stable_cholesky(precision)
    covariance = np.linalg.solve(precision, np.eye(dimension))
    covariance = 0.5 * (covariance + covariance.T)
    _, covariance = stable_cholesky(covariance)
    mean = rng.multivariate_normal(
        posterior.log_mean_location,
        covariance / posterior.mean_strength,
    )
    if not np.all(np.isfinite(mean)):
        raise FloatingPointError("IDM hierarchy mean draw is not finite.")
    return np.asarray(mean, dtype=float), np.asarray(precision, dtype=float)


def sample_scenario_emissions(
    trajectories: Sequence[Trajectory],
    paths: Sequence[IntArray],
    state_space: JointStateSpace,
    prior: NormalInverseWishartPrior,
    rng: np.random.Generator,
) -> Tuple[FloatArray, FloatArray]:
    trajectories, paths = _validate_assignments(trajectories, paths, state_space)
    means = np.empty((state_space.num_scenario_states, prior.mean.size))
    covariances = np.empty(
        (state_space.num_scenario_states, prior.mean.size, prior.mean.size)
    )
    scenario_paths = tuple(state_space.decode_array(path)[1] for path in paths)
    for scenario in range(state_space.num_scenario_states):
        selected = [
            trajectory.standardized_context[scenario_path == scenario]
            for trajectory, scenario_path in zip(trajectories, scenario_paths)
        ]
        observations = np.concatenate(selected, axis=0)
        posterior = normal_inverse_wishart_posterior(prior, observations)
        means[scenario], covariances[scenario] = sample_normal_inverse_wishart(
            posterior, rng
        )
    return means, covariances


def log_idm_state_prior(
    log_parameters: ArrayLike,
    prior: LogIDMPrior,
    *,
    hierarchy_mean: Optional[ArrayLike] = None,
    hierarchy_precision: Optional[ArrayLike] = None,
) -> float:
    """Evaluate one behavior-state log prior in the active parameterization."""

    values = np.asarray(log_parameters, dtype=float)
    if values.shape != (5,) or not np.all(np.isfinite(values)):
        raise ValueError("One log-IDM state must be a finite length-five vector.")
    if (hierarchy_mean is None) != (hierarchy_precision is None):
        raise ValueError(
            "Hierarchy mean and precision must be supplied together."
        )
    if hierarchy_mean is None:
        standardized = (
            values - prior.log_mean
        ) / prior.log_standard_deviation
        return float(
            -0.5 * np.sum(standardized * standardized)
            - np.log(prior.log_standard_deviation).sum()
            - 0.5 * values.size * np.log(2.0 * np.pi)
        )

    mean = np.asarray(hierarchy_mean, dtype=float)
    precision = np.asarray(hierarchy_precision, dtype=float)
    if (
        mean.shape != (5,)
        or precision.shape != (5, 5)
        or not np.all(np.isfinite(mean))
        or not np.all(np.isfinite(precision))
        or not np.allclose(precision, precision.T, atol=1.0e-12)
    ):
        raise ValueError("Hierarchy mean/precision shape or symmetry is invalid.")
    try:
        cholesky = np.linalg.cholesky(precision)
    except np.linalg.LinAlgError as error:
        raise ValueError(
            "Hierarchy precision must be positive definite."
        ) from error
    log_determinant = float(2.0 * np.log(np.diag(cholesky)).sum())
    difference = values - mean
    quadratic = float(difference @ precision @ difference)
    if not np.isfinite(quadratic) or quadratic < 0.0:
        raise ValueError("Hierarchy precision quadratic form is invalid.")
    return float(
        0.5 * log_determinant
        - 0.5 * values.size * np.log(2.0 * np.pi)
        - 0.5 * quadratic
    )


def _log_idm_prior(log_parameters: FloatArray, prior: LogIDMPrior) -> float:
    """Backward-compatible fixed-independent prior helper."""

    return log_idm_state_prior(log_parameters, prior)


def _behavior_observations(
    trajectories: Sequence[Trajectory],
    paths: Sequence[IntArray],
    state_space: JointStateSpace,
    behavior: int,
) -> Tuple[FloatArray, FloatArray]:
    context_parts = []
    acceleration_parts = []
    for trajectory, path in zip(trajectories, paths):
        behavior_path = state_space.decode_array(path)[0]
        selected = behavior_path == behavior
        context_parts.append(trajectory.physical_context[selected])
        acceleration_parts.append(trajectory.acceleration[selected])
    return np.concatenate(context_parts, axis=0), np.concatenate(
        acceleration_parts, axis=0
    )


def sample_idm_parameters_mh(
    current_parameters: ArrayLike,
    residual_variance: ArrayLike,
    trajectories: Sequence[Trajectory],
    paths: Sequence[IntArray],
    state_space: JointStateSpace,
    prior: LogIDMPrior,
    rng: np.random.Generator,
    proposal_standard_deviation: Optional[ArrayLike] = None,
    *,
    hierarchy_mean: Optional[ArrayLike] = None,
    hierarchy_precision: Optional[ArrayLike] = None,
) -> IDMMHUpdate:
    """Make bounded coordinate-wise log-space proposals per behavior state.

    A single five-dimensional proposal has vanishing acceptance once a state
    contains thousands of observations.  Sequential coordinate updates retain
    the same truncated log-parameter posterior while yielding a separately
    auditable acceptance rate for each physical parameter.
    """

    current = np.asarray(current_parameters, dtype=float)
    variance = np.asarray(residual_variance, dtype=float)
    trajectories, paths = _validate_assignments(trajectories, paths, state_space)
    expected = (state_space.num_behavior_states, 5)
    if current.shape != expected or variance.shape != (expected[0],):
        raise ValueError("IDM parameter or residual-variance shape is invalid.")
    if (
        not np.all(np.isfinite(current))
        or np.any(current <= 0.0)
        or not np.all(np.isfinite(variance))
        or np.any(variance <= 0.0)
    ):
        raise ValueError("IDM parameters and residual variances must be finite and positive.")
    if np.any(current < prior.lower) or np.any(current > prior.upper):
        raise ValueError("Current IDM parameters violate the configured bounds.")
    if proposal_standard_deviation is None:
        proposal = np.broadcast_to(
            prior.proposal_standard_deviation, expected
        )
    else:
        proposal_input = np.asarray(proposal_standard_deviation, dtype=float)
        if proposal_input.shape == (5,):
            proposal = np.broadcast_to(proposal_input, expected)
        elif proposal_input.shape == expected:
            proposal = proposal_input
        else:
            raise ValueError(
                "proposal_standard_deviation must have shape (5,) or "
                f"{expected}."
            )
        if not np.all(np.isfinite(proposal)) or np.any(proposal <= 0.0):
            raise ValueError(
                "Proposal standard deviations must be finite and positive."
            )
    updated = current.copy()
    accepted = np.zeros((state_space.num_behavior_states, 5), dtype=bool)
    log_lower = np.log(prior.lower)
    log_upper = np.log(prior.upper)

    for behavior in range(state_space.num_behavior_states):
        context, observed = _behavior_observations(
            trajectories, paths, state_space, behavior
        )
        current_log = np.log(updated[behavior])
        if observed.size:
            current_mean = idm_acceleration(updated[behavior], context)
            current_likelihood = normal_log_density(
                observed, current_mean, float(variance[behavior])
            ).sum()
        else:
            current_likelihood = 0.0

        for parameter_index in range(5):
            proposed_log = current_log.copy()
            proposed_log[parameter_index] += rng.normal(
                0.0, proposal[behavior, parameter_index]
            )
            if (
                proposed_log[parameter_index] < log_lower[parameter_index]
                or proposed_log[parameter_index] > log_upper[parameter_index]
            ):
                continue
            proposed = np.exp(proposed_log)
            if observed.size:
                proposed_mean = idm_acceleration(proposed, context)
                proposed_likelihood = normal_log_density(
                    observed, proposed_mean, float(variance[behavior])
                ).sum()
            else:
                proposed_likelihood = 0.0
            log_acceptance = (
                proposed_likelihood
                - current_likelihood
                + log_idm_state_prior(
                    proposed_log,
                    prior,
                    hierarchy_mean=hierarchy_mean,
                    hierarchy_precision=hierarchy_precision,
                )
                - log_idm_state_prior(
                    current_log,
                    prior,
                    hierarchy_mean=hierarchy_mean,
                    hierarchy_precision=hierarchy_precision,
                )
            )
            if np.isfinite(log_acceptance) and np.log(rng.random()) < min(
                0.0, float(log_acceptance)
            ):
                current_log = proposed_log
                current_likelihood = proposed_likelihood
                updated[behavior] = proposed
                accepted[behavior, parameter_index] = True
    return IDMMHUpdate(updated, accepted)


def sample_inverse_gamma(
    shape: float, scale: float, rng: np.random.Generator
) -> float:
    draw = 1.0 / rng.gamma(shape=shape, scale=1.0 / scale)
    if not np.isfinite(draw) or draw <= 0.0:
        raise FloatingPointError("Inverse-gamma draw is invalid.")
    return float(draw)


def sample_residual_variances(
    idm_parameters: ArrayLike,
    trajectories: Sequence[Trajectory],
    paths: Sequence[IntArray],
    state_space: JointStateSpace,
    prior: InverseGammaPrior,
    rng: np.random.Generator,
) -> FloatArray:
    parameters = np.asarray(idm_parameters, dtype=float)
    trajectories, paths = _validate_assignments(trajectories, paths, state_space)
    if parameters.shape != (state_space.num_behavior_states, 5):
        raise ValueError("idm_parameters has an invalid shape.")
    if not np.all(np.isfinite(parameters)) or np.any(parameters <= 0.0):
        raise ValueError("idm_parameters must be finite and positive.")
    variances = np.empty(state_space.num_behavior_states)
    for behavior in range(state_space.num_behavior_states):
        context, observed = _behavior_observations(
            trajectories, paths, state_space, behavior
        )
        if observed.size:
            residual = observed - idm_acceleration(parameters[behavior], context)
            squared_error = float(residual @ residual)
        else:
            squared_error = 0.0
        shape = prior.shape + 0.5 * observed.size
        scale = prior.scale + 0.5 * squared_error
        variances[behavior] = sample_inverse_gamma(shape, scale, rng)
    return variances
