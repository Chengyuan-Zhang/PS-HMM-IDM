"""Causal held-out filtering and acceleration-prediction metrics.

At time ``t`` the filter may use the current observed kinematic covariates, but
it scores acceleration before conditioning the latent state on that
acceleration. This ordering separates descriptive smoothing from predictive
validation and prevents future or target leakage.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from scipy.optimize import brentq
from scipy.special import logsumexp
from scipy.stats import norm


@dataclass(frozen=True)
class CausalFilterResult:
    """Per-time outputs from causal acceleration filtering."""

    state_weights_before_acceleration: np.ndarray
    filtered_state_probabilities: np.ndarray
    log_predictive_density: np.ndarray
    predictive_mean: np.ndarray
    predictive_variance: np.ndarray


@dataclass(frozen=True)
class PredictiveMetrics:
    """Aggregate held-out acceleration metrics."""

    mean_log_predictive_density: float
    rmse: float
    mae: float
    interval_level: float
    interval_coverage: float
    mean_interval_width: float
    num_observations: int


def _validate_probabilities(values: np.ndarray, name: str) -> np.ndarray:
    probabilities = np.asarray(values, dtype=float)
    if probabilities.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if not np.all(np.isfinite(probabilities)) or np.any(probabilities < 0.0):
        raise ValueError(f"{name} must contain finite, non-negative values")
    total = float(probabilities.sum())
    if not np.isclose(total, 1.0, atol=1e-10):
        raise ValueError(f"{name} must sum to one, got {total}")
    return probabilities


def _validate_transition(transition: np.ndarray, num_states: int) -> np.ndarray:
    matrix = np.asarray(transition, dtype=float)
    if matrix.shape != (num_states, num_states):
        raise ValueError(
            f"transition must have shape {(num_states, num_states)}, got {matrix.shape}"
        )
    if not np.all(np.isfinite(matrix)) or np.any(matrix < 0.0):
        raise ValueError("transition must be finite and non-negative")
    if not np.allclose(matrix.sum(axis=1), 1.0, atol=1e-10):
        raise ValueError("each transition row must sum to one")
    return matrix


def causal_acceleration_filter(
    *,
    log_response_emission: np.ndarray,
    component_means: np.ndarray,
    component_standard_deviations: np.ndarray,
    initial_probabilities: np.ndarray,
    transition: np.ndarray,
    log_context_emission: Optional[np.ndarray] = None,
) -> CausalFilterResult:
    """Score a held-out sequence in the correct causal order.

    Arrays are over the joint latent state space. Current context may affect
    the weights used to predict acceleration. Current acceleration only affects
    the filtered probabilities propagated to the next time step.
    """

    log_response = np.asarray(log_response_emission, dtype=float)
    means = np.asarray(component_means, dtype=float)
    standard_deviations = np.asarray(component_standard_deviations, dtype=float)
    if log_response.ndim != 2:
        raise ValueError("log_response_emission must have shape (time, state)")
    if means.shape != log_response.shape or standard_deviations.shape != log_response.shape:
        raise ValueError("component arrays must match log_response_emission shape")
    if not np.all(np.isfinite(log_response)) or not np.all(np.isfinite(means)):
        raise ValueError("response log densities and component means must be finite")
    if not np.all(np.isfinite(standard_deviations)) or np.any(standard_deviations <= 0.0):
        raise ValueError("component standard deviations must be finite and positive")

    num_times, num_states = log_response.shape
    if num_times == 0:
        raise ValueError("at least one observation is required")
    initial = _validate_probabilities(initial_probabilities, "initial_probabilities")
    if len(initial) != num_states:
        raise ValueError("initial probabilities do not match emission state dimension")
    transition_matrix = _validate_transition(transition, num_states)

    if log_context_emission is None:
        log_context = np.zeros_like(log_response)
    else:
        log_context = np.asarray(log_context_emission, dtype=float)
        if log_context.shape != log_response.shape:
            raise ValueError("log_context_emission must match response emission shape")
        if not np.all(np.isfinite(log_context)):
            raise ValueError("context log densities must be finite")

    before_response = np.empty_like(log_response)
    filtered = np.empty_like(log_response)
    log_predictive = np.empty(num_times, dtype=float)
    predictive_mean = np.empty(num_times, dtype=float)
    predictive_variance = np.empty(num_times, dtype=float)

    prior = initial.copy()
    for time_index in range(num_times):
        log_weights = np.log(np.maximum(prior, np.finfo(float).tiny)) + log_context[time_index]
        log_weights -= logsumexp(log_weights)
        weights = np.exp(log_weights)
        before_response[time_index] = weights

        component_mean = means[time_index]
        component_variance = standard_deviations[time_index] ** 2
        mixture_mean = float(weights @ component_mean)
        mixture_second_moment = float(weights @ (component_variance + component_mean**2))
        predictive_mean[time_index] = mixture_mean
        predictive_variance[time_index] = max(mixture_second_moment - mixture_mean**2, 0.0)

        log_predictive[time_index] = logsumexp(log_weights + log_response[time_index])
        log_filtered = log_weights + log_response[time_index]
        log_filtered -= logsumexp(log_filtered)
        filtered[time_index] = np.exp(log_filtered)
        prior = filtered[time_index] @ transition_matrix

    return CausalFilterResult(
        state_weights_before_acceleration=before_response,
        filtered_state_probabilities=filtered,
        log_predictive_density=log_predictive,
        predictive_mean=predictive_mean,
        predictive_variance=predictive_variance,
    )


def normal_mixture_interval(
    weights: np.ndarray,
    means: np.ndarray,
    standard_deviations: np.ndarray,
    level: float = 0.9,
) -> Tuple[float, float]:
    """Compute an equal-tailed interval for a finite Normal mixture."""

    mixture_weights = _validate_probabilities(weights, "weights")
    component_means = np.asarray(means, dtype=float)
    component_scales = np.asarray(standard_deviations, dtype=float)
    if component_means.shape != mixture_weights.shape or component_scales.shape != mixture_weights.shape:
        raise ValueError("weights, means, and standard deviations must have identical shapes")
    if not 0.0 < level < 1.0:
        raise ValueError("level must lie strictly between zero and one")
    if np.any(component_scales <= 0.0) or not np.all(np.isfinite(component_scales)):
        raise ValueError("standard deviations must be finite and positive")

    tail = (1.0 - level) / 2.0
    lower_bound = float(np.min(component_means - 12.0 * component_scales))
    upper_bound = float(np.max(component_means + 12.0 * component_scales))

    def mixture_cdf(value: float) -> float:
        return float(mixture_weights @ norm.cdf((value - component_means) / component_scales))

    lower = brentq(lambda value: mixture_cdf(value) - tail, lower_bound, upper_bound)
    upper = brentq(lambda value: mixture_cdf(value) - (1.0 - tail), lower_bound, upper_bound)
    return float(lower), float(upper)


def summarize_causal_predictions(
    *,
    observed_acceleration: np.ndarray,
    filter_result: CausalFilterResult,
    component_means: np.ndarray,
    component_standard_deviations: np.ndarray,
    interval_level: float = 0.9,
) -> PredictiveMetrics:
    """Summarize density, point, and calibrated-interval performance."""

    observed = np.asarray(observed_acceleration, dtype=float)
    if observed.ndim != 1 or len(observed) != len(filter_result.predictive_mean):
        raise ValueError("observed_acceleration has incompatible shape")
    if not np.all(np.isfinite(observed)):
        raise ValueError("observed_acceleration must be finite")
    means = np.asarray(component_means, dtype=float)
    scales = np.asarray(component_standard_deviations, dtype=float)
    if means.shape != filter_result.state_weights_before_acceleration.shape:
        raise ValueError("component_means have incompatible shape")
    if scales.shape != means.shape:
        raise ValueError("component_standard_deviations have incompatible shape")

    bounds = np.array(
        [
            normal_mixture_interval(weights, row_means, row_scales, interval_level)
            for weights, row_means, row_scales in zip(
                filter_result.state_weights_before_acceleration, means, scales
            )
        ]
    )
    errors = filter_result.predictive_mean - observed
    coverage = (observed >= bounds[:, 0]) & (observed <= bounds[:, 1])
    return PredictiveMetrics(
        mean_log_predictive_density=float(np.mean(filter_result.log_predictive_density)),
        rmse=float(np.sqrt(np.mean(errors**2))),
        mae=float(np.mean(np.abs(errors))),
        interval_level=float(interval_level),
        interval_coverage=float(np.mean(coverage)),
        mean_interval_width=float(np.mean(bounds[:, 1] - bounds[:, 0])),
        num_observations=int(len(observed)),
    )
