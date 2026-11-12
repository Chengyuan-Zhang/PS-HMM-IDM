"""Conditional fixed-preconditioner MALA updates for corrected-IDM parameters.

The kernel in this module is deliberately standalone.  It updates each
behavior state's five log-IDM parameters conditional on fixed latent paths and
residual variances, but it is not wired into the main sampler.  A caller must
provide an :class:`~.idm_block.IDMBlockMetric` built before posterior sampling
and keep that metric fixed while this kernel is used.

Unlike the symmetric random-walk block kernel, MALA has a state-dependent
drift.  The forward and reverse Gaussian proposal densities therefore appear
explicitly in the Metropolis-Hastings ratio.  Hard parameter bounds define the
support of the target; proposals outside that support are rejected before a
uniform variate is drawn.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, Tuple

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ..model import JointStateSpace
from .emissions import idm_acceleration
from .idm_block import (
    IDMBlockMetric,
    _behavior_observations,
    _log_idm_target,
    _validate_kernel_inputs,
    log_idm_acceleration_jacobian,
)
from .structures import LogIDMPrior, Trajectory


FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]
BoolArray = NDArray[np.bool_]
LOG_TWO_PI = float(np.log(2.0 * np.pi))


@dataclass(frozen=True)
class IDMConditionalEvaluation:
    """Log conditional density and analytic gradient in log-parameter space."""

    log_target: float
    gradient: FloatArray


@dataclass(frozen=True)
class IDMGradientCheck:
    """Numerical audit of the analytic corrected-IDM conditional gradient."""

    analytic_gradient: FloatArray
    finite_difference_gradient: FloatArray
    absolute_error: FloatArray
    relative_error: FloatArray
    maximum_absolute_error: float
    maximum_relative_error: float
    finite_difference_step: float


@dataclass(frozen=True)
class IDMMALAAudit:
    """Complete per-behavior audit trail for one MALA sweep.

    Fields associated with a reverse proposal remain ``NaN`` when the forward
    proposal is outside the hard bounds or cannot be evaluated finitely.
    ``uniform_draw`` and ``log_uniform_draw`` remain ``NaN`` whenever the
    proposal is rejected before an MH uniform is required.
    """

    current_log_parameters: FloatArray
    proposed_log_parameters: FloatArray
    current_gradient: FloatArray
    proposed_gradient: FloatArray
    forward_mean: FloatArray
    reverse_mean: FloatArray
    standard_normal_draw: FloatArray
    current_log_target: FloatArray
    proposed_log_target: FloatArray
    log_forward_proposal_density: FloatArray
    log_reverse_proposal_density: FloatArray
    log_target_ratio: FloatArray
    log_proposal_ratio: FloatArray
    log_acceptance_ratio: FloatArray
    step_size: FloatArray
    within_bounds: BoolArray
    finite_evaluation: BoolArray
    uniform_draw: FloatArray
    log_uniform_draw: FloatArray
    accepted: BoolArray


@dataclass(frozen=True)
class IDMMALAUpdate:
    """Updated physical IDM parameters and the corresponding MALA audit."""

    parameters: FloatArray
    accepted: BoolArray
    audit: IDMMALAAudit


def evaluate_idm_log_conditional(
    log_parameters: ArrayLike,
    physical_context: ArrayLike,
    observed_acceleration: ArrayLike,
    residual_variance: float,
    prior: LogIDMPrior,
) -> IDMConditionalEvaluation:
    """Evaluate the bounded corrected-IDM conditional inside its support.

    The likelihood is Gaussian in observed acceleration, while the prior is
    Gaussian directly in the five log parameters.  Consequently, for IDM
    response Jacobian ``J`` and residual vector ``r = y - mu(eta)``,

    ``grad log p(eta | ...) = J.T @ r / variance
                              - (eta - prior_mean) / prior_sd**2``.

    The derivative follows the inactive branch at the corrected desired-gap
    ``max`` kink, consistent with :func:`log_idm_acceleration_jacobian`.
    """

    eta, context, observed, variance = _validate_conditional_inputs(
        log_parameters,
        physical_context,
        observed_acceleration,
        residual_variance,
        prior,
    )
    predicted = idm_acceleration(np.exp(eta), context)
    jacobian = log_idm_acceleration_jacobian(eta, context, method="analytic")
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        likelihood_gradient = jacobian.T @ (observed - predicted) / variance
        prior_gradient = -(
            (eta - prior.log_mean) / prior.log_standard_deviation**2
        )
        gradient = likelihood_gradient + prior_gradient
    log_target = _log_idm_target(eta, context, observed, variance, prior)
    return IDMConditionalEvaluation(
        log_target=float(log_target),
        gradient=np.asarray(gradient, dtype=float).copy(),
    )


def finite_difference_idm_log_conditional_gradient(
    log_parameters: ArrayLike,
    physical_context: ArrayLike,
    observed_acceleration: ArrayLike,
    residual_variance: float,
    prior: LogIDMPrior,
    *,
    step: float = 1.0e-6,
) -> FloatArray:
    """Return a central finite-difference check of the log conditional.

    This diagnostic is defined only when every ``eta +/- step`` point remains
    strictly inside the hard bounds.  It is not used by the sampling kernel.
    """

    eta, context, observed, variance = _validate_conditional_inputs(
        log_parameters,
        physical_context,
        observed_acceleration,
        residual_variance,
        prior,
    )
    if not np.isfinite(step) or step <= 0.0:
        raise ValueError("step must be finite and strictly positive.")
    log_lower = np.log(prior.lower)
    log_upper = np.log(prior.upper)
    if np.any(eta - step <= log_lower) or np.any(eta + step >= log_upper):
        raise ValueError(
            "Central finite differences require eta +/- step inside bounds."
        )

    gradient = np.empty(5, dtype=float)
    for parameter_index in range(5):
        perturbation = np.zeros(5, dtype=float)
        perturbation[parameter_index] = step
        upper = _log_idm_target(
            eta + perturbation, context, observed, variance, prior
        )
        lower = _log_idm_target(
            eta - perturbation, context, observed, variance, prior
        )
        with np.errstate(over="ignore", invalid="ignore"):
            gradient[parameter_index] = (upper - lower) / (2.0 * step)
    return gradient


def check_idm_log_conditional_gradient(
    log_parameters: ArrayLike,
    physical_context: ArrayLike,
    observed_acceleration: ArrayLike,
    residual_variance: float,
    prior: LogIDMPrior,
    *,
    step: float = 1.0e-6,
) -> IDMGradientCheck:
    """Compare the analytic gradient with a central finite difference."""

    analytic = evaluate_idm_log_conditional(
        log_parameters,
        physical_context,
        observed_acceleration,
        residual_variance,
        prior,
    ).gradient
    numerical = finite_difference_idm_log_conditional_gradient(
        log_parameters,
        physical_context,
        observed_acceleration,
        residual_variance,
        prior,
        step=step,
    )
    absolute = np.abs(analytic - numerical)
    denominator = np.maximum(
        np.maximum(np.abs(analytic), np.abs(numerical)),
        np.finfo(float).eps,
    )
    relative = absolute / denominator
    return IDMGradientCheck(
        analytic_gradient=analytic.copy(),
        finite_difference_gradient=numerical.copy(),
        absolute_error=absolute,
        relative_error=relative,
        maximum_absolute_error=float(np.max(absolute)),
        maximum_relative_error=float(np.max(relative)),
        finite_difference_step=float(step),
    )


def log_preconditioned_mala_proposal_density(
    destination: ArrayLike,
    origin: ArrayLike,
    origin_gradient: ArrayLike,
    step_size: float,
    preconditioner: ArrayLike,
    preconditioner_cholesky: ArrayLike,
) -> float:
    """Evaluate the exact Gaussian MALA proposal log density.

    The proposal is

    ``N(origin + 0.5*h**2*C*gradient, h**2*C)``,

    where ``C`` is a fixed symmetric positive-definite preconditioner.  This
    helper is public so tests and later integrations can audit both directions
    of the non-symmetric proposal without reimplementing the algebra.
    """

    destination_array = _finite_vector(destination, "destination")
    origin_array = _finite_vector(origin, "origin")
    gradient_array = _finite_vector(origin_gradient, "origin_gradient")
    covariance, factor = _validate_preconditioner(
        preconditioner, preconditioner_cholesky
    )
    if not np.isfinite(step_size) or step_size <= 0.0:
        raise ValueError("step_size must be finite and strictly positive.")

    mean = origin_array + 0.5 * step_size**2 * (
        covariance @ gradient_array
    )
    centered = destination_array - mean
    standardized = np.linalg.solve(factor, centered) / step_size
    quadratic = float(standardized @ standardized)
    log_determinant_factor = float(np.log(np.diag(factor)).sum())
    dimension = destination_array.size
    return float(
        -0.5 * dimension * LOG_TWO_PI
        - dimension * np.log(step_size)
        - log_determinant_factor
        - 0.5 * quadratic
    )


def sample_idm_parameters_preconditioned_mala(
    current_parameters: ArrayLike,
    residual_variance: ArrayLike,
    trajectories: Sequence[Trajectory],
    paths: Sequence[IntArray],
    state_space: JointStateSpace,
    prior: LogIDMPrior,
    metric: IDMBlockMetric,
    step_size: ArrayLike,
    rng: np.random.Generator,
) -> IDMMALAUpdate:
    """Apply one exact fixed-preconditioner MALA proposal per behavior.

    The preconditioner is ``metric.covariance``.  It must remain fixed for the
    period in which this transition kernel is used.  State-dependent gradient
    drift is allowed because the returned MH decision includes exact forward
    and reverse Gaussian proposal densities.
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
    steps = np.asarray(step_size, dtype=float)
    if steps.shape != (behavior_count,):
        raise ValueError(f"step_size must have shape ({behavior_count},).")
    if not np.all(np.isfinite(steps)) or np.any(steps <= 0.0):
        raise ValueError("step_size must be finite and strictly positive.")
    covariance, cholesky = _validate_metric(metric, behavior_count)

    updated = current.copy()
    accepted = np.zeros(behavior_count, dtype=bool)
    current_log_parameters = np.log(current)
    proposed_log_parameters = np.full_like(current_log_parameters, np.nan)
    current_gradient = np.full_like(current_log_parameters, np.nan)
    proposed_gradient = np.full_like(current_log_parameters, np.nan)
    forward_mean = np.full_like(current_log_parameters, np.nan)
    reverse_mean = np.full_like(current_log_parameters, np.nan)
    standard_normal_draw = np.full_like(current_log_parameters, np.nan)
    current_log_target = np.full(behavior_count, np.nan, dtype=float)
    proposed_log_target = np.full(behavior_count, np.nan, dtype=float)
    log_forward_density = np.full(behavior_count, np.nan, dtype=float)
    log_reverse_density = np.full(behavior_count, np.nan, dtype=float)
    log_target_ratio = np.full(behavior_count, np.nan, dtype=float)
    log_proposal_ratio = np.full(behavior_count, np.nan, dtype=float)
    log_acceptance_ratio = np.full(behavior_count, -np.inf, dtype=float)
    within_bounds = np.zeros(behavior_count, dtype=bool)
    finite_evaluation = np.zeros(behavior_count, dtype=bool)
    uniform_draw = np.full(behavior_count, np.nan, dtype=float)
    log_uniform_draw = np.full(behavior_count, np.nan, dtype=float)
    log_lower = np.log(prior.lower)
    log_upper = np.log(prior.upper)

    for behavior in range(behavior_count):
        context, observed = _behavior_observations(
            trajectory_tuple, path_tuple, state_space, behavior
        )
        current_evaluation = evaluate_idm_log_conditional(
            current_log_parameters[behavior],
            context,
            observed,
            float(variance[behavior]),
            prior,
        )
        current_log_target[behavior] = current_evaluation.log_target
        current_gradient[behavior] = current_evaluation.gradient
        if not np.isfinite(current_evaluation.log_target) or not np.all(
            np.isfinite(current_evaluation.gradient)
        ):
            raise FloatingPointError(
                "The current IDM conditional target or gradient is non-finite."
            )

        h = float(steps[behavior])
        forward_mean[behavior] = current_log_parameters[behavior] + 0.5 * h**2 * (
            covariance[behavior] @ current_evaluation.gradient
        )
        if not np.all(np.isfinite(forward_mean[behavior])):
            raise FloatingPointError("The current MALA drift is non-finite.")
        standard_normal_draw[behavior] = rng.standard_normal(5)
        proposed_log_parameters[behavior] = forward_mean[behavior] + h * (
            cholesky[behavior] @ standard_normal_draw[behavior]
        )
        if not np.all(np.isfinite(proposed_log_parameters[behavior])):
            continue
        log_forward_density[behavior] = log_preconditioned_mala_proposal_density(
            proposed_log_parameters[behavior],
            current_log_parameters[behavior],
            current_evaluation.gradient,
            h,
            covariance[behavior],
            cholesky[behavior],
        )
        within_bounds[behavior] = bool(
            np.all(proposed_log_parameters[behavior] >= log_lower)
            and np.all(proposed_log_parameters[behavior] <= log_upper)
        )
        if not within_bounds[behavior]:
            continue

        proposed_evaluation = evaluate_idm_log_conditional(
            proposed_log_parameters[behavior],
            context,
            observed,
            float(variance[behavior]),
            prior,
        )
        proposed_log_target[behavior] = proposed_evaluation.log_target
        proposed_gradient[behavior] = proposed_evaluation.gradient
        if not np.isfinite(proposed_evaluation.log_target) or not np.all(
            np.isfinite(proposed_evaluation.gradient)
        ):
            continue

        reverse_mean[behavior] = proposed_log_parameters[behavior] + 0.5 * h**2 * (
            covariance[behavior] @ proposed_evaluation.gradient
        )
        if not np.all(np.isfinite(reverse_mean[behavior])):
            continue
        log_reverse_density[behavior] = log_preconditioned_mala_proposal_density(
            current_log_parameters[behavior],
            proposed_log_parameters[behavior],
            proposed_evaluation.gradient,
            h,
            covariance[behavior],
            cholesky[behavior],
        )
        if not np.isfinite(log_forward_density[behavior]) or not np.isfinite(
            log_reverse_density[behavior]
        ):
            continue

        with np.errstate(over="ignore", invalid="ignore"):
            log_target_ratio[behavior] = (
                proposed_log_target[behavior] - current_log_target[behavior]
            )
            log_proposal_ratio[behavior] = (
                log_reverse_density[behavior] - log_forward_density[behavior]
            )
            log_acceptance_ratio[behavior] = (
                log_target_ratio[behavior] + log_proposal_ratio[behavior]
            )
        if np.isnan(log_acceptance_ratio[behavior]):
            continue
        finite_evaluation[behavior] = True
        uniform_draw[behavior] = float(rng.random())
        with np.errstate(divide="ignore"):
            log_uniform_draw[behavior] = np.log(uniform_draw[behavior])
        if log_uniform_draw[behavior] < min(
            0.0, float(log_acceptance_ratio[behavior])
        ):
            updated[behavior] = np.exp(proposed_log_parameters[behavior])
            accepted[behavior] = True

    audit = IDMMALAAudit(
        current_log_parameters=current_log_parameters.copy(),
        proposed_log_parameters=proposed_log_parameters.copy(),
        current_gradient=current_gradient.copy(),
        proposed_gradient=proposed_gradient.copy(),
        forward_mean=forward_mean.copy(),
        reverse_mean=reverse_mean.copy(),
        standard_normal_draw=standard_normal_draw.copy(),
        current_log_target=current_log_target.copy(),
        proposed_log_target=proposed_log_target.copy(),
        log_forward_proposal_density=log_forward_density.copy(),
        log_reverse_proposal_density=log_reverse_density.copy(),
        log_target_ratio=log_target_ratio.copy(),
        log_proposal_ratio=log_proposal_ratio.copy(),
        log_acceptance_ratio=log_acceptance_ratio.copy(),
        step_size=steps.copy(),
        within_bounds=within_bounds.copy(),
        finite_evaluation=finite_evaluation.copy(),
        uniform_draw=uniform_draw.copy(),
        log_uniform_draw=log_uniform_draw.copy(),
        accepted=accepted.copy(),
    )
    return IDMMALAUpdate(
        parameters=updated,
        accepted=accepted.copy(),
        audit=audit,
    )


def _validate_conditional_inputs(
    log_parameters: ArrayLike,
    physical_context: ArrayLike,
    observed_acceleration: ArrayLike,
    residual_variance: float,
    prior: LogIDMPrior,
) -> Tuple[FloatArray, FloatArray, FloatArray, float]:
    eta = _finite_vector(log_parameters, "log_parameters")
    context = np.asarray(physical_context, dtype=float)
    observed = np.asarray(observed_acceleration, dtype=float)
    if context.ndim != 2 or context.shape[1] != 3:
        raise ValueError("physical_context must have shape (observations, 3).")
    if observed.shape != (context.shape[0],):
        raise ValueError(
            "observed_acceleration must match the physical-context row count."
        )
    if not np.all(np.isfinite(context)) or not np.all(np.isfinite(observed)):
        raise ValueError("Context and observed acceleration must be finite.")
    if np.any(context[:, 0] < 0.0) or np.any(context[:, 1] <= 0.0):
        raise ValueError("Speeds must be non-negative and gaps strictly positive.")
    if not np.isfinite(residual_variance) or residual_variance <= 0.0:
        raise ValueError("residual_variance must be finite and strictly positive.")
    if np.any(eta < np.log(prior.lower)) or np.any(eta > np.log(prior.upper)):
        raise ValueError("log_parameters violate the configured hard bounds.")
    return eta, context, observed, float(residual_variance)


def _finite_vector(value: ArrayLike, name: str) -> FloatArray:
    array = np.asarray(value, dtype=float)
    if array.shape != (5,):
        raise ValueError(f"{name} must have shape (5,).")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be finite.")
    return array


def _validate_preconditioner(
    preconditioner: ArrayLike,
    preconditioner_cholesky: ArrayLike,
) -> Tuple[FloatArray, FloatArray]:
    covariance = np.asarray(preconditioner, dtype=float)
    factor = np.asarray(preconditioner_cholesky, dtype=float)
    if covariance.shape != (5, 5) or factor.shape != (5, 5):
        raise ValueError("Preconditioner and Cholesky factor must have shape (5, 5).")
    if not np.all(np.isfinite(covariance)) or not np.all(np.isfinite(factor)):
        raise ValueError("Preconditioner and Cholesky factor must be finite.")
    if not np.allclose(covariance, covariance.T, rtol=1.0e-10, atol=1.0e-12):
        raise ValueError("Preconditioner must be symmetric.")
    if np.any(np.diag(factor) <= 0.0) or not np.allclose(
        factor, np.tril(factor), rtol=0.0, atol=1.0e-14
    ):
        raise ValueError("Preconditioner Cholesky factor is invalid.")
    if not np.allclose(
        factor @ factor.T, covariance, rtol=1.0e-8, atol=1.0e-10
    ):
        raise ValueError("Preconditioner and Cholesky factor are inconsistent.")
    return covariance, factor


def _validate_metric(
    metric: IDMBlockMetric, behavior_count: int
) -> Tuple[FloatArray, FloatArray]:
    covariance = np.asarray(metric.covariance, dtype=float)
    cholesky = np.asarray(metric.cholesky, dtype=float)
    expected = (behavior_count, 5, 5)
    if covariance.shape != expected or cholesky.shape != expected:
        raise ValueError("IDM block metric is incompatible with behavior states.")
    for behavior in range(behavior_count):
        _validate_preconditioner(covariance[behavior], cholesky[behavior])
    return covariance, cholesky


__all__ = [
    "IDMConditionalEvaluation",
    "IDMGradientCheck",
    "IDMMALAAudit",
    "IDMMALAUpdate",
    "check_idm_log_conditional_gradient",
    "evaluate_idm_log_conditional",
    "finite_difference_idm_log_conditional_gradient",
    "log_preconditioned_mala_proposal_density",
    "sample_idm_parameters_preconditioned_mala",
]
