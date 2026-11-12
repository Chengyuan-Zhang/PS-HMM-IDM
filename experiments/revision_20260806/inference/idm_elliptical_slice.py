"""Elliptical slice updates for bounded corrected-IDM conditionals.

The Gaussian prior is defined on the five log-IDM parameters and supplies the
ellipse.  The acceleration likelihood and the original hard parameter bounds
form the slice likelihood.  Keeping this factorization explicit is essential:
including the Gaussian log prior in the slice height would count it twice.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ..model import JointStateSpace
from .emissions import idm_acceleration, normal_log_density
from .idm_block import _behavior_observations, _validate_kernel_inputs
from .structures import LogIDMPrior, Trajectory


FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]
IntArray = NDArray[np.int64]
TWO_PI = float(2.0 * np.pi)


@dataclass(frozen=True)
class IDMEllipticalSliceAudit:
    """Detailed audit trail for one joint five-dimensional update per state.

    Proposal-level arrays are flattened across behavior states.  The proposals
    for behavior ``b`` occupy
    ``behavior_proposal_offsets[b]:behavior_proposal_offsets[b + 1]``.
    ``likelihood_evaluations`` counts the current-state evaluation, every
    in-bounds ellipse evaluation, and the final canonical-state check.  Bounds
    rejections do not evaluate the acceleration likelihood.
    """

    current_log_parameters: FloatArray
    final_log_parameters: FloatArray
    standard_normal_direction: FloatArray
    prior_direction: FloatArray
    slice_uniform: FloatArray
    log_slice_uniform: FloatArray
    current_log_likelihood: FloatArray
    slice_log_height: FloatArray
    initial_angle: FloatArray
    accepted_angle: FloatArray
    initial_bracket_lower: FloatArray
    initial_bracket_upper: FloatArray
    final_bracket_lower: FloatArray
    final_bracket_upper: FloatArray
    behavior_proposal_offsets: IntArray
    proposal_angle: FloatArray
    proposal_bracket_lower: FloatArray
    proposal_bracket_upper: FloatArray
    proposal_within_bounds: BoolArray
    proposal_log_likelihood: FloatArray
    proposal_accepted: BoolArray
    likelihood_evaluations: IntArray
    bound_rejections: IntArray
    nonfinite_rejections: IntArray
    shrink_count: IntArray
    accepted_log_likelihood: FloatArray
    final_log_likelihood: FloatArray
    accepted_above_slice_height: BoolArray
    final_above_slice_height: BoolArray
    final_within_bounds: BoolArray
    moved: BoolArray
    max_bracket_shrinks: int


@dataclass(frozen=True)
class IDMEllipticalSliceUpdate:
    """Updated physical IDM parameters and their elliptical-slice audit."""

    parameters: FloatArray
    audit: IDMEllipticalSliceAudit

    @property
    def moved(self) -> BoolArray:
        """Return whether each behavior state's physical parameters moved."""

        return self.audit.moved


def bounded_idm_log_likelihood(
    log_parameters: ArrayLike,
    physical_context: ArrayLike,
    observed_acceleration: ArrayLike,
    residual_variance: float,
    prior: LogIDMPrior,
) -> float:
    """Return acceleration log likelihood times the hard-bound indicator.

    This function deliberately excludes the Gaussian log-IDM prior.  It is the
    likelihood factor required by elliptical slice sampling when the
    untruncated Gaussian supplies the ellipse and the original truncation is
    represented by an indicator.  An out-of-bounds point returns ``-inf``.
    NaN or positive-infinite likelihood evaluations fail closed.
    """

    eta = np.asarray(log_parameters, dtype=float)
    context = np.asarray(physical_context, dtype=float)
    observed = np.asarray(observed_acceleration, dtype=float)
    variance = float(residual_variance)
    if eta.shape != (5,) or not np.all(np.isfinite(eta)):
        raise ValueError("log_parameters must be a finite vector with shape (5,).")
    if context.ndim != 2 or context.shape[1] != 3:
        raise ValueError("physical_context must have shape (observations, 3).")
    if observed.shape != (context.shape[0],):
        raise ValueError(
            "observed_acceleration must match the physical-context rows."
        )
    if not np.all(np.isfinite(context)) or not np.all(np.isfinite(observed)):
        raise ValueError("Context and observed acceleration must be finite.")
    if np.any(context[:, 0] < 0.0) or np.any(context[:, 1] <= 0.0):
        raise ValueError("Speeds must be non-negative and gaps strictly positive.")
    if not np.isfinite(variance) or variance <= 0.0:
        raise ValueError("residual_variance must be finite and strictly positive.")

    log_lower = np.log(prior.lower)
    log_upper = np.log(prior.upper)
    if np.any(eta < log_lower) or np.any(eta > log_upper):
        return float("-inf")
    if observed.size == 0:
        return 0.0

    predicted = idm_acceleration(np.exp(eta), context)
    contributions = normal_log_density(observed, predicted, variance)
    if np.any(np.isnan(contributions)) or np.any(np.isposinf(contributions)):
        raise FloatingPointError(
            "The bounded IDM acceleration likelihood contains NaN or +inf."
        )
    value = float(np.sum(contributions))
    if np.isnan(value) or np.isposinf(value):
        raise FloatingPointError(
            "The bounded IDM acceleration likelihood is NaN or +inf."
        )
    return value


def sample_idm_parameters_elliptical_slice(
    current_parameters: ArrayLike,
    residual_variance: ArrayLike,
    trajectories: Sequence[Trajectory],
    paths: Sequence[IntArray],
    state_space: JointStateSpace,
    prior: LogIDMPrior,
    rng: np.random.Generator,
    *,
    max_bracket_shrinks: int = 1_000,
) -> IDMEllipticalSliceUpdate:
    """Apply one exact Gaussian-prior elliptical slice update per behavior.

    The five log-IDM parameters are updated jointly.  Each behavior first draws
    one standard-normal direction, then a slice-height uniform, then an initial
    angle uniform.  Every rejected angle consumes one additional uniform after
    the bracket is shrunk around angle zero.  A numerical failure or exhausted
    shrink limit raises instead of silently returning the current state.
    """

    if (
        isinstance(max_bracket_shrinks, (bool, np.bool_))
        or not isinstance(max_bracket_shrinks, (int, np.integer))
        or max_bracket_shrinks < 0
    ):
        raise ValueError("max_bracket_shrinks must be a non-negative integer.")
    maximum_shrinks = int(max_bracket_shrinks)
    current, variance, trajectory_tuple, path_tuple = _validate_kernel_inputs(
        current_parameters,
        residual_variance,
        trajectories,
        paths,
        state_space,
        prior,
    )

    behavior_count = state_space.num_behavior_states
    initial_eta = np.log(current)
    final_eta = initial_eta.copy()
    updated = current.copy()
    standard_direction = np.empty((behavior_count, 5), dtype=float)
    prior_direction = np.empty_like(standard_direction)
    slice_uniform = np.empty(behavior_count, dtype=float)
    log_slice_uniform = np.empty(behavior_count, dtype=float)
    current_likelihood = np.empty(behavior_count, dtype=float)
    slice_height = np.empty(behavior_count, dtype=float)
    initial_angle = np.empty(behavior_count, dtype=float)
    accepted_angle = np.empty(behavior_count, dtype=float)
    initial_lower = np.empty(behavior_count, dtype=float)
    initial_upper = np.empty(behavior_count, dtype=float)
    final_lower = np.empty(behavior_count, dtype=float)
    final_upper = np.empty(behavior_count, dtype=float)
    likelihood_evaluations = np.zeros(behavior_count, dtype=np.int64)
    bound_rejections = np.zeros(behavior_count, dtype=np.int64)
    nonfinite_rejections = np.zeros(behavior_count, dtype=np.int64)
    shrink_count = np.zeros(behavior_count, dtype=np.int64)
    accepted_likelihood = np.empty(behavior_count, dtype=float)
    final_likelihood = np.empty(behavior_count, dtype=float)
    accepted_above = np.zeros(behavior_count, dtype=bool)
    final_above = np.zeros(behavior_count, dtype=bool)
    final_within = np.zeros(behavior_count, dtype=bool)
    moved = np.zeros(behavior_count, dtype=bool)

    proposal_angles: list[float] = []
    proposal_lower: list[float] = []
    proposal_upper: list[float] = []
    proposal_within: list[bool] = []
    proposal_likelihood: list[float] = []
    proposal_acceptance: list[bool] = []
    offsets = [0]
    log_lower = np.log(prior.lower)
    log_upper = np.log(prior.upper)

    for behavior in range(behavior_count):
        context, observed = _behavior_observations(
            trajectory_tuple, path_tuple, state_space, behavior
        )
        eta_zero = initial_eta[behavior].copy()
        log_likelihood_zero = bounded_idm_log_likelihood(
            eta_zero,
            context,
            observed,
            float(variance[behavior]),
            prior,
        )
        likelihood_evaluations[behavior] += 1
        if not np.isfinite(log_likelihood_zero):
            raise FloatingPointError(
                "The current bounded IDM likelihood must be finite."
            )
        current_likelihood[behavior] = log_likelihood_zero

        direction_draw = np.asarray(rng.standard_normal(5), dtype=float)
        if direction_draw.shape != (5,) or not np.all(np.isfinite(direction_draw)):
            raise FloatingPointError(
                "The elliptical-slice Gaussian direction must be finite "
                "with shape (5,)."
            )
        standard_direction[behavior] = direction_draw
        direction = prior.log_standard_deviation * direction_draw
        prior_direction[behavior] = direction

        height_uniform = _unit_uniform(rng, "slice-height")
        slice_uniform[behavior] = height_uniform
        with np.errstate(divide="ignore", invalid="ignore"):
            log_height_uniform = float(np.log(height_uniform))
        log_slice_uniform[behavior] = log_height_uniform
        height = float(log_likelihood_zero + log_height_uniform)
        if np.isnan(height) or np.isposinf(height):
            raise FloatingPointError("The elliptical-slice height is invalid.")
        slice_height[behavior] = height

        angle = TWO_PI * _unit_uniform(rng, "initial-angle")
        lower = angle - TWO_PI
        upper = angle
        initial_angle[behavior] = angle
        initial_lower[behavior] = lower
        initial_upper[behavior] = upper
        behavior_accepted_eta: FloatArray | None = None
        behavior_accepted_likelihood = float("nan")

        while True:
            proposal_angles.append(angle)
            proposal_lower.append(lower)
            proposal_upper.append(upper)
            proposed_eta = (
                prior.log_mean
                + (eta_zero - prior.log_mean) * np.cos(angle)
                + direction * np.sin(angle)
            )
            if not np.all(np.isfinite(proposed_eta)):
                raise FloatingPointError(
                    "The elliptical-slice proposal is not finite."
                )

            is_within = bool(
                np.all(proposed_eta >= log_lower)
                and np.all(proposed_eta <= log_upper)
            )
            proposal_within.append(is_within)
            if is_within:
                proposed_likelihood = bounded_idm_log_likelihood(
                    proposed_eta,
                    context,
                    observed,
                    float(variance[behavior]),
                    prior,
                )
                likelihood_evaluations[behavior] += 1
                if np.isnan(proposed_likelihood) or np.isposinf(
                    proposed_likelihood
                ):
                    raise FloatingPointError(
                        "The elliptical-slice proposal likelihood is invalid."
                    )
                if np.isneginf(proposed_likelihood):
                    nonfinite_rejections[behavior] += 1
            else:
                proposed_likelihood = float("-inf")
                bound_rejections[behavior] += 1

            # When the slice-height uniform is exactly zero, ``height`` is
            # ``-inf``.  An out-of-bounds or numerically ``-inf`` proposal must
            # still be rejected rather than passing the ``-inf >= -inf``
            # comparison.
            is_accepted = bool(
                is_within
                and np.isfinite(proposed_likelihood)
                and proposed_likelihood >= height
            )
            proposal_likelihood.append(proposed_likelihood)
            proposal_acceptance.append(is_accepted)
            if is_accepted:
                behavior_accepted_eta = np.asarray(proposed_eta, dtype=float)
                behavior_accepted_likelihood = float(proposed_likelihood)
                break

            if shrink_count[behavior] >= maximum_shrinks:
                raise RuntimeError(
                    "Elliptical-slice bracket shrink limit was exhausted."
                )
            if angle < 0.0:
                lower = angle
            elif angle > 0.0:
                upper = angle
            else:
                raise FloatingPointError(
                    "Angle zero reproduced the current state below its slice height."
                )
            if not lower < 0.0 < upper:
                raise FloatingPointError(
                    "The elliptical-slice bracket no longer contains angle zero."
                )
            shrink_count[behavior] += 1
            angle_uniform = _unit_uniform(rng, "shrunken-angle")
            angle = lower + angle_uniform * (upper - lower)

        if behavior_accepted_eta is None:
            raise RuntimeError("Elliptical slice terminated without a proposal.")
        accepted_angle[behavior] = angle
        final_lower[behavior] = lower
        final_upper[behavior] = upper
        accepted_likelihood[behavior] = behavior_accepted_likelihood
        accepted_above[behavior] = bool(behavior_accepted_likelihood >= height)

        canonical_parameters, canonical_eta = _canonical_physical_state(
            behavior_accepted_eta, prior
        )
        canonical_likelihood = bounded_idm_log_likelihood(
            canonical_eta,
            context,
            observed,
            float(variance[behavior]),
            prior,
        )
        likelihood_evaluations[behavior] += 1
        if not np.isfinite(canonical_likelihood):
            raise FloatingPointError(
                "The canonical elliptical-slice state has non-finite likelihood."
            )
        final_above[behavior] = bool(canonical_likelihood >= height)
        if not final_above[behavior]:
            raise FloatingPointError(
                "The canonical elliptical-slice state is below its slice height."
            )
        final_likelihood[behavior] = canonical_likelihood
        final_eta[behavior] = canonical_eta
        updated[behavior] = canonical_parameters
        final_within[behavior] = bool(
            np.all(canonical_parameters >= prior.lower)
            and np.all(canonical_parameters <= prior.upper)
        )
        if not final_within[behavior]:
            raise FloatingPointError(
                "The canonical elliptical-slice state violates hard bounds."
            )
        moved[behavior] = not np.array_equal(
            canonical_parameters, current[behavior]
        )
        offsets.append(len(proposal_angles))

    audit = IDMEllipticalSliceAudit(
        current_log_parameters=initial_eta.copy(),
        final_log_parameters=final_eta.copy(),
        standard_normal_direction=standard_direction,
        prior_direction=prior_direction,
        slice_uniform=slice_uniform,
        log_slice_uniform=log_slice_uniform,
        current_log_likelihood=current_likelihood,
        slice_log_height=slice_height,
        initial_angle=initial_angle,
        accepted_angle=accepted_angle,
        initial_bracket_lower=initial_lower,
        initial_bracket_upper=initial_upper,
        final_bracket_lower=final_lower,
        final_bracket_upper=final_upper,
        behavior_proposal_offsets=np.asarray(offsets, dtype=np.int64),
        proposal_angle=np.asarray(proposal_angles, dtype=float),
        proposal_bracket_lower=np.asarray(proposal_lower, dtype=float),
        proposal_bracket_upper=np.asarray(proposal_upper, dtype=float),
        proposal_within_bounds=np.asarray(proposal_within, dtype=bool),
        proposal_log_likelihood=np.asarray(proposal_likelihood, dtype=float),
        proposal_accepted=np.asarray(proposal_acceptance, dtype=bool),
        likelihood_evaluations=likelihood_evaluations,
        bound_rejections=bound_rejections,
        nonfinite_rejections=nonfinite_rejections,
        shrink_count=shrink_count,
        accepted_log_likelihood=accepted_likelihood,
        final_log_likelihood=final_likelihood,
        accepted_above_slice_height=accepted_above,
        final_above_slice_height=final_above,
        final_within_bounds=final_within,
        moved=moved,
        max_bracket_shrinks=maximum_shrinks,
    )
    return IDMEllipticalSliceUpdate(updated, audit)


def _unit_uniform(rng: np.random.Generator, purpose: str) -> float:
    draw = float(rng.random())
    if not np.isfinite(draw) or draw < 0.0 or draw >= 1.0:
        raise FloatingPointError(
            f"The {purpose} uniform must be finite and lie in [0, 1)."
        )
    return draw


def _canonical_physical_state(
    accepted_log_parameters: FloatArray, prior: LogIDMPrior
) -> tuple[FloatArray, FloatArray]:
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        physical = np.exp(accepted_log_parameters)
    if not np.all(np.isfinite(physical)) or np.any(physical <= 0.0):
        raise FloatingPointError(
            "The elliptical-slice physical parameters are not finite and positive."
        )
    inclusive_lower = np.nextafter(prior.lower, -np.inf)
    inclusive_upper = np.nextafter(prior.upper, np.inf)
    if np.any(physical < inclusive_lower) or np.any(physical > inclusive_upper):
        raise FloatingPointError(
            "The elliptical-slice physical parameters violate hard bounds."
        )
    canonical = np.clip(physical, prior.lower, prior.upper)
    return canonical, np.log(canonical)


__all__ = [
    "IDMEllipticalSliceAudit",
    "IDMEllipticalSliceUpdate",
    "bounded_idm_log_likelihood",
    "sample_idm_parameters_elliptical_slice",
]
