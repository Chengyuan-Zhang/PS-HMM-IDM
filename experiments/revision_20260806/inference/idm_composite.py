"""Composite coordinates for the corrected-IDM conditional posterior.

The two gap parameters are represented by a reference-speed effective gap
and a composition coordinate,

``gamma = log(s0 + v_ref * T)`` and
``kappa = log(v_ref * T / s0)``.

The map from ``(gamma, kappa)`` to ``(log(s0), log(T))`` has determinant one.
Consequently, the existing posterior target in log-IDM coordinates can be
evaluated without a variable change-of-measure correction.  ``v_ref`` is a
fixed design constant supplied by the caller; this module never estimates or
adapts it inside a chain.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ..model import JointStateSpace
from .idm_block import (
    _behavior_observations,
    _log_idm_target,
    _validate_kernel_inputs,
)
from .structures import LogIDMPrior, Trajectory


FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]
IntArray = NDArray[np.int64]


@dataclass(frozen=True)
class IDMCompositeCoordinateMHAudit:
    """Proposal-level audit trail for one sequential coordinate sweep.

    ``proposed_composite_coordinates[b, j]`` is the complete five-vector
    proposed for behavior ``b`` while updating coordinate ``j``.  A proposal
    outside the original IDM bounds has ``within_original_bounds=False`` and
    consumes no Metropolis uniform; its target and uniform fields remain NaN.
    """

    initial_composite_coordinates: FloatArray
    final_composite_coordinates: FloatArray
    proposed_composite_coordinates: FloatArray
    normal_draw: FloatArray
    within_original_bounds: BoolArray
    log_acceptance_ratio: FloatArray
    uniform_draw: FloatArray
    log_uniform_draw: FloatArray
    accepted: BoolArray


@dataclass(frozen=True)
class IDMCompositeCoordinateMHUpdate:
    """Updated physical IDM parameters and coordinate-proposal audit."""

    parameters: FloatArray
    audit: IDMCompositeCoordinateMHAudit

    @property
    def accepted(self) -> BoolArray:
        """Return the per-behavior, per-coordinate acceptance indicators."""

        return self.audit.accepted


def log_idm_to_composite(
    log_parameters: ArrayLike, reference_speed: float
) -> FloatArray:
    """Map log ``[v0, s0, T, a, b]`` to composite coordinates.

    The returned order is ``[log(v0), gamma, kappa, log(a), log(b)]``.
    ``numpy.logaddexp`` keeps ``gamma`` stable when either ``s0`` or
    ``v_ref*T`` dominates the effective gap.
    """

    eta = _finite_parameter_array(log_parameters, "log_parameters")
    log_reference_speed = np.log(_validate_reference_speed(reference_speed))
    result = eta.copy()
    result[..., 1] = np.logaddexp(
        eta[..., 1], log_reference_speed + eta[..., 2]
    )
    result[..., 2] = log_reference_speed + eta[..., 2] - eta[..., 1]
    if not np.all(np.isfinite(result)):
        raise FloatingPointError("Composite coordinates are not finite.")
    return result


def composite_to_log_idm(
    composite_coordinates: ArrayLike, reference_speed: float
) -> FloatArray:
    """Invert composite coordinates to log ``[v0, s0, T, a, b]``.

    The inverse uses softplus identities rather than subtracting
    ``exp(gamma) - v_ref*T``, so it remains stable for highly imbalanced gap
    compositions.
    """

    coordinates = _finite_parameter_array(
        composite_coordinates, "composite_coordinates"
    )
    log_reference_speed = np.log(_validate_reference_speed(reference_speed))
    gamma = coordinates[..., 1]
    kappa = coordinates[..., 2]
    result = coordinates.copy()
    result[..., 1] = gamma - np.logaddexp(0.0, kappa)
    result[..., 2] = (
        gamma - log_reference_speed - np.logaddexp(0.0, -kappa)
    )
    if not np.all(np.isfinite(result)):
        raise FloatingPointError("Mapped log-IDM parameters are not finite.")
    return result


def idm_parameters_to_composite(
    parameters: ArrayLike, reference_speed: float
) -> FloatArray:
    """Map strictly positive physical IDM parameters to composite space."""

    physical = _finite_parameter_array(parameters, "parameters")
    if np.any(physical <= 0.0):
        raise ValueError("Physical IDM parameters must be strictly positive.")
    return log_idm_to_composite(np.log(physical), reference_speed)


def composite_to_idm_parameters(
    composite_coordinates: ArrayLike, reference_speed: float
) -> FloatArray:
    """Map composite coordinates to strictly positive physical parameters."""

    log_parameters = composite_to_log_idm(
        composite_coordinates, reference_speed
    )
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        parameters = np.exp(log_parameters)
    if not np.all(np.isfinite(parameters)) or np.any(parameters <= 0.0):
        raise FloatingPointError(
            "Mapped physical IDM parameters are not finite and positive."
        )
    return np.asarray(parameters, dtype=float)


def eta_from_composite_jacobian(
    composite_coordinates: ArrayLike,
) -> FloatArray:
    """Return ``d eta / d z`` for one composite coordinate vector.

    Both ``eta`` and ``z`` have order documented by the conversion helpers.
    The determinant is analytically one for every finite ``kappa``.
    """

    coordinates = np.asarray(composite_coordinates, dtype=float)
    if coordinates.shape != (5,) or not np.all(np.isfinite(coordinates)):
        raise ValueError(
            "composite_coordinates must be a finite vector with shape (5,)."
        )
    kappa = float(coordinates[2])
    q = float(np.exp(-np.logaddexp(0.0, -kappa)))
    jacobian = np.zeros((5, 5), dtype=float)
    jacobian[0, 0] = 1.0
    jacobian[1, 1] = 1.0
    jacobian[1, 2] = -q
    jacobian[2, 1] = 1.0
    jacobian[2, 2] = 1.0 - q
    jacobian[3, 3] = 1.0
    jacobian[4, 4] = 1.0
    return jacobian


def transform_log_idm_gradient_to_composite(
    log_parameter_gradient: ArrayLike,
    composite_coordinates: ArrayLike,
) -> FloatArray:
    """Transform an ``eta``-space gradient to composite coordinates."""

    gradient = np.asarray(log_parameter_gradient, dtype=float)
    if gradient.shape != (5,) or not np.all(np.isfinite(gradient)):
        raise ValueError(
            "log_parameter_gradient must be a finite vector with shape (5,)."
        )
    jacobian = eta_from_composite_jacobian(composite_coordinates)
    return np.asarray(jacobian.T @ gradient, dtype=float)


def composite_within_original_bounds(
    composite_coordinates: ArrayLike,
    reference_speed: float,
    prior: LogIDMPrior,
) -> bool:
    """Return whether a composite point is strictly inside the physical box.

    The open and closed IDM boxes define the same continuous truncated target;
    they differ only on a measure-zero boundary.  The strict finite-precision
    predicate lets the slice kernel decode accepted coordinates without a
    recurrent clipping or re-encoding projection.
    """

    return (
        decode_supported_composite(composite_coordinates, reference_speed, prior)
        is not None
    )


def decode_supported_composite(
    composite_coordinates: ArrayLike,
    reference_speed: float,
    prior: LogIDMPrior,
) -> Optional[Tuple[FloatArray, FloatArray]]:
    """Decode one composite state without projection when strictly supported.

    Returns ``(log_parameters, physical_parameters)`` or ``None``.  This helper
    never expands the target support, clips a value, or re-encodes the supplied
    composite coordinates.
    """

    eta = composite_to_log_idm(composite_coordinates, reference_speed)
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        physical = np.exp(eta)
    if (
        not np.all(np.isfinite(physical))
        or np.any(physical <= prior.lower)
        or np.any(physical >= prior.upper)
    ):
        return None
    return np.asarray(eta, dtype=float), np.asarray(physical, dtype=float)


def composite_log_idm_target(
    composite_coordinates: ArrayLike,
    physical_context: ArrayLike,
    observed_acceleration: ArrayLike,
    residual_variance: float,
    prior: LogIDMPrior,
    reference_speed: float,
) -> float:
    """Evaluate the unchanged bounded IDM target in composite coordinates.

    The original target is a density in the five log-IDM parameters.  Because
    ``abs(det(d eta / d z)) == 1``, no change-of-measure term is added here.
    Points outside the original hard bounds have log density ``-inf``.
    """

    decoded = decode_supported_composite(
        composite_coordinates, reference_speed, prior
    )
    if decoded is None:
        return float("-inf")
    eta, _ = decoded
    return _log_idm_target(
        eta,
        np.asarray(physical_context, dtype=float),
        np.asarray(observed_acceleration, dtype=float),
        float(residual_variance),
        prior,
    )


def sample_idm_parameters_composite_coordinate_mh(
    current_parameters: ArrayLike,
    residual_variance: ArrayLike,
    trajectories: Sequence[Trajectory],
    paths: Sequence[IntArray],
    state_space: JointStateSpace,
    prior: LogIDMPrior,
    reference_speed: float,
    proposal_standard_deviation: ArrayLike,
    rng: np.random.Generator,
) -> IDMCompositeCoordinateMHUpdate:
    """Apply sequential symmetric random-walk MH in composite coordinates.

    Each behavior receives five scalar proposals in the fixed order
    ``[log(v0), gamma, kappa, log(a), log(b)]``.  A proposed point is mapped
    back to the original log-IDM coordinates before its hard bounds and target
    are evaluated.  Out-of-bounds points are rejected without consuming an MH
    uniform.  No Jacobian term is required because the transform has unit
    determinant.
    """

    current, variance, trajectory_tuple, path_tuple = _validate_kernel_inputs(
        current_parameters,
        residual_variance,
        trajectories,
        paths,
        state_space,
        prior,
    )
    fixed_reference_speed = _validate_reference_speed(reference_speed)
    proposal = _proposal_scale(
        proposal_standard_deviation, state_space.num_behavior_states
    )
    behavior_count = state_space.num_behavior_states
    initial_coordinates = idm_parameters_to_composite(
        current, fixed_reference_speed
    )
    final_coordinates = initial_coordinates.copy()
    proposed_coordinates = np.full(
        (behavior_count, 5, 5), np.nan, dtype=float
    )
    normal_draw = np.full((behavior_count, 5), np.nan, dtype=float)
    within_bounds = np.zeros((behavior_count, 5), dtype=bool)
    log_acceptance = np.full((behavior_count, 5), np.nan, dtype=float)
    uniform_draw = np.full((behavior_count, 5), np.nan, dtype=float)
    log_uniform_draw = np.full((behavior_count, 5), np.nan, dtype=float)
    accepted = np.zeros((behavior_count, 5), dtype=bool)
    updated = current.copy()

    for behavior in range(behavior_count):
        context, observed = _behavior_observations(
            trajectory_tuple, path_tuple, state_space, behavior
        )
        current_z = final_coordinates[behavior].copy()
        current_eta = composite_to_log_idm(
            current_z, fixed_reference_speed
        )
        current_target = _log_idm_target(
            current_eta,
            context,
            observed,
            float(variance[behavior]),
            prior,
        )
        if not np.isfinite(current_target):
            raise FloatingPointError(
                "The current composite IDM conditional target is non-finite."
            )

        for coordinate in range(5):
            draw = float(
                rng.normal(0.0, proposal[behavior, coordinate])
            )
            normal_draw[behavior, coordinate] = draw
            proposed_z = current_z.copy()
            proposed_z[coordinate] += draw
            proposed_coordinates[behavior, coordinate] = proposed_z
            is_within_bounds = composite_within_original_bounds(
                proposed_z, fixed_reference_speed, prior
            )
            within_bounds[behavior, coordinate] = is_within_bounds
            if not is_within_bounds:
                continue

            proposed_eta = composite_to_log_idm(
                proposed_z, fixed_reference_speed
            )
            proposed_target = _log_idm_target(
                proposed_eta,
                context,
                observed,
                float(variance[behavior]),
                prior,
            )
            ratio = float(proposed_target - current_target)
            log_acceptance[behavior, coordinate] = ratio
            if np.isnan(ratio):
                continue

            uniform = float(rng.random())
            uniform_draw[behavior, coordinate] = uniform
            with np.errstate(divide="ignore", invalid="ignore"):
                log_uniform = float(np.log(uniform))
            log_uniform_draw[behavior, coordinate] = log_uniform
            if log_uniform < min(0.0, ratio):
                current_z = proposed_z
                current_eta = proposed_eta
                current_target = proposed_target
                accepted[behavior, coordinate] = True

        # The composite round trip can move a mathematically exact physical
        # boundary by one ULP.  Canonicalize the returned state to the original
        # closed support so that a legal boundary state remains legal on the
        # next sweep.  Recompute the audited final coordinates from that
        # canonical physical state to keep the audit and returned state equal.
        canonical_parameters = np.clip(
            np.exp(current_eta), prior.lower, prior.upper
        )
        updated[behavior] = canonical_parameters
        final_coordinates[behavior] = idm_parameters_to_composite(
            canonical_parameters, fixed_reference_speed
        )

    audit = IDMCompositeCoordinateMHAudit(
        initial_composite_coordinates=initial_coordinates.copy(),
        final_composite_coordinates=final_coordinates.copy(),
        proposed_composite_coordinates=proposed_coordinates,
        normal_draw=normal_draw,
        within_original_bounds=within_bounds,
        log_acceptance_ratio=log_acceptance,
        uniform_draw=uniform_draw,
        log_uniform_draw=log_uniform_draw,
        accepted=accepted,
    )
    return IDMCompositeCoordinateMHUpdate(updated, audit)


def _finite_parameter_array(values: ArrayLike, name: str) -> FloatArray:
    array = np.asarray(values, dtype=float)
    if array.ndim < 1 or array.shape[-1] != 5:
        raise ValueError(f"{name} must have final dimension 5.")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be finite.")
    return array.copy()


def _validate_reference_speed(reference_speed: float) -> float:
    value = float(reference_speed)
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError("reference_speed must be finite and strictly positive.")
    return value


def _proposal_scale(
    proposal_standard_deviation: ArrayLike, behavior_count: int
) -> FloatArray:
    supplied = np.asarray(proposal_standard_deviation, dtype=float)
    expected = (behavior_count, 5)
    if supplied.shape == (5,):
        proposal = np.broadcast_to(supplied, expected).copy()
    elif supplied.shape == expected:
        proposal = supplied.copy()
    else:
        raise ValueError(
            "proposal_standard_deviation must have shape (5,) or "
            f"{expected}."
        )
    if not np.all(np.isfinite(proposal)) or np.any(proposal <= 0.0):
        raise ValueError(
            "Proposal standard deviations must be finite and strictly positive."
        )
    return proposal
