"""Bounded coordinate slice updates in corrected-IDM composite space.

The five sampling coordinates are

``z = (log(v0), gamma, kappa, log(a), log(b))``,

where ``gamma = log(s0 + v_ref * T)`` and
``kappa = log(v_ref * T / s0)``.  The map from ``z`` to the five log-IDM
parameters has unit Jacobian determinant, so the complete existing log-IDM
target can be evaluated directly in composite space without a
change-of-measure term.

Each scalar update starts from the finite global interval induced by the
original physical IDM box and a frozen reference speed.  The interval is not
silently narrowed to a local approximation: proposals that lie in the global
composite box but map outside the original IDM box receive target ``-inf`` and
shrink the bracket around the current coordinate.  There is no proposal width,
metric, or adaptation parameter.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ..model import JointStateSpace
from .idm_block import _behavior_observations, _validate_kernel_inputs
from .idm_composite import (
    composite_log_idm_target,
    composite_to_log_idm,
    composite_within_original_bounds,
    decode_supported_composite,
    log_idm_to_composite,
)
from .structures import LogIDMPrior, Trajectory


FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]
IntArray = NDArray[np.int64]


@dataclass(frozen=True)
class IDMCompositeCoordinateSliceAudit:
    """Complete transcript for one five-coordinate sweep per behavior.

    Proposal-level arrays are flattened in behavior-major, coordinate-minor
    order.  Segment ``u = 5 * behavior + coordinate`` occupies
    ``coordinate_proposal_offsets[u]:coordinate_proposal_offsets[u + 1]``.
    The saved bracket is the bracket *before* the corresponding proposal.
    """

    input_parameters: FloatArray
    canonical_initial_parameters: FloatArray
    initialized_from_physical: BoolArray
    initial_state_adjusted: BoolArray
    initial_boundary_canonicalized: BoolArray
    initialization_inward_steps: IntArray
    initial_composite_coordinates: FloatArray
    final_composite_coordinates: FloatArray
    coordinate_current_composite_coordinates: FloatArray
    coordinate_accepted_composite_coordinates: FloatArray
    global_interval_lower: FloatArray
    global_interval_upper: FloatArray
    slice_uniform: FloatArray
    log_slice_uniform: FloatArray
    current_log_target: FloatArray
    slice_log_height: FloatArray
    accepted_coordinate_value: FloatArray
    accepted_log_target: FloatArray
    accepted_within_original_bounds: BoolArray
    accepted_above_slice_height: BoolArray
    coordinate_proposal_offsets: IntArray
    proposal_uniform: FloatArray
    proposal_coordinate_value: FloatArray
    proposal_bracket_lower: FloatArray
    proposal_bracket_upper: FloatArray
    proposal_within_original_bounds: BoolArray
    proposal_log_target: FloatArray
    proposal_accepted: BoolArray
    proposal_count: IntArray
    shrink_count: IntArray
    bound_rejections: IntArray
    nonfinite_rejections: IntArray
    final_log_target: FloatArray
    final_within_original_bounds: BoolArray
    final_above_last_slice_height: BoolArray
    moved: BoolArray
    max_bracket_shrinks: int


@dataclass(frozen=True)
class IDMCompositeCoordinateSliceUpdate:
    """Updated physical and authoritative composite states plus transcript."""

    parameters: FloatArray
    composite_coordinates: FloatArray
    audit: IDMCompositeCoordinateSliceAudit

    @property
    def moved(self) -> BoolArray:
        """Return whether each behavior's authoritative composite state moved."""

        return self.audit.moved


def composite_global_bounding_intervals(
    prior: LogIDMPrior, reference_speed: float
) -> Tuple[FloatArray, FloatArray]:
    """Return finite global bounds for all five composite coordinates.

    For the two transformed gap coordinates, monotonicity gives

    ``gamma_min = log(s0_min + v_ref*T_min)``,
    ``gamma_max = log(s0_max + v_ref*T_max)``,
    ``kappa_min = log(v_ref) + log(T_min) - log(s0_max)``, and
    ``kappa_max = log(v_ref) + log(T_max) - log(s0_min)``.

    This is the global bounding rectangle of the transformed physical box,
    not a claim that every ``(gamma, kappa)`` pair in that rectangle is in the
    original support.
    """

    fixed_reference_speed = _validate_reference_speed(reference_speed)
    log_reference_speed = float(np.log(fixed_reference_speed))
    log_lower = np.log(prior.lower)
    log_upper = np.log(prior.upper)
    lower = log_lower.copy()
    upper = log_upper.copy()
    lower[1] = np.logaddexp(
        log_lower[1], log_reference_speed + log_lower[2]
    )
    upper[1] = np.logaddexp(
        log_upper[1], log_reference_speed + log_upper[2]
    )
    lower[2] = log_reference_speed + log_lower[2] - log_upper[1]
    upper[2] = log_reference_speed + log_upper[2] - log_lower[1]
    # Preserve mathematically exact physical corners whose log/add/subtract
    # round trip lands one representable float beyond the analytic endpoint.
    # Original physical support is still checked separately for every proposal.
    lower = np.nextafter(lower, -np.inf)
    upper = np.nextafter(upper, np.inf)
    if (
        not np.all(np.isfinite(lower))
        or not np.all(np.isfinite(upper))
        or np.any(upper <= lower)
    ):
        raise FloatingPointError(
            "The induced composite bounding intervals are invalid."
        )
    return lower, upper


def sample_idm_parameters_composite_coordinate_slice(
    current_parameters: ArrayLike,
    residual_variance: ArrayLike,
    trajectories: Sequence[Trajectory],
    paths: Sequence[IntArray],
    state_space: JointStateSpace,
    prior: LogIDMPrior,
    reference_speed: float,
    rng: np.random.Generator,
    *,
    current_composite_coordinates: Optional[ArrayLike] = None,
    max_bracket_shrinks: int = 10_000,
) -> IDMCompositeCoordinateSliceUpdate:
    """Apply one target-invariant full-bracket coordinate-slice sweep.

    Coordinates are updated sequentially in the documented order.  For each
    scalar update, a slice height is drawn from the *complete* composite log
    target and proposals are uniform on the current finite bracket.  A
    rejected proposal replaces the bracket endpoint on its side of the
    current coordinate.  Each scalar kernel satisfies the usual slice-sampler
    detailed-balance construction; their fixed-order composition preserves the
    target but is not claimed to be reversible as a complete sweep.  The caller
    must persist the returned ``composite_coordinates`` and pass them back on
    the next sweep; this avoids a recurrent physical-to-composite projection.
    Exact physical-boundary inputs are moved inward only when no authoritative
    composite state is supplied.  Exhausting ``max_bracket_shrinks`` raises
    instead of returning an unaudited fallback state.
    """

    if (
        isinstance(max_bracket_shrinks, (bool, np.bool_))
        or not isinstance(max_bracket_shrinks, (int, np.integer))
        or max_bracket_shrinks < 0
    ):
        raise ValueError("max_bracket_shrinks must be a non-negative integer.")
    maximum_shrinks = int(max_bracket_shrinks)
    fixed_reference_speed = _validate_reference_speed(reference_speed)
    input_parameters_array, variance, trajectory_tuple, path_tuple = _validate_kernel_inputs(
        current_parameters,
        residual_variance,
        trajectories,
        paths,
        state_space,
        prior,
    )
    global_lower, global_upper = composite_global_bounding_intervals(
        prior, fixed_reference_speed
    )

    behavior_count = state_space.num_behavior_states
    input_parameters_array = input_parameters_array.copy()
    current = np.empty_like(input_parameters_array)
    initial_z = np.empty_like(current)
    initialized_from_physical = np.zeros(behavior_count, dtype=bool)
    initial_state_adjusted = np.zeros(behavior_count, dtype=bool)
    initial_boundary_canonicalized = np.zeros(behavior_count, dtype=bool)
    initialization_inward_steps = np.zeros((behavior_count, 5), dtype=np.int64)
    if current_composite_coordinates is None:
        initialized_from_physical[:] = True
        for behavior in range(behavior_count):
            canonical_parameters, canonical_z, inward_steps = (
                _initialize_composite_state(
                    input_parameters_array[behavior],
                    fixed_reference_speed,
                    prior,
                )
            )
            current[behavior] = canonical_parameters
            initial_z[behavior] = canonical_z
            initialization_inward_steps[behavior] = inward_steps
            initial_boundary_canonicalized[behavior] = bool(
                np.any(inward_steps > 0)
            )
            initial_state_adjusted[behavior] = not np.array_equal(
                canonical_parameters, input_parameters_array[behavior]
            )
    else:
        supplied_z = np.asarray(current_composite_coordinates, dtype=float)
        if supplied_z.shape != (behavior_count, 5) or not np.all(
            np.isfinite(supplied_z)
        ):
            raise ValueError(
                "current_composite_coordinates must be finite with shape "
                f"({behavior_count}, 5)."
            )
        for behavior in range(behavior_count):
            decoded = decode_supported_composite(
                supplied_z[behavior], fixed_reference_speed, prior
            )
            if decoded is None:
                raise ValueError(
                    "current_composite_coordinates must lie strictly inside "
                    "the original physical IDM bounds."
                )
            _, decoded_parameters = decoded
            if not np.array_equal(
                decoded_parameters, input_parameters_array[behavior]
            ):
                raise ValueError(
                    "current_parameters must exactly match the physical state "
                    "decoded from current_composite_coordinates."
                )
            current[behavior] = decoded_parameters
            initial_z[behavior] = supplied_z[behavior]
    final_z = initial_z.copy()
    updated = current.copy()
    coordinate_current = np.empty((behavior_count, 5, 5), dtype=float)
    coordinate_accepted = np.empty_like(coordinate_current)
    slice_uniform = np.empty((behavior_count, 5), dtype=float)
    log_slice_uniform = np.empty_like(slice_uniform)
    current_target = np.empty_like(slice_uniform)
    slice_height = np.empty_like(slice_uniform)
    accepted_value = np.empty_like(slice_uniform)
    accepted_target = np.empty_like(slice_uniform)
    accepted_within = np.zeros((behavior_count, 5), dtype=bool)
    accepted_above = np.zeros((behavior_count, 5), dtype=bool)
    proposal_count = np.zeros((behavior_count, 5), dtype=np.int64)
    shrink_count = np.zeros_like(proposal_count)
    bound_rejections = np.zeros_like(proposal_count)
    nonfinite_rejections = np.zeros_like(proposal_count)
    canonical_target = np.empty(behavior_count, dtype=float)
    canonical_within = np.zeros(behavior_count, dtype=bool)
    canonical_above = np.zeros(behavior_count, dtype=bool)
    moved = np.zeros(behavior_count, dtype=bool)

    proposal_values: list[float] = []
    proposal_uniforms: list[float] = []
    proposal_lower: list[float] = []
    proposal_upper: list[float] = []
    proposal_within: list[bool] = []
    proposal_targets: list[float] = []
    proposal_acceptance: list[bool] = []
    offsets = [0]

    for behavior in range(behavior_count):
        context, observed = _behavior_observations(
            trajectory_tuple, path_tuple, state_space, behavior
        )
        behavior_z = final_z[behavior].copy()

        for coordinate in range(5):
            coordinate_current[behavior, coordinate] = behavior_z
            log_target_zero = composite_log_idm_target(
                behavior_z,
                context,
                observed,
                float(variance[behavior]),
                prior,
                fixed_reference_speed,
            )
            if not np.isfinite(log_target_zero):
                raise FloatingPointError(
                    "The current composite slice target must be finite."
                )
            current_target[behavior, coordinate] = log_target_zero

            height_uniform = _unit_uniform(rng, "slice-height")
            slice_uniform[behavior, coordinate] = height_uniform
            with np.errstate(divide="ignore", invalid="ignore"):
                log_height_uniform = float(np.log(height_uniform))
            log_slice_uniform[behavior, coordinate] = log_height_uniform
            height = float(log_target_zero + log_height_uniform)
            if np.isnan(height) or np.isposinf(height):
                raise FloatingPointError(
                    "The composite coordinate slice height is invalid."
                )
            slice_height[behavior, coordinate] = height

            lower = float(global_lower[coordinate])
            upper = float(global_upper[coordinate])
            anchor = float(behavior_z[coordinate])
            if not lower <= anchor <= upper:
                raise FloatingPointError(
                    "The current composite coordinate is outside its global interval."
                )

            accepted_z: FloatArray | None = None
            accepted_log_target = float("nan")
            while True:
                proposal_uniform = _unit_uniform(rng, "full-bracket proposal")
                value = float(lower + proposal_uniform * (upper - lower))
                if not np.isfinite(value) or value < lower or value > upper:
                    raise FloatingPointError(
                        "The full-bracket composite proposal is invalid."
                    )
                proposed_z = behavior_z.copy()
                proposed_z[coordinate] = value

                proposal_uniforms.append(proposal_uniform)
                proposal_values.append(value)
                proposal_lower.append(lower)
                proposal_upper.append(upper)
                is_within = composite_within_original_bounds(
                    proposed_z, fixed_reference_speed, prior
                )
                proposal_within.append(is_within)
                if is_within:
                    proposed_target = composite_log_idm_target(
                        proposed_z,
                        context,
                        observed,
                        float(variance[behavior]),
                        prior,
                        fixed_reference_speed,
                    )
                    if np.isnan(proposed_target) or np.isposinf(proposed_target):
                        raise FloatingPointError(
                            "The composite slice proposal target is invalid."
                        )
                    if np.isneginf(proposed_target):
                        nonfinite_rejections[behavior, coordinate] += 1
                else:
                    proposed_target = float("-inf")
                    bound_rejections[behavior, coordinate] += 1

                # A zero slice uniform gives height -inf.  Requiring a finite
                # proposal target prevents either an out-of-support or
                # numerically -inf proposal from passing -inf >= -inf.
                is_accepted = bool(
                    is_within
                    and np.isfinite(proposed_target)
                    and proposed_target >= height
                )
                proposal_targets.append(float(proposed_target))
                proposal_acceptance.append(is_accepted)
                proposal_count[behavior, coordinate] += 1
                if is_accepted:
                    accepted_z = proposed_z
                    accepted_log_target = float(proposed_target)
                    break

                if shrink_count[behavior, coordinate] >= maximum_shrinks:
                    raise RuntimeError(
                        "Composite coordinate slice bracket shrink limit was "
                        "exhausted."
                    )
                if value < anchor:
                    lower = value
                elif value > anchor:
                    upper = value
                else:
                    raise FloatingPointError(
                        "The current composite coordinate was rejected below "
                        "its own slice height."
                    )
                if not lower <= anchor <= upper:
                    raise FloatingPointError(
                        "The composite slice bracket lost its current coordinate."
                    )
                shrink_count[behavior, coordinate] += 1

            if accepted_z is None:
                raise RuntimeError(
                    "Composite coordinate slice terminated without a proposal."
                )
            coordinate_accepted[behavior, coordinate] = accepted_z
            accepted_value[behavior, coordinate] = accepted_z[coordinate]
            accepted_target[behavior, coordinate] = accepted_log_target
            accepted_within[behavior, coordinate] = bool(
                composite_within_original_bounds(
                    accepted_z, fixed_reference_speed, prior
                )
            )
            accepted_above[behavior, coordinate] = bool(
                accepted_log_target >= height
            )
            if not accepted_within[behavior, coordinate] or not accepted_above[
                behavior, coordinate
            ]:
                raise FloatingPointError(
                    "The accepted composite coordinate failed its support or "
                    "slice-height audit."
                )

            behavior_z = accepted_z
            offsets.append(len(proposal_values))

        decoded_final = decode_supported_composite(
            behavior_z, fixed_reference_speed, prior
        )
        if decoded_final is None:
            raise FloatingPointError(
                "The final authoritative composite state is unsupported."
            )
        _, final_parameters = decoded_final
        final_log_target = composite_log_idm_target(
            behavior_z,
            context,
            observed,
            float(variance[behavior]),
            prior,
            fixed_reference_speed,
        )
        if not np.isfinite(final_log_target):
            raise FloatingPointError(
                "The final composite slice target is non-finite."
            )
        if final_log_target != accepted_target[behavior, -1]:
            raise FloatingPointError(
                "The final target differs from the last accepted target."
            )
        canonical_target[behavior] = final_log_target
        canonical_within[behavior] = True
        canonical_above[behavior] = bool(
            final_log_target >= slice_height[behavior, -1]
        )
        if not canonical_within[behavior] or not canonical_above[behavior]:
            raise FloatingPointError(
                "The final composite slice state failed its audit."
            )
        updated[behavior] = final_parameters
        final_z[behavior] = behavior_z
        moved[behavior] = not np.array_equal(
            behavior_z, initial_z[behavior]
        )

    audit = IDMCompositeCoordinateSliceAudit(
        input_parameters=input_parameters_array.copy(),
        canonical_initial_parameters=current.copy(),
        initialized_from_physical=initialized_from_physical,
        initial_state_adjusted=initial_state_adjusted,
        initial_boundary_canonicalized=initial_boundary_canonicalized,
        initialization_inward_steps=initialization_inward_steps,
        initial_composite_coordinates=initial_z.copy(),
        final_composite_coordinates=final_z.copy(),
        coordinate_current_composite_coordinates=coordinate_current,
        coordinate_accepted_composite_coordinates=coordinate_accepted,
        global_interval_lower=np.broadcast_to(
            global_lower, (behavior_count, 5)
        ).copy(),
        global_interval_upper=np.broadcast_to(
            global_upper, (behavior_count, 5)
        ).copy(),
        slice_uniform=slice_uniform,
        log_slice_uniform=log_slice_uniform,
        current_log_target=current_target,
        slice_log_height=slice_height,
        accepted_coordinate_value=accepted_value,
        accepted_log_target=accepted_target,
        accepted_within_original_bounds=accepted_within,
        accepted_above_slice_height=accepted_above,
        coordinate_proposal_offsets=np.asarray(offsets, dtype=np.int64),
        proposal_uniform=np.asarray(proposal_uniforms, dtype=float),
        proposal_coordinate_value=np.asarray(proposal_values, dtype=float),
        proposal_bracket_lower=np.asarray(proposal_lower, dtype=float),
        proposal_bracket_upper=np.asarray(proposal_upper, dtype=float),
        proposal_within_original_bounds=np.asarray(proposal_within, dtype=bool),
        proposal_log_target=np.asarray(proposal_targets, dtype=float),
        proposal_accepted=np.asarray(proposal_acceptance, dtype=bool),
        proposal_count=proposal_count,
        shrink_count=shrink_count,
        bound_rejections=bound_rejections,
        nonfinite_rejections=nonfinite_rejections,
        final_log_target=canonical_target,
        final_within_original_bounds=canonical_within,
        final_above_last_slice_height=canonical_above,
        moved=moved,
        max_bracket_shrinks=maximum_shrinks,
    )
    return IDMCompositeCoordinateSliceUpdate(updated, final_z.copy(), audit)


def _initialize_composite_state(
    parameters: FloatArray,
    reference_speed: float,
    prior: LogIDMPrior,
    *,
    max_inward_steps: int = 64,
) -> Tuple[FloatArray, FloatArray, IntArray]:
    """Encode one physical initialization into strict open composite support.

    This deterministic map is initialization-only.  Exact physical-boundary or
    round-trip-invalid components move one representable log-space value at a
    time toward the box midpoint.  Recurrent sweeps instead carry the accepted
    composite state directly and never call this helper.
    """

    physical_input = np.asarray(parameters, dtype=float)
    if physical_input.shape != (5,) or not np.all(np.isfinite(physical_input)):
        raise ValueError("parameters must be a finite vector with shape (5,).")
    if np.any(physical_input < prior.lower) or np.any(
        physical_input > prior.upper
    ):
        raise ValueError("parameters must lie within the closed IDM bounds.")
    if (
        isinstance(max_inward_steps, (bool, np.bool_))
        or not isinstance(max_inward_steps, (int, np.integer))
        or max_inward_steps <= 0
    ):
        raise ValueError("max_inward_steps must be a positive integer.")

    eta = np.log(physical_input)
    log_lower = np.log(prior.lower)
    log_upper = np.log(prior.upper)
    midpoint = 0.5 * (log_lower + log_upper)
    inward_steps = np.zeros(5, dtype=np.int64)
    lower_boundary = physical_input <= prior.lower
    upper_boundary = physical_input >= prior.upper
    eta[lower_boundary] = np.nextafter(log_lower[lower_boundary], midpoint[lower_boundary])
    eta[upper_boundary] = np.nextafter(log_upper[upper_boundary], midpoint[upper_boundary])
    inward_steps[lower_boundary | upper_boundary] += 1

    for _ in range(int(max_inward_steps) + 1):
        composite = log_idm_to_composite(eta, reference_speed)
        decoded = decode_supported_composite(composite, reference_speed, prior)
        if decoded is not None:
            _, decoded_parameters = decoded
            return decoded_parameters, composite, inward_steps

        recovered_eta = composite_to_log_idm(composite, reference_speed)
        with np.errstate(over="ignore", under="ignore", invalid="ignore"):
            recovered_parameters = np.exp(recovered_eta)
        move_lower = (~np.isfinite(recovered_parameters)) | (
            recovered_parameters <= prior.lower
        )
        move_upper = (~np.isfinite(recovered_parameters)) | (
            recovered_parameters >= prior.upper
        )
        move = move_lower | move_upper
        if not np.any(move):
            raise FloatingPointError(
                "Unsupported composite initialization had no inward direction."
            )
        if np.any(inward_steps[move] >= int(max_inward_steps)):
            raise FloatingPointError(
                "Composite initialization exhausted its inward ULP limit."
            )
        eta[move] = np.nextafter(eta[move], midpoint[move])
        inward_steps[move] += 1

    raise FloatingPointError(
        "Composite initialization did not enter strict open support."
    )


def _validate_reference_speed(reference_speed: float) -> float:
    value = float(reference_speed)
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError("reference_speed must be finite and strictly positive.")
    return value


def _unit_uniform(rng: np.random.Generator, purpose: str) -> float:
    draw = float(rng.random())
    if not np.isfinite(draw) or draw < 0.0 or draw >= 1.0:
        raise FloatingPointError(
            f"The {purpose} uniform must be finite and lie in [0, 1)."
        )
    return draw


__all__ = [
    "IDMCompositeCoordinateSliceAudit",
    "IDMCompositeCoordinateSliceUpdate",
    "composite_global_bounding_intervals",
    "sample_idm_parameters_composite_coordinate_slice",
]
