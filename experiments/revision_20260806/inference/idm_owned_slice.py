"""Full-bracket slice updates for authoritative owned log-IDM parameters.

The Markov-chain state in this module is the reduced vector ``psi`` of unique
parameter owners defined by :mod:`idm_ownership`.  Expanded physical IDM
parameters are decoded views used only by the emission likelihood.  The prior
is consequently evaluated once per unique owner, including shared owners.

Every scalar update starts from its complete finite prior-bound interval and
shrinks around the current coordinate.  Physical bounds are strict-open for
the recurrent chain.  Closed-boundary physical values are canonicalized only
when initializing without an authoritative ``psi``; recurrent sweeps never
project, clip, collapse, or re-encode their state.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ..model import JointStateSpace
from .emissions import idm_acceleration, normal_log_density
from .idm_block import _behavior_observations, _validate_kernel_inputs
from .idm_ownership import IDMOwnershipLayout
from .structures import LogIDMPrior, Trajectory
from .updates import log_idm_state_prior


FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]
IntArray = NDArray[np.int64]


@dataclass(frozen=True)
class OwnedLogIDMSliceAudit:
    """Complete transcript for one fixed-order owned-coordinate sweep.

    Proposal-level arrays are flattened in owner order.  Owner ``u`` occupies
    ``coordinate_proposal_offsets[u]:coordinate_proposal_offsets[u + 1]``.
    The recorded bracket is the bracket before its proposal.  Full current and
    accepted authoritative vectors, together with their decoded physical
    views, make shared-owner effects and coordinate continuity replayable.
    """

    layout_name: str
    owner_index: IntArray
    owner_parameter_index: IntArray
    owner_behavior_index: IntArray
    affected_behavior_mask: BoolArray
    coordinate_owner_order: IntArray
    input_parameters: FloatArray
    canonical_initial_parameters: FloatArray
    initialized_from_physical: bool
    initial_state_adjusted: bool
    initial_boundary_canonicalized: bool
    initialization_inward_steps: IntArray
    initial_owned_log_parameters: FloatArray
    final_owned_log_parameters: FloatArray
    coordinate_current_owned_log_parameters: FloatArray
    coordinate_accepted_owned_log_parameters: FloatArray
    coordinate_current_parameters: FloatArray
    coordinate_accepted_parameters: FloatArray
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
    final_log_target: float
    final_within_original_bounds: bool
    final_above_last_slice_height: bool
    moved: bool
    max_bracket_shrinks: int


@dataclass(frozen=True)
class OwnedLogIDMSliceUpdate:
    """Updated expanded physical view and authoritative owned log state."""

    parameters: FloatArray
    owned_log_parameters: FloatArray
    audit: OwnedLogIDMSliceAudit

    @property
    def moved(self) -> bool:
        """Return whether any authoritative owner changed in this sweep."""

        return self.audit.moved


def owned_log_global_bounding_intervals(
    prior: LogIDMPrior, layout: IDMOwnershipLayout
) -> Tuple[FloatArray, FloatArray]:
    """Return outward-rounded finite log intervals for all unique owners."""

    parameter_index = layout.owner_parameter_index
    lower = np.nextafter(np.log(prior.lower[parameter_index]), -np.inf)
    upper = np.nextafter(np.log(prior.upper[parameter_index]), np.inf)
    if (
        not np.all(np.isfinite(lower))
        or not np.all(np.isfinite(upper))
        or np.any(upper <= lower)
    ):
        raise FloatingPointError("The owned log-IDM intervals are invalid.")
    return np.asarray(lower, dtype=float), np.asarray(upper, dtype=float)


def decode_supported_owned_log(
    owned_log_parameters: ArrayLike,
    prior: LogIDMPrior,
    layout: IDMOwnershipLayout,
) -> Optional[Tuple[FloatArray, FloatArray]]:
    """Decode an authoritative state only when it is in strict-open support.

    Returns ``(expanded_log_parameters, physical_parameters)`` or ``None``.
    This function never clips, projects, collapses, or re-encodes ``psi``.
    """

    expanded_log = layout.expand_log(owned_log_parameters)
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        physical = np.exp(expanded_log)
    if (
        not np.all(np.isfinite(physical))
        or np.any(physical <= prior.lower[None, :])
        or np.any(physical >= prior.upper[None, :])
    ):
        return None
    return expanded_log, np.asarray(physical, dtype=float)


def owned_log_within_original_bounds(
    owned_log_parameters: ArrayLike,
    prior: LogIDMPrior,
    layout: IDMOwnershipLayout,
) -> bool:
    """Return whether ``psi`` decodes strictly inside all physical bounds."""

    return decode_supported_owned_log(
        owned_log_parameters, prior, layout
    ) is not None


def owned_log_idm_target(
    owned_log_parameters: ArrayLike,
    residual_variance: ArrayLike,
    trajectories: Sequence[Trajectory],
    paths: Sequence[IntArray],
    state_space: JointStateSpace,
    prior: LogIDMPrior,
    layout: IDMOwnershipLayout,
    *,
    hierarchy_mean: Optional[ArrayLike] = None,
    hierarchy_precision: Optional[ArrayLike] = None,
) -> float:
    """Evaluate the fixed-path posterior target on bounded log support.

    The fixed-independent model retains one Gaussian prior per unique owner.
    The optional manuscript hierarchy is supported only for the original
    behavior-specific layout and contributes one multivariate Gaussian density
    per response state. Unsupported points have log density ``-inf``.
    """

    _validate_layout(layout, state_space)
    probe = np.broadcast_to(
        np.sqrt(prior.lower * prior.upper),
        (state_space.num_behavior_states, 5),
    ).copy()
    _, variance, trajectory_tuple, path_tuple = _validate_kernel_inputs(
        probe,
        residual_variance,
        trajectories,
        paths,
        state_space,
        prior,
    )
    decoded = decode_supported_owned_log(
        owned_log_parameters, prior, layout
    )
    if decoded is None:
        return float("-inf")
    _, physical = decoded
    behavior_data = _collect_behavior_data(
        trajectory_tuple, path_tuple, state_space
    )
    owned = np.asarray(owned_log_parameters, dtype=float)
    return _complete_owned_log_target(
        owned,
        physical,
        variance,
        behavior_data,
        prior,
        layout,
        hierarchy_mean=hierarchy_mean,
        hierarchy_precision=hierarchy_precision,
    )


def full_bracket_owned_log_idm_slice_update(
    current_parameters: ArrayLike,
    residual_variance: ArrayLike,
    trajectories: Sequence[Trajectory],
    paths: Sequence[IntArray],
    state_space: JointStateSpace,
    prior: LogIDMPrior,
    layout: IDMOwnershipLayout,
    rng: np.random.Generator,
    *,
    hierarchy_mean: Optional[ArrayLike] = None,
    hierarchy_precision: Optional[ArrayLike] = None,
    current_owned_log_parameters: Optional[ArrayLike] = None,
    max_bracket_shrinks: int = 10_000,
) -> OwnedLogIDMSliceUpdate:
    """Apply one fixed-order, full-bracket scalar sweep over unique owners.

    With no ``current_owned_log_parameters``, the physical input is collapsed
    once and closed-boundary values are moved inward by representable log-space
    steps.  When an authoritative state is supplied, it must decode bitwise to
    ``current_parameters`` and is used directly.  Thus recurrent transitions
    contain no physical-to-owner projection.

    Reaching ``max_bracket_shrinks`` raises before any additional shrink and
    returns no fallback state.
    """

    if (
        isinstance(max_bracket_shrinks, (bool, np.bool_))
        or not isinstance(max_bracket_shrinks, (int, np.integer))
        or max_bracket_shrinks < 0
    ):
        raise ValueError("max_bracket_shrinks must be a non-negative integer.")
    maximum_shrinks = int(max_bracket_shrinks)
    _validate_layout(layout, state_space)
    input_parameters, variance, trajectory_tuple, path_tuple = (
        _validate_kernel_inputs(
            current_parameters,
            residual_variance,
            trajectories,
            paths,
            state_space,
            prior,
        )
    )
    input_parameters = input_parameters.copy()
    behavior_data = _collect_behavior_data(
        trajectory_tuple, path_tuple, state_space
    )
    global_lower, global_upper = owned_log_global_bounding_intervals(
        prior, layout
    )

    initialized_from_physical = current_owned_log_parameters is None
    initialization_inward_steps = np.zeros(
        layout.num_owned_parameters, dtype=np.int64
    )
    if initialized_from_physical:
        canonical_initial, initial_owned, initialization_inward_steps = (
            _initialize_owned_log_state(input_parameters, prior, layout)
        )
    else:
        supplied = np.asarray(current_owned_log_parameters, dtype=float)
        expected = (layout.num_owned_parameters,)
        if supplied.shape != expected or not np.all(np.isfinite(supplied)):
            raise ValueError(
                "current_owned_log_parameters must be finite with shape "
                f"{expected}."
            )
        decoded = decode_supported_owned_log(supplied, prior, layout)
        if decoded is None:
            raise ValueError(
                "current_owned_log_parameters must lie strictly inside the "
                "original physical IDM bounds."
            )
        _, decoded_parameters = decoded
        if not np.array_equal(decoded_parameters, input_parameters):
            raise ValueError(
                "current_parameters must exactly match the physical state "
                "decoded from current_owned_log_parameters."
            )
        initial_owned = supplied.copy()
        canonical_initial = decoded_parameters.copy()

    initial_state_adjusted = not np.array_equal(
        canonical_initial, input_parameters
    )
    initial_boundary_canonicalized = bool(
        np.any(initialization_inward_steps > 0)
    )
    owner_count = layout.num_owned_parameters
    final_owned = initial_owned.copy()
    coordinate_current = np.empty((owner_count, owner_count), dtype=float)
    coordinate_accepted = np.empty_like(coordinate_current)
    coordinate_current_parameters = np.empty(
        (owner_count, layout.num_behavior_states, 5), dtype=float
    )
    coordinate_accepted_parameters = np.empty_like(
        coordinate_current_parameters
    )
    slice_uniform = np.empty(owner_count, dtype=float)
    log_slice_uniform = np.empty(owner_count, dtype=float)
    current_target = np.empty(owner_count, dtype=float)
    slice_height = np.empty(owner_count, dtype=float)
    accepted_value = np.empty(owner_count, dtype=float)
    accepted_target = np.empty(owner_count, dtype=float)
    accepted_within = np.zeros(owner_count, dtype=bool)
    accepted_above = np.zeros(owner_count, dtype=bool)
    proposal_count = np.zeros(owner_count, dtype=np.int64)
    shrink_count = np.zeros(owner_count, dtype=np.int64)
    bound_rejections = np.zeros(owner_count, dtype=np.int64)
    nonfinite_rejections = np.zeros(owner_count, dtype=np.int64)

    proposal_uniforms: List[float] = []
    proposal_values: List[float] = []
    proposal_lower: List[float] = []
    proposal_upper: List[float] = []
    proposal_within: List[bool] = []
    proposal_targets: List[float] = []
    proposal_acceptance: List[bool] = []
    offsets: List[int] = [0]

    for owner in range(owner_count):
        coordinate_current[owner] = final_owned
        current_decoded = decode_supported_owned_log(
            final_owned, prior, layout
        )
        if current_decoded is None:
            raise FloatingPointError(
                "The current authoritative owned log state is unsupported."
            )
        _, current_physical = current_decoded
        coordinate_current_parameters[owner] = current_physical
        log_target_zero = _complete_owned_log_target(
            final_owned,
            current_physical,
            variance,
            behavior_data,
            prior,
            layout,
            hierarchy_mean=hierarchy_mean,
            hierarchy_precision=hierarchy_precision,
        )
        if not np.isfinite(log_target_zero):
            raise FloatingPointError(
                "The current owned log-IDM slice target must be finite."
            )
        current_target[owner] = log_target_zero

        height_uniform = _unit_uniform(rng, "slice-height")
        slice_uniform[owner] = height_uniform
        with np.errstate(divide="ignore", invalid="ignore"):
            log_height_uniform = float(np.log(height_uniform))
        log_slice_uniform[owner] = log_height_uniform
        height = float(log_target_zero + log_height_uniform)
        if np.isnan(height) or np.isposinf(height):
            raise FloatingPointError(
                "The owned log-IDM slice height is invalid."
            )
        slice_height[owner] = height

        lower = float(global_lower[owner])
        upper = float(global_upper[owner])
        anchor = float(final_owned[owner])
        if not lower <= anchor <= upper:
            raise FloatingPointError(
                "The current owned coordinate is outside its global interval."
            )

        accepted_owned: Optional[FloatArray] = None
        accepted_physical: Optional[FloatArray] = None
        accepted_log_target = float("nan")
        while True:
            proposal_uniform = _unit_uniform(rng, "full-bracket proposal")
            value = float(lower + proposal_uniform * (upper - lower))
            if not np.isfinite(value) or value < lower or value > upper:
                raise FloatingPointError(
                    "The full-bracket owned proposal is invalid."
                )
            proposed_owned = final_owned.copy()
            proposed_owned[owner] = value

            proposal_uniforms.append(proposal_uniform)
            proposal_values.append(value)
            proposal_lower.append(lower)
            proposal_upper.append(upper)
            proposed_decoded = decode_supported_owned_log(
                proposed_owned, prior, layout
            )
            is_within = proposed_decoded is not None
            proposal_within.append(is_within)
            if proposed_decoded is None:
                proposed_target = float("-inf")
                proposed_physical = None
                bound_rejections[owner] += 1
            else:
                _, proposed_physical = proposed_decoded
                proposed_target = _complete_owned_log_target(
                    proposed_owned,
                    proposed_physical,
                    variance,
                    behavior_data,
                    prior,
                    layout,
                    hierarchy_mean=hierarchy_mean,
                    hierarchy_precision=hierarchy_precision,
                )
                if np.isnan(proposed_target) or np.isposinf(proposed_target):
                    raise FloatingPointError(
                        "The owned log-IDM proposal target is invalid."
                    )
                if np.isneginf(proposed_target):
                    nonfinite_rejections[owner] += 1

            is_accepted = bool(
                is_within
                and np.isfinite(proposed_target)
                and proposed_target >= height
            )
            proposal_targets.append(float(proposed_target))
            proposal_acceptance.append(is_accepted)
            proposal_count[owner] += 1
            if is_accepted:
                if proposed_physical is None:
                    raise RuntimeError(
                        "An accepted owned proposal lacks a physical decode."
                    )
                accepted_owned = proposed_owned
                accepted_physical = proposed_physical
                accepted_log_target = float(proposed_target)
                break

            if shrink_count[owner] >= maximum_shrinks:
                raise RuntimeError(
                    "Owned log-IDM coordinate slice bracket shrink limit was "
                    "exhausted."
                )
            if value < anchor:
                lower = value
            elif value > anchor:
                upper = value
            else:
                raise FloatingPointError(
                    "The current owned coordinate was rejected below its own "
                    "slice height."
                )
            if not lower <= anchor <= upper:
                raise FloatingPointError(
                    "The owned coordinate slice bracket lost its anchor."
                )
            shrink_count[owner] += 1

        if accepted_owned is None or accepted_physical is None:
            raise RuntimeError(
                "Owned coordinate slice terminated without an accepted proposal."
            )
        coordinate_accepted[owner] = accepted_owned
        coordinate_accepted_parameters[owner] = accepted_physical
        accepted_value[owner] = accepted_owned[owner]
        accepted_target[owner] = accepted_log_target
        accepted_within[owner] = bool(
            owned_log_within_original_bounds(accepted_owned, prior, layout)
        )
        accepted_above[owner] = bool(accepted_log_target >= height)
        if not accepted_within[owner] or not accepted_above[owner]:
            raise FloatingPointError(
                "The accepted owned coordinate failed its support or height audit."
            )
        final_owned = accepted_owned
        offsets.append(len(proposal_values))

    final_decoded = decode_supported_owned_log(final_owned, prior, layout)
    if final_decoded is None:
        raise FloatingPointError(
            "The final authoritative owned log state is unsupported."
        )
    _, final_parameters = final_decoded
    final_log_target = _complete_owned_log_target(
        final_owned,
        final_parameters,
        variance,
        behavior_data,
        prior,
        layout,
        hierarchy_mean=hierarchy_mean,
        hierarchy_precision=hierarchy_precision,
    )
    if not np.isfinite(final_log_target):
        raise FloatingPointError("The final owned log-IDM target is non-finite.")
    if final_log_target != accepted_target[-1]:
        raise FloatingPointError(
            "The final owned log-IDM target differs from the last accepted target."
        )
    final_within = bool(
        owned_log_within_original_bounds(final_owned, prior, layout)
    )
    final_above = bool(final_log_target >= slice_height[-1])
    if not final_within or not final_above:
        raise FloatingPointError(
            "The final owned log-IDM state failed its support or height audit."
        )

    affected_behavior_mask = np.zeros(
        (owner_count, layout.num_behavior_states), dtype=bool
    )
    for owner in range(owner_count):
        affected_behavior_mask[owner, layout.affected_behaviors(owner)] = True

    audit = OwnedLogIDMSliceAudit(
        layout_name=layout.name,
        owner_index=layout.owner_index.copy(),
        owner_parameter_index=layout.owner_parameter_index.copy(),
        owner_behavior_index=layout.owner_behavior_index.copy(),
        affected_behavior_mask=affected_behavior_mask,
        coordinate_owner_order=np.arange(owner_count, dtype=np.int64),
        input_parameters=input_parameters.copy(),
        canonical_initial_parameters=canonical_initial.copy(),
        initialized_from_physical=initialized_from_physical,
        initial_state_adjusted=initial_state_adjusted,
        initial_boundary_canonicalized=initial_boundary_canonicalized,
        initialization_inward_steps=initialization_inward_steps.copy(),
        initial_owned_log_parameters=initial_owned.copy(),
        final_owned_log_parameters=final_owned.copy(),
        coordinate_current_owned_log_parameters=coordinate_current,
        coordinate_accepted_owned_log_parameters=coordinate_accepted,
        coordinate_current_parameters=coordinate_current_parameters,
        coordinate_accepted_parameters=coordinate_accepted_parameters,
        global_interval_lower=global_lower.copy(),
        global_interval_upper=global_upper.copy(),
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
        final_log_target=float(final_log_target),
        final_within_original_bounds=final_within,
        final_above_last_slice_height=final_above,
        moved=not np.array_equal(final_owned, initial_owned),
        max_bracket_shrinks=maximum_shrinks,
    )
    return OwnedLogIDMSliceUpdate(
        final_parameters.copy(), final_owned.copy(), audit
    )


def _initialize_owned_log_state(
    parameters: FloatArray,
    prior: LogIDMPrior,
    layout: IDMOwnershipLayout,
    *,
    max_inward_steps: int = 64,
) -> Tuple[FloatArray, FloatArray, IntArray]:
    """Collapse once and move closed-boundary owners into strict support."""

    if (
        isinstance(max_inward_steps, (bool, np.bool_))
        or not isinstance(max_inward_steps, (int, np.integer))
        or max_inward_steps <= 0
    ):
        raise ValueError("max_inward_steps must be a positive integer.")
    owned = layout.collapse_physical(parameters)
    parameter_index = layout.owner_parameter_index
    log_lower = np.log(prior.lower[parameter_index])
    log_upper = np.log(prior.upper[parameter_index])
    midpoint = 0.5 * (log_lower + log_upper)
    owner_physical = np.empty(layout.num_owned_parameters, dtype=float)
    for owner in range(layout.num_owned_parameters):
        owner_physical[owner] = parameters[layout.owner_index == owner][0]

    inward_steps = np.zeros(layout.num_owned_parameters, dtype=np.int64)
    lower_boundary = owner_physical <= prior.lower[parameter_index]
    upper_boundary = owner_physical >= prior.upper[parameter_index]
    owned[lower_boundary] = np.nextafter(
        log_lower[lower_boundary], midpoint[lower_boundary]
    )
    owned[upper_boundary] = np.nextafter(
        log_upper[upper_boundary], midpoint[upper_boundary]
    )
    inward_steps[lower_boundary | upper_boundary] += 1

    for _ in range(int(max_inward_steps) + 1):
        decoded = decode_supported_owned_log(owned, prior, layout)
        if decoded is not None:
            _, decoded_parameters = decoded
            return decoded_parameters, owned, inward_steps

        with np.errstate(over="ignore", under="ignore", invalid="ignore"):
            recovered_physical = np.exp(owned)
        move_lower = (~np.isfinite(recovered_physical)) | (
            recovered_physical <= prior.lower[parameter_index]
        )
        move_upper = (~np.isfinite(recovered_physical)) | (
            recovered_physical >= prior.upper[parameter_index]
        )
        move = move_lower | move_upper
        if not np.any(move):
            raise FloatingPointError(
                "Unsupported owned initialization had no inward direction."
            )
        if np.any(inward_steps[move] >= int(max_inward_steps)):
            raise FloatingPointError(
                "Owned initialization exhausted its inward ULP limit."
            )
        owned[move] = np.nextafter(owned[move], midpoint[move])
        inward_steps[move] += 1

    raise FloatingPointError(
        "Owned initialization did not enter strict open support."
    )


def _collect_behavior_data(
    trajectories: Tuple[Trajectory, ...],
    paths: Tuple[IntArray, ...],
    state_space: JointStateSpace,
) -> Tuple[Tuple[FloatArray, FloatArray], ...]:
    return tuple(
        _behavior_observations(trajectories, paths, state_space, behavior)
        for behavior in range(state_space.num_behavior_states)
    )


def _complete_owned_log_target(
    owned_log_parameters: FloatArray,
    physical_parameters: FloatArray,
    residual_variance: FloatArray,
    behavior_data: Tuple[Tuple[FloatArray, FloatArray], ...],
    prior: LogIDMPrior,
    layout: IDMOwnershipLayout,
    *,
    hierarchy_mean: Optional[ArrayLike] = None,
    hierarchy_precision: Optional[ArrayLike] = None,
) -> float:
    log_likelihood = 0.0
    for behavior, (context, observed) in enumerate(behavior_data):
        if observed.size:
            predicted = idm_acceleration(
                physical_parameters[behavior], context
            )
            log_likelihood += float(
                normal_log_density(
                    observed,
                    predicted,
                    float(residual_variance[behavior]),
                ).sum()
            )
    if (hierarchy_mean is None) != (hierarchy_precision is None):
        raise ValueError(
            "Hierarchy mean and precision must be supplied together."
        )
    if hierarchy_mean is None:
        parameter_index = layout.owner_parameter_index
        prior_mean = prior.log_mean[parameter_index]
        prior_standard_deviation = prior.log_standard_deviation[
            parameter_index
        ]
        standardized = (
            owned_log_parameters - prior_mean
        ) / prior_standard_deviation
        unique_log_prior = float(
            -0.5 * np.sum(standardized * standardized)
            - np.log(prior_standard_deviation).sum()
            - 0.5 * owned_log_parameters.size * np.log(2.0 * np.pi)
        )
    else:
        if layout.name != "behavior_specific":
            raise ValueError(
                "The hierarchical IDM prior is defined only for the original "
                "behavior-specific state vectors."
            )
        expanded_log = layout.expand_log(owned_log_parameters)
        unique_log_prior = float(
            sum(
                log_idm_state_prior(
                    expanded_log[behavior],
                    prior,
                    hierarchy_mean=hierarchy_mean,
                    hierarchy_precision=hierarchy_precision,
                )
                for behavior in range(expanded_log.shape[0])
            )
        )
    return float(log_likelihood + unique_log_prior)


def _validate_layout(
    layout: IDMOwnershipLayout, state_space: JointStateSpace
) -> None:
    if not isinstance(layout, IDMOwnershipLayout):
        raise TypeError("layout must be an IDMOwnershipLayout.")
    if layout.num_behavior_states != state_space.num_behavior_states:
        raise ValueError(
            "The ownership layout and state space have different behavior counts."
        )


def _unit_uniform(rng: np.random.Generator, purpose: str) -> float:
    draw = float(rng.random())
    if not np.isfinite(draw) or draw < 0.0 or draw >= 1.0:
        raise FloatingPointError(
            f"The {purpose} uniform must be finite and lie in [0, 1)."
        )
    return draw


__all__ = [
    "OwnedLogIDMSliceAudit",
    "OwnedLogIDMSliceUpdate",
    "decode_supported_owned_log",
    "full_bracket_owned_log_idm_slice_update",
    "owned_log_global_bounding_intervals",
    "owned_log_idm_target",
    "owned_log_within_original_bounds",
]
