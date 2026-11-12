"""Structured transition matrices and explicit Dirichlet updates."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .state_space import JointStateSpace


FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


def _validate_stochastic_matrix(
    matrix: ArrayLike,
    expected_shape: Tuple[int, int],
    name: str,
) -> FloatArray:
    probability = np.asarray(matrix, dtype=float)
    if probability.shape != expected_shape:
        raise ValueError(
            f"{name} must have shape {expected_shape}, got {probability.shape}."
        )
    if not np.all(np.isfinite(probability)) or np.any(probability < 0.0):
        raise ValueError(f"{name} must contain finite non-negative probabilities.")
    if not np.allclose(probability.sum(axis=1), 1.0, atol=1.0e-12):
        raise ValueError(f"Every row of {name} must sum to one.")
    return probability


def _validate_paths(
    paths: Sequence[ArrayLike],
    num_states: int,
) -> Tuple[IntArray, ...]:
    validated = []
    for sequence_index, path_like in enumerate(paths):
        path = np.asarray(path_like)
        if path.ndim != 1 or path.size == 0:
            raise ValueError(
                f"Path {sequence_index} must be a non-empty one-dimensional array."
            )
        if not np.issubdtype(path.dtype, np.integer):
            if not (
                np.issubdtype(path.dtype, np.floating)
                and np.all(np.isfinite(path))
                and np.all(path == np.floor(path))
            ):
                raise ValueError(f"Path {sequence_index} must contain integer states.")
        path = path.astype(np.int64, copy=False)
        if np.any((path < 0) | (path >= num_states)):
            raise ValueError(f"Path {sequence_index} contains an out-of-range state.")
        validated.append(path)
    if not validated:
        raise ValueError("At least one independent path is required.")
    return tuple(validated)


def build_independent_transition(
    behavior_transition: ArrayLike,
    scenario_transition: ArrayLike,
    state_space: JointStateSpace,
) -> FloatArray:
    """Build ``p(b', s' | b, s) = p(b' | b) p(s' | s)``."""

    behavior = _validate_stochastic_matrix(
        behavior_transition,
        (state_space.num_behavior_states, state_space.num_behavior_states),
        "behavior_transition",
    )
    scenario = _validate_stochastic_matrix(
        scenario_transition,
        (state_space.num_scenario_states, state_space.num_scenario_states),
        "scenario_transition",
    )
    joint = np.empty(
        (state_space.num_joint_states, state_space.num_joint_states), dtype=float
    )
    for behavior_current in range(state_space.num_behavior_states):
        for scenario_current in range(state_space.num_scenario_states):
            row = state_space.encode(behavior_current, scenario_current)
            for behavior_next in range(state_space.num_behavior_states):
                for scenario_next in range(state_space.num_scenario_states):
                    column = state_space.encode(behavior_next, scenario_next)
                    joint[row, column] = (
                        behavior[behavior_current, behavior_next]
                        * scenario[scenario_current, scenario_next]
                    )
    return joint


def build_context_conditioned_transition(
    scenario_transition: ArrayLike,
    behavior_transition_by_context: ArrayLike,
    state_space: JointStateSpace,
) -> FloatArray:
    """Build ``p(s' | s) p(b' | b, s)`` using the current context ``s``."""

    scenario = _validate_stochastic_matrix(
        scenario_transition,
        (state_space.num_scenario_states, state_space.num_scenario_states),
        "scenario_transition",
    )
    behavior_by_context = np.asarray(behavior_transition_by_context, dtype=float)
    expected_shape = (
        state_space.num_scenario_states,
        state_space.num_behavior_states,
        state_space.num_behavior_states,
    )
    if behavior_by_context.shape != expected_shape:
        raise ValueError(
            "behavior_transition_by_context must have shape "
            f"{expected_shape}, got {behavior_by_context.shape}."
        )
    for scenario_current in range(state_space.num_scenario_states):
        _validate_stochastic_matrix(
            behavior_by_context[scenario_current],
            (state_space.num_behavior_states, state_space.num_behavior_states),
            f"behavior_transition_by_context[{scenario_current}]",
        )

    joint = np.empty(
        (state_space.num_joint_states, state_space.num_joint_states), dtype=float
    )
    for behavior_current in range(state_space.num_behavior_states):
        for scenario_current in range(state_space.num_scenario_states):
            row = state_space.encode(behavior_current, scenario_current)
            for behavior_next in range(state_space.num_behavior_states):
                for scenario_next in range(state_space.num_scenario_states):
                    column = state_space.encode(behavior_next, scenario_next)
                    joint[row, column] = (
                        scenario[scenario_current, scenario_next]
                        * behavior_by_context[
                            scenario_current, behavior_current, behavior_next
                        ]
                    )
    return joint


def build_full_joint_transition(
    joint_transition: ArrayLike,
    state_space: JointStateSpace,
) -> FloatArray:
    """Validate and copy an unrestricted Cartesian-state transition matrix."""

    joint = _validate_stochastic_matrix(
        joint_transition,
        (state_space.num_joint_states, state_space.num_joint_states),
        "joint_transition",
    )
    return joint.copy()


@dataclass(frozen=True)
class IndependentTransitions:
    """Separate behavior and scenario Markov transition matrices."""

    state_space: JointStateSpace
    behavior: FloatArray
    scenario: FloatArray

    def __post_init__(self) -> None:
        behavior = _validate_stochastic_matrix(
            self.behavior,
            (
                self.state_space.num_behavior_states,
                self.state_space.num_behavior_states,
            ),
            "behavior",
        )
        scenario = _validate_stochastic_matrix(
            self.scenario,
            (
                self.state_space.num_scenario_states,
                self.state_space.num_scenario_states,
            ),
            "scenario",
        )
        object.__setattr__(self, "behavior", behavior)
        object.__setattr__(self, "scenario", scenario)

    def joint_matrix(self) -> FloatArray:
        return build_independent_transition(
            self.behavior, self.scenario, self.state_space
        )

    @property
    def num_free_parameters(self) -> int:
        return independent_transition_parameter_count(self.state_space)


@dataclass(frozen=True)
class ContextConditionedTransitions:
    """Scenario dynamics plus behavior switching conditioned on current context."""

    state_space: JointStateSpace
    scenario: FloatArray
    behavior_by_context: NDArray[np.float64]

    def __post_init__(self) -> None:
        joint = build_context_conditioned_transition(
            self.scenario, self.behavior_by_context, self.state_space
        )
        del joint
        object.__setattr__(self, "scenario", np.asarray(self.scenario, dtype=float))
        object.__setattr__(
            self,
            "behavior_by_context",
            np.asarray(self.behavior_by_context, dtype=float),
        )

    def joint_matrix(self) -> FloatArray:
        return build_context_conditioned_transition(
            self.scenario, self.behavior_by_context, self.state_space
        )

    @property
    def num_free_parameters(self) -> int:
        return context_conditioned_transition_parameter_count(self.state_space)


@dataclass(frozen=True)
class FullJointTransitions:
    """Unrestricted transition matrix on the Cartesian-product state space."""

    state_space: JointStateSpace
    joint: FloatArray

    def __post_init__(self) -> None:
        joint = build_full_joint_transition(self.joint, self.state_space)
        object.__setattr__(self, "joint", joint)

    def joint_matrix(self) -> FloatArray:
        return self.joint.copy()

    @property
    def num_free_parameters(self) -> int:
        return full_joint_transition_parameter_count(self.state_space)


def independent_transition_parameter_count(state_space: JointStateSpace) -> int:
    behavior = state_space.num_behavior_states
    scenario = state_space.num_scenario_states
    return behavior * (behavior - 1) + scenario * (scenario - 1)


def context_conditioned_transition_parameter_count(
    state_space: JointStateSpace,
) -> int:
    behavior = state_space.num_behavior_states
    scenario = state_space.num_scenario_states
    return scenario * (scenario - 1) + scenario * behavior * (behavior - 1)


def full_joint_transition_parameter_count(state_space: JointStateSpace) -> int:
    joint = state_space.num_joint_states
    return joint * (joint - 1)


def initial_distribution_parameter_count(state_space: JointStateSpace) -> int:
    return state_space.num_joint_states - 1


def dirichlet_prior_parameters(
    num_rows: int,
    num_columns: int,
    total_concentration: float,
    base_probabilities: Optional[ArrayLike] = None,
) -> FloatArray:
    """Return row-wise Dirichlet parameters with explicit total concentration.

    ``total_concentration`` is the sum of prior parameters in *each row*.
    Under the default uniform base distribution, each cell therefore receives
    ``total_concentration / num_columns``.
    """

    if num_rows <= 0 or num_columns <= 0:
        raise ValueError("Dirichlet dimensions must be positive.")
    if not np.isfinite(total_concentration) or total_concentration <= 0.0:
        raise ValueError("total_concentration must be finite and positive.")

    if base_probabilities is None:
        base = np.full((num_rows, num_columns), 1.0 / num_columns, dtype=float)
    else:
        base_input = np.asarray(base_probabilities, dtype=float)
        if base_input.shape == (num_columns,):
            base = np.broadcast_to(base_input, (num_rows, num_columns)).copy()
        elif base_input.shape == (num_rows, num_columns):
            base = base_input.copy()
        else:
            raise ValueError(
                "base_probabilities must have shape "
                f"({num_columns},) or ({num_rows}, {num_columns})."
            )
        if not np.all(np.isfinite(base)) or np.any(base <= 0.0):
            raise ValueError("Dirichlet base probabilities must be finite and positive.")
        if not np.allclose(base.sum(axis=1), 1.0, atol=1.0e-12):
            raise ValueError("Every Dirichlet base-probability row must sum to one.")
    return total_concentration * base


def sample_dirichlet_posterior_rows(
    counts: ArrayLike,
    total_concentration: float,
    rng: np.random.Generator,
    base_probabilities: Optional[ArrayLike] = None,
) -> FloatArray:
    """Sample independent stochastic rows from a Dirichlet posterior."""

    count_array = np.asarray(counts, dtype=float)
    if count_array.ndim != 2 or count_array.shape[0] == 0 or count_array.shape[1] == 0:
        raise ValueError("counts must be a non-empty two-dimensional array.")
    if not np.all(np.isfinite(count_array)) or np.any(count_array < 0.0):
        raise ValueError("counts must be finite and non-negative.")
    prior = dirichlet_prior_parameters(
        count_array.shape[0],
        count_array.shape[1],
        total_concentration,
        base_probabilities,
    )
    draws = np.empty_like(prior)
    for row in range(prior.shape[0]):
        draws[row] = rng.dirichlet(prior[row] + count_array[row])
    return draws


def count_joint_transitions(
    paths: Sequence[ArrayLike],
    num_states: int,
) -> IntArray:
    """Count transitions within, never across, independent trajectories."""

    if num_states <= 0:
        raise ValueError("num_states must be positive.")
    validated = _validate_paths(paths, num_states)
    counts = np.zeros((num_states, num_states), dtype=np.int64)
    for path in validated:
        if path.size > 1:
            np.add.at(counts, (path[:-1], path[1:]), 1)
    return counts


def count_initial_states(
    paths: Sequence[ArrayLike],
    num_states: int,
) -> IntArray:
    """Count exactly one initial state from each independent trajectory."""

    if num_states <= 0:
        raise ValueError("num_states must be positive.")
    validated = _validate_paths(paths, num_states)
    return np.bincount(
        np.asarray([path[0] for path in validated], dtype=np.int64),
        minlength=num_states,
    ).astype(np.int64, copy=False)


def sample_initial_distribution(
    paths: Sequence[ArrayLike],
    num_states: int,
    total_concentration: float,
    rng: np.random.Generator,
    base_probabilities: Optional[ArrayLike] = None,
) -> FloatArray:
    """Sample the initial-state distribution from its Dirichlet posterior."""

    counts = count_initial_states(paths, num_states).reshape(1, num_states)
    return sample_dirichlet_posterior_rows(
        counts, total_concentration, rng, base_probabilities
    )[0]


def count_independent_transitions(
    paths: Sequence[ArrayLike],
    state_space: JointStateSpace,
) -> Tuple[IntArray, IntArray]:
    """Return behavior and scenario counts for the independent model."""

    validated = _validate_paths(paths, state_space.num_joint_states)
    behavior_counts = np.zeros(
        (state_space.num_behavior_states, state_space.num_behavior_states),
        dtype=np.int64,
    )
    scenario_counts = np.zeros(
        (state_space.num_scenario_states, state_space.num_scenario_states),
        dtype=np.int64,
    )
    for path in validated:
        behavior, scenario = state_space.decode_array(path)
        if path.size > 1:
            np.add.at(behavior_counts, (behavior[:-1], behavior[1:]), 1)
            np.add.at(scenario_counts, (scenario[:-1], scenario[1:]), 1)
    return behavior_counts, scenario_counts


def count_context_conditioned_transitions(
    paths: Sequence[ArrayLike],
    state_space: JointStateSpace,
) -> Tuple[IntArray, IntArray]:
    """Return scenario counts and ``(current s, current b, next b)`` counts."""

    validated = _validate_paths(paths, state_space.num_joint_states)
    scenario_counts = np.zeros(
        (state_space.num_scenario_states, state_space.num_scenario_states),
        dtype=np.int64,
    )
    behavior_counts = np.zeros(
        (
            state_space.num_scenario_states,
            state_space.num_behavior_states,
            state_space.num_behavior_states,
        ),
        dtype=np.int64,
    )
    for path in validated:
        behavior, scenario = state_space.decode_array(path)
        if path.size > 1:
            np.add.at(scenario_counts, (scenario[:-1], scenario[1:]), 1)
            np.add.at(
                behavior_counts,
                (scenario[:-1], behavior[:-1], behavior[1:]),
                1,
            )
    return scenario_counts, behavior_counts


def sample_independent_transitions(
    paths: Sequence[ArrayLike],
    state_space: JointStateSpace,
    behavior_total_concentration: float,
    scenario_total_concentration: float,
    rng: np.random.Generator,
) -> IndependentTransitions:
    behavior_counts, scenario_counts = count_independent_transitions(
        paths, state_space
    )
    behavior = sample_dirichlet_posterior_rows(
        behavior_counts, behavior_total_concentration, rng
    )
    scenario = sample_dirichlet_posterior_rows(
        scenario_counts, scenario_total_concentration, rng
    )
    return IndependentTransitions(state_space, behavior, scenario)


def sample_context_conditioned_transitions(
    paths: Sequence[ArrayLike],
    state_space: JointStateSpace,
    behavior_total_concentration: float,
    scenario_total_concentration: float,
    rng: np.random.Generator,
) -> ContextConditionedTransitions:
    scenario_counts, behavior_counts = count_context_conditioned_transitions(
        paths, state_space
    )
    scenario = sample_dirichlet_posterior_rows(
        scenario_counts, scenario_total_concentration, rng
    )
    behavior_by_context = np.empty_like(behavior_counts, dtype=float)
    for context in range(state_space.num_scenario_states):
        behavior_by_context[context] = sample_dirichlet_posterior_rows(
            behavior_counts[context], behavior_total_concentration, rng
        )
    return ContextConditionedTransitions(
        state_space, scenario, behavior_by_context
    )


def sample_full_joint_transitions(
    paths: Sequence[ArrayLike],
    state_space: JointStateSpace,
    total_concentration: float,
    rng: np.random.Generator,
) -> FullJointTransitions:
    counts = count_joint_transitions(paths, state_space.num_joint_states)
    joint = sample_dirichlet_posterior_rows(counts, total_concentration, rng)
    return FullJointTransitions(state_space, joint)
