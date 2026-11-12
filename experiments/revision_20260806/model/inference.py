"""Log-space filtering and exact forward-filter backward-sampling."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.special import logsumexp


FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


@dataclass(frozen=True)
class ForwardFilterResult:
    """Normalized log probabilities and evidence from one trajectory."""

    log_filtered_probabilities: FloatArray
    log_predictive_state_probabilities: FloatArray
    log_scaling_factors: FloatArray
    log_likelihood: float

    @property
    def filtered_probabilities(self) -> FloatArray:
        return np.exp(self.log_filtered_probabilities)

    @property
    def predictive_state_probabilities(self) -> FloatArray:
        return np.exp(self.log_predictive_state_probabilities)


def _validate_probability_vector(
    probability: ArrayLike,
    num_states: int,
    name: str,
) -> FloatArray:
    vector = np.asarray(probability, dtype=float)
    if vector.shape != (num_states,):
        raise ValueError(f"{name} must have shape ({num_states},).")
    if not np.all(np.isfinite(vector)) or np.any(vector < 0.0):
        raise ValueError(f"{name} must contain finite non-negative probabilities.")
    if not np.isclose(vector.sum(), 1.0, atol=1.0e-12):
        raise ValueError(f"{name} must sum to one.")
    return vector


def _validate_transition(transition_probability: ArrayLike) -> FloatArray:
    transition = np.asarray(transition_probability, dtype=float)
    if transition.ndim != 2 or transition.shape[0] != transition.shape[1]:
        raise ValueError("transition_probability must be a square matrix.")
    if not np.all(np.isfinite(transition)) or np.any(transition < 0.0):
        raise ValueError(
            "transition_probability must contain finite non-negative values."
        )
    if not np.allclose(transition.sum(axis=1), 1.0, atol=1.0e-12):
        raise ValueError("Every transition-probability row must sum to one.")
    return transition


def _validate_log_emission(
    log_emission_likelihood: ArrayLike,
    num_states: int,
) -> FloatArray:
    emission = np.asarray(log_emission_likelihood, dtype=float)
    if emission.ndim != 2 or emission.shape[1] != num_states or emission.shape[0] == 0:
        raise ValueError(
            "log_emission_likelihood must have shape "
            f"(positive_time, {num_states})."
        )
    if np.any(np.isnan(emission)) or np.any(np.isposinf(emission)):
        raise ValueError("Log emission likelihoods may be finite or -inf, not NaN/+inf.")
    return emission


def _log_probability(probability: FloatArray) -> FloatArray:
    with np.errstate(divide="ignore"):
        return np.log(probability)


def log_forward_filter(
    log_emission_likelihood: ArrayLike,
    initial_probability: ArrayLike,
    transition_probability: ArrayLike,
) -> ForwardFilterResult:
    """Run a normalized forward filter entirely in log space."""

    transition = _validate_transition(transition_probability)
    num_states = transition.shape[0]
    initial = _validate_probability_vector(
        initial_probability, num_states, "initial_probability"
    )
    emission = _validate_log_emission(log_emission_likelihood, num_states)

    num_time = emission.shape[0]
    log_transition = _log_probability(transition)
    log_filtered = np.empty((num_time, num_states), dtype=float)
    log_predictive = np.empty((num_time, num_states), dtype=float)
    log_scaling = np.empty(num_time, dtype=float)

    log_predictive[0] = _log_probability(initial)
    unnormalized = log_predictive[0] + emission[0]
    log_scaling[0] = logsumexp(unnormalized)
    if not np.isfinite(log_scaling[0]):
        raise ValueError("All states have zero posterior support at time 0.")
    log_filtered[0] = unnormalized - log_scaling[0]

    for time_index in range(1, num_time):
        log_predictive[time_index] = logsumexp(
            log_filtered[time_index - 1, :, np.newaxis] + log_transition,
            axis=0,
        )
        unnormalized = log_predictive[time_index] + emission[time_index]
        log_scaling[time_index] = logsumexp(unnormalized)
        if not np.isfinite(log_scaling[time_index]):
            raise ValueError(
                f"All states have zero posterior support at time {time_index}."
            )
        log_filtered[time_index] = unnormalized - log_scaling[time_index]

    return ForwardFilterResult(
        log_filtered_probabilities=log_filtered,
        log_predictive_state_probabilities=log_predictive,
        log_scaling_factors=log_scaling,
        log_likelihood=float(log_scaling.sum()),
    )


def log_forward_filter_sequences(
    log_emission_sequences: Sequence[ArrayLike],
    initial_probability: ArrayLike,
    transition_probability: ArrayLike,
) -> Tuple[ForwardFilterResult, ...]:
    """Filter independent trajectories without crossing sequence boundaries.

    Sequences with the same length are evaluated as one NumPy batch.  This is
    algebraically identical to calling :func:`log_forward_filter` one sequence
    at a time, but avoids repeating thousands of small Python/SciPy calls for a
    cohort of equal-length car-following segments.
    """

    if len(log_emission_sequences) == 0:
        raise ValueError("At least one emission sequence is required.")
    transition = _validate_transition(transition_probability)
    num_states = transition.shape[0]
    initial = _validate_probability_vector(
        initial_probability, num_states, "initial_probability"
    )
    emissions = tuple(
        _validate_log_emission(sequence, num_states)
        for sequence in log_emission_sequences
    )
    groups: Dict[int, List[int]] = {}
    for sequence_index, emission in enumerate(emissions):
        groups.setdefault(emission.shape[0], []).append(sequence_index)

    results: List[Optional[ForwardFilterResult]] = [None] * len(emissions)
    log_transition = _log_probability(transition)
    log_initial = _log_probability(initial)
    for time_count, sequence_indices in groups.items():
        batch = np.stack([emissions[index] for index in sequence_indices], axis=0)
        batch_size = batch.shape[0]
        log_filtered = np.empty((batch_size, time_count, num_states), dtype=float)
        log_predictive = np.empty_like(log_filtered)
        log_scaling = np.empty((batch_size, time_count), dtype=float)

        log_predictive[:, 0, :] = log_initial
        unnormalized = log_predictive[:, 0, :] + batch[:, 0, :]
        log_scaling[:, 0] = logsumexp(unnormalized, axis=1)
        invalid = np.flatnonzero(~np.isfinite(log_scaling[:, 0]))
        if invalid.size:
            sequence_index = sequence_indices[int(invalid[0])]
            raise ValueError(
                "All states have zero posterior support at time 0 in sequence "
                f"{sequence_index}."
            )
        log_filtered[:, 0, :] = unnormalized - log_scaling[:, 0, np.newaxis]

        for time_index in range(1, time_count):
            log_predictive[:, time_index, :] = logsumexp(
                log_filtered[:, time_index - 1, :, np.newaxis]
                + log_transition[np.newaxis, :, :],
                axis=1,
            )
            unnormalized = log_predictive[:, time_index, :] + batch[:, time_index, :]
            log_scaling[:, time_index] = logsumexp(unnormalized, axis=1)
            invalid = np.flatnonzero(~np.isfinite(log_scaling[:, time_index]))
            if invalid.size:
                sequence_index = sequence_indices[int(invalid[0])]
                raise ValueError(
                    "All states have zero posterior support at time "
                    f"{time_index} in sequence {sequence_index}."
                )
            log_filtered[:, time_index, :] = (
                unnormalized - log_scaling[:, time_index, np.newaxis]
            )

        for batch_index, sequence_index in enumerate(sequence_indices):
            results[sequence_index] = ForwardFilterResult(
                log_filtered_probabilities=log_filtered[batch_index],
                log_predictive_state_probabilities=log_predictive[batch_index],
                log_scaling_factors=log_scaling[batch_index],
                log_likelihood=float(log_scaling[batch_index].sum()),
            )
    if any(result is None for result in results):
        raise AssertionError("A validated emission sequence was not filtered.")
    return tuple(result for result in results if result is not None)


def _sample_from_log_probabilities(
    log_probability: FloatArray,
    rng: np.random.Generator,
) -> int:
    log_normalizer = logsumexp(log_probability)
    if not np.isfinite(log_normalizer):
        raise ValueError("Cannot sample from a distribution with zero support.")
    probability = np.exp(log_probability - log_normalizer)
    return int(rng.choice(probability.size, p=probability))


def ffbs_sample_path(
    filter_result: ForwardFilterResult,
    transition_probability: ArrayLike,
    rng: np.random.Generator,
) -> IntArray:
    """Draw one complete path from its joint smoothing distribution."""

    transition = _validate_transition(transition_probability)
    log_filtered = np.asarray(filter_result.log_filtered_probabilities, dtype=float)
    if log_filtered.ndim != 2 or log_filtered.shape[1] != transition.shape[0]:
        raise ValueError("filter_result and transition_probability are incompatible.")
    log_transition = _log_probability(transition)
    path = np.empty(log_filtered.shape[0], dtype=np.int64)
    path[-1] = _sample_from_log_probabilities(log_filtered[-1], rng)
    for time_index in range(path.size - 2, -1, -1):
        backward_log_probability = (
            log_filtered[time_index] + log_transition[:, path[time_index + 1]]
        )
        path[time_index] = _sample_from_log_probabilities(
            backward_log_probability, rng
        )
    return path


def ffbs_sample_paths(
    log_emission_sequences: Sequence[ArrayLike],
    initial_probability: ArrayLike,
    transition_probability: ArrayLike,
    rng: np.random.Generator,
) -> Tuple[IntArray, ...]:
    """Filter and jointly sample each independent latent-state trajectory."""

    filtered = log_forward_filter_sequences(
        log_emission_sequences, initial_probability, transition_probability
    )
    return tuple(
        ffbs_sample_path(result, transition_probability, rng)
        for result in filtered
    )
