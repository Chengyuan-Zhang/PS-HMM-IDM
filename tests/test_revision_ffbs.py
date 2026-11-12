import itertools

import numpy as np
import pytest
from scipy.special import logsumexp

from experiments.revision_20260806.model import (
    ffbs_sample_path,
    ffbs_sample_paths,
    log_forward_filter,
    log_forward_filter_sequences,
)


def _exact_path_posterior(
    initial: np.ndarray,
    transition: np.ndarray,
    log_emission: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float]:
    paths = np.asarray(
        list(itertools.product(range(initial.size), repeat=log_emission.shape[0])),
        dtype=int,
    )
    log_weights = np.empty(paths.shape[0], dtype=float)
    for index, path in enumerate(paths):
        log_weight = np.log(initial[path[0]]) + log_emission[0, path[0]]
        for time_index in range(1, path.size):
            log_weight += np.log(transition[path[time_index - 1], path[time_index]])
            log_weight += log_emission[time_index, path[time_index]]
        log_weights[index] = log_weight
    log_evidence = float(logsumexp(log_weights))
    return paths, np.exp(log_weights - log_evidence), log_evidence


@pytest.fixture
def tiny_hmm() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    initial = np.array([0.65, 0.35])
    transition = np.array([[0.82, 0.18], [0.27, 0.73]])
    emission_probability = np.array(
        [
            [0.75, 0.25],
            [0.35, 0.65],
            [0.60, 0.40],
        ]
    )
    return initial, transition, np.log(emission_probability)


def test_log_forward_filter_matches_exact_enumeration(
    tiny_hmm: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> None:
    initial, transition, log_emission = tiny_hmm
    paths, posterior, log_evidence = _exact_path_posterior(
        initial, transition, log_emission
    )

    result = log_forward_filter(log_emission, initial, transition)
    exact_marginals = np.vstack(
        [
            [posterior[paths[:, time] == state].sum() for state in range(2)]
            for time in range(log_emission.shape[0])
        ]
    )

    assert result.log_likelihood == pytest.approx(log_evidence)
    np.testing.assert_allclose(result.filtered_probabilities[-1], exact_marginals[-1])
    np.testing.assert_allclose(result.filtered_probabilities.sum(axis=1), np.ones(3))


def test_ffbs_empirical_path_distribution_matches_exact_posterior(
    tiny_hmm: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> None:
    initial, transition, log_emission = tiny_hmm
    paths, exact_probability, _ = _exact_path_posterior(
        initial, transition, log_emission
    )
    path_to_index = {tuple(path): index for index, path in enumerate(paths)}
    filtered = log_forward_filter(log_emission, initial, transition)
    rng = np.random.default_rng(20260806)
    counts = np.zeros(paths.shape[0], dtype=int)

    for _ in range(30_000):
        sampled = ffbs_sample_path(filtered, transition, rng)
        counts[path_to_index[tuple(sampled)]] += 1

    empirical_probability = counts / counts.sum()
    np.testing.assert_allclose(empirical_probability, exact_probability, atol=0.01)

    exact_marginals = np.vstack(
        [
            [exact_probability[paths[:, time] == state].sum() for state in range(2)]
            for time in range(log_emission.shape[0])
        ]
    )
    empirical_marginals = np.vstack(
        [
            [
                empirical_probability[paths[:, time] == state].sum()
                for state in range(2)
            ]
            for time in range(log_emission.shape[0])
        ]
    )
    np.testing.assert_allclose(empirical_marginals, exact_marginals, atol=0.01)


def test_multiple_sequences_are_filtered_and_sampled_independently(
    tiny_hmm: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> None:
    initial, transition, log_emission = tiny_hmm
    sequences = [log_emission, log_emission[:2][::-1]]

    filtered = log_forward_filter_sequences(sequences, initial, transition)
    first = ffbs_sample_paths(
        sequences, initial, transition, np.random.default_rng(91)
    )
    second = ffbs_sample_paths(
        sequences, initial, transition, np.random.default_rng(91)
    )

    assert len(filtered) == 2
    assert [result.log_filtered_probabilities.shape for result in filtered] == [
        (3, 2),
        (2, 2),
    ]
    assert [path.shape for path in first] == [(3,), (2,)]
    for first_path, second_path in zip(first, second):
        np.testing.assert_array_equal(first_path, second_path)


def test_batched_equal_length_filter_is_numerically_equivalent_to_sequential(
    tiny_hmm: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> None:
    initial, transition, log_emission = tiny_hmm
    sequences = (
        log_emission,
        log_emission + np.log(np.array([[1.0, 0.8], [0.9, 1.1], [1.2, 0.7]])),
        log_emission[:2],
        log_emission[:2][::-1],
    )
    sequential = tuple(
        log_forward_filter(sequence, initial, transition) for sequence in sequences
    )
    batched = log_forward_filter_sequences(sequences, initial, transition)

    for expected, actual in zip(sequential, batched):
        np.testing.assert_allclose(
            actual.log_filtered_probabilities,
            expected.log_filtered_probabilities,
            rtol=0.0,
            atol=1.0e-15,
        )
        np.testing.assert_allclose(
            actual.log_predictive_state_probabilities,
            expected.log_predictive_state_probabilities,
            rtol=0.0,
            atol=1.0e-15,
        )
        np.testing.assert_allclose(
            actual.log_scaling_factors,
            expected.log_scaling_factors,
            rtol=0.0,
            atol=1.0e-15,
        )
        assert actual.log_likelihood == pytest.approx(
            expected.log_likelihood, rel=0.0, abs=1.0e-15
        )


def test_batched_filter_preserves_ffbs_rng_order_and_stream(
    tiny_hmm: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> None:
    initial, transition, log_emission = tiny_hmm
    sequences = (log_emission, log_emission + 0.1, log_emission[:2])
    batched_rng = np.random.default_rng(744)
    batched_paths = ffbs_sample_paths(
        sequences, initial, transition, batched_rng
    )
    batched_tail = batched_rng.random(8)

    sequential_rng = np.random.default_rng(744)
    sequential_paths = tuple(
        ffbs_sample_path(
            log_forward_filter(sequence, initial, transition),
            transition,
            sequential_rng,
        )
        for sequence in sequences
    )
    sequential_tail = sequential_rng.random(8)

    for batched_path, sequential_path in zip(batched_paths, sequential_paths):
        np.testing.assert_array_equal(batched_path, sequential_path)
    np.testing.assert_array_equal(batched_tail, sequential_tail)


def test_filter_rejects_time_step_with_no_supported_state() -> None:
    initial = np.array([0.5, 0.5])
    transition = np.eye(2)
    log_emission = np.array([[0.0, 0.0], [-np.inf, -np.inf]])

    with pytest.raises(ValueError, match="zero posterior support"):
        log_forward_filter(log_emission, initial, transition)
