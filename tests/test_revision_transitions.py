import numpy as np
import pytest

from experiments.revision_20260806.model import (
    ContextConditionedTransitions,
    JointStateSpace,
    build_context_conditioned_transition,
    build_full_joint_transition,
    build_independent_transition,
    context_conditioned_transition_parameter_count,
    count_context_conditioned_transitions,
    count_initial_states,
    count_joint_transitions,
    dirichlet_prior_parameters,
    full_joint_transition_parameter_count,
    independent_transition_parameter_count,
    sample_dirichlet_posterior_rows,
    sample_initial_distribution,
)


def test_joint_state_index_round_trip() -> None:
    state_space = JointStateSpace(3, 2)

    for behavior in range(3):
        for scenario in range(2):
            joint = state_space.encode(behavior, scenario)
            assert joint == behavior * 2 + scenario
            assert state_space.decode(joint) == (behavior, scenario)

    behavior, scenario = state_space.decode_array(np.arange(6))
    np.testing.assert_array_equal(behavior, np.array([0, 0, 1, 1, 2, 2]))
    np.testing.assert_array_equal(scenario, np.array([0, 1, 0, 1, 0, 1]))
    np.testing.assert_array_equal(
        state_space.encode_array(behavior, scenario), np.arange(6)
    )


def test_independent_transition_builder_uses_both_factors() -> None:
    state_space = JointStateSpace(2, 2)
    behavior = np.array([[0.8, 0.2], [0.3, 0.7]])
    scenario = np.array([[0.6, 0.4], [0.1, 0.9]])

    joint = build_independent_transition(behavior, scenario, state_space)

    assert np.allclose(joint.sum(axis=1), 1.0)
    current = state_space.encode(0, 1)
    following = state_space.encode(1, 0)
    assert joint[current, following] == pytest.approx(behavior[0, 1] * scenario[1, 0])


def test_context_conditioned_transition_uses_current_scenario() -> None:
    state_space = JointStateSpace(2, 2)
    scenario = np.array([[0.7, 0.3], [0.4, 0.6]])
    behavior_by_context = np.array(
        [
            [[0.9, 0.1], [0.2, 0.8]],
            [[0.4, 0.6], [0.7, 0.3]],
        ]
    )

    joint = build_context_conditioned_transition(
        scenario, behavior_by_context, state_space
    )
    model = ContextConditionedTransitions(
        state_space, scenario, behavior_by_context
    )

    current = state_space.encode(0, 1)
    following = state_space.encode(1, 0)
    expected = scenario[1, 0] * behavior_by_context[1, 0, 1]
    assert joint[current, following] == pytest.approx(expected)
    np.testing.assert_allclose(model.joint_matrix(), joint)
    np.testing.assert_allclose(joint.sum(axis=1), 1.0)


def test_full_joint_transition_builder_validates_and_copies() -> None:
    state_space = JointStateSpace(2, 1)
    source = np.array([[0.8, 0.2], [0.1, 0.9]])

    joint = build_full_joint_transition(source, state_space)
    source[0, 0] = 0.0

    np.testing.assert_allclose(joint, np.array([[0.8, 0.2], [0.1, 0.9]]))


def test_joint_state_space_rejects_noninteger_dimensions() -> None:
    with pytest.raises(ValueError, match="integers"):
        JointStateSpace(2.5, 2)  # type: ignore[arg-type]


def test_transition_counts_do_not_cross_trajectory_boundaries() -> None:
    paths = [np.array([0, 1]), np.array([2, 0])]

    counts = count_joint_transitions(paths, num_states=3)

    expected = np.zeros((3, 3), dtype=int)
    expected[0, 1] = 1
    expected[2, 0] = 1
    np.testing.assert_array_equal(counts, expected)
    assert counts[1, 2] == 0


def test_initial_distribution_uses_one_state_per_trajectory() -> None:
    paths = [np.array([0, 1, 1]), np.array([2, 0]), np.array([2])]

    counts = count_initial_states(paths, num_states=3)

    np.testing.assert_array_equal(counts, np.array([1, 0, 2]))


def test_context_conditioned_counts_use_current_context_only() -> None:
    state_space = JointStateSpace(2, 2)
    path = np.array(
        [
            state_space.encode(0, 1),
            state_space.encode(1, 0),
            state_space.encode(0, 1),
        ]
    )

    scenario_counts, behavior_counts = count_context_conditioned_transitions(
        [path], state_space
    )

    assert scenario_counts[1, 0] == 1
    assert scenario_counts[0, 1] == 1
    assert behavior_counts[1, 0, 1] == 1
    assert behavior_counts[0, 1, 0] == 1
    assert behavior_counts.sum() == 2


def test_dirichlet_total_concentration_has_explicit_row_sum() -> None:
    prior = dirichlet_prior_parameters(
        num_rows=2, num_columns=4, total_concentration=2.0
    )

    np.testing.assert_allclose(prior, np.full((2, 4), 0.5))
    np.testing.assert_allclose(prior.sum(axis=1), np.array([2.0, 2.0]))


def test_dirichlet_posterior_sampling_is_seed_reproducible() -> None:
    counts = np.array([[3, 1], [0, 4]])

    first = sample_dirichlet_posterior_rows(
        counts, total_concentration=1.0, rng=np.random.default_rng(123)
    )
    second = sample_dirichlet_posterior_rows(
        counts, total_concentration=1.0, rng=np.random.default_rng(123)
    )

    np.testing.assert_allclose(first, second)
    np.testing.assert_allclose(first.sum(axis=1), np.ones(2))


def test_initial_distribution_sampling_is_seed_reproducible() -> None:
    paths = [np.array([0, 1]), np.array([1, 0])]

    first = sample_initial_distribution(
        paths, 2, 1.0, np.random.default_rng(77)
    )
    second = sample_initial_distribution(
        paths, 2, 1.0, np.random.default_rng(77)
    )

    np.testing.assert_allclose(first, second)
    assert first.sum() == pytest.approx(1.0)


def test_transition_parameter_counts_match_closed_forms() -> None:
    state_space = JointStateSpace(5, 5)

    assert independent_transition_parameter_count(state_space) == 40
    assert context_conditioned_transition_parameter_count(state_space) == 120
    assert full_joint_transition_parameter_count(state_space) == 600
