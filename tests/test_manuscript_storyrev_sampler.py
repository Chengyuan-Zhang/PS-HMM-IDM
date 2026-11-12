from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from experiments.manuscript_storyrev_20260808.checkpoint import (
    load_checkpoint,
    write_checkpoint,
)

from experiments.manuscript_storyrev_20260808.sampler import (
    ReconstructionConfig,
    ReconstructionState,
    run_chain,
    sample_scenario_emissions,
    standardize_trajectories,
)
from experiments.revision_20260806.model import (
    JointStateSpace,
    count_initial_states,
    count_joint_transitions,
)


CONFIG_PATH = (
    Path(__file__).resolve().parents[1]
    / "experiments"
    / "manuscript_storyrev_20260808"
    / "configs"
    / "b2s2.json"
)


def _tiny_problem() -> tuple:
    first_context = np.array(
        [
            [10.0, 18.0, -0.5],
            [10.5, 17.5, 0.0],
            [11.0, 17.0, 0.5],
            [11.5, 16.5, 1.0],
            [12.0, 16.0, 0.2],
            [12.3, 15.7, -0.2],
        ]
    )
    second_context = np.array(
        [
            [20.0, 30.0, -1.0],
            [20.5, 29.0, -0.5],
            [21.0, 28.0, 0.0],
            [21.5, 27.0, 0.5],
            [22.0, 26.0, 1.0],
            [22.5, 25.0, 1.5],
        ]
    )
    parameters = np.array([33.0, 2.0, 1.6, 1.5, 1.67])
    from experiments.manuscript_storyrev_20260808.target import (
        original_idm_acceleration,
    )

    first_acceleration = original_idm_acceleration(parameters, first_context) + np.array(
        [0.1, -0.1, 0.05, -0.05, 0.08, -0.08]
    )
    second_acceleration = original_idm_acceleration(parameters, second_context) + np.array(
        [-0.08, 0.08, -0.04, 0.04, -0.02, 0.02]
    )
    trajectories, _ = standardize_trajectories(
        (first_acceleration, second_acceleration),
        (first_context, second_context),
        ("trajectory-0", "trajectory-1"),
    )
    config = replace(
        ReconstructionConfig.from_json(CONFIG_PATH),
        expected_num_trajectories=2,
        allow_observed_negative_speed=False,
        minimum_allowed_speed_mps=0.0,
        expected_negative_speed_count=0,
        expected_negative_speed_fraction=0.0,
        expected_negative_speed_affected_trajectories=0,
        negative_speed_unaffected_case_indices=(),
        preserve_raw_observations=False,
        negative_speed_justification="",
        num_iterations=4,
        burn_in=2,
        thin=1,
    )
    return trajectories, config


def test_tiny_chain_is_seed_deterministic_and_stores_joint_paths() -> None:
    trajectories, config = _tiny_problem()

    first = run_chain(trajectories, config)
    second = run_chain(trajectories, config)

    for key, first_value in first.as_npz_payload().items():
        np.testing.assert_array_equal(first_value, second.as_npz_payload()[key])
    assert first.paths is not None
    assert first.paths.shape == (2, 12)
    assert first.transition_probability.shape == (2, 4, 4)
    np.testing.assert_allclose(
        first.transition_probability.sum(axis=2), np.ones((2, 4))
    )
    np.testing.assert_allclose(first.initial_probability.sum(axis=1), np.ones(2))
    assert np.all(np.isfinite(first.log_likelihood))
    assert np.all(first.residual_variance > 0.0)
    assert first.initialization_runtime_seconds >= 0.0
    assert first.iteration_runtime_seconds.shape == (config.num_iterations,)
    assert np.all(np.isfinite(first.iteration_runtime_seconds))
    assert np.all(first.iteration_runtime_seconds >= 0.0)
    assert "iteration_runtime_seconds" not in first.as_npz_payload()


def test_checkpoint_resume_matches_uninterrupted_chain_and_rng_bitwise(
    tmp_path: Path,
) -> None:
    trajectories, base_config = _tiny_problem()
    config = replace(base_config, num_iterations=6, burn_in=2, thin=1)
    full_rng = config.make_rng()
    uninterrupted = run_chain(trajectories, config, rng=full_rng)
    checkpoint_root = tmp_path / "checkpoints"
    checkpoint_paths = None

    class PlannedInterruption(RuntimeError):
        pass

    def checkpoint_and_interrupt(snapshot) -> None:
        nonlocal checkpoint_paths
        checkpoint_paths = write_checkpoint(
            checkpoint_root,
            completed_iterations=snapshot.completed_iterations,
            retained_index=snapshot.retained_index,
            state=snapshot.state,
            rng=snapshot.rng,
            retained_history=snapshot.retained_history,
            trajectory_offsets=snapshot.trajectory_offsets,
            target_fingerprint="a" * 64,
            config_sha256="b" * 64,
            cohort_content_sha256="c" * 64,
            source_sha256={"sampler.py": "d" * 64},
            environment_receipt={"python": "test", "threads": 1},
            initialization_runtime_seconds=(
                snapshot.initialization_runtime_seconds
            ),
            iteration_runtime_seconds=snapshot.iteration_runtime_seconds,
        )
        raise PlannedInterruption

    with pytest.raises(PlannedInterruption):
        run_chain(
            trajectories,
            config,
            checkpoint_interval=3,
            checkpoint_callback=checkpoint_and_interrupt,
        )
    assert checkpoint_paths is not None
    restored = load_checkpoint(
        checkpoint_paths.directory,
        expected_target_fingerprint="a" * 64,
        expected_config_sha256="b" * 64,
        expected_cohort_content_sha256="c" * 64,
        expected_source_sha256={"sampler.py": "d" * 64},
        expected_environment_receipt={"python": "test", "threads": 1},
    )
    resume_rng = restored.restore_generator()
    resumed = run_chain(
        trajectories,
        config,
        rng=resume_rng,
        resume_checkpoint=restored,
    )

    for name, expected in uninterrupted.as_npz_payload().items():
        np.testing.assert_array_equal(resumed.as_npz_payload()[name], expected)
    for expected_path, resumed_path in zip(
        uninterrupted.final_state.paths, resumed.final_state.paths
    ):
        np.testing.assert_array_equal(resumed_path, expected_path)
    for name in (
        "initial_probability",
        "transition_probability",
        "idm_parameters",
        "residual_variance",
        "scenario_mean",
        "scenario_covariance",
    ):
        np.testing.assert_array_equal(
            getattr(resumed.final_state, name),
            getattr(uninterrupted.final_state, name),
        )
    np.testing.assert_array_equal(resume_rng.random(32), full_rng.random(32))

    invalid_schedule = replace(restored, retained_index=0)
    with pytest.raises(ValueError, match="retained_index"):
        run_chain(
            trajectories,
            config,
            resume_checkpoint=invalid_schedule,
        )


def test_transition_and_initial_counts_respect_trajectory_boundaries() -> None:
    paths = (np.array([0, 1]), np.array([2, 0]))

    transition_counts = count_joint_transitions(paths, 4)
    initial_counts = count_initial_states(paths, 4)

    assert transition_counts.sum() == 2
    assert transition_counts[1, 2] == 0
    np.testing.assert_array_equal(initial_counts, np.array([1, 0, 1, 0]))


def test_empty_scenario_state_is_drawn_from_niw_prior_reproducibly() -> None:
    trajectories, config = _tiny_problem()
    state_space = JointStateSpace(2, 2)
    empty_scenario_paths = tuple(
        np.where(np.arange(trajectory.acceleration.size) % 2 == 0, 0, 2).astype(int)
        for trajectory in trajectories
    )
    state = ReconstructionState(
        paths=empty_scenario_paths,
        initial_probability=np.full(4, 0.25),
        transition_probability=np.full((4, 4), 0.25),
        idm_parameters=np.tile(np.array([33.0, 2.0, 1.6, 1.5, 1.67]), (2, 1)),
        residual_variance=np.ones(2),
        scenario_mean=np.zeros((2, 3)),
        scenario_covariance=np.tile(np.eye(3), (2, 1, 1)),
    )

    first = sample_scenario_emissions(
        state, trajectories, state_space, config, np.random.default_rng(44)
    )
    second = sample_scenario_emissions(
        state, trajectories, state_space, config, np.random.default_rng(44)
    )

    np.testing.assert_array_equal(first[0], second[0])
    np.testing.assert_array_equal(first[1], second[1])
    assert np.all(np.isfinite(first[0][1]))
    assert np.all(np.linalg.eigvalsh(first[1][1]) > 0.0)
