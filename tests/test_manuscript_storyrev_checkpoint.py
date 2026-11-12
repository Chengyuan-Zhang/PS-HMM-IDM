from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pytest

from experiments.manuscript_storyrev_20260808.checkpoint import (
    load_checkpoint,
    write_checkpoint,
)
from experiments.manuscript_storyrev_20260808.sampler import ReconstructionState


TARGET = "a" * 64
CONFIG = "b" * 64
COHORT = "c" * 64
SOURCES = {
    "experiments/manuscript_storyrev_20260808/sampler.py": "d" * 64,
    "experiments/revision_20260806/model/inference.py": "e" * 64,
}
ENVIRONMENT = {
    "python": "3.9.13",
    "numpy": "1.23.5",
    "scipy": "1.9.3",
    "threads": {"MKL_NUM_THREADS": "1", "OMP_NUM_THREADS": "1"},
}


def _state() -> ReconstructionState:
    return ReconstructionState(
        paths=(np.array([0, 1, 3]), np.array([2, 0])),
        initial_probability=np.array([0.4, 0.3, 0.2, 0.1]),
        transition_probability=np.array(
            [
                [0.7, 0.1, 0.1, 0.1],
                [0.2, 0.5, 0.2, 0.1],
                [0.1, 0.2, 0.6, 0.1],
                [0.1, 0.1, 0.2, 0.6],
            ]
        ),
        idm_parameters=np.array(
            [[33.0, 2.0, 1.6, 1.5, 1.67], [30.0, 1.8, 1.4, 1.3, 1.8]]
        ),
        residual_variance=np.array([0.2, 0.3]),
        scenario_mean=np.array([[0.5, -0.2, 0.1], [-0.5, 0.2, -0.1]]),
        scenario_covariance=np.array(
            [np.diag([0.5, 0.6, 0.7]), np.diag([0.8, 0.9, 1.0])]
        ),
    )


def _history(retained: int = 2) -> dict[str, np.ndarray]:
    state = _state()
    return {
        "idm_parameters": np.stack(
            [state.idm_parameters + 0.01 * index for index in range(retained)]
        ),
        "residual_variance": np.stack(
            [state.residual_variance + 0.01 * index for index in range(retained)]
        ),
        "scenario_mean": np.stack(
            [state.scenario_mean + 0.01 * index for index in range(retained)]
        ),
        "scenario_covariance": np.stack(
            [state.scenario_covariance + 0.01 * index * np.eye(3) for index in range(retained)]
        ),
        "initial_probability": np.stack(
            [state.initial_probability for _ in range(retained)]
        ),
        "transition_probability": np.stack(
            [state.transition_probability for _ in range(retained)]
        ),
        "log_likelihood": np.array([-120.0, -119.5])[:retained],
        "slice_step_out_evaluations": np.ones((retained, 2, 5), dtype=np.int64),
        "slice_shrink_evaluations": np.full(
            (retained, 2, 5), 2, dtype=np.int64
        ),
        "paths": np.stack(
            [np.concatenate(state.paths) for _ in range(retained)]
        ).astype(np.int16),
    }


def _write(checkpoint_root: Path):
    rng = np.random.default_rng(20260808)
    rng.random(7)
    state_before = copy.deepcopy(rng.bit_generator.state)
    paths = write_checkpoint(
        checkpoint_root,
        completed_iterations=5,
        retained_index=2,
        state=_state(),
        rng=rng,
        retained_history=_history(),
        trajectory_offsets=np.array([0, 3, 5], dtype=np.int64),
        target_fingerprint=TARGET,
        config_sha256=CONFIG,
        cohort_content_sha256=COHORT,
        source_sha256=SOURCES,
        environment_receipt=ENVIRONMENT,
        initialization_runtime_seconds=0.25,
        iteration_runtime_seconds=np.array([0.5, 0.4, 0.45, 0.42, 0.41]),
    )
    assert rng.bit_generator.state == state_before
    return paths, rng


def _load(checkpoint_directory: Path):
    return load_checkpoint(
        checkpoint_directory,
        expected_target_fingerprint=TARGET,
        expected_config_sha256=CONFIG,
        expected_cohort_content_sha256=COHORT,
        expected_source_sha256=SOURCES,
        expected_environment_receipt=ENVIRONMENT,
    )


def test_checkpoint_round_trip_is_immutable_and_does_not_advance_rng(
    tmp_path: Path,
) -> None:
    paths, original_rng = _write(tmp_path / "checkpoints")

    restored = _load(paths.directory)

    assert restored.completed_iterations == 5
    assert restored.retained_index == 2
    assert restored.initialization_runtime_seconds == pytest.approx(0.25)
    np.testing.assert_array_equal(
        restored.iteration_runtime_seconds,
        np.array([0.5, 0.4, 0.45, 0.42, 0.41]),
    )
    np.testing.assert_array_equal(restored.trajectory_offsets, [0, 3, 5])
    for expected, actual in zip(_state().paths, restored.state.paths):
        np.testing.assert_array_equal(actual, expected)
    for name in (
        "initial_probability",
        "transition_probability",
        "idm_parameters",
        "residual_variance",
        "scenario_mean",
        "scenario_covariance",
    ):
        np.testing.assert_array_equal(
            getattr(restored.state, name), getattr(_state(), name)
        )
    expected_history = _history()
    assert set(restored.retained_history) == set(expected_history)
    for name, expected in expected_history.items():
        np.testing.assert_array_equal(restored.retained_history[name], expected)
    np.testing.assert_array_equal(
        restored.restore_generator().random(12), original_rng.random(12)
    )
    assert paths.directory.name == "iter_00000005"
    assert paths.payload.is_file() and paths.receipt.is_file()

    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        _write(tmp_path / "checkpoints")


def test_checkpoint_rejects_payload_corruption(tmp_path: Path) -> None:
    paths, _ = _write(tmp_path / "checkpoints")
    payload = bytearray(paths.payload.read_bytes())
    payload[len(payload) // 2] ^= 0x01
    paths.payload.write_bytes(payload)

    with pytest.raises(ValueError, match="payload SHA-256 mismatch"):
        _load(paths.directory)


def test_checkpoint_supports_an_empty_burn_in_history_prefix(tmp_path: Path) -> None:
    rng = np.random.default_rng(91)
    empty_history = {name: value[:0] for name, value in _history().items()}
    paths = write_checkpoint(
        tmp_path / "checkpoints",
        completed_iterations=3,
        retained_index=0,
        state=_state(),
        rng=rng,
        retained_history=empty_history,
        trajectory_offsets=np.array([0, 3, 5], dtype=np.int64),
        target_fingerprint=TARGET,
        config_sha256=CONFIG,
        cohort_content_sha256=COHORT,
        source_sha256=SOURCES,
        environment_receipt=ENVIRONMENT,
        iteration_runtime_seconds=np.array([0.4, 0.3, 0.35]),
    )

    restored = _load(paths.directory)

    assert restored.completed_iterations == 3
    assert restored.retained_index == 0
    assert all(value.shape[0] == 0 for value in restored.retained_history.values())


def test_checkpoint_rejects_receipt_corruption_and_identity_mismatch(
    tmp_path: Path,
) -> None:
    paths, _ = _write(tmp_path / "checkpoints")

    with pytest.raises(ValueError, match="target_fingerprint"):
        load_checkpoint(
            paths.directory,
            expected_target_fingerprint="f" * 64,
            expected_config_sha256=CONFIG,
            expected_cohort_content_sha256=COHORT,
            expected_source_sha256=SOURCES,
            expected_environment_receipt=ENVIRONMENT,
        )
    with pytest.raises(ValueError, match="source_sha256"):
        load_checkpoint(
            paths.directory,
            expected_target_fingerprint=TARGET,
            expected_config_sha256=CONFIG,
            expected_cohort_content_sha256=COHORT,
            expected_source_sha256={**SOURCES, "extra.py": "f" * 64},
            expected_environment_receipt=ENVIRONMENT,
        )
    with pytest.raises(ValueError, match="config_sha256"):
        load_checkpoint(
            paths.directory,
            expected_target_fingerprint=TARGET,
            expected_config_sha256="f" * 64,
            expected_cohort_content_sha256=COHORT,
            expected_source_sha256=SOURCES,
            expected_environment_receipt=ENVIRONMENT,
        )
    with pytest.raises(ValueError, match="cohort_content_sha256"):
        load_checkpoint(
            paths.directory,
            expected_target_fingerprint=TARGET,
            expected_config_sha256=CONFIG,
            expected_cohort_content_sha256="f" * 64,
            expected_source_sha256=SOURCES,
            expected_environment_receipt=ENVIRONMENT,
        )
    with pytest.raises(ValueError, match="environment_receipt"):
        load_checkpoint(
            paths.directory,
            expected_target_fingerprint=TARGET,
            expected_config_sha256=CONFIG,
            expected_cohort_content_sha256=COHORT,
            expected_source_sha256=SOURCES,
            expected_environment_receipt={**ENVIRONMENT, "python": "changed"},
        )

    receipt = json.loads(paths.receipt.read_text(encoding="utf-8"))
    receipt["retained_index"] = 1
    paths.receipt.write_text(json.dumps(receipt), encoding="utf-8")
    with pytest.raises(ValueError, match="receipt integrity hash mismatch"):
        _load(paths.directory)
