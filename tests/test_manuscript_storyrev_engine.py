import json
from pathlib import Path
from typing import Optional

import numpy as np
import pytest

from experiments.manuscript_storyrev_20260808.run_exact_reconstruction import (
    _canonical_cohort_hash,
    _sha256_array,
    _sha256_file,
    execute_run,
    preflight_run,
    verify_target_fingerprint,
)


CONFIG_PATH = (
    Path(__file__).resolve().parents[1]
    / "experiments"
    / "manuscript_storyrev_20260808"
    / "configs"
    / "b2s2.json"
)


def _write_tiny_artifacts(
    tmp_path: Path,
    *,
    negative_speed: Optional[float] = None,
    opt_in: bool = False,
) -> tuple[Path, Path, Path]:
    first = np.array(
        [[10.0, 15.0, -1.0], [11.0, 14.0, 0.0], [12.0, 13.0, 1.0]]
    )
    second = np.array(
        [[20.0, 30.0, -2.0], [21.0, 28.0, 0.5], [22.0, 26.0, 2.0]]
    )
    arrays = {
        "physical_context": np.vstack((first, second)).astype(np.float64),
        "acceleration": np.array([0.2, 0.1, -0.1, 0.3, 0.0, -0.2], dtype=np.float64),
        "offsets": np.array([0, 3, 6], dtype=np.int64),
        "frames": np.array([0, 5, 10, 0, 5, 10], dtype=np.int64),
        "trajectory_ids": np.array(["r01-f101", "r02-f202"]),
        "recording_ids": np.array([1, 2], dtype=np.int64),
        "follower_ids": np.array([101, 202], dtype=np.int64),
        "pair_nos": np.array([0, 1], dtype=np.int64),
        "lengths": np.array([3, 3], dtype=np.int64),
    }
    if negative_speed is not None:
        arrays["physical_context"][0, 0] = negative_speed
    cohort_path = tmp_path / "cohort_data.npz"
    np.savez(cohort_path, **arrays)
    manifest = {
        "arrays": {
            name: {
                "dtype": str(array.dtype),
                "shape": list(array.shape),
                "sha256": _sha256_array(array),
            }
            for name, array in arrays.items()
        },
        "cohort_content_sha256": _canonical_cohort_hash(arrays),
    }
    if opt_in:
        manifest["cohort"] = {
            "domain_audit": {
                "legacy_continuity_decision": {
                    "decision_source": (
                        "scheme_b_scope_and_implementation_disclosure"
                    ),
                    "allow_observed_negative_speed": True,
                    "minimum_observed_speed_mps": negative_speed,
                    "minimum_allowed_speed_mps": -0.1,
                    "negative_speed_values": 1,
                    "negative_speed_fraction": 1.0 / 6.0,
                    "affected_trajectory_count": 1,
                    "affected_trajectory_indices_zero_based": [0],
                    "unaffected_case_indices_zero_based": [1],
                    "raw_observations_preserved": True,
                    "speed_clipped": False,
                    "observations_deleted": False,
                    "relative_speed_recomputed": False,
                    "justification": "Synthetic test preserves the observed value.",
                }
            }
        }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    raw_config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    raw_config["data"]["expected_num_trajectories"] = 2
    if opt_in:
        raw_config["data"]["observed_negative_speed_policy"] = {
            "allow_observed_negative_speed": True,
            "minimum_allowed_speed_mps": -0.1,
            "expected_negative_speed_count": 1,
            "expected_negative_speed_fraction": 1.0 / 6.0,
            "expected_affected_trajectory_count": 1,
            "unaffected_case_indices_zero_based": [1],
            "raw_observations_preserved": True,
            "justification": "Synthetic test of explicit legacy continuity.",
        }
    else:
        raw_config["data"].pop("observed_negative_speed_policy", None)
    raw_config["chain"]["num_iterations"] = 2
    raw_config["chain"]["burn_in"] = 1
    raw_config["chain"]["store_latent_paths"] = True
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(raw_config), encoding="utf-8")
    return config_path, cohort_path, manifest_path


def test_execute_run_writes_append_only_provenance_and_chain(tmp_path: Path) -> None:
    config, cohort, manifest = _write_tiny_artifacts(tmp_path)
    results_root = tmp_path / "results"

    run_directory = execute_run(config, cohort, manifest, results_root)

    assert run_directory.parent == results_root
    assert (run_directory / "chain.npz").is_file()
    assert (run_directory / "target.json").is_file()
    assert (run_directory / "run_metadata.json").is_file()
    assert (run_directory / "artifact_manifest.json").is_file()
    assert (run_directory / "checkpoint_provenance.json").is_file()
    assert (run_directory / "COMPLETED.json").is_file()
    with np.load(run_directory / "chain.npz", allow_pickle=False) as chain:
        assert chain["paths"].shape == (1, 6)
        assert chain["transition_probability"].shape == (1, 4, 4)
        assert "iteration_runtime_seconds" not in chain.files
    runtime = json.loads(
        (run_directory / "runtime_diagnostics.json").read_text(encoding="utf-8")
    )
    assert runtime["diagnostic_only"] is True
    assert runtime["scientific_target_affected"] is False
    assert len(runtime["iteration_runtime_seconds"]) == 2
    assert runtime["iteration_runtime_summary_seconds"]["count"] == 2
    assert runtime["storage"]["chain_compressed_bytes"] > 0
    assert len(runtime["storage"]["chain_sha256"]) == 64
    metadata = json.loads(
        (run_directory / "run_metadata.json").read_text(encoding="utf-8")
    )
    required_sources = {
        "experiments/manuscript_storyrev_20260808/target.py",
        "experiments/manuscript_storyrev_20260808/sampler.py",
        "experiments/manuscript_storyrev_20260808/checkpoint.py",
        "experiments/manuscript_storyrev_20260808/run_exact_reconstruction.py",
        "experiments/revision_20260806/model/inference.py",
        "experiments/revision_20260806/model/state_space.py",
        "experiments/revision_20260806/model/transitions.py",
        "experiments/revision_20260806/inference/emissions.py",
        "experiments/revision_20260806/inference/structures.py",
        "experiments/revision_20260806/inference/updates.py",
    }
    assert required_sources <= set(metadata["source_sha256"])
    repository_root = Path(__file__).resolve().parents[1]
    for relative_path, expected_sha256 in metadata["source_sha256"].items():
        assert _sha256_file(repository_root / relative_path) == expected_sha256
    thread_environment = metadata["thread_environment"]
    assert thread_environment["logical_cpu_count"] is not None
    assert {
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "PYTHONHASHSEED",
    } <= set(thread_environment["environment_variables"])


def test_runner_checkpoint_resume_matches_uninterrupted_chain_and_binds_identity(
    tmp_path: Path,
) -> None:
    config, cohort, manifest = _write_tiny_artifacts(tmp_path)
    raw_config = json.loads(config.read_text(encoding="utf-8"))
    raw_config["chain"]["num_iterations"] = 4
    raw_config["chain"]["burn_in"] = 1
    config.write_text(json.dumps(raw_config), encoding="utf-8")

    uninterrupted_directory = execute_run(
        config,
        cohort,
        manifest,
        tmp_path / "uninterrupted",
        checkpoint_interval=2,
    )
    checkpoint_directory = (
        uninterrupted_directory / "checkpoints" / "iter_00000002"
    )
    assert checkpoint_directory.is_dir()
    resumed_directory = execute_run(
        config,
        cohort,
        manifest,
        tmp_path / "resumed",
        checkpoint_interval=2,
        resume_checkpoint=checkpoint_directory,
    )

    with np.load(
        uninterrupted_directory / "chain.npz", allow_pickle=False
    ) as uninterrupted, np.load(
        resumed_directory / "chain.npz", allow_pickle=False
    ) as resumed:
        assert set(uninterrupted.files) == set(resumed.files)
        for name in uninterrupted.files:
            np.testing.assert_array_equal(resumed[name], uninterrupted[name])
    checkpoint_provenance = json.loads(
        (resumed_directory / "checkpoint_provenance.json").read_text(
            encoding="utf-8"
        )
    )
    assert checkpoint_provenance["scientific_target_affected"] is False
    assert checkpoint_provenance["resumed_from"]["completed_iterations"] == 2
    assert checkpoint_provenance["resumed_from"]["retained_index"] == 1
    assert checkpoint_provenance["checkpoints_written"][0][
        "completed_iterations"
    ] == 4

    mismatch = json.loads(config.read_text(encoding="utf-8"))
    mismatch["arm_id"] = "CONFIG_SHA_MISMATCH"
    mismatch_path = tmp_path / "mismatch.json"
    mismatch_path.write_text(json.dumps(mismatch), encoding="utf-8")
    mismatch_root = tmp_path / "mismatch_results"
    with pytest.raises(ValueError, match="config_sha256"):
        execute_run(
            mismatch_path,
            cohort,
            manifest,
            mismatch_root,
            checkpoint_interval=2,
            resume_checkpoint=checkpoint_directory,
        )
    assert not mismatch_root.exists()


def test_preflight_validates_without_sampling_or_creating_results(tmp_path: Path) -> None:
    config, cohort, manifest = _write_tiny_artifacts(tmp_path)

    report = preflight_run(config, cohort, manifest)

    assert report["status"] == "PREFLIGHT_PASSED"
    assert report["would_create_output"] is False
    assert report["would_sample"] is False
    assert not (tmp_path / "results").exists()


def test_manifest_hash_mismatch_stops_before_a_run_directory_is_created(
    tmp_path: Path,
) -> None:
    config, cohort, manifest = _write_tiny_artifacts(tmp_path)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["arrays"]["acceleration"]["sha256"] = "0" * 64
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    results_root = tmp_path / "results"

    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        execute_run(config, cohort, manifest, results_root)

    assert not results_root.exists()


def test_negative_speed_is_rejected_without_explicit_opt_in(tmp_path: Path) -> None:
    config, cohort, manifest = _write_tiny_artifacts(
        tmp_path, negative_speed=-0.05, opt_in=False
    )

    with pytest.raises(ValueError, match="strict default"):
        execute_run(config, cohort, manifest, tmp_path / "results")

    assert not (tmp_path / "results").exists()


def test_limited_negative_speed_opt_in_preserves_input_bytes(tmp_path: Path) -> None:
    config, cohort, manifest = _write_tiny_artifacts(
        tmp_path, negative_speed=-0.05, opt_in=True
    )
    before = cohort.read_bytes()

    run_directory = execute_run(config, cohort, manifest, tmp_path / "results")

    assert cohort.read_bytes() == before
    metadata = json.loads(
        (run_directory / "run_metadata.json").read_text(encoding="utf-8")
    )
    assert metadata["domain_audit"]["negative_speed_values"] == 1
    assert metadata["domain_audit"]["minimum_observed_speed_mps"] == -0.05
    assert metadata["domain_audit"]["raw_observations_preserved"] is True


def test_negative_speed_below_approved_bound_is_rejected(tmp_path: Path) -> None:
    config, cohort, manifest = _write_tiny_artifacts(
        tmp_path, negative_speed=-0.25, opt_in=True
    )

    with pytest.raises(ValueError, match="below the explicitly disclosed"):
        execute_run(config, cohort, manifest, tmp_path / "results")

    assert not (tmp_path / "results").exists()


def test_incompatible_target_fingerprint_is_rejected(tmp_path: Path) -> None:
    receipt = tmp_path / "target.json"
    receipt.write_text(
        json.dumps({"scientific_target_fingerprint": "a" * 64}),
        encoding="utf-8",
    )

    verify_target_fingerprint(receipt, "a" * 64)
    with pytest.raises(ValueError, match="incompatible"):
        verify_target_fingerprint(receipt, "b" * 64)
