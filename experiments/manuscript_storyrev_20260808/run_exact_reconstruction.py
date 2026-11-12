"""Run one append-only exact reconstruction from a frozen cohort artifact."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np
import scipy
from numpy.typing import NDArray

from .checkpoint import CheckpointPaths, RestoredCheckpoint, load_checkpoint, write_checkpoint
from .sampler import (
    ReconstructionCheckpointSnapshot,
    ReconstructionConfig,
    run_chain,
    standardize_trajectories,
)
from .target import (
    COVARIATE_ORDER,
    ContextStandardizer,
    StoryTrajectory,
    scientific_target_fingerprint,
)


FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]
REQUIRED_ARRAYS = (
    "physical_context",
    "acceleration",
    "offsets",
    "frames",
    "trajectory_ids",
    "recording_ids",
    "follower_ids",
    "pair_nos",
    "lengths",
)
THREAD_ENVIRONMENT_VARIABLES = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "MKL_DYNAMIC",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "BLIS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "PYTHONHASHSEED",
)


@dataclass(frozen=True)
class LoadedCohort:
    trajectories: Tuple[StoryTrajectory, ...]
    standardizer: ContextStandardizer
    content_sha256: str
    num_observations: int
    num_trajectories: int
    domain_audit: Mapping[str, Any]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256_array(array: NDArray[np.generic]) -> str:
    materialized = np.asarray(array)
    if materialized.dtype.hasobject:
        raise TypeError("Object arrays are not valid provenance payloads.")
    if materialized.dtype.kind in "iufcbmM":
        canonical_dtype = materialized.dtype.newbyteorder("<")
        materialized = materialized.astype(canonical_dtype, copy=False)
    contiguous = np.ascontiguousarray(materialized)
    header = {
        "dtype": contiguous.dtype.str,
        "shape": list(contiguous.shape),
    }
    digest = hashlib.sha256()
    digest.update(
        json.dumps(
            header,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
    )
    digest.update(b"\0")
    digest.update(contiguous.tobytes(order="C"))
    return digest.hexdigest()


def _canonical_cohort_hash(arrays: Mapping[str, NDArray[np.generic]]) -> str:
    per_array = {
        name: {
            "dtype": str(arrays[name].dtype),
            "shape": list(arrays[name].shape),
            "sha256": _sha256_array(arrays[name]),
        }
        for name in REQUIRED_ARRAYS
    }
    canonical = json.dumps(
        per_array,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _manifest_array_metadata(
    manifest: Mapping[str, Any], name: str
) -> Optional[Mapping[str, Any]]:
    arrays = manifest.get("arrays")
    if isinstance(arrays, Mapping) and isinstance(arrays.get(name), Mapping):
        return arrays[name]
    artifact = manifest.get("cohort_data")
    if isinstance(artifact, Mapping):
        nested = artifact.get("arrays")
        if isinstance(nested, Mapping) and isinstance(nested.get(name), Mapping):
            return nested[name]
    cohort = manifest.get("cohort")
    if isinstance(cohort, Mapping):
        nested = cohort.get("array_manifest")
        if isinstance(nested, Mapping) and isinstance(nested.get(name), Mapping):
            return nested[name]
    return None


def _verified_manifest_input_fingerprint(
    manifest: Mapping[str, Any],
) -> Optional[str]:
    receipt = manifest.get("target_input_fingerprint")
    if not isinstance(receipt, Mapping):
        return None
    inputs = receipt.get("inputs")
    expected = str(receipt.get("sha256", "")).lower()
    if not isinstance(inputs, Mapping) or len(expected) != 64:
        raise ValueError("Manifest target_input_fingerprint is malformed.")
    canonical = json.dumps(
        inputs,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    actual = hashlib.sha256(canonical).hexdigest()
    if actual != expected:
        raise ValueError("Manifest target_input_fingerprint is internally inconsistent.")
    return actual


def _verify_manifest_arrays(
    arrays: Mapping[str, NDArray[np.generic]], manifest: Mapping[str, Any]
) -> None:
    for name in REQUIRED_ARRAYS:
        metadata = _manifest_array_metadata(manifest, name)
        if metadata is None:
            raise ValueError(f"Manifest lacks audited metadata for array {name!r}.")
        expected_shape = metadata.get("shape")
        expected_dtype = metadata.get("dtype")
        expected_sha256 = metadata.get("sha256")
        if list(arrays[name].shape) != list(expected_shape):
            raise ValueError(f"Manifest shape mismatch for {name}.")
        if str(arrays[name].dtype) != str(expected_dtype):
            raise ValueError(f"Manifest dtype mismatch for {name}.")
        if _sha256_array(arrays[name]) != str(expected_sha256).lower():
            raise ValueError(f"Manifest SHA-256 mismatch for {name}.")


def _legacy_continuity_manifest_decision(
    manifest: Mapping[str, Any],
) -> Mapping[str, Any]:
    cohort = manifest.get("cohort")
    if not isinstance(cohort, Mapping):
        raise ValueError("Manifest lacks cohort metadata.")
    domain_audit = cohort.get("domain_audit")
    if not isinstance(domain_audit, Mapping):
        raise ValueError("Manifest lacks a cohort domain audit.")
    decision = domain_audit.get("legacy_continuity_decision")
    if not isinstance(decision, Mapping):
        raise ValueError(
            "Manifest lacks the approved legacy-continuity speed decision."
        )
    return decision


def _audit_observed_speed(
    physical_context: FloatArray,
    offsets: IntArray,
    config: ReconstructionConfig,
    manifest: Mapping[str, Any],
) -> Dict[str, Any]:
    speed = physical_context[:, 0]
    negative = speed < 0.0
    count = int(negative.sum())
    fraction = float(count / speed.size)
    minimum = float(speed.min())
    affected = [
        index
        for index in range(offsets.size - 1)
        if np.any(negative[offsets[index] : offsets[index + 1]])
    ]
    unaffected_cases = list(config.negative_speed_unaffected_case_indices)
    for case_index in unaffected_cases:
        if case_index >= offsets.size - 1:
            raise ValueError("Configured unaffected case index is outside the cohort.")
        if np.any(negative[offsets[case_index] : offsets[case_index + 1]]):
            raise ValueError(
                f"Case index {case_index} is not unaffected by negative speed."
            )

    audit = {
        "negative_speed_values": count,
        "negative_speed_fraction": fraction,
        "minimum_observed_speed_mps": minimum,
        "affected_trajectory_count": len(affected),
        "affected_trajectory_indices_zero_based": affected,
        "unaffected_case_indices_zero_based": unaffected_cases,
        "raw_observations_preserved": True,
    }
    if count == 0:
        if config.expected_negative_speed_count != 0:
            raise ValueError("Observed negative-speed count disagrees with config.")
        return audit
    if not config.allow_observed_negative_speed:
        raise ValueError(
            "Cohort contains negative follower speed and the strict default "
            "policy does not opt in to preserving it."
        )
    if minimum < config.minimum_allowed_speed_mps - 1.0e-12:
        raise ValueError(
            "Observed speed is below the explicitly disclosed frozen-cohort bound."
        )
    if count != config.expected_negative_speed_count:
        raise ValueError("Observed negative-speed count disagrees with config.")
    if not np.isclose(
        fraction,
        config.expected_negative_speed_fraction,
        rtol=0.0,
        atol=1.0e-15,
    ):
        raise ValueError("Observed negative-speed fraction disagrees with config.")
    if len(affected) != config.expected_negative_speed_affected_trajectories:
        raise ValueError("Affected-trajectory count disagrees with config.")
    if not config.preserve_raw_observations:
        raise ValueError("Legacy-continuity opt-in must preserve raw observations.")

    decision = _legacy_continuity_manifest_decision(manifest)
    required_values = {
        "decision_source": "scheme_b_scope_and_implementation_disclosure",
        "allow_observed_negative_speed": True,
        "negative_speed_values": count,
        "affected_trajectory_count": len(affected),
        "affected_trajectory_indices_zero_based": affected,
        "unaffected_case_indices_zero_based": unaffected_cases,
        "raw_observations_preserved": True,
        "speed_clipped": False,
        "observations_deleted": False,
        "relative_speed_recomputed": False,
    }
    for name, expected in required_values.items():
        if decision.get(name) != expected:
            raise ValueError(
                f"Manifest legacy-continuity decision has invalid {name!r}."
            )
    float_checks = {
        "minimum_observed_speed_mps": minimum,
        "minimum_allowed_speed_mps": config.minimum_allowed_speed_mps,
        "negative_speed_fraction": fraction,
    }
    for name, expected in float_checks.items():
        try:
            actual = float(decision.get(name))
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"Manifest legacy-continuity decision lacks numeric {name!r}."
            ) from error
        if not np.isclose(actual, expected, rtol=0.0, atol=1.0e-12):
            raise ValueError(
                f"Manifest legacy-continuity decision has invalid {name!r}."
            )
    if str(decision.get("justification", "")).strip() == "":
        raise ValueError("Manifest legacy-continuity decision lacks justification.")
    audit["decision_source"] = decision["decision_source"]
    return audit


def load_cohort_artifact(
    cohort_npz: Path,
    manifest_path: Path,
    config: ReconstructionConfig,
) -> LoadedCohort:
    """Validate the frozen flattened cohort and build independent sequences."""

    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    with np.load(cohort_npz, allow_pickle=False) as loaded:
        missing = [name for name in REQUIRED_ARRAYS if name not in loaded.files]
        if missing:
            raise ValueError(f"Cohort artifact lacks required arrays: {missing}.")
        arrays = {name: loaded[name].copy() for name in REQUIRED_ARRAYS}
    _verify_manifest_arrays(arrays, manifest)

    physical_context = np.asarray(arrays["physical_context"], dtype=float)
    acceleration = np.asarray(arrays["acceleration"], dtype=float)
    offsets = np.asarray(arrays["offsets"], dtype=np.int64)
    frames = np.asarray(arrays["frames"], dtype=np.int64)
    trajectory_ids = np.asarray(arrays["trajectory_ids"])
    recording_ids = np.asarray(arrays["recording_ids"], dtype=np.int64)
    follower_ids = np.asarray(arrays["follower_ids"], dtype=np.int64)
    pair_nos = np.asarray(arrays["pair_nos"], dtype=np.int64)
    lengths = np.asarray(arrays["lengths"], dtype=np.int64)
    num_trajectories = trajectory_ids.size
    num_observations = acceleration.size

    if physical_context.shape != (num_observations, 3):
        raise ValueError("physical_context must have shape (N, 3).")
    if frames.shape != (num_observations,):
        raise ValueError("frames must align with observations.")
    if offsets.shape != (num_trajectories + 1,):
        raise ValueError("offsets must have length D + 1.")
    if offsets[0] != 0 or offsets[-1] != num_observations:
        raise ValueError("offset endpoints do not cover the flattened observations.")
    if np.any(np.diff(offsets) <= 0):
        raise ValueError("Every frozen trajectory must be non-empty.")
    if not np.array_equal(np.diff(offsets), lengths):
        raise ValueError("lengths do not match differences of offsets.")
    for name, array in (
        ("recording_ids", recording_ids),
        ("follower_ids", follower_ids),
        ("pair_nos", pair_nos),
        ("lengths", lengths),
    ):
        if array.shape != (num_trajectories,):
            raise ValueError(f"{name} must have shape (D,).")
    if num_trajectories != config.expected_num_trajectories:
        raise ValueError(
            f"Expected {config.expected_num_trajectories} trajectories, "
            f"found {num_trajectories}."
        )
    if np.unique(follower_ids).size != num_trajectories:
        raise ValueError("The manuscript cohort must contain distinct follower IDs.")
    identifiers = tuple(str(identifier) for identifier in trajectory_ids.tolist())
    if len(set(identifiers)) != num_trajectories:
        raise ValueError("trajectory_ids must be unique.")
    if not np.all(np.isfinite(physical_context)) or not np.all(
        np.isfinite(acceleration)
    ):
        raise ValueError("Cohort observations must be finite.")
    if np.any(physical_context[:, 1] <= 0.0):
        raise ValueError("Cohort contains non-positive net gap.")
    domain_audit = _audit_observed_speed(
        physical_context, offsets, config, manifest
    )

    acceleration_sequences = tuple(
        acceleration[offsets[index] : offsets[index + 1]]
        for index in range(num_trajectories)
    )
    context_sequences = tuple(
        physical_context[offsets[index] : offsets[index + 1]]
        for index in range(num_trajectories)
    )
    trajectories, standardizer = standardize_trajectories(
        acceleration_sequences,
        context_sequences,
        identifiers,
        minimum_allowed_speed_mps=config.minimum_allowed_speed_mps,
    )
    content_sha256 = _verified_manifest_input_fingerprint(manifest)
    if content_sha256 is None:
        content_sha256 = _canonical_cohort_hash(arrays)
    manifest_content_hash = manifest.get("cohort_content_sha256")
    if manifest_content_hash is not None and (
        str(manifest_content_hash).lower() != content_sha256
    ):
        raise ValueError("Manifest cohort_content_sha256 does not match arrays.")
    return LoadedCohort(
        trajectories=trajectories,
        standardizer=standardizer,
        content_sha256=content_sha256,
        num_observations=num_observations,
        num_trajectories=num_trajectories,
        domain_audit=domain_audit,
    )


def verify_target_fingerprint(path: Path, expected: str) -> None:
    """Reject any warm start or cache from a different posterior target."""

    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    actual = payload.get("scientific_target_fingerprint")
    if actual != expected:
        raise ValueError(
            "Cached artifact target fingerprint is incompatible with this run."
        )


def _git_commit(repository_root: Path) -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository_root,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        return "UNAVAILABLE"
    return completed.stdout.strip()


def _git_worktree_dirty(repository_root: Path) -> Optional[bool]:
    completed = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=repository_root,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        return None
    return bool(completed.stdout.strip())


def _scientific_source_hashes(repository_root: Path) -> Dict[str, str]:
    """Hash the complete local source closure used by the reconstruction."""

    storyrev_root = Path(__file__).resolve().parent
    revision_root = repository_root / "experiments" / "revision_20260806"
    source_paths = [
        storyrev_root / "__init__.py",
        storyrev_root / "checkpoint.py",
        storyrev_root / "target.py",
        storyrev_root / "sampler.py",
        storyrev_root / "run_exact_reconstruction.py",
    ]
    for package_directory in (
        revision_root / "model",
        revision_root / "inference",
    ):
        source_paths.extend(sorted(package_directory.glob("*.py")))
    unique_paths = sorted({path.resolve() for path in source_paths})
    if not unique_paths or any(not path.is_file() for path in unique_paths):
        raise FileNotFoundError("The declared scientific source closure is incomplete.")
    return {
        path.relative_to(repository_root.resolve()).as_posix(): _sha256_file(path)
        for path in unique_paths
    }


def _thread_environment_receipt() -> Dict[str, Any]:
    receipt: Dict[str, Any] = {
        "environment_variables": {
            name: os.environ.get(name, "UNSET")
            for name in THREAD_ENVIRONMENT_VARIABLES
        },
        "logical_cpu_count": os.cpu_count(),
    }
    try:
        from threadpoolctl import threadpool_info
    except ImportError:
        receipt["threadpoolctl"] = "UNAVAILABLE"
    else:
        receipt["threadpoolctl"] = sorted(
            threadpool_info(),
            key=lambda item: json.dumps(item, sort_keys=True, default=str),
        )
    return receipt


def _stable_execution_environment(
    thread_environment: Mapping[str, Any],
) -> Dict[str, Any]:
    """Return only restart-stable environment fields used for identity checks."""

    return {
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "thread_environment": thread_environment,
    }


def _checkpoint_artifact_record(
    paths: CheckpointPaths, *, relative_to: Optional[Path] = None
) -> Dict[str, Any]:
    directory = paths.directory
    recorded_directory = (
        directory.relative_to(relative_to).as_posix()
        if relative_to is not None
        else str(directory.resolve())
    )
    return {
        "directory": recorded_directory,
        "payload_sha256": _sha256_file(paths.payload),
        "payload_bytes": paths.payload.stat().st_size,
        "receipt_sha256": _sha256_file(paths.receipt),
        "receipt_bytes": paths.receipt.stat().st_size,
    }


def _timestamped_run_directory(results_root: Path, arm_id: str) -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    run_directory = results_root / f"{timestamp}_{arm_id}"
    run_directory.mkdir(parents=True, exist_ok=False)
    return run_directory


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite {path}.")
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _process_memory_receipt() -> Dict[str, Any]:
    """Return a best-effort process-memory receipt without changing the chain."""

    try:
        import psutil
    except ImportError:
        return {
            "status": "UNAVAILABLE",
            "reason": "psutil is not installed",
        }
    information = psutil.Process(os.getpid()).memory_info()
    receipt: Dict[str, Any] = {
        "status": "AVAILABLE",
        "source": "psutil.Process.memory_info",
        "measurement_scope": "current process lifetime; baseline is not subtracted",
        "rss_bytes_at_receipt": int(information.rss),
    }
    optional_fields = {
        "peak_wset": "process_lifetime_peak_working_set_bytes",
        "wset": "working_set_bytes_at_receipt",
        "private": "private_bytes_at_receipt",
        "pagefile": "pagefile_bytes_at_receipt",
        "peak_pagefile": "process_lifetime_peak_pagefile_bytes",
    }
    for source_name, receipt_name in optional_fields.items():
        value = getattr(information, source_name, None)
        if value is not None:
            receipt[receipt_name] = int(value)
    return receipt


def _runtime_diagnostics(
    *,
    result: Any,
    sampling_runtime_seconds: float,
    chain_write_runtime_seconds: float,
    runtime_through_chain_write_seconds: float,
    chain_path: Path,
    uncompressed_array_bytes: int,
) -> Dict[str, Any]:
    iteration_seconds = np.asarray(result.iteration_runtime_seconds, dtype=float)
    if iteration_seconds.ndim != 1 or iteration_seconds.size == 0:
        raise ValueError("Iteration timing receipt must be a non-empty vector.")
    if not np.all(np.isfinite(iteration_seconds)) or np.any(iteration_seconds < 0.0):
        raise ValueError("Iteration timing receipt contains an invalid duration.")
    compressed_bytes = chain_path.stat().st_size
    post_first = iteration_seconds[1:] if iteration_seconds.size > 1 else iteration_seconds
    return {
        "schema_version": 1,
        "diagnostic_only": True,
        "scientific_target_affected": False,
        "timing_clock": "time.perf_counter",
        "timing_scope": {
            "initialization": "initialize_state plus state validation",
            "iteration": "one complete gibbs_step",
            "sampling": "run_chain including initialization and all Gibbs iterations",
            "runtime_through_chain_write": (
                "append-only run setup, sampling, and compressed chain write"
            ),
        },
        "initialization_runtime_seconds": float(
            result.initialization_runtime_seconds
        ),
        "iteration_runtime_seconds": iteration_seconds.tolist(),
        "iteration_runtime_summary_seconds": {
            "count": int(iteration_seconds.size),
            "sum": float(np.sum(iteration_seconds)),
            "mean": float(np.mean(iteration_seconds)),
            "median": float(np.median(iteration_seconds)),
            "minimum": float(np.min(iteration_seconds)),
            "maximum": float(np.max(iteration_seconds)),
            "post_first_mean": float(np.mean(post_first)),
            "post_first_median": float(np.median(post_first)),
        },
        "sampling_runtime_seconds": float(sampling_runtime_seconds),
        "chain_write_runtime_seconds": float(chain_write_runtime_seconds),
        "runtime_through_chain_write_seconds": float(
            runtime_through_chain_write_seconds
        ),
        "storage": {
            "chain_sha256": _sha256_file(chain_path),
            "chain_compressed_bytes": int(compressed_bytes),
            "chain_uncompressed_array_bytes": int(uncompressed_array_bytes),
            "compressed_to_uncompressed_ratio": float(
                compressed_bytes / uncompressed_array_bytes
            ),
        },
        "memory": _process_memory_receipt(),
    }


def execute_run(
    config_path: Path,
    cohort_npz: Path,
    manifest_path: Path,
    results_root: Path,
    *,
    checkpoint_interval: int = 500,
    resume_checkpoint: Optional[Path] = None,
) -> Path:
    if checkpoint_interval <= 0:
        raise ValueError("checkpoint_interval must be positive.")
    config = ReconstructionConfig.from_json(config_path)
    cohort = load_cohort_artifact(cohort_npz, manifest_path, config)
    target_fingerprint = scientific_target_fingerprint(
        config.scientific_target_payload(), cohort_sha256=cohort.content_sha256
    )
    repository_root = Path(__file__).resolve().parents[2]
    source_hashes = _scientific_source_hashes(repository_root)
    thread_environment = _thread_environment_receipt()
    stable_environment = _stable_execution_environment(thread_environment)
    config_sha256 = _sha256_file(config_path)
    restored: Optional[RestoredCheckpoint] = None
    resumed_from: Optional[Dict[str, Any]] = None
    if resume_checkpoint is not None:
        restored = load_checkpoint(
            resume_checkpoint,
            expected_target_fingerprint=target_fingerprint,
            expected_config_sha256=config_sha256,
            expected_cohort_content_sha256=cohort.content_sha256,
            expected_source_sha256=source_hashes,
            expected_environment_receipt=stable_environment,
        )
        resumed_from = {
            **_checkpoint_artifact_record(restored.checkpoint_paths),
            "completed_iterations": restored.completed_iterations,
            "retained_index": restored.retained_index,
        }
    run_directory = _timestamped_run_directory(results_root, config.arm_id)
    checkpoint_records = []

    def persist_checkpoint(snapshot: ReconstructionCheckpointSnapshot) -> None:
        paths = write_checkpoint(
            run_directory / "checkpoints",
            completed_iterations=snapshot.completed_iterations,
            retained_index=snapshot.retained_index,
            state=snapshot.state,
            rng=snapshot.rng,
            retained_history=snapshot.retained_history,
            trajectory_offsets=snapshot.trajectory_offsets,
            target_fingerprint=target_fingerprint,
            config_sha256=config_sha256,
            cohort_content_sha256=cohort.content_sha256,
            source_sha256=source_hashes,
            environment_receipt=stable_environment,
            initialization_runtime_seconds=(
                snapshot.initialization_runtime_seconds
            ),
            iteration_runtime_seconds=snapshot.iteration_runtime_seconds,
        )
        checkpoint_records.append(
            {
                **_checkpoint_artifact_record(paths, relative_to=run_directory),
                "completed_iterations": snapshot.completed_iterations,
                "retained_index": snapshot.retained_index,
            }
        )

    started = time.perf_counter()
    with config_path.open("r", encoding="utf-8") as handle:
        raw_config = json.load(handle)
    _write_json(run_directory / "config.json", raw_config)
    _write_json(
        run_directory / "RUNNING.json",
        {
            "arm_id": config.arm_id,
            "started_at_utc": datetime.now(timezone.utc).isoformat(),
            "scientific_target_fingerprint": target_fingerprint,
            "checkpoint_interval": checkpoint_interval,
            "resumed_from": resumed_from,
        },
    )
    sampling_started = time.perf_counter()
    result = run_chain(
        cohort.trajectories,
        config,
        resume_checkpoint=restored,
        checkpoint_interval=checkpoint_interval,
        checkpoint_callback=persist_checkpoint,
    )
    sampling_runtime_seconds = time.perf_counter() - sampling_started
    chain_path = run_directory / "chain.npz"
    if chain_path.exists():
        raise FileExistsError(f"Refusing to overwrite {chain_path}.")
    chain_payload = result.as_npz_payload()
    uncompressed_array_bytes = sum(
        int(np.asarray(array).nbytes) for array in chain_payload.values()
    )
    chain_write_started = time.perf_counter()
    np.savez_compressed(chain_path, **chain_payload)
    chain_write_runtime_seconds = time.perf_counter() - chain_write_started
    runtime_seconds = time.perf_counter() - started
    runtime_diagnostics = _runtime_diagnostics(
        result=result,
        sampling_runtime_seconds=sampling_runtime_seconds,
        chain_write_runtime_seconds=chain_write_runtime_seconds,
        runtime_through_chain_write_seconds=runtime_seconds,
        chain_path=chain_path,
        uncompressed_array_bytes=uncompressed_array_bytes,
    )
    _write_json(run_directory / "runtime_diagnostics.json", runtime_diagnostics)
    checkpoint_provenance = {
        "schema_version": 1,
        "run_control_only": True,
        "scientific_target_affected": False,
        "checkpoint_interval": checkpoint_interval,
        "resumed_from": resumed_from,
        "checkpoints_written": checkpoint_records,
    }
    _write_json(
        run_directory / "checkpoint_provenance.json", checkpoint_provenance
    )

    _write_json(
        run_directory / "standardizer.json",
        {
            "context_order": list(COVARIATE_ORDER),
            "mean": cohort.standardizer.mean.tolist(),
            "scale": cohort.standardizer.scale.tolist(),
        },
    )
    _write_json(
        run_directory / "target.json",
        {
            "scientific_target_fingerprint": target_fingerprint,
            "cohort_content_sha256": cohort.content_sha256,
            "scientific_target": config.scientific_target_payload(),
        },
    )
    _write_json(
        run_directory / "seed_lineage.json",
        {
            "bit_generator": "PCG64",
            "master_seed": config.master_seed,
            "spawn_key": [config.seed_spawn_index],
            "derived_seed_receipt": config.seed,
            "source": "NumPy SeedSequence(master_seed, spawn_key=(arm_index,))",
        },
    )
    metadata = {
        "arm_id": config.arm_id,
        "purpose": raw_config.get("purpose", "FORMAL_RECONSTRUCTION"),
        "model_id": config.model_id,
        "scientific_target_fingerprint": target_fingerprint,
        "cohort_content_sha256": cohort.content_sha256,
        "cohort_npz_sha256": _sha256_file(cohort_npz),
        "cohort_manifest_sha256": _sha256_file(manifest_path),
        "config_sha256": config_sha256,
        "num_trajectories": cohort.num_trajectories,
        "num_observations": cohort.num_observations,
        "domain_audit": cohort.domain_audit,
        "num_iterations": config.num_iterations,
        "burn_in": config.burn_in,
        "thin": config.thin,
        "num_retained_draws": config.num_retained_draws,
        "runtime_seconds": runtime_seconds,
        "sampling_runtime_seconds": sampling_runtime_seconds,
        "chain_write_runtime_seconds": chain_write_runtime_seconds,
        "runtime_diagnostics": {
            "path": "runtime_diagnostics.json",
            "sha256": _sha256_file(run_directory / "runtime_diagnostics.json"),
        },
        "checkpoint_provenance": {
            "path": "checkpoint_provenance.json",
            "sha256": _sha256_file(
                run_directory / "checkpoint_provenance.json"
            ),
            "checkpoint_interval": checkpoint_interval,
            "resumed_from": resumed_from,
            "checkpoints_written": len(checkpoint_records),
        },
        "git_commit": _git_commit(repository_root),
        "git_worktree_dirty": _git_worktree_dirty(repository_root),
        "source_dependency_scope": {
            "policy": "conservative complete Python-module closure",
            "storyrev_files": [
                "__init__.py",
                "checkpoint.py",
                "target.py",
                "sampler.py",
                "run_exact_reconstruction.py",
            ],
            "reused_packages": [
                "experiments/revision_20260806/model/*.py",
                "experiments/revision_20260806/inference/*.py",
            ],
        },
        "source_sha256": source_hashes,
        "thread_environment": thread_environment,
        "stable_execution_environment": stable_environment,
        "invocation_argv": sys.argv,
        "working_directory": str(Path.cwd()),
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "process_id": os.getpid(),
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    _write_json(run_directory / "run_metadata.json", metadata)
    artifacts = {
        name: {
            "sha256": _sha256_file(run_directory / name),
            "bytes": (run_directory / name).stat().st_size,
        }
        for name in (
            "chain.npz",
            "config.json",
            "standardizer.json",
            "target.json",
            "seed_lineage.json",
            "runtime_diagnostics.json",
            "checkpoint_provenance.json",
            "run_metadata.json",
        )
    }
    _write_json(run_directory / "artifact_manifest.json", {"artifacts": artifacts})
    _write_json(
        run_directory / "COMPLETED.json",
        {
            "arm_id": config.arm_id,
            "runtime_seconds": runtime_seconds,
            "scientific_target_fingerprint": target_fingerprint,
        },
    )
    return run_directory


def preflight_run(
    config_path: Path,
    cohort_npz: Path,
    manifest_path: Path,
    *,
    checkpoint_interval: int = 500,
    resume_checkpoint: Optional[Path] = None,
) -> Dict[str, Any]:
    """Validate an arm and cohort without creating output or sampling."""

    if checkpoint_interval <= 0:
        raise ValueError("checkpoint_interval must be positive.")
    config = ReconstructionConfig.from_json(config_path)
    cohort = load_cohort_artifact(cohort_npz, manifest_path, config)
    fingerprint = scientific_target_fingerprint(
        config.scientific_target_payload(), cohort_sha256=cohort.content_sha256
    )
    resume_receipt: Optional[Dict[str, Any]] = None
    if resume_checkpoint is not None:
        repository_root = Path(__file__).resolve().parents[2]
        source_hashes = _scientific_source_hashes(repository_root)
        stable_environment = _stable_execution_environment(
            _thread_environment_receipt()
        )
        restored = load_checkpoint(
            resume_checkpoint,
            expected_target_fingerprint=fingerprint,
            expected_config_sha256=_sha256_file(config_path),
            expected_cohort_content_sha256=cohort.content_sha256,
            expected_source_sha256=source_hashes,
            expected_environment_receipt=stable_environment,
        )
        expected_retained = len(
            range(config.burn_in, restored.completed_iterations, config.thin)
        )
        if restored.completed_iterations > config.num_iterations:
            raise ValueError(
                "Checkpoint completed_iterations is outside the configured schedule."
            )
        if restored.retained_index != expected_retained:
            raise ValueError(
                "Checkpoint retained_index disagrees with burn-in/thinning schedule."
            )
        resume_receipt = {
            **_checkpoint_artifact_record(restored.checkpoint_paths),
            "completed_iterations": restored.completed_iterations,
            "retained_index": restored.retained_index,
        }
    return {
        "status": "PREFLIGHT_PASSED",
        "arm_id": config.arm_id,
        "model_id": config.model_id,
        "num_trajectories": cohort.num_trajectories,
        "num_observations": cohort.num_observations,
        "cohort_content_sha256": cohort.content_sha256,
        "scientific_target_fingerprint": fingerprint,
        "domain_audit": cohort.domain_audit,
        "checkpoint_interval": checkpoint_interval,
        "resume_checkpoint": resume_receipt,
        "would_create_output": False,
        "would_sample": False,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--cohort-npz", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument(
        "--results-root",
        type=Path,
        default=Path(__file__).resolve().parent / "results",
    )
    parser.add_argument(
        "--checkpoint-interval",
        type=int,
        default=500,
        help="Write an immutable checkpoint after this many complete Gibbs steps.",
    )
    parser.add_argument(
        "--resume-checkpoint",
        type=Path,
        help="Resume from one immutable iter_XXXXXXXX checkpoint directory.",
    )
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="Validate hashes, cohort, and target without writing or sampling.",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.preflight_only:
        report = preflight_run(
            args.config.resolve(),
            args.cohort_npz.resolve(),
            args.manifest.resolve(),
            checkpoint_interval=args.checkpoint_interval,
            resume_checkpoint=(
                args.resume_checkpoint.resolve()
                if args.resume_checkpoint is not None
                else None
            ),
        )
        print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    else:
        run_directory = execute_run(
            args.config.resolve(),
            args.cohort_npz.resolve(),
            args.manifest.resolve(),
            args.results_root.resolve(),
            checkpoint_interval=args.checkpoint_interval,
            resume_checkpoint=(
                args.resume_checkpoint.resolve()
                if args.resume_checkpoint is not None
                else None
            ),
        )
        print(run_directory)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
