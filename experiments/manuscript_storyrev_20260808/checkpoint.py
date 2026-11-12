"""Immutable deterministic checkpoints for the manuscript reconstruction chain.

This module only serializes already materialized state.  It never draws from a
random-number generator.  A checkpoint is committed by atomically renaming a
temporary directory that already contains an fsynced NPZ payload and JSON
receipt; the JSON receipt is the commit metadata for the payload.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple

import numpy as np
from numpy.typing import NDArray

from .sampler import ReconstructionState


CHECKPOINT_SCHEMA_VERSION = 1
PAYLOAD_FILENAME = "checkpoint.npz"
RECEIPT_FILENAME = "receipt.json"
RETAINED_HISTORY_KEYS = (
    "idm_parameters",
    "residual_variance",
    "scenario_mean",
    "scenario_covariance",
    "initial_probability",
    "transition_probability",
    "log_likelihood",
    "slice_step_out_evaluations",
    "slice_shrink_evaluations",
)
OPTIONAL_RETAINED_HISTORY_KEYS = ("paths",)


Array = NDArray[np.generic]


@dataclass(frozen=True)
class CheckpointPaths:
    directory: Path
    payload: Path
    receipt: Path


@dataclass(frozen=True)
class RestoredCheckpoint:
    completed_iterations: int
    retained_index: int
    state: ReconstructionState
    rng_state: Mapping[str, Any]
    retained_history: Mapping[str, Array]
    trajectory_offsets: NDArray[np.int64]
    target_fingerprint: str
    config_sha256: str
    cohort_content_sha256: str
    source_sha256: Mapping[str, str]
    environment_receipt: Mapping[str, Any]
    initialization_runtime_seconds: Optional[float]
    iteration_runtime_seconds: Optional[NDArray[np.float64]]
    checkpoint_paths: CheckpointPaths

    def restore_generator(self) -> np.random.Generator:
        """Restore PCG64 without consuming a draw from the saved stream."""

        bit_generator = np.random.PCG64(0)
        bit_generator.state = copy.deepcopy(dict(self.rng_state))
        return np.random.Generator(bit_generator)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256_array(array: Array) -> str:
    materialized = np.asarray(array)
    if materialized.dtype.hasobject:
        raise TypeError("Object arrays are not valid checkpoint payloads.")
    if materialized.dtype.kind in "iufcbmM":
        materialized = materialized.astype(
            materialized.dtype.newbyteorder("<"), copy=False
        )
    contiguous = np.ascontiguousarray(materialized)
    header = {
        "dtype": contiguous.dtype.str,
        "shape": list(contiguous.shape),
    }
    digest = hashlib.sha256()
    digest.update(_canonical_json_bytes(header))
    digest.update(b"\0")
    digest.update(contiguous.tobytes(order="C"))
    return digest.hexdigest()


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        value = float(value)
    if isinstance(value, float):
        if not np.isfinite(value):
            raise ValueError("Checkpoint JSON values must be finite.")
        return value
    if value is None or isinstance(value, (str, int, bool)):
        return value
    raise TypeError(f"Unsupported checkpoint JSON value: {type(value).__name__}.")


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        _jsonable(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _hex_digest(value: str, name: str) -> str:
    normalized = str(value).lower()
    if len(normalized) != 64 or any(
        character not in "0123456789abcdef" for character in normalized
    ):
        raise ValueError(f"{name} must be a hexadecimal SHA-256 digest.")
    return normalized


def _normalized_source_hashes(source_sha256: Mapping[str, str]) -> Dict[str, str]:
    if not source_sha256:
        raise ValueError("source_sha256 must contain every scientific dependency.")
    normalized: Dict[str, str] = {}
    for name, digest in source_sha256.items():
        key = str(name)
        if not key or key in normalized:
            raise ValueError("Scientific source paths must be non-empty and unique.")
        normalized[key] = _hex_digest(digest, f"source_sha256[{key!r}]")
    return dict(sorted(normalized.items()))


def _validate_offsets(
    trajectory_offsets: Array, paths: Tuple[NDArray[np.int64], ...]
) -> NDArray[np.int64]:
    offsets = np.asarray(trajectory_offsets)
    if offsets.ndim != 1 or not np.issubdtype(offsets.dtype, np.integer):
        raise ValueError("trajectory_offsets must be a one-dimensional integer array.")
    offsets = offsets.astype(np.int64, copy=True)
    if offsets.shape != (len(paths) + 1,) or offsets[0] != 0:
        raise ValueError("trajectory_offsets do not match the number of paths.")
    if np.any(np.diff(offsets) <= 0):
        raise ValueError("Every checkpoint trajectory must be non-empty.")
    lengths = np.asarray([path.size for path in paths], dtype=np.int64)
    if not np.array_equal(np.diff(offsets), lengths):
        raise ValueError("trajectory_offsets do not match latent-path lengths.")
    return offsets


def _validate_state(state: ReconstructionState) -> Tuple[int, int, int]:
    parameters = np.asarray(state.idm_parameters)
    variance = np.asarray(state.residual_variance)
    scenario_mean = np.asarray(state.scenario_mean)
    scenario_covariance = np.asarray(state.scenario_covariance)
    initial = np.asarray(state.initial_probability)
    transition = np.asarray(state.transition_probability)
    if parameters.ndim != 2 or parameters.shape[1] != 5:
        raise ValueError("Checkpoint IDM parameters must have shape (B, 5).")
    behavior_count = parameters.shape[0]
    if variance.shape != (behavior_count,):
        raise ValueError("Checkpoint residual variances do not match behavior states.")
    if scenario_mean.ndim != 2 or scenario_mean.shape[1] != 3:
        raise ValueError("Checkpoint scenario means must have shape (S, 3).")
    scenario_count = scenario_mean.shape[0]
    if scenario_covariance.shape != (scenario_count, 3, 3):
        raise ValueError("Checkpoint scenario covariances have an invalid shape.")
    joint_count = behavior_count * scenario_count
    if initial.shape != (joint_count,) or transition.shape != (
        joint_count,
        joint_count,
    ):
        raise ValueError("Checkpoint joint probabilities have an invalid shape.")
    continuous = (
        parameters,
        variance,
        scenario_mean,
        scenario_covariance,
        initial,
        transition,
    )
    if any(not np.all(np.isfinite(array)) for array in continuous):
        raise ValueError("Checkpoint state contains a non-finite value.")
    if np.any(parameters <= 0.0) or np.any(variance <= 0.0):
        raise ValueError("Checkpoint IDM parameters and variances must be positive.")
    if np.any(initial < 0.0) or not np.allclose(
        initial.sum(), 1.0, rtol=0.0, atol=1.0e-12
    ):
        raise ValueError("Checkpoint initial probabilities are invalid.")
    if np.any(transition < 0.0) or not np.allclose(
        transition.sum(axis=1), 1.0, rtol=0.0, atol=1.0e-12
    ):
        raise ValueError("Checkpoint transition probabilities are invalid.")
    for covariance in scenario_covariance:
        if not np.allclose(covariance, covariance.T, rtol=0.0, atol=1.0e-12):
            raise ValueError("Checkpoint scenario covariance is not symmetric.")
        try:
            np.linalg.cholesky(covariance)
        except np.linalg.LinAlgError as error:
            raise ValueError(
                "Checkpoint scenario covariance is not positive definite."
            ) from error
    if not state.paths:
        raise ValueError("Checkpoint state must contain latent paths.")
    for path_like in state.paths:
        path = np.asarray(path_like)
        if path.ndim != 1 or not np.issubdtype(path.dtype, np.integer):
            raise ValueError("Every checkpoint latent path must be an integer vector.")
        if np.any(path < 0) or np.any(path >= joint_count):
            raise ValueError("A checkpoint latent state is outside the joint state space.")
    return behavior_count, scenario_count, joint_count


def _validate_retained_history(
    history: Mapping[str, Array],
    *,
    retained_index: int,
    behavior_count: int,
    scenario_count: int,
    joint_count: int,
    total_observations: int,
) -> Dict[str, Array]:
    keys = set(history)
    required = set(RETAINED_HISTORY_KEYS)
    allowed = required | set(OPTIONAL_RETAINED_HISTORY_KEYS)
    missing = sorted(required - keys)
    unexpected = sorted(keys - allowed)
    if missing or unexpected:
        raise ValueError(
            f"Retained history keys are incomplete or unexpected; "
            f"missing={missing}, unexpected={unexpected}."
        )
    expected_shapes = {
        "idm_parameters": (retained_index, behavior_count, 5),
        "residual_variance": (retained_index, behavior_count),
        "scenario_mean": (retained_index, scenario_count, 3),
        "scenario_covariance": (retained_index, scenario_count, 3, 3),
        "initial_probability": (retained_index, joint_count),
        "transition_probability": (retained_index, joint_count, joint_count),
        "log_likelihood": (retained_index,),
        "slice_step_out_evaluations": (retained_index, behavior_count, 5),
        "slice_shrink_evaluations": (retained_index, behavior_count, 5),
    }
    if "paths" in history:
        expected_shapes["paths"] = (retained_index, total_observations)
    normalized: Dict[str, Array] = {}
    for name, expected_shape in expected_shapes.items():
        value = np.asarray(history[name])
        if value.shape != expected_shape or value.dtype.hasobject:
            raise ValueError(
                f"Retained history {name!r} must have shape {expected_shape}."
            )
        if name in (
            "slice_step_out_evaluations",
            "slice_shrink_evaluations",
            "paths",
        ):
            if not np.issubdtype(value.dtype, np.integer) or np.any(value < 0):
                raise ValueError(f"Retained history {name!r} must be non-negative integer data.")
            if name == "paths" and np.any(value >= joint_count):
                raise ValueError("A retained latent state is outside the joint state space.")
        elif not np.all(np.isfinite(value)):
            raise ValueError(f"Retained history {name!r} contains non-finite values.")
        normalized[name] = value.copy()
    if np.any(normalized["idm_parameters"] <= 0.0) or np.any(
        normalized["residual_variance"] <= 0.0
    ):
        raise ValueError("Retained IDM parameters and variances must be positive.")
    initial_history = normalized["initial_probability"]
    transition_history = normalized["transition_probability"]
    if np.any(initial_history < 0.0) or not np.allclose(
        initial_history.sum(axis=1), 1.0, rtol=0.0, atol=1.0e-12
    ):
        raise ValueError("Retained initial probabilities are invalid.")
    if np.any(transition_history < 0.0) or not np.allclose(
        transition_history.sum(axis=2), 1.0, rtol=0.0, atol=1.0e-12
    ):
        raise ValueError("Retained transition probabilities are invalid.")
    for covariance in normalized["scenario_covariance"].reshape(-1, 3, 3):
        if not np.allclose(covariance, covariance.T, rtol=0.0, atol=1.0e-12):
            raise ValueError("A retained scenario covariance is not symmetric.")
        try:
            np.linalg.cholesky(covariance)
        except np.linalg.LinAlgError as error:
            raise ValueError(
                "A retained scenario covariance is not positive definite."
            ) from error
    return normalized


def _payload_arrays(
    *,
    state: ReconstructionState,
    retained_history: Mapping[str, Array],
    trajectory_offsets: NDArray[np.int64],
    iteration_runtime_seconds: Optional[Array],
) -> Dict[str, Array]:
    flat_path = np.concatenate(
        [np.asarray(path, dtype=np.int64) for path in state.paths]
    )
    arrays: Dict[str, Array] = {
        "state_paths": flat_path,
        "state_initial_probability": np.asarray(state.initial_probability).copy(),
        "state_transition_probability": np.asarray(state.transition_probability).copy(),
        "state_idm_parameters": np.asarray(state.idm_parameters).copy(),
        "state_residual_variance": np.asarray(state.residual_variance).copy(),
        "state_scenario_mean": np.asarray(state.scenario_mean).copy(),
        "state_scenario_covariance": np.asarray(state.scenario_covariance).copy(),
        "trajectory_offsets": trajectory_offsets.copy(),
    }
    arrays.update(
        {f"history_{name}": np.asarray(value).copy() for name, value in retained_history.items()}
    )
    if iteration_runtime_seconds is not None:
        arrays["iteration_runtime_seconds"] = np.asarray(
            iteration_runtime_seconds, dtype=np.float64
        ).copy()
    return arrays


def _fsync_directory(path: Path) -> None:
    try:
        descriptor = os.open(str(path), os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    except OSError:
        pass
    finally:
        os.close(descriptor)


def _write_npz_exclusive(path: Path, arrays: Mapping[str, Array]) -> None:
    with path.open("xb") as handle:
        np.savez_compressed(handle, **arrays)
        handle.flush()
        os.fsync(handle.fileno())


def _write_json_exclusive(path: Path, payload: Mapping[str, Any]) -> None:
    encoded = json.dumps(
        _jsonable(payload),
        indent=2,
        sort_keys=True,
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8") + b"\n"
    with path.open("xb") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())


def write_checkpoint(
    checkpoint_root: Path,
    *,
    completed_iterations: int,
    retained_index: int,
    state: ReconstructionState,
    rng: np.random.Generator,
    retained_history: Mapping[str, Array],
    trajectory_offsets: Array,
    target_fingerprint: str,
    config_sha256: str,
    cohort_content_sha256: str,
    source_sha256: Mapping[str, str],
    environment_receipt: Mapping[str, Any],
    initialization_runtime_seconds: Optional[float] = None,
    iteration_runtime_seconds: Optional[Array] = None,
) -> CheckpointPaths:
    """Write one immutable checkpoint without advancing ``rng``."""

    if completed_iterations < 0 or retained_index < 0:
        raise ValueError("Checkpoint iteration counters must be non-negative.")
    if retained_index > completed_iterations:
        raise ValueError("retained_index cannot exceed completed_iterations.")
    if not isinstance(rng.bit_generator, np.random.PCG64):
        raise ValueError("The frozen reconstruction requires the PCG64 bit generator.")
    behavior_count, scenario_count, joint_count = _validate_state(state)
    paths = tuple(np.asarray(path, dtype=np.int64) for path in state.paths)
    offsets = _validate_offsets(trajectory_offsets, paths)
    history = _validate_retained_history(
        retained_history,
        retained_index=retained_index,
        behavior_count=behavior_count,
        scenario_count=scenario_count,
        joint_count=joint_count,
        total_observations=int(offsets[-1]),
    )
    if initialization_runtime_seconds is not None and (
        not np.isfinite(initialization_runtime_seconds)
        or initialization_runtime_seconds < 0.0
    ):
        raise ValueError("initialization_runtime_seconds must be finite and non-negative.")
    timing: Optional[NDArray[np.float64]] = None
    if iteration_runtime_seconds is not None:
        timing = np.asarray(iteration_runtime_seconds, dtype=np.float64)
        if timing.shape != (completed_iterations,) or not np.all(np.isfinite(timing)):
            raise ValueError(
                "iteration_runtime_seconds must cover every completed iteration."
            )
        if np.any(timing < 0.0):
            raise ValueError("Iteration runtimes must be non-negative.")

    target = _hex_digest(target_fingerprint, "target_fingerprint")
    config = _hex_digest(config_sha256, "config_sha256")
    cohort = _hex_digest(cohort_content_sha256, "cohort_content_sha256")
    sources = _normalized_source_hashes(source_sha256)
    environment = _jsonable(environment_receipt)
    rng_state = _jsonable(copy.deepcopy(rng.bit_generator.state))
    if rng_state.get("bit_generator") != "PCG64":
        raise ValueError("The captured RNG state is not PCG64.")

    arrays = _payload_arrays(
        state=state,
        retained_history=history,
        trajectory_offsets=offsets,
        iteration_runtime_seconds=timing,
    )
    checkpoint_root = Path(checkpoint_root)
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    final_directory = checkpoint_root / f"iter_{completed_iterations:08d}"
    if final_directory.exists():
        raise FileExistsError(f"Refusing to overwrite checkpoint {final_directory}.")
    temporary_directory = checkpoint_root / f".p{completed_iterations:08d}"
    temporary_directory.mkdir(exist_ok=False)
    payload_partial = temporary_directory / f"{PAYLOAD_FILENAME}.partial"
    payload_path = temporary_directory / PAYLOAD_FILENAME
    _write_npz_exclusive(payload_partial, arrays)
    os.replace(payload_partial, payload_path)
    _fsync_directory(temporary_directory)

    array_manifest = {
        name: {
            "dtype": str(np.asarray(value).dtype),
            "shape": list(np.asarray(value).shape),
            "sha256": _sha256_array(np.asarray(value)),
        }
        for name, value in sorted(arrays.items())
    }
    identity = {
        "target_fingerprint": target,
        "config_sha256": config,
        "cohort_content_sha256": cohort,
        "source_sha256": sources,
        "environment_receipt": environment,
    }
    receipt: Dict[str, Any] = {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "completed_iterations": completed_iterations,
        "retained_index": retained_index,
        "stores_latent_paths": "paths" in history,
        "initialization_runtime_seconds": initialization_runtime_seconds,
        "rng_state": rng_state,
        "execution_identity": identity,
        "execution_identity_sha256": hashlib.sha256(
            _canonical_json_bytes(identity)
        ).hexdigest(),
        "payload": {
            "filename": PAYLOAD_FILENAME,
            "bytes": payload_path.stat().st_size,
            "sha256": _sha256_file(payload_path),
            "arrays": array_manifest,
        },
    }
    receipt["receipt_content_sha256"] = hashlib.sha256(
        _canonical_json_bytes(receipt)
    ).hexdigest()
    receipt_partial = temporary_directory / f"{RECEIPT_FILENAME}.partial"
    receipt_path = temporary_directory / RECEIPT_FILENAME
    _write_json_exclusive(receipt_partial, receipt)
    os.replace(receipt_partial, receipt_path)
    _fsync_directory(temporary_directory)
    if final_directory.exists():
        raise FileExistsError(f"Refusing to overwrite checkpoint {final_directory}.")
    os.replace(temporary_directory, final_directory)
    _fsync_directory(checkpoint_root)
    return CheckpointPaths(
        directory=final_directory,
        payload=final_directory / PAYLOAD_FILENAME,
        receipt=final_directory / RECEIPT_FILENAME,
    )


def _verify_receipt_integrity(receipt: Mapping[str, Any]) -> None:
    stored = str(receipt.get("receipt_content_sha256", "")).lower()
    unsigned = dict(receipt)
    unsigned.pop("receipt_content_sha256", None)
    actual = hashlib.sha256(_canonical_json_bytes(unsigned)).hexdigest()
    if stored != actual:
        raise ValueError("Checkpoint receipt integrity hash mismatch.")


def _verify_expected_identity(
    receipt: Mapping[str, Any],
    *,
    expected_target_fingerprint: str,
    expected_config_sha256: str,
    expected_cohort_content_sha256: str,
    expected_source_sha256: Mapping[str, str],
    expected_environment_receipt: Mapping[str, Any],
) -> Mapping[str, Any]:
    identity = receipt.get("execution_identity")
    if not isinstance(identity, Mapping):
        raise ValueError("Checkpoint lacks an execution identity.")
    expected = {
        "target_fingerprint": _hex_digest(
            expected_target_fingerprint, "expected_target_fingerprint"
        ),
        "config_sha256": _hex_digest(expected_config_sha256, "expected_config_sha256"),
        "cohort_content_sha256": _hex_digest(
            expected_cohort_content_sha256, "expected_cohort_content_sha256"
        ),
        "source_sha256": _normalized_source_hashes(expected_source_sha256),
        "environment_receipt": _jsonable(expected_environment_receipt),
    }
    identity_hash = hashlib.sha256(_canonical_json_bytes(identity)).hexdigest()
    if receipt.get("execution_identity_sha256") != identity_hash:
        raise ValueError("Checkpoint execution-identity hash mismatch.")
    for name, expected_value in expected.items():
        if identity.get(name) != expected_value:
            raise ValueError(f"Checkpoint {name} does not match the requested execution.")
    return identity


def _load_payload(
    payload_path: Path, payload_receipt: Mapping[str, Any]
) -> Dict[str, Array]:
    if payload_receipt.get("filename") != PAYLOAD_FILENAME:
        raise ValueError("Checkpoint payload filename is invalid.")
    if not payload_path.is_file():
        raise ValueError("Checkpoint payload is missing.")
    if payload_path.stat().st_size != int(payload_receipt.get("bytes", -1)):
        raise ValueError("Checkpoint payload size mismatch.")
    if _sha256_file(payload_path) != payload_receipt.get("sha256"):
        raise ValueError("Checkpoint payload SHA-256 mismatch.")
    manifest = payload_receipt.get("arrays")
    if not isinstance(manifest, Mapping):
        raise ValueError("Checkpoint payload lacks an array manifest.")
    with np.load(payload_path, allow_pickle=False) as loaded:
        if set(loaded.files) != set(manifest):
            raise ValueError("Checkpoint payload array names do not match the receipt.")
        arrays = {name: loaded[name].copy() for name in loaded.files}
    for name, array in arrays.items():
        metadata = manifest[name]
        if not isinstance(metadata, Mapping):
            raise ValueError(f"Checkpoint array metadata for {name!r} is invalid.")
        if list(array.shape) != list(metadata.get("shape", [])):
            raise ValueError(f"Checkpoint array shape mismatch for {name!r}.")
        if str(array.dtype) != str(metadata.get("dtype")):
            raise ValueError(f"Checkpoint array dtype mismatch for {name!r}.")
        if _sha256_array(array) != metadata.get("sha256"):
            raise ValueError(f"Checkpoint array SHA-256 mismatch for {name!r}.")
    return arrays


def load_checkpoint(
    checkpoint_directory: Path,
    *,
    expected_target_fingerprint: str,
    expected_config_sha256: str,
    expected_cohort_content_sha256: str,
    expected_source_sha256: Mapping[str, str],
    expected_environment_receipt: Mapping[str, Any],
) -> RestoredCheckpoint:
    """Load a complete checkpoint and fail closed on any identity mismatch."""

    directory = Path(checkpoint_directory)
    receipt_path = directory / RECEIPT_FILENAME
    payload_path = directory / PAYLOAD_FILENAME
    if not receipt_path.is_file():
        raise ValueError("Checkpoint receipt is missing; partial checkpoint rejected.")
    with receipt_path.open("r", encoding="utf-8") as handle:
        receipt = json.load(handle)
    if receipt.get("schema_version") != CHECKPOINT_SCHEMA_VERSION:
        raise ValueError("Unsupported checkpoint schema version.")
    _verify_receipt_integrity(receipt)
    identity = _verify_expected_identity(
        receipt,
        expected_target_fingerprint=expected_target_fingerprint,
        expected_config_sha256=expected_config_sha256,
        expected_cohort_content_sha256=expected_cohort_content_sha256,
        expected_source_sha256=expected_source_sha256,
        expected_environment_receipt=expected_environment_receipt,
    )
    completed_iterations = int(receipt.get("completed_iterations", -1))
    retained_index = int(receipt.get("retained_index", -1))
    if completed_iterations < 0 or retained_index < 0 or retained_index > completed_iterations:
        raise ValueError("Checkpoint iteration counters are invalid.")
    payload_receipt = receipt.get("payload")
    if not isinstance(payload_receipt, Mapping):
        raise ValueError("Checkpoint lacks payload metadata.")
    arrays = _load_payload(payload_path, payload_receipt)
    offsets = np.asarray(arrays.pop("trajectory_offsets"), dtype=np.int64)
    flat_path = np.asarray(arrays.pop("state_paths"), dtype=np.int64)
    if offsets.ndim != 1 or offsets.size < 2 or offsets[0] != 0:
        raise ValueError("Checkpoint trajectory offsets are invalid.")
    if offsets[-1] != flat_path.size or np.any(np.diff(offsets) <= 0):
        raise ValueError("Checkpoint trajectory offsets do not cover state paths.")
    paths = tuple(
        flat_path[offsets[index] : offsets[index + 1]].copy()
        for index in range(offsets.size - 1)
    )
    state = ReconstructionState(
        paths=paths,
        initial_probability=np.asarray(
            arrays.pop("state_initial_probability"), dtype=float
        ),
        transition_probability=np.asarray(
            arrays.pop("state_transition_probability"), dtype=float
        ),
        idm_parameters=np.asarray(arrays.pop("state_idm_parameters"), dtype=float),
        residual_variance=np.asarray(
            arrays.pop("state_residual_variance"), dtype=float
        ),
        scenario_mean=np.asarray(arrays.pop("state_scenario_mean"), dtype=float),
        scenario_covariance=np.asarray(
            arrays.pop("state_scenario_covariance"), dtype=float
        ),
    )
    behavior_count, scenario_count, joint_count = _validate_state(state)
    offsets = _validate_offsets(offsets, paths)
    timing = arrays.pop("iteration_runtime_seconds", None)
    if timing is not None:
        timing = np.asarray(timing, dtype=np.float64)
        if timing.shape != (completed_iterations,) or not np.all(np.isfinite(timing)):
            raise ValueError("Checkpoint iteration timing prefix is invalid.")
        if np.any(timing < 0.0):
            raise ValueError("Checkpoint iteration timing prefix is negative.")
    history = {
        name.removeprefix("history_"): value
        for name, value in arrays.items()
        if name.startswith("history_")
    }
    if len(history) != len(arrays):
        raise ValueError("Checkpoint payload contains an unrecognized array.")
    history = _validate_retained_history(
        history,
        retained_index=retained_index,
        behavior_count=behavior_count,
        scenario_count=scenario_count,
        joint_count=joint_count,
        total_observations=int(offsets[-1]),
    )
    stores_paths = bool(receipt.get("stores_latent_paths"))
    if stores_paths != ("paths" in history):
        raise ValueError("Checkpoint latent-history policy is internally inconsistent.")
    rng_state = receipt.get("rng_state")
    if not isinstance(rng_state, Mapping) or rng_state.get("bit_generator") != "PCG64":
        raise ValueError("Checkpoint PCG64 state is missing or invalid.")
    initialization_runtime = receipt.get("initialization_runtime_seconds")
    if initialization_runtime is not None:
        initialization_runtime = float(initialization_runtime)
        if not np.isfinite(initialization_runtime) or initialization_runtime < 0.0:
            raise ValueError("Checkpoint initialization runtime is invalid.")
    checkpoint_paths = CheckpointPaths(directory, payload_path, receipt_path)
    return RestoredCheckpoint(
        completed_iterations=completed_iterations,
        retained_index=retained_index,
        state=state,
        rng_state=copy.deepcopy(dict(rng_state)),
        retained_history=history,
        trajectory_offsets=offsets,
        target_fingerprint=str(identity["target_fingerprint"]),
        config_sha256=str(identity["config_sha256"]),
        cohort_content_sha256=str(identity["cohort_content_sha256"]),
        source_sha256=dict(identity["source_sha256"]),
        environment_receipt=copy.deepcopy(identity["environment_receipt"]),
        initialization_runtime_seconds=initialization_runtime,
        iteration_runtime_seconds=timing,
        checkpoint_paths=checkpoint_paths,
    )
