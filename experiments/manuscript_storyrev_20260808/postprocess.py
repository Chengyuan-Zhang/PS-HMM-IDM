"""Reproducible posterior alignment and summaries for the Scheme B manuscript.

This module never falls back to legacy caches and never invents a result.  Its CLI
requires all five completed formal arms by default.  Label alignment is performed
independently in the behavior and scenario coordinates for every retained draw;
the resulting permutations are then applied synchronously to every object whose
indices carry latent-state meaning.
"""

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
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple

import numpy as np
import scipy
from numpy.typing import NDArray
from scipy.optimize import linear_sum_assignment

from .run_exact_reconstruction import (
    REQUIRED_ARRAYS,
    _canonical_cohort_hash,
    _verified_manifest_input_fingerprint,
)
from .target import COVARIATE_ORDER, original_idm_acceleration


FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]
BoolArray = NDArray[np.bool_]

FORMAL_NUM_ITERATIONS = 8000
FORMAL_BURN_IN = 6000
FORMAL_THIN = 1
FORMAL_RETAINED_DRAWS = 2000
PARAMETER_NAMES = (
    "desired_speed",
    "minimum_gap",
    "time_headway",
    "maximum_acceleration",
    "comfortable_deceleration",
)
CONTEXT_NAMES = ("speed", "gap", "relative_speed")
REQUIRED_CHAIN_ARRAYS = (
    "idm_parameters",
    "residual_variance",
    "scenario_mean",
    "scenario_covariance",
    "initial_probability",
    "transition_probability",
    "log_likelihood",
    "slice_step_out_evaluations",
    "slice_shrink_evaluations",
    "trajectory_offsets",
    "paths",
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Cannot serialize {type(value).__name__}.")


def _write_json_new(path: Path, payload: Mapping[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite {path}.")
    path.write_text(
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
            allow_nan=False,
            default=_json_default,
        )
        + "\n",
        encoding="utf-8",
    )


def _write_text_new(path: Path, content: str) -> None:
    """Write UTF-8 text with LF newlines while refusing an existing target."""

    with path.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(content)


def _git_commit(repository_root: Path) -> Optional[str]:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository_root,
        capture_output=True,
        text=True,
        check=False,
    )
    return completed.stdout.strip() if completed.returncode == 0 else None


def _git_dirty(repository_root: Path) -> Optional[bool]:
    completed = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=repository_root,
        capture_output=True,
        text=True,
        check=False,
    )
    return bool(completed.stdout.strip()) if completed.returncode == 0 else None


@dataclass(frozen=True)
class PostprocessConfig:
    """Frozen postprocessing choices, including explicit story-gate thresholds."""

    required_arm_ids: Tuple[str, ...]
    manuscript_arm_id: str
    pooled_arm_id: str
    fixed_case_indices: Tuple[int, ...]
    credible_interval: Tuple[float, float]
    rare_occupancy_threshold: float
    empty_draw_fraction_threshold: float
    case_supported_occupancy_threshold: float
    minimum_passing_fixed_cases: int
    response_anchors: FloatArray
    feature_scale_floor: float
    time_space_max_trajectories: int
    time_space_lane_ids: Tuple[int, ...]
    sampling_frequency_hz: float

    @classmethod
    def from_json(cls, path: Path) -> "PostprocessConfig":
        with path.open("r", encoding="utf-8") as handle:
            raw = json.load(handle)
        if int(raw.get("schema_version", -1)) != 1:
            raise ValueError("Unsupported postprocess schema version.")
        alignment = raw["alignment"]
        if alignment.get("reference_draw") != "maximum_log_likelihood":
            raise ValueError("Only the audited maximum-log-likelihood pivot is supported.")
        figure = raw["figure"]
        config = cls(
            required_arm_ids=tuple(str(value) for value in raw["required_arm_ids"]),
            manuscript_arm_id=str(raw["manuscript_arm_id"]),
            pooled_arm_id=str(raw["pooled_arm_id"]),
            fixed_case_indices=tuple(
                int(value) for value in raw["fixed_case_indices_zero_based"]
            ),
            credible_interval=tuple(
                float(value) for value in raw["credible_interval_probabilities"]
            ),
            rare_occupancy_threshold=float(raw["rare_occupancy_threshold"]),
            empty_draw_fraction_threshold=float(
                raw["empty_draw_fraction_threshold"]
            ),
            case_supported_occupancy_threshold=float(
                raw["case_supported_occupancy_threshold"]
            ),
            minimum_passing_fixed_cases=int(raw["minimum_passing_fixed_cases"]),
            response_anchors=np.asarray(
                alignment["response_anchors_physical_order_v_s_dv"], dtype=float
            ),
            feature_scale_floor=float(alignment["feature_scale_floor"]),
            time_space_max_trajectories=int(
                figure["time_space_max_trajectories"]
            ),
            time_space_lane_ids=tuple(int(value) for value in figure["time_space_lane_ids"]),
            sampling_frequency_hz=float(figure["sampling_frequency_hz"]),
        )
        config.validate()
        return config

    def validate(self) -> None:
        if len(self.required_arm_ids) != len(set(self.required_arm_ids)):
            raise ValueError("required_arm_ids must be unique.")
        if self.manuscript_arm_id not in self.required_arm_ids:
            raise ValueError("The manuscript arm must be required.")
        if self.pooled_arm_id not in self.required_arm_ids:
            raise ValueError("The pooled arm must be required.")
        if self.fixed_case_indices != (2, 3, 6, 9):
            raise ValueError("Scheme B fixes the manuscript cases at indices 2, 3, 6, and 9.")
        if self.credible_interval != (0.025, 0.975):
            raise ValueError("The manuscript interval must be the equal-tail 95% interval.")
        expected_gate_values = {
            "rare_occupancy_threshold": (self.rare_occupancy_threshold, 0.01),
            "empty_draw_fraction_threshold": (
                self.empty_draw_fraction_threshold,
                0.50,
            ),
            "case_supported_occupancy_threshold": (
                self.case_supported_occupancy_threshold,
                0.05,
            ),
        }
        for name, (actual, expected) in expected_gate_values.items():
            if actual != expected:
                raise ValueError(
                    f"Scheme B fixes {name} at {expected}; received {actual}."
                )
        if self.minimum_passing_fixed_cases != 3:
            raise ValueError(
                "Scheme B requires exactly three of the four fixed cases to pass."
            )
        if self.response_anchors.ndim != 2 or self.response_anchors.shape[1] != 3:
            raise ValueError("Response anchors must have shape (A, 3).")
        if not np.all(np.isfinite(self.response_anchors)):
            raise ValueError("Response anchors must be finite.")
        if np.any(self.response_anchors[:, 0] < 0.0) or np.any(
            self.response_anchors[:, 1] <= 0.0
        ):
            raise ValueError("Response anchors must have nonnegative speed and positive gap.")
        if self.feature_scale_floor <= 0.0:
            raise ValueError("feature_scale_floor must be positive.")
        if self.time_space_max_trajectories <= 0 or not self.time_space_lane_ids:
            raise ValueError("Time-space figure configuration is invalid.")
        if self.sampling_frequency_hz <= 0.0:
            raise ValueError("sampling_frequency_hz must be positive.")


@dataclass(frozen=True)
class CohortArrays:
    physical_context: FloatArray
    acceleration: FloatArray
    offsets: IntArray
    frames: IntArray
    trajectory_ids: NDArray[np.str_]
    recording_ids: IntArray
    follower_ids: IntArray
    pair_nos: IntArray
    lengths: IntArray
    content_sha256: str
    npz_sha256: str
    manifest_sha256: str

    @property
    def num_observations(self) -> int:
        return int(self.acceleration.size)

    @property
    def num_trajectories(self) -> int:
        return int(self.lengths.size)


@dataclass(frozen=True)
class PosteriorRun:
    run_directory: Path
    arm_id: str
    num_behavior_states: int
    num_scenario_states: int
    idm_parameters: FloatArray
    residual_variance: FloatArray
    scenario_mean: FloatArray
    scenario_covariance: FloatArray
    initial_probability: FloatArray
    transition_probability: FloatArray
    log_likelihood: FloatArray
    paths: IntArray
    trajectory_offsets: IntArray
    standardizer_mean: FloatArray
    standardizer_scale: FloatArray
    config: Mapping[str, Any]
    run_metadata: Mapping[str, Any]

    @property
    def num_draws(self) -> int:
        return int(self.idm_parameters.shape[0])

    @property
    def num_joint_states(self) -> int:
        return self.num_behavior_states * self.num_scenario_states


@dataclass(frozen=True)
class AlignedPosterior:
    idm_parameters: FloatArray
    residual_variance: FloatArray
    scenario_mean: FloatArray
    scenario_covariance: FloatArray
    initial_probability: FloatArray
    transition_probability: FloatArray
    paths: IntArray
    behavior_old_for_new: IntArray
    scenario_old_for_new: IntArray
    reference_draw_index: int
    behavior_feature_center: FloatArray
    behavior_feature_scale: FloatArray
    scenario_feature_center: FloatArray
    scenario_feature_scale: FloatArray


@dataclass(frozen=True)
class ArmAnalysis:
    arm_id: str
    summary: Mapping[str, Any]
    arrays: Mapping[str, NDArray[np.generic]]


def _manifest_array_metadata(manifest: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    cohort = manifest.get("cohort")
    if not isinstance(cohort, Mapping):
        raise ValueError("Cohort manifest lacks cohort metadata.")
    arrays = cohort.get("array_manifest")
    if not isinstance(arrays, Mapping) or not isinstance(arrays.get(name), Mapping):
        raise ValueError(f"Cohort manifest lacks array metadata for {name!r}.")
    return arrays[name]


def _sha256_array(array: NDArray[np.generic]) -> str:
    materialized = np.asarray(array)
    if materialized.dtype.hasobject:
        raise TypeError("Object arrays are not valid provenance payloads.")
    if materialized.dtype.kind in "iufcbmM":
        materialized = materialized.astype(materialized.dtype.newbyteorder("<"), copy=False)
    contiguous = np.ascontiguousarray(materialized)
    header = {"dtype": contiguous.dtype.str, "shape": list(contiguous.shape)}
    digest = hashlib.sha256()
    digest.update(
        json.dumps(header, sort_keys=True, separators=(",", ":")).encode("utf-8")
    )
    digest.update(b"\0")
    digest.update(contiguous.tobytes(order="C"))
    return digest.hexdigest()


def load_cohort(cohort_npz: Path, manifest_path: Path) -> CohortArrays:
    """Load and independently re-verify the frozen cohort used by every arm."""

    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    with np.load(cohort_npz, allow_pickle=False) as loaded:
        missing = [name for name in REQUIRED_ARRAYS if name not in loaded.files]
        if missing:
            raise ValueError(f"Cohort artifact lacks required arrays: {missing}.")
        raw = {name: loaded[name].copy() for name in REQUIRED_ARRAYS}
    for name, array in raw.items():
        metadata = _manifest_array_metadata(manifest, name)
        if list(array.shape) != list(metadata.get("shape", [])):
            raise ValueError(f"Cohort shape mismatch for {name}.")
        if str(array.dtype) != str(metadata.get("dtype")):
            raise ValueError(f"Cohort dtype mismatch for {name}.")
        if _sha256_array(array) != str(metadata.get("sha256", "")).lower():
            raise ValueError(f"Cohort SHA-256 mismatch for {name}.")
    content_sha256 = _verified_manifest_input_fingerprint(manifest)
    if content_sha256 is None:
        content_sha256 = _canonical_cohort_hash(raw)
    physical = np.asarray(raw["physical_context"], dtype=float)
    acceleration = np.asarray(raw["acceleration"], dtype=float)
    offsets = np.asarray(raw["offsets"], dtype=np.int64)
    lengths = np.asarray(raw["lengths"], dtype=np.int64)
    if physical.shape != (acceleration.size, 3):
        raise ValueError("Cohort context and acceleration shapes do not align.")
    if offsets.shape != (lengths.size + 1,) or not np.array_equal(
        np.diff(offsets), lengths
    ):
        raise ValueError("Cohort offsets and lengths do not align.")
    return CohortArrays(
        physical_context=physical,
        acceleration=acceleration,
        offsets=offsets,
        frames=np.asarray(raw["frames"], dtype=np.int64),
        trajectory_ids=np.asarray(raw["trajectory_ids"]).astype(str),
        recording_ids=np.asarray(raw["recording_ids"], dtype=np.int64),
        follower_ids=np.asarray(raw["follower_ids"], dtype=np.int64),
        pair_nos=np.asarray(raw["pair_nos"], dtype=np.int64),
        lengths=lengths,
        content_sha256=content_sha256,
        npz_sha256=_sha256_file(cohort_npz),
        manifest_sha256=_sha256_file(manifest_path),
    )


def _verify_run_artifacts(run_directory: Path) -> None:
    completed_path = run_directory / "COMPLETED.json"
    artifact_path = run_directory / "artifact_manifest.json"
    if not completed_path.is_file() or not artifact_path.is_file():
        raise ValueError(f"Run is not complete: {run_directory}.")
    with artifact_path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise ValueError("Run artifact manifest is malformed.")
    for name in (
        "chain.npz",
        "config.json",
        "standardizer.json",
        "target.json",
        "seed_lineage.json",
        "run_metadata.json",
    ):
        metadata = artifacts.get(name)
        path = run_directory / name
        if not isinstance(metadata, Mapping) or not path.is_file():
            raise ValueError(f"Run artifact {name!r} is missing.")
        if int(metadata.get("bytes", -1)) != path.stat().st_size:
            raise ValueError(f"Run artifact size mismatch for {name}.")
        if str(metadata.get("sha256", "")).lower() != _sha256_file(path):
            raise ValueError(f"Run artifact hash mismatch for {name}.")


def load_posterior_run(run_directory: Path, *, require_formal: bool = True) -> PosteriorRun:
    """Load one completed posterior run and validate its full output schema."""

    run_directory = run_directory.resolve()
    _verify_run_artifacts(run_directory)
    with (run_directory / "config.json").open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    with (run_directory / "run_metadata.json").open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)
    with (run_directory / "target.json").open("r", encoding="utf-8") as handle:
        target = json.load(handle)
    with (run_directory / "standardizer.json").open("r", encoding="utf-8") as handle:
        standardizer = json.load(handle)
    if tuple(standardizer.get("context_order", ())) != COVARIATE_ORDER:
        raise ValueError("Run standardizer context order is not canonical [v, s, dv].")
    arm_id = str(config["arm_id"])
    if str(metadata.get("arm_id")) != arm_id:
        raise ValueError("Run metadata arm ID disagrees with config.")
    if str(metadata.get("scientific_target_fingerprint")) != str(
        target.get("scientific_target_fingerprint")
    ):
        raise ValueError("Run target fingerprint receipt is inconsistent.")
    behavior = int(config["states"]["behavior"])
    scenario = int(config["states"]["scenario"])
    joint = behavior * scenario
    chain_config = config["chain"]
    if require_formal:
        formal_values = (
            int(chain_config["num_iterations"]) == FORMAL_NUM_ITERATIONS,
            int(chain_config["burn_in"]) == FORMAL_BURN_IN,
            int(chain_config["thin"]) == FORMAL_THIN,
            bool(chain_config["store_latent_paths"]),
            int(metadata.get("num_retained_draws", -1)) == FORMAL_RETAINED_DRAWS,
        )
        if not all(formal_values) or arm_id.startswith("SMOKE_"):
            raise ValueError(f"Non-formal run rejected by default: {run_directory}.")
    with np.load(run_directory / "chain.npz", allow_pickle=False) as loaded:
        missing = [name for name in REQUIRED_CHAIN_ARRAYS if name not in loaded.files]
        if missing:
            raise ValueError(f"Posterior chain lacks required arrays: {missing}.")
        arrays = {name: loaded[name].copy() for name in REQUIRED_CHAIN_ARRAYS}
    idm = np.asarray(arrays["idm_parameters"], dtype=float)
    draws = idm.shape[0] if idm.ndim == 3 else -1
    observations = int(np.asarray(arrays["trajectory_offsets"])[-1])
    expected_shapes = {
        "idm_parameters": (draws, behavior, 5),
        "residual_variance": (draws, behavior),
        "scenario_mean": (draws, scenario, 3),
        "scenario_covariance": (draws, scenario, 3, 3),
        "initial_probability": (draws, joint),
        "transition_probability": (draws, joint, joint),
        "log_likelihood": (draws,),
        "slice_step_out_evaluations": (draws, behavior, 5),
        "slice_shrink_evaluations": (draws, behavior, 5),
        "paths": (draws, observations),
    }
    for name, shape in expected_shapes.items():
        if arrays[name].shape != shape:
            raise ValueError(
                f"Posterior array {name} has shape {arrays[name].shape}, expected {shape}."
            )
    finite_names = tuple(name for name in expected_shapes if name != "paths")
    for name in finite_names:
        if not np.all(np.isfinite(arrays[name])):
            raise ValueError(f"Posterior array {name} contains non-finite values.")
    raw_paths = np.asarray(arrays["paths"])
    if raw_paths.dtype.kind not in "iu":
        raise ValueError("Posterior paths must use an integer dtype.")
    paths = raw_paths.astype(np.int16, copy=False)
    if np.any(paths < 0) or np.any(paths >= joint):
        raise ValueError("Posterior paths contain out-of-range joint states.")
    if np.any(np.asarray(arrays["residual_variance"]) <= 0.0):
        raise ValueError("Residual variances must be positive.")
    transition = np.asarray(arrays["transition_probability"], dtype=float)
    initial = np.asarray(arrays["initial_probability"], dtype=float)
    if not np.allclose(transition.sum(axis=2), 1.0, atol=1.0e-10):
        raise ValueError("Transition rows are not stochastic.")
    if not np.allclose(initial.sum(axis=1), 1.0, atol=1.0e-10):
        raise ValueError("Initial distributions are not stochastic.")
    standardizer_mean = np.asarray(standardizer["mean"], dtype=float)
    standardizer_scale = np.asarray(standardizer["scale"], dtype=float)
    if standardizer_mean.shape != (3,) or standardizer_scale.shape != (3,):
        raise ValueError("Run standardizer vectors must have length three.")
    if np.any(standardizer_scale <= 0.0):
        raise ValueError("Run standardizer scales must be positive.")
    return PosteriorRun(
        run_directory=run_directory,
        arm_id=arm_id,
        num_behavior_states=behavior,
        num_scenario_states=scenario,
        idm_parameters=idm,
        residual_variance=np.asarray(arrays["residual_variance"], dtype=float),
        scenario_mean=np.asarray(arrays["scenario_mean"], dtype=float),
        scenario_covariance=np.asarray(arrays["scenario_covariance"], dtype=float),
        initial_probability=initial,
        transition_probability=transition,
        log_likelihood=np.asarray(arrays["log_likelihood"], dtype=float),
        paths=paths,
        trajectory_offsets=np.asarray(arrays["trajectory_offsets"], dtype=np.int64),
        standardizer_mean=standardizer_mean,
        standardizer_scale=standardizer_scale,
        config=config,
        run_metadata=metadata,
    )


def _standardize_feature_cube(
    features: FloatArray, scale_floor: float
) -> Tuple[FloatArray, FloatArray, FloatArray]:
    if features.ndim != 3 or not np.all(np.isfinite(features)):
        raise ValueError("Alignment features must be a finite draw-state-feature cube.")
    flattened = features.reshape(-1, features.shape[-1])
    center = np.median(flattened, axis=0)
    q25, q75 = np.quantile(flattened, (0.25, 0.75), axis=0)
    robust_scale = (q75 - q25) / 1.349
    standard_scale = np.std(flattened, axis=0)
    scale = np.where(robust_scale > scale_floor, robust_scale, standard_scale)
    scale = np.where(scale > scale_floor, scale, 1.0)
    return (features - center) / scale, center, scale


def behavior_alignment_features(
    idm_parameters: FloatArray,
    response_anchors: FloatArray,
    *,
    minimum_allowed_speed_mps: float,
) -> FloatArray:
    """Combine log-IDM profiles with responses at fixed physical anchors."""

    parameters = np.asarray(idm_parameters, dtype=float)
    if parameters.ndim != 3 or parameters.shape[2] != 5:
        raise ValueError("IDM draw array must have shape (D, B, 5).")
    if np.any(parameters <= 0.0):
        raise ValueError("IDM parameters must be positive before taking logarithms.")
    responses = behavior_response_at_anchors(
        parameters,
        response_anchors,
        minimum_allowed_speed_mps=minimum_allowed_speed_mps,
    )
    return np.concatenate((np.log(parameters), responses), axis=2)


def behavior_response_at_anchors(
    idm_parameters: FloatArray,
    response_anchors: FloatArray,
    *,
    minimum_allowed_speed_mps: float,
) -> FloatArray:
    """Evaluate every response-state draw at the frozen physical anchors."""

    parameters = np.asarray(idm_parameters, dtype=float)
    anchors = np.asarray(response_anchors, dtype=float)
    if parameters.ndim != 3 or parameters.shape[2] != 5:
        raise ValueError("IDM draw array must have shape (D, B, 5).")
    if anchors.ndim != 2 or anchors.shape[1] != 3:
        raise ValueError("Response anchors must have shape (A, 3).")
    responses = np.empty((parameters.shape[0], parameters.shape[1], anchors.shape[0]))
    for draw in range(parameters.shape[0]):
        for state in range(parameters.shape[1]):
            responses[draw, state] = original_idm_acceleration(
                parameters[draw, state],
                anchors,
                minimum_allowed_speed_mps=minimum_allowed_speed_mps,
            )
    if not np.all(np.isfinite(responses)):
        raise ValueError("Fixed-anchor IDM responses are non-finite.")
    return responses


def scenario_alignment_features(
    scenario_mean: FloatArray, scenario_covariance: FloatArray
) -> FloatArray:
    """Use standardized-space Gaussian means and covariance lower triangles."""

    means = np.asarray(scenario_mean, dtype=float)
    covariance = np.asarray(scenario_covariance, dtype=float)
    if means.ndim != 3 or means.shape[2] != 3:
        raise ValueError("Scenario means must have shape (D, S, 3).")
    if covariance.shape != means.shape[:2] + (3, 3):
        raise ValueError("Scenario covariances do not align with means.")
    lower = np.tril_indices(3)
    covariance_features = covariance[:, :, lower[0], lower[1]]
    return np.concatenate((means, covariance_features), axis=2)


def hungarian_old_for_new(
    standardized_features: FloatArray, reference_draw_index: int
) -> IntArray:
    """Return, for every draw, the old state index assigned to each new label."""

    features = np.asarray(standardized_features, dtype=float)
    if features.ndim != 3:
        raise ValueError("standardized_features must have shape (D, K, F).")
    draws, states, _ = features.shape
    if not 0 <= reference_draw_index < draws:
        raise ValueError("reference_draw_index is out of range.")
    reference = features[reference_draw_index]
    permutations = np.empty((draws, states), dtype=np.int64)
    for draw in range(draws):
        delta = reference[:, None, :] - features[draw][None, :, :]
        cost = np.einsum("rsf,rsf->rs", delta, delta)
        rows, columns = linear_sum_assignment(cost)
        if not np.array_equal(rows, np.arange(states)):
            raise AssertionError("Hungarian solver did not return every reference row.")
        permutations[draw] = columns
    return permutations


def apply_coordinate_permutations(
    run: PosteriorRun,
    behavior_old_for_new: IntArray,
    scenario_old_for_new: IntArray,
    *,
    reference_draw_index: int,
    behavior_feature_center: FloatArray,
    behavior_feature_scale: FloatArray,
    scenario_feature_center: FloatArray,
    scenario_feature_scale: FloatArray,
) -> AlignedPosterior:
    """Synchronously reindex parameters, paths, rho, and both transition axes."""

    draws = run.num_draws
    behavior = run.num_behavior_states
    scenario = run.num_scenario_states
    if behavior_old_for_new.shape != (draws, behavior):
        raise ValueError("Behavior permutations have an invalid shape.")
    if scenario_old_for_new.shape != (draws, scenario):
        raise ValueError("Scenario permutations have an invalid shape.")
    aligned_idm = np.empty_like(run.idm_parameters)
    aligned_variance = np.empty_like(run.residual_variance)
    aligned_mean = np.empty_like(run.scenario_mean)
    aligned_covariance = np.empty_like(run.scenario_covariance)
    aligned_initial = np.empty_like(run.initial_probability)
    aligned_transition = np.empty_like(run.transition_probability)
    aligned_paths = np.empty_like(run.paths)
    canonical_behavior = np.arange(behavior, dtype=np.int64)
    canonical_scenario = np.arange(scenario, dtype=np.int64)
    for draw in range(draws):
        behavior_permutation = np.asarray(behavior_old_for_new[draw], dtype=np.int64)
        scenario_permutation = np.asarray(scenario_old_for_new[draw], dtype=np.int64)
        if not np.array_equal(np.sort(behavior_permutation), canonical_behavior):
            raise ValueError("A behavior assignment is not a permutation.")
        if not np.array_equal(np.sort(scenario_permutation), canonical_scenario):
            raise ValueError("A scenario assignment is not a permutation.")
        aligned_idm[draw] = run.idm_parameters[draw, behavior_permutation]
        aligned_variance[draw] = run.residual_variance[draw, behavior_permutation]
        aligned_mean[draw] = run.scenario_mean[draw, scenario_permutation]
        aligned_covariance[draw] = run.scenario_covariance[
            draw, scenario_permutation
        ]
        old_joint_for_new = (
            behavior_permutation[:, None] * scenario
            + scenario_permutation[None, :]
        ).reshape(-1)
        aligned_initial[draw] = run.initial_probability[draw, old_joint_for_new]
        aligned_transition[draw] = run.transition_probability[draw][
            np.ix_(old_joint_for_new, old_joint_for_new)
        ]
        new_for_old_behavior = np.empty(behavior, dtype=np.int64)
        new_for_old_behavior[behavior_permutation] = canonical_behavior
        new_for_old_scenario = np.empty(scenario, dtype=np.int64)
        new_for_old_scenario[scenario_permutation] = canonical_scenario
        old_paths = run.paths[draw]
        old_behavior = old_paths // scenario
        old_scenario = old_paths % scenario
        aligned_paths[draw] = (
            new_for_old_behavior[old_behavior] * scenario
            + new_for_old_scenario[old_scenario]
        )
    if not np.allclose(aligned_transition.sum(axis=2), 1.0, atol=1.0e-10):
        raise AssertionError("Aligned transition rows lost stochasticity.")
    if not np.allclose(aligned_initial.sum(axis=1), 1.0, atol=1.0e-10):
        raise AssertionError("Aligned initial distributions lost stochasticity.")
    return AlignedPosterior(
        idm_parameters=aligned_idm,
        residual_variance=aligned_variance,
        scenario_mean=aligned_mean,
        scenario_covariance=aligned_covariance,
        initial_probability=aligned_initial,
        transition_probability=aligned_transition,
        paths=aligned_paths,
        behavior_old_for_new=np.asarray(behavior_old_for_new, dtype=np.int64),
        scenario_old_for_new=np.asarray(scenario_old_for_new, dtype=np.int64),
        reference_draw_index=int(reference_draw_index),
        behavior_feature_center=np.asarray(behavior_feature_center, dtype=float),
        behavior_feature_scale=np.asarray(behavior_feature_scale, dtype=float),
        scenario_feature_center=np.asarray(scenario_feature_center, dtype=float),
        scenario_feature_scale=np.asarray(scenario_feature_scale, dtype=float),
    )


def align_posterior(run: PosteriorRun, config: PostprocessConfig) -> AlignedPosterior:
    """Align all retained draws to a deterministic, data-retained pivot draw."""

    reference = int(np.argmax(run.log_likelihood))
    speed_policy = run.config["data"]["observed_negative_speed_policy"]
    behavior_raw = behavior_alignment_features(
        run.idm_parameters,
        config.response_anchors,
        minimum_allowed_speed_mps=float(speed_policy["minimum_allowed_speed_mps"]),
    )
    behavior_features, behavior_center, behavior_scale = _standardize_feature_cube(
        behavior_raw, config.feature_scale_floor
    )
    scenario_raw = scenario_alignment_features(
        run.scenario_mean, run.scenario_covariance
    )
    scenario_features, scenario_center, scenario_scale = _standardize_feature_cube(
        scenario_raw, config.feature_scale_floor
    )
    behavior_permutations = hungarian_old_for_new(behavior_features, reference)
    scenario_permutations = hungarian_old_for_new(scenario_features, reference)
    return apply_coordinate_permutations(
        run,
        behavior_permutations,
        scenario_permutations,
        reference_draw_index=reference,
        behavior_feature_center=behavior_center,
        behavior_feature_scale=behavior_scale,
        scenario_feature_center=scenario_center,
        scenario_feature_scale=scenario_scale,
    )


def pointwise_modal_categories(paths: IntArray, num_states: int) -> Tuple[IntArray, FloatArray]:
    """Return categorical modes and probabilities; never average integer labels."""

    values = np.asarray(paths, dtype=np.int64)
    if values.ndim != 2 or num_states <= 0:
        raise ValueError("paths must have shape (D, N) and num_states must be positive.")
    if np.any(values < 0) or np.any(values >= num_states):
        raise ValueError("paths contain out-of-range categories.")
    probabilities = np.stack(
        [np.mean(values == state, axis=0) for state in range(num_states)], axis=1
    )
    modal = np.argmax(probabilities, axis=1).astype(np.int64)
    return modal, probabilities


def _quantile_summary(values: FloatArray, interval: Tuple[float, float]) -> Dict[str, float]:
    array = np.asarray(values, dtype=float)
    lower, upper = np.quantile(array, interval)
    return {
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "lower_95": float(lower),
        "upper_95": float(upper),
    }


def _autocorrelation_fft(values: FloatArray) -> FloatArray:
    """Compute a normalized single-chain autocorrelation sequence by FFT."""

    series = np.asarray(values, dtype=float)
    if series.ndim != 1 or series.size == 0 or not np.all(np.isfinite(series)):
        raise ValueError("A diagnostic trace must be a non-empty finite vector.")
    centered = series - np.mean(series)
    variance = float(np.dot(centered, centered))
    if variance == 0.0:
        result = np.zeros(series.size, dtype=float)
        result[0] = 1.0
        return result
    transform_length = 1 << (2 * series.size - 1).bit_length()
    transform = np.fft.rfft(centered, n=transform_length)
    autocovariance = np.fft.irfft(transform * np.conjugate(transform), n=transform_length)[
        : series.size
    ]
    autocovariance /= np.arange(series.size, 0, -1)
    return np.asarray(autocovariance / autocovariance[0], dtype=float)


def _effective_sample_size(values: FloatArray) -> float:
    """Estimate ESS using Geyer's initial positive sequence for one chain."""

    autocorrelation = _autocorrelation_fft(values)
    if autocorrelation.size < 2 or np.all(autocorrelation[1:] == 0.0):
        return float(autocorrelation.size)
    positive_pairs = []
    lag = 0
    while lag + 1 < autocorrelation.size:
        pair_sum = float(autocorrelation[lag] + autocorrelation[lag + 1])
        if pair_sum <= 0.0:
            break
        positive_pairs.append(pair_sum)
        lag += 2
    integrated_time = max(
        -1.0 + 2.0 * float(np.sum(positive_pairs)),
        1.0,
    )
    return float(np.clip(autocorrelation.size / integrated_time, 1.0, autocorrelation.size))


def _single_trace_diagnostics(values: FloatArray) -> Mapping[str, Any]:
    """Window stability, selected ACF lags, and ESS for one retained trace."""

    series = np.asarray(values, dtype=float)
    if series.ndim != 1 or series.size == 0 or not np.all(np.isfinite(series)):
        raise ValueError("Single-chain diagnostics require a non-empty finite trace.")
    if series.size == 1:
        return {
            "num_retained_draws": 1,
            "window_stability_available": False,
            "window_stability_unavailable_reason": (
                "At least two retained draws are required; this is expected only for "
                "an explicitly allowed smoke artifact."
            ),
            "autocorrelation": {},
            "effective_sample_size_geyer_initial_positive_sequence": 1.0,
        }
    midpoint = series.size // 2
    first = series[:midpoint]
    second = series[midpoint:]
    first_mean = float(np.mean(first))
    second_mean = float(np.mean(second))
    posterior_sd = float(np.std(series, ddof=1))
    mean_difference = second_mean - first_mean
    autocorrelation = _autocorrelation_fft(series)
    requested_lags = (1, 5, 10, 20, 50)
    return {
        "num_retained_draws": int(series.size),
        "window_stability_available": True,
        "first_half_mean": first_mean,
        "second_half_mean": second_mean,
        "second_minus_first_mean": mean_difference,
        "absolute_window_shift_in_posterior_sd": (
            float(abs(mean_difference) / posterior_sd) if posterior_sd > 0.0 else 0.0
        ),
        "relative_window_shift": float(
            abs(mean_difference) / max(abs(float(np.mean(series))), 1.0e-12)
        ),
        "autocorrelation": {
            str(lag): float(autocorrelation[lag])
            for lag in requested_lags
            if lag < autocorrelation.size
        },
        "effective_sample_size_geyer_initial_positive_sequence": _effective_sample_size(
            series
        ),
    }


def _single_chain_parameter_diagnostics(
    aligned: AlignedPosterior, run: PosteriorRun
) -> Mapping[str, Any]:
    idm_records = []
    for state in range(run.num_behavior_states):
        idm_records.append(
            {
                "state_index_zero_based": state,
                "parameters": {
                    name: _single_trace_diagnostics(
                        aligned.idm_parameters[:, state, parameter_index]
                    )
                    for parameter_index, name in enumerate(PARAMETER_NAMES)
                },
            }
        )
    variance_records = [
        {
            "state_index_zero_based": state,
            "residual_variance": _single_trace_diagnostics(
                aligned.residual_variance[:, state]
            ),
        }
        for state in range(run.num_behavior_states)
    ]
    return {
        "diagnostic_type": "single-chain retained-draw diagnostics",
        "rank_normalized_rhat": None,
        "rank_normalized_rhat_not_reported_reason": (
            "Only one original chain was run. Artificially treating windows from that "
            "chain as independent chains would not provide a genuine multi-chain R-hat."
        ),
        "interpretation": (
            "Window stability, autocorrelation, and ESS diagnose this retained trace; "
            "they do not establish convergence to the full posterior."
        ),
        "idm_parameters": idm_records,
        "residual_variance": variance_records,
        "log_likelihood": _single_trace_diagnostics(run.log_likelihood),
    }


def _array_interval_summary(
    values: FloatArray, interval: Tuple[float, float]
) -> Mapping[str, Any]:
    array = np.asarray(values, dtype=float)
    return {
        "mean": np.mean(array, axis=0).tolist(),
        "median": np.median(array, axis=0).tolist(),
        "lower_95": np.quantile(array, interval[0], axis=0).tolist(),
        "upper_95": np.quantile(array, interval[1], axis=0).tolist(),
    }


def _audit_interval_order(summary: Mapping[str, Any]) -> Mapping[str, Any]:
    lower = np.asarray(summary["lower_95"], dtype=float)
    median = np.asarray(summary["median"], dtype=float)
    upper = np.asarray(summary["upper_95"], dtype=float)
    ordered = bool(
        np.all(np.isfinite(lower))
        and np.all(np.isfinite(median))
        and np.all(np.isfinite(upper))
        and np.all(lower <= median)
        and np.all(median <= upper)
    )
    if not ordered:
        raise AssertionError("A posterior interval summary is non-finite or unordered.")
    return {"num_scalar_parameters": int(lower.size), "all_intervals_ordered": True}


def _physical_scenario_parameters(
    aligned: AlignedPosterior, run: PosteriorRun
) -> Tuple[FloatArray, FloatArray]:
    scale = run.standardizer_scale
    mean = aligned.scenario_mean * scale[None, None, :] + run.standardizer_mean[
        None, None, :
    ]
    covariance = (
        aligned.scenario_covariance
        * scale[None, None, :, None]
        * scale[None, None, None, :]
    )
    return mean, covariance


def _occupancy(paths: IntArray, num_states: int) -> FloatArray:
    return np.stack(
        [np.mean(paths == state, axis=1) for state in range(num_states)], axis=1
    )


def _conditional_predictions(
    aligned: AlignedPosterior,
    run: PosteriorRun,
    cohort: CohortArrays,
    fixed_cases: Sequence[int],
) -> Tuple[FloatArray, FloatArray, Dict[int, FloatArray]]:
    draws = run.num_draws
    observations = cohort.num_observations
    behavior = run.num_behavior_states
    scenario = run.num_scenario_states
    prediction_sum = np.zeros(observations, dtype=float)
    draw_rmse = np.empty(draws, dtype=float)
    case_predictions = {
        int(case_index): np.empty(
            (draws, int(cohort.lengths[case_index])), dtype=np.float32
        )
        for case_index in fixed_cases
    }
    minimum_speed = float(
        run.config["data"]["observed_negative_speed_policy"][
            "minimum_allowed_speed_mps"
        ]
    )
    for draw in range(draws):
        behavior_path = aligned.paths[draw] // scenario
        prediction = np.empty(observations, dtype=float)
        for state in range(behavior):
            selected = behavior_path == state
            if np.any(selected):
                prediction[selected] = original_idm_acceleration(
                    aligned.idm_parameters[draw, state],
                    cohort.physical_context[selected],
                    minimum_allowed_speed_mps=minimum_speed,
                )
        if not np.all(np.isfinite(prediction)):
            raise ValueError("A conditional acceleration reconstruction is non-finite.")
        prediction_sum += prediction
        draw_rmse[draw] = float(
            np.sqrt(np.mean((prediction - cohort.acceleration) ** 2))
        )
        for case_index, storage in case_predictions.items():
            start, stop = cohort.offsets[case_index : case_index + 2]
            storage[draw] = prediction[start:stop]
    return prediction_sum / draws, draw_rmse, case_predictions


def _state_support_records(
    occupancy: FloatArray,
    config: PostprocessConfig,
    *,
    label_prefix: str,
) -> Sequence[Mapping[str, Any]]:
    records = []
    for state in range(occupancy.shape[1]):
        values = occupancy[:, state]
        empty_fraction = float(np.mean(values == 0.0))
        mean_occupancy = float(np.mean(values))
        rare = (
            mean_occupancy < config.rare_occupancy_threshold
            or empty_fraction > config.empty_draw_fraction_threshold
        )
        records.append(
            {
                "state_index_zero_based": state,
                "display_label": f"{label_prefix} #{state + 1}",
                "occupancy": _quantile_summary(values, config.credible_interval),
                "empty_draw_fraction": empty_fraction,
                "classification": "rare/submode" if rare else "supported",
                "semantic_interpretation_allowed": not rare,
            }
        )
    return records


def _modal_run_length_segments(
    modal_path: IntArray,
    sampling_frequency_hz: float,
    *,
    label_prefix: str,
) -> Sequence[Mapping[str, Any]]:
    """Encode one case-local modal path as contiguous half-open segments."""

    path = np.asarray(modal_path, dtype=np.int64)
    if path.ndim != 1 or path.size == 0:
        raise ValueError("A modal case path must be a nonempty one-dimensional array.")
    if np.any(path < 0):
        raise ValueError("Modal state indices must be nonnegative.")
    if not np.isfinite(sampling_frequency_hz) or sampling_frequency_hz <= 0.0:
        raise ValueError("sampling_frequency_hz must be finite and positive.")

    starts = np.concatenate(
        (np.array([0], dtype=np.int64), np.flatnonzero(path[1:] != path[:-1]) + 1)
    )
    stops = np.concatenate((starts[1:], np.array([path.size], dtype=np.int64)))
    records = []
    for start, stop in zip(starts, stops):
        start_index = int(start)
        stop_index = int(stop)
        state = int(path[start_index])
        num_samples = stop_index - start_index
        records.append(
            {
                "state_index_zero_based": state,
                "display_label": f"{label_prefix} #{state + 1}",
                "start_sample_zero_based": start_index,
                "end_sample_exclusive": stop_index,
                "start_seconds": start_index / sampling_frequency_hz,
                "end_seconds_exclusive": stop_index / sampling_frequency_hz,
                "last_observation_seconds": (stop_index - 1) / sampling_frequency_hz,
                "duration_seconds": num_samples / sampling_frequency_hz,
                "num_samples": num_samples,
                "case_fraction": num_samples / path.size,
            }
        )
    return records


def _case_structural_gate(
    behavior_modal: IntArray,
    scenario_modal: IntArray,
    cohort: CohortArrays,
    config: PostprocessConfig,
) -> Sequence[Mapping[str, Any]]:
    records = []
    for case_index in config.fixed_case_indices:
        if case_index >= cohort.num_trajectories:
            raise ValueError(f"Fixed case {case_index} is outside the cohort.")
        start, stop = cohort.offsets[case_index : case_index + 2]
        behavior = behavior_modal[start:stop]
        scenario = scenario_modal[start:stop]
        behavior_counts = np.bincount(behavior) / behavior.size
        scenario_counts = np.bincount(scenario) / scenario.size
        supported_behavior = int(
            np.sum(behavior_counts >= config.case_supported_occupancy_threshold)
        )
        supported_scenario = int(
            np.sum(scenario_counts >= config.case_supported_occupancy_threshold)
        )
        behavior_switches = int(np.sum(behavior[1:] != behavior[:-1]))
        scenario_switches = int(np.sum(scenario[1:] != scenario[:-1]))
        passed = (
            supported_behavior >= 2
            and supported_scenario >= 2
            and behavior_switches >= 1
            and scenario_switches >= 1
        )
        records.append(
            {
                "case_index_zero_based": case_index,
                "trajectory_id": str(cohort.trajectory_ids[case_index]),
                "supported_behavior_states_at_least_5pct": supported_behavior,
                "supported_scenario_states_at_least_5pct": supported_scenario,
                "behavior_modal_switches": behavior_switches,
                "scenario_modal_switches": scenario_switches,
                "broad_switching_story_retained": bool(passed),
                "trajectory_length_samples": int(behavior.size),
                "sampling_frequency_hz": config.sampling_frequency_hz,
                "segment_interval_convention": (
                    "Case-local samples use zero-based half-open intervals "
                    "[start_sample_zero_based, end_sample_exclusive); seconds equal "
                    "sample index divided by sampling_frequency_hz."
                ),
                "behavior_modal_segments": _modal_run_length_segments(
                    behavior,
                    config.sampling_frequency_hz,
                    label_prefix="B",
                ),
                "scenario_modal_segments": _modal_run_length_segments(
                    scenario,
                    config.sampling_frequency_hz,
                    label_prefix="S",
                ),
                "scope": (
                    "Structural gate only: it does not validate psychological labels or "
                    "causal driver attributes."
                ),
            }
        )
    return records


def analyze_arm(
    run: PosteriorRun, cohort: CohortArrays, config: PostprocessConfig
) -> ArmAnalysis:
    """Align one arm and compute posterior summaries from all retained draws."""

    if not np.array_equal(run.trajectory_offsets, cohort.offsets):
        raise ValueError(f"Run {run.arm_id} trajectory offsets disagree with cohort.")
    if str(run.run_metadata.get("cohort_content_sha256")) != cohort.content_sha256:
        raise ValueError(f"Run {run.arm_id} used a different cohort content hash.")
    aligned = align_posterior(run, config)
    minimum_speed = float(
        run.config["data"]["observed_negative_speed_policy"][
            "minimum_allowed_speed_mps"
        ]
    )
    anchor_responses = behavior_response_at_anchors(
        aligned.idm_parameters,
        config.response_anchors,
        minimum_allowed_speed_mps=minimum_speed,
    )
    behavior_paths = aligned.paths // run.num_scenario_states
    scenario_paths = aligned.paths % run.num_scenario_states
    behavior_occupancy = _occupancy(behavior_paths, run.num_behavior_states)
    scenario_occupancy = _occupancy(scenario_paths, run.num_scenario_states)
    joint_occupancy = _occupancy(aligned.paths, run.num_joint_states)
    cooccurrence_by_draw = joint_occupancy.reshape(
        run.num_draws, run.num_behavior_states, run.num_scenario_states
    )
    cooccurrence = np.mean(cooccurrence_by_draw, axis=0)
    behavior_modal, behavior_probability = pointwise_modal_categories(
        behavior_paths, run.num_behavior_states
    )
    scenario_modal, scenario_probability = pointwise_modal_categories(
        scenario_paths, run.num_scenario_states
    )
    expected_acceleration, draw_rmse, case_prediction_draws = _conditional_predictions(
        aligned, run, cohort, config.fixed_case_indices
    )
    trajectory_rmse = np.empty(cohort.num_trajectories, dtype=float)
    for trajectory in range(cohort.num_trajectories):
        start, stop = cohort.offsets[trajectory : trajectory + 2]
        trajectory_rmse[trajectory] = np.sqrt(
            np.mean(
                (
                    expected_acceleration[start:stop]
                    - cohort.acceleration[start:stop]
                )
                ** 2
            )
        )
    physical_mean, physical_covariance = _physical_scenario_parameters(aligned, run)
    idm_interval = _array_interval_summary(
        aligned.idm_parameters, config.credible_interval
    )
    residual_variance_interval = _array_interval_summary(
        aligned.residual_variance, config.credible_interval
    )
    anchor_response_interval = _array_interval_summary(
        anchor_responses, config.credible_interval
    )
    scenario_mean_interval = _array_interval_summary(
        physical_mean, config.credible_interval
    )
    scenario_covariance_interval = _array_interval_summary(
        physical_covariance, config.credible_interval
    )
    initial_interval = _array_interval_summary(
        aligned.initial_probability, config.credible_interval
    )
    transition_interval = _array_interval_summary(
        aligned.transition_probability, config.credible_interval
    )
    interval_audits = {
        "idm_parameters": _audit_interval_order(idm_interval),
        "residual_variance": _audit_interval_order(residual_variance_interval),
        "behavior_response_at_fixed_anchors": _audit_interval_order(
            anchor_response_interval
        ),
        "scenario_mean_physical": _audit_interval_order(scenario_mean_interval),
        "scenario_covariance_physical": _audit_interval_order(
            scenario_covariance_interval
        ),
        "initial_probability": _audit_interval_order(initial_interval),
        "transition_probability": _audit_interval_order(transition_interval),
    }
    interval_scalar_total = int(
        sum(item["num_scalar_parameters"] for item in interval_audits.values())
    )
    behavior_records = []
    behavior_support = _state_support_records(
        behavior_occupancy, config, label_prefix="Regime"
    )
    for state in range(run.num_behavior_states):
        parameter_record = {
            PARAMETER_NAMES[index]: _quantile_summary(
                aligned.idm_parameters[:, state, index], config.credible_interval
            )
            for index in range(5)
        }
        behavior_records.append(
            {
                **behavior_support[state],
                "idm_parameters": parameter_record,
                "residual_standard_deviation_mps2": _quantile_summary(
                    np.sqrt(aligned.residual_variance[:, state]),
                    config.credible_interval,
                ),
                "residual_variance_m2ps4": _quantile_summary(
                    aligned.residual_variance[:, state],
                    config.credible_interval,
                ),
                "response_at_fixed_anchors_mps2": [
                    {
                        "anchor_physical_v_s_dv": config.response_anchors[
                            anchor_index
                        ].tolist(),
                        "conditional_acceleration": _quantile_summary(
                            anchor_responses[:, state, anchor_index],
                            config.credible_interval,
                        ),
                    }
                    for anchor_index in range(config.response_anchors.shape[0])
                ],
            }
        )
    scenario_records = []
    scenario_support = _state_support_records(
        scenario_occupancy, config, label_prefix="Scenario"
    )
    for state in range(run.num_scenario_states):
        scenario_records.append(
            {
                **scenario_support[state],
                "physical_mean": {
                    CONTEXT_NAMES[index]: _quantile_summary(
                        physical_mean[:, state, index], config.credible_interval
                    )
                    for index in range(3)
                },
                "physical_standard_deviation": {
                    CONTEXT_NAMES[index]: _quantile_summary(
                        np.sqrt(physical_covariance[:, state, index, index]),
                        config.credible_interval,
                    )
                    for index in range(3)
                },
                "physical_covariance": {
                    row_name: {
                        column_name: _quantile_summary(
                            physical_covariance[:, state, row_index, column_index],
                            config.credible_interval,
                        )
                        for column_index, column_name in enumerate(CONTEXT_NAMES)
                    }
                    for row_index, row_name in enumerate(CONTEXT_NAMES)
                },
            }
        )
    summary: Dict[str, Any] = {
        "arm_id": run.arm_id,
        "states": {
            "behavior": run.num_behavior_states,
            "scenario": run.num_scenario_states,
            "joint": run.num_joint_states,
        },
        "num_retained_draws": run.num_draws,
        "alignment": {
            "method": "separate_coordinate_hungarian",
            "reference_draw_rule": "maximum retained marginal log likelihood",
            "reference_draw_index_zero_based": aligned.reference_draw_index,
            "behavior_features": "log IDM parameters plus original-IDM response at fixed physical anchors",
            "scenario_features": "standardized-space Gaussian mean plus covariance lower triangle",
            "synchronous_joint_reindexing": [
                "initial_probability",
                "transition_rows",
                "transition_columns",
                "latent_paths",
            ],
            "response_anchors_physical_order_v_s_dv": config.response_anchors.tolist(),
        },
        "conditional_reconstruction": {
            "scope": "in-sample observed-covariate posterior conditional reconstruction",
            "not_open_loop": True,
            "posterior_mean_rmse_mps2": float(
                np.sqrt(np.mean((expected_acceleration - cohort.acceleration) ** 2))
            ),
            "drawwise_rmse_mps2": _quantile_summary(
                draw_rmse, config.credible_interval
            ),
            "trajectory_rmse_mps2": {
                "mean": float(np.mean(trajectory_rmse)),
                "standard_deviation": float(np.std(trajectory_rmse)),
                "median": float(np.median(trajectory_rmse)),
            },
        },
        "behavior_states": behavior_records,
        "scenario_states": scenario_records,
        "posterior_expected_cooccurrence": cooccurrence.tolist(),
        "initial_distribution_summary": initial_interval,
        "transition_summary": {
            **transition_interval,
            "diagonal_offset_added": False,
        },
        "posterior_interval_audit": {
            "credible_interval_probabilities": list(config.credible_interval),
            "coverage": interval_audits,
            "total_scalar_parameters_covered": interval_scalar_total,
            "all_intervals_finite_and_ordered": True,
        },
        "single_chain_diagnostics": _single_chain_parameter_diagnostics(
            aligned, run
        ),
        "input_run": {
            "directory": str(run.run_directory),
            "chain_sha256": _sha256_file(run.run_directory / "chain.npz"),
            "scientific_target_fingerprint": str(
                run.run_metadata["scientific_target_fingerprint"]
            ),
            "runtime_seconds": float(run.run_metadata["runtime_seconds"]),
        },
    }
    arrays: Dict[str, NDArray[np.generic]] = {
        "behavior_old_for_new": aligned.behavior_old_for_new,
        "scenario_old_for_new": aligned.scenario_old_for_new,
        "behavior_feature_center": aligned.behavior_feature_center,
        "behavior_feature_scale": aligned.behavior_feature_scale,
        "scenario_feature_center": aligned.scenario_feature_center,
        "scenario_feature_scale": aligned.scenario_feature_scale,
        "transition_mean": np.mean(aligned.transition_probability, axis=0),
        "transition_lower_95": np.quantile(
            aligned.transition_probability, config.credible_interval[0], axis=0
        ),
        "transition_upper_95": np.quantile(
            aligned.transition_probability, config.credible_interval[1], axis=0
        ),
        "initial_mean": np.mean(aligned.initial_probability, axis=0),
        "cooccurrence_mean": cooccurrence,
        "cooccurrence_lower_95": np.quantile(
            cooccurrence_by_draw, config.credible_interval[0], axis=0
        ),
        "cooccurrence_upper_95": np.quantile(
            cooccurrence_by_draw, config.credible_interval[1], axis=0
        ),
        "behavior_modal": behavior_modal,
        "scenario_modal": scenario_modal,
        "behavior_probability": behavior_probability.astype(np.float32),
        "scenario_probability": scenario_probability.astype(np.float32),
        "trajectory_offsets": cohort.offsets.copy(),
        "expected_acceleration": expected_acceleration,
        "trajectory_rmse": trajectory_rmse,
        "behavior_parameters_median": np.median(aligned.idm_parameters, axis=0),
        "behavior_parameters_lower_95": np.quantile(
            aligned.idm_parameters, config.credible_interval[0], axis=0
        ),
        "behavior_parameters_upper_95": np.quantile(
            aligned.idm_parameters, config.credible_interval[1], axis=0
        ),
        "response_anchors_physical_v_s_dv": config.response_anchors,
        "behavior_anchor_response_median": np.median(anchor_responses, axis=0),
        "behavior_anchor_response_lower_95": np.quantile(
            anchor_responses, config.credible_interval[0], axis=0
        ),
        "behavior_anchor_response_upper_95": np.quantile(
            anchor_responses, config.credible_interval[1], axis=0
        ),
        "residual_sd_median": np.median(np.sqrt(aligned.residual_variance), axis=0),
        "scenario_mean_physical_median": np.median(physical_mean, axis=0),
        "scenario_mean_physical_lower_95": np.quantile(
            physical_mean, config.credible_interval[0], axis=0
        ),
        "scenario_mean_physical_upper_95": np.quantile(
            physical_mean, config.credible_interval[1], axis=0
        ),
        "scenario_covariance_physical_median": np.median(
            physical_covariance, axis=0
        ),
        "scenario_covariance_physical_mean": np.mean(physical_covariance, axis=0),
        "log_likelihood_trace": run.log_likelihood,
    }
    for case_index, prediction_draws in case_prediction_draws.items():
        arrays[f"case_{case_index}_prediction_median"] = np.median(
            prediction_draws, axis=0
        )
        arrays[f"case_{case_index}_prediction_lower_95"] = np.quantile(
            prediction_draws, config.credible_interval[0], axis=0
        )
        arrays[f"case_{case_index}_prediction_upper_95"] = np.quantile(
            prediction_draws, config.credible_interval[1], axis=0
        )
    return ArmAnalysis(arm_id=run.arm_id, summary=summary, arrays=arrays)


def _slug(arm_id: str) -> str:
    return arm_id.lower().replace("_full_joint", "").replace("_", "")


def _latex_escape(value: str) -> str:
    return value.replace("_", r"\_").replace("%", r"\%")


def build_tables_tex(
    summaries: Mapping[str, Mapping[str, Any]], config: PostprocessConfig
) -> str:
    """Create auditable LaTeX tables; every displayed number comes from JSON."""

    lines = [
        "% Auto-generated by manuscript_storyrev_20260808/postprocess.py.",
        "% Do not edit by hand; regenerate from completed formal chains.",
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{In-sample observed-covariate conditional reconstruction by model arm.}",
        r"\begin{tabular}{lrrr}",
        r"\toprule",
        r"Arm & $K^{[\mathrm B]}$ & $K^{[\mathrm S]}$ & RMSE (m/s$^2$) \\",
        r"\midrule",
    ]
    for arm_id in config.required_arm_ids:
        summary = summaries[arm_id]
        states = summary["states"]
        rmse = summary["conditional_reconstruction"]["posterior_mean_rmse_mps2"]
        lines.append(
            f"{_latex_escape(arm_id)} & {states['behavior']} & "
            f"{states['scenario']} & {rmse:.3f} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}", ""])
    manuscript = summaries[config.manuscript_arm_id]
    lines.extend(
        [
            r"\begin{table}[t]",
            r"\centering",
            r"\caption{Posterior behavior-state summaries. Values are medians with equal-tail 95\% intervals.}",
            r"\begin{tabular}{lrrrrrr}",
            r"\toprule",
            r"State & $v_f$ & $s_0$ & $T$ & $a$ & $b$ & $\sigma_a$ \\",
            r"\midrule",
        ]
    )
    for state in manuscript["behavior_states"]:
        cells = []
        for name in PARAMETER_NAMES:
            item = state["idm_parameters"][name]
            cells.append(
                f"{item['median']:.2f} [{item['lower_95']:.2f}, {item['upper_95']:.2f}]"
            )
        residual = state["residual_standard_deviation_mps2"]
        cells.append(
            f"{residual['median']:.2f} [{residual['lower_95']:.2f}, {residual['upper_95']:.2f}]"
        )
        lines.append(
            f"{_latex_escape(state['display_label'])} & " + " & ".join(cells) + r" \\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}", ""])
    lines.extend(
        [
            r"\begin{table}[t]",
            r"\centering",
            r"\caption{Posterior physical-scale scenario means. Values are medians with equal-tail 95\% intervals.}",
            r"\begin{tabular}{lrrr}",
            r"\toprule",
            r"State & Speed (m/s) & Gap (m) & Relative speed (m/s) \\",
            r"\midrule",
        ]
    )
    for state in manuscript["scenario_states"]:
        cells = []
        for name in CONTEXT_NAMES:
            item = state["physical_mean"][name]
            cells.append(
                f"{item['median']:.2f} [{item['lower_95']:.2f}, {item['upper_95']:.2f}]"
            )
        lines.append(
            f"{_latex_escape(state['display_label'])} & " + " & ".join(cells) + r" \\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}", ""])
    return "\n".join(lines)


def _timestamped_directory(root: Path) -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    path = root / f"{timestamp}_scheme_b_postprocess"
    path.mkdir(parents=True, exist_ok=False)
    return path


def execute_postprocess(
    run_directories: Sequence[Path],
    cohort_npz: Path,
    cohort_manifest: Path,
    config_path: Path,
    output_root: Path,
    *,
    require_formal: bool = True,
) -> Path:
    """Postprocess all arms into one append-only, hash-receipted artifact."""

    started = time.perf_counter()
    config = PostprocessConfig.from_json(config_path)
    cohort = load_cohort(cohort_npz, cohort_manifest)
    runs: Dict[str, PosteriorRun] = {}
    for path in run_directories:
        run = load_posterior_run(path, require_formal=require_formal)
        if run.arm_id in runs:
            raise ValueError(f"Duplicate arm supplied: {run.arm_id}.")
        runs[run.arm_id] = run
    missing = [arm for arm in config.required_arm_ids if arm not in runs]
    unexpected = sorted(set(runs) - set(config.required_arm_ids))
    if missing or unexpected:
        raise ValueError(f"Arm set mismatch; missing={missing}, unexpected={unexpected}.")
    output = _timestamped_directory(output_root)
    _write_json_new(
        output / "RUNNING.json",
        {
            "started_at_utc": datetime.now(timezone.utc).isoformat(),
            "required_arm_ids": config.required_arm_ids,
        },
    )
    (output / "config.json").write_bytes(config_path.read_bytes())
    summaries: Dict[str, Mapping[str, Any]] = {}
    arm_files: Dict[str, str] = {}
    analyses: Dict[str, ArmAnalysis] = {}
    for arm_id in config.required_arm_ids:
        analysis = analyze_arm(runs[arm_id], cohort, config)
        analyses[arm_id] = analysis
        summaries[arm_id] = analysis.summary
        filename = f"arm_{_slug(arm_id)}.npz"
        np.savez_compressed(output / filename, **analysis.arrays)
        arm_files[arm_id] = filename
    manuscript_analysis = analyses[config.manuscript_arm_id]
    manuscript_arrays = manuscript_analysis.arrays
    gate_cases = _case_structural_gate(
        np.asarray(manuscript_arrays["behavior_modal"], dtype=np.int64),
        np.asarray(manuscript_arrays["scenario_modal"], dtype=np.int64),
        cohort,
        config,
    )
    passed_cases = sum(
        bool(record["broad_switching_story_retained"]) for record in gate_cases
    )
    story_gate = {
        "status": (
            "PASS"
            if passed_cases >= config.minimum_passing_fixed_cases
            else "STOP_BEFORE_RESULTS_REWRITE"
        ),
        "decision_rule": (
            f"At least {config.minimum_passing_fixed_cases} of "
            f"{len(config.fixed_case_indices)} fixed cases must retain structural "
            "behavior and scenario switching."
        ),
        "passing_fixed_cases": passed_cases,
        "required_passing_fixed_cases": config.minimum_passing_fixed_cases,
        "fixed_cases": gate_cases,
        "state_support_policy": {
            "rare_if_posterior_mean_occupancy_below": config.rare_occupancy_threshold,
            "rare_if_empty_draw_fraction_above": config.empty_draw_fraction_threshold,
            "rare_states_must_be_described_as": "rare/submode",
        },
        "behavior_state_support": [
            {
                "state_index_zero_based": item["state_index_zero_based"],
                "classification": item["classification"],
                "semantic_interpretation_allowed": item[
                    "semantic_interpretation_allowed"
                ],
            }
            for item in summaries[config.manuscript_arm_id]["behavior_states"]
        ],
        "scenario_state_support": [
            {
                "state_index_zero_based": item["state_index_zero_based"],
                "classification": item["classification"],
                "semantic_interpretation_allowed": item[
                    "semantic_interpretation_allowed"
                ],
            }
            for item in summaries[config.manuscript_arm_id]["scenario_states"]
        ],
        "interpretation_boundary": (
            "Passing permits a descriptive model-derived archetype narrative only; "
            "it does not validate stable personality traits, psychology, or causality."
        ),
    }
    pooled_arrays = analyses[config.pooled_arm_id].arrays
    for case_index in config.fixed_case_indices:
        start, stop = cohort.offsets[case_index : case_index + 2]
        observed = cohort.acceleration[start:stop]
        full = np.asarray(
            manuscript_arrays[f"case_{case_index}_prediction_median"], dtype=float
        )
        pooled = np.asarray(
            pooled_arrays[f"case_{case_index}_prediction_median"], dtype=float
        )
        for record in story_gate["fixed_cases"]:
            if int(record["case_index_zero_based"]) == case_index:
                record["conditional_reconstruction_rmse_mps2"] = {
                    "product_state": float(np.sqrt(np.mean((full - observed) ** 2))),
                    "pooled_b1s1": float(np.sqrt(np.mean((pooled - observed) ** 2))),
                }
                break
    posterior_summary = {
        "schema_version": 1,
        "scientific_scope": {
            "dataset": "frozen 100-trajectory highD cohort",
            "evaluation": "in-sample descriptive and observed-covariate conditional reconstruction",
            "not_claimed": [
                "out-of-sample prediction",
                "open-loop simulation validity",
                "psychological trait validation",
                "independent latent-factor dynamics",
            ],
        },
        "cohort": {
            "content_sha256": cohort.content_sha256,
            "npz_sha256": cohort.npz_sha256,
            "manifest_sha256": cohort.manifest_sha256,
            "num_trajectories": cohort.num_trajectories,
            "num_observations": cohort.num_observations,
        },
        "arm_files": arm_files,
        "arms": summaries,
        "story_gate_status": story_gate["status"],
    }
    _write_json_new(output / "posterior_summary.json", posterior_summary)
    _write_json_new(output / "story_gate.json", story_gate)
    _write_json_new(
        output / "single_chain_diagnostics.json",
        {
            "schema_version": 1,
            "scope": (
                "Single-chain trace-window stability, autocorrelation, and ESS. "
                "These quantities are diagnostics, not a multi-chain convergence proof."
            ),
            "rank_normalized_rhat_reported": False,
            "arms": {
                arm_id: summaries[arm_id]["single_chain_diagnostics"]
                for arm_id in config.required_arm_ids
            },
        },
    )
    _write_text_new(output / "tables.tex", build_tables_tex(summaries, config))
    repository_root = Path(__file__).resolve().parents[2]
    metadata = {
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "runtime_seconds": time.perf_counter() - started,
        "require_formal": require_formal,
        "git_commit": _git_commit(repository_root),
        "git_worktree_dirty": _git_dirty(repository_root),
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "source_sha256": {
            "postprocess.py": _sha256_file(Path(__file__)),
            "run_exact_reconstruction.py": _sha256_file(
                Path(__file__).with_name("run_exact_reconstruction.py")
            ),
            "target.py": _sha256_file(Path(__file__).with_name("target.py")),
            "postprocess_config": _sha256_file(config_path),
        },
        "input_runs": {
            arm_id: {
                "directory": str(runs[arm_id].run_directory),
                "chain_sha256": _sha256_file(runs[arm_id].run_directory / "chain.npz"),
                "artifact_manifest_sha256": _sha256_file(
                    runs[arm_id].run_directory / "artifact_manifest.json"
                ),
            }
            for arm_id in config.required_arm_ids
        },
    }
    _write_json_new(output / "postprocess_metadata.json", metadata)
    artifact_names = [
        "config.json",
        "posterior_summary.json",
        "story_gate.json",
        "single_chain_diagnostics.json",
        "tables.tex",
        "postprocess_metadata.json",
        *arm_files.values(),
    ]
    _write_json_new(
        output / "artifact_manifest.json",
        {
            "artifacts": {
                name: {
                    "sha256": _sha256_file(output / name),
                    "bytes": (output / name).stat().st_size,
                }
                for name in artifact_names
            }
        },
    )
    _write_json_new(
        output / "COMPLETED.json",
        {
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            "story_gate_status": story_gate["status"],
            "posterior_summary_sha256": _sha256_file(
                output / "posterior_summary.json"
            ),
        },
    )
    return output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run",
        dest="runs",
        action="append",
        type=Path,
        required=True,
        help="Completed run directory; pass exactly once for each required arm.",
    )
    parser.add_argument("--cohort-npz", type=Path, required=True)
    parser.add_argument("--cohort-manifest", type=Path, required=True)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).resolve().parent / "configs" / "postprocess.json",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(__file__).resolve().parent / "postprocessed",
    )
    parser.add_argument(
        "--allow-nonformal",
        action="store_true",
        help="Testing only: permit smoke schedules. Never use for manuscript figures.",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    output = execute_postprocess(
        args.runs,
        args.cohort_npz,
        args.cohort_manifest,
        args.config,
        args.output_root,
        require_formal=not args.allow_nonformal,
    )
    print(json.dumps({"status": "COMPLETED", "output": str(output)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
