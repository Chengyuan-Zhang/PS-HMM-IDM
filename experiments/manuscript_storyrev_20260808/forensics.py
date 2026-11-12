"""Read-only provenance audit for the legacy 100-follower manuscript cohort.

This module never invokes the MCMC sampler and never rewrites legacy caches.  It
reconstructs the exact ordering used by ``MCMC/MCMC_main.py``, compares the
currently available preprocessing cache with the historical Git cache used near
the February 2025 runs, and writes append-only forensic artifacts.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import io
import json
import os
import pickle
import re
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


SCHEMA_VERSION = "manuscript-storyrev-forensics-v2"
DEFAULT_HISTORICAL_COMMIT = "7c4f5e69857547bd7abba64883034729c7d02ea1"
DEFAULT_RECORDINGS = tuple(range(1, 61))
MANUSCRIPT_CASE_INDICES = (2, 3, 6, 9)
CONTEXT_COLUMNS = ("v_follower_m_s", "gap_m", "delta_v_follower_minus_leader_m_s")
REQUIRED_TRACK_KEYS = (
    "pair_No",
    "id_Foll",
    "frame_Foll",
    "precedingId_Foll",
    "aFollReal",
    "vFollReal",
    "vLeaderReal",
    "sReal",
    "dvReal",
)
EXPECTED_LEGACY_ARMS = (
    ("B1S1", 1, 1),
    ("B2S2", 2, 2),
    ("B5S5", 5, 5),
    ("B5S1", 5, 1),
    ("B1S5", 1, 5),
)
LEGACY_CACHE_PATTERN = re.compile(
    r"^FHMM_Slice_(?P<n_pairs>\d+)pairs_KB(?P<k_b>\d+)_KS(?P<k_s>\d+)"
    r"_w(?P<slice_width>[0-9.]+)_alpha(?P<alpha>\d+)"
    r"_gamma(?P<gamma_shape>\d+)_(?P<gamma_scale>\d+)v(?P<version>\d+)\.npy$"
)


@dataclass(frozen=True)
class LocatedTrackPair:
    """A legacy track-pair dictionary with its otherwise missing recording ID."""

    recording_id: int
    payload: Mapping[str, Any]


def _json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def stable_array_sha256(array: np.ndarray) -> str:
    """Hash array values together with their shape and canonical dtype."""

    materialized = np.asarray(array)
    if materialized.dtype.hasobject:
        raise TypeError("Object arrays are not valid stable provenance payloads.")
    if materialized.dtype.kind in "iufcbmM":
        canonical_dtype = materialized.dtype.newbyteorder("<")
        materialized = materialized.astype(canonical_dtype, copy=False)
    contiguous = np.ascontiguousarray(materialized)
    header = {
        "dtype": contiguous.dtype.str,
        "shape": list(contiguous.shape),
    }
    digest = hashlib.sha256()
    digest.update(_json_bytes(header))
    digest.update(b"\0")
    digest.update(contiguous.tobytes(order="C"))
    return digest.hexdigest()


def legacy_preprocessing_cache_path(
    repo_root: Path,
    recording_id: int,
    *,
    min_duration_seconds: int = 50,
    output_hz: int = 5,
) -> Path:
    name = (
        f"track_pair_list_temp{recording_id:02d}_MinLength{min_duration_seconds}"
        f"_freq_{output_hz}.pkl"
    )
    return repo_root / "data" / "cache" / name


def _validate_pickle_payload(payload: Any, source: str) -> list[Mapping[str, Any]]:
    if not isinstance(payload, list):
        raise TypeError(f"{source} must contain a list, found {type(payload).__name__}.")
    for index, item in enumerate(payload):
        if not isinstance(item, Mapping):
            raise TypeError(
                f"{source}[{index}] must be a mapping, found {type(item).__name__}."
            )
    return payload


def load_current_preprocessing_cache(
    repo_root: Path,
    recording_id: int,
) -> list[Mapping[str, Any]]:
    path = legacy_preprocessing_cache_path(repo_root, recording_id)
    if not path.is_file():
        raise FileNotFoundError(f"Missing preprocessing cache: {path}")
    with path.open("rb") as handle:
        payload = pickle.load(handle)
    return _validate_pickle_payload(payload, str(path))


def git_blob_bytes(repo_root: Path, commit: str, relative_path: str) -> bytes:
    """Read a historical file without checking it out or modifying the worktree."""

    result = subprocess.run(
        ["git", "show", f"{commit}:{relative_path}"],
        cwd=repo_root,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return result.stdout


def load_historical_preprocessing_cache(
    repo_root: Path,
    commit: str,
    recording_id: int,
) -> tuple[list[Mapping[str, Any]], bytes]:
    relative = (
        f"data/cache/track_pair_list_temp{recording_id:02d}_MinLength50_freq_5.pkl"
    )
    blob = git_blob_bytes(repo_root, commit, relative)
    payload = pickle.load(io.BytesIO(blob))
    return _validate_pickle_payload(payload, f"{commit}:{relative}"), blob


def load_located_pairs(
    repo_root: Path,
    recording_ids: Sequence[int] = DEFAULT_RECORDINGS,
    *,
    limit: int | None = None,
) -> tuple[list[LocatedTrackPair], list[dict[str, Any]]]:
    located: list[LocatedTrackPair] = []
    cache_inventory: list[dict[str, Any]] = []
    for recording_id in recording_ids:
        path = legacy_preprocessing_cache_path(repo_root, int(recording_id))
        pairs = load_current_preprocessing_cache(repo_root, int(recording_id))
        cache_inventory.append(
            {
                "recording_id": int(recording_id),
                "path": path.relative_to(repo_root).as_posix(),
                "size_bytes": path.stat().st_size,
                "n_pairs": len(pairs),
            }
        )
        located.extend(
            LocatedTrackPair(recording_id=int(recording_id), payload=pair)
            for pair in pairs
        )
        if limit is not None and len(located) >= limit:
            break
    if limit is not None:
        located = located[:limit]
    return located, cache_inventory


def _one_dimensional_array(
    pair: Mapping[str, Any], key: str, expected_length: int | None = None
) -> np.ndarray:
    if key not in pair:
        raise KeyError(f"Track pair is missing required key {key!r}.")
    value = np.asarray(pair[key])
    if value.ndim != 1:
        raise ValueError(f"Track key {key!r} must be one-dimensional; got {value.shape}.")
    if expected_length is not None and value.shape[0] != expected_length:
        raise ValueError(
            f"Track key {key!r} has length {value.shape[0]}, expected {expected_length}."
        )
    return value


def canonicalize_track_pair(located: LocatedTrackPair) -> dict[str, Any]:
    pair = located.payload
    missing = [key for key in REQUIRED_TRACK_KEYS if key not in pair]
    if missing:
        raise KeyError(f"Track pair is missing required keys: {missing}")

    acceleration = _one_dimensional_array(pair, "aFollReal").astype(np.float64)
    length = acceleration.shape[0]
    speed = _one_dimensional_array(pair, "vFollReal", length).astype(np.float64)
    leader_speed = _one_dimensional_array(pair, "vLeaderReal", length).astype(np.float64)
    gap = _one_dimensional_array(pair, "sReal", length).astype(np.float64)
    stored_delta_v = _one_dimensional_array(pair, "dvReal", length).astype(np.float64)
    frames = _one_dimensional_array(pair, "frame_Foll", length).astype(np.int64)
    leader_ids = _one_dimensional_array(pair, "precedingId_Foll", length).astype(np.int64)
    delta_v = speed - leader_speed
    if not np.allclose(delta_v, stored_delta_v, rtol=0.0, atol=1e-12, equal_nan=True):
        max_difference = float(np.nanmax(np.abs(delta_v - stored_delta_v)))
        raise ValueError(
            "Stored dvReal does not equal vFollReal - vLeaderReal; "
            f"maximum absolute difference is {max_difference}."
        )
    context = np.column_stack((speed, gap, delta_v)).astype(np.float64, copy=False)

    pair_no = int(np.asarray(pair["pair_No"]).item())
    follower_id = int(np.asarray(pair["id_Foll"]).item())
    trajectory_id = (
        f"legacy_pair_{pair_no:04d}_r{located.recording_id:02d}_f{follower_id:04d}"
    )
    payload_hashes = {
        "physical_context": stable_array_sha256(context),
        "acceleration": stable_array_sha256(acceleration),
        "frames": stable_array_sha256(frames),
        "leader_ids": stable_array_sha256(leader_ids),
    }
    trajectory_digest = sha256_bytes(_json_bytes(payload_hashes))
    finite_context = np.isfinite(context)
    finite_acceleration = np.isfinite(acceleration)
    frame_steps = np.diff(frames)
    return {
        "trajectory_id": trajectory_id,
        "recording_id": located.recording_id,
        "follower_id": follower_id,
        "pair_no": pair_no,
        "follower_class": str(pair.get("class_foll", "UNKNOWN")),
        "length": length,
        "physical_context": context,
        "acceleration": acceleration,
        "frames": frames,
        "leader_ids": leader_ids,
        "hashes": payload_hashes,
        "trajectory_payload_sha256": trajectory_digest,
        "audit": {
            "nonfinite_context_values": int((~finite_context).sum()),
            "nonfinite_acceleration_values": int((~finite_acceleration).sum()),
            "negative_speed_values": int((speed < 0).sum()),
            "nonpositive_gap_values": int((gap <= 0).sum()),
            "frame_step_not_five": int((frame_steps != 5).sum()),
            "speed_min": float(np.nanmin(speed)),
            "speed_max": float(np.nanmax(speed)),
            "gap_min": float(np.nanmin(gap)),
            "gap_max": float(np.nanmax(gap)),
            "delta_v_min": float(np.nanmin(delta_v)),
            "delta_v_max": float(np.nanmax(delta_v)),
            "acceleration_min": float(np.nanmin(acceleration)),
            "acceleration_max": float(np.nanmax(acceleration)),
        },
    }


def build_cohort_arrays(located_pairs: Sequence[LocatedTrackPair]) -> dict[str, Any]:
    if not located_pairs:
        raise ValueError("The selected cohort is empty.")
    tracks = [canonicalize_track_pair(located) for located in located_pairs]
    pair_nos = np.asarray([track["pair_no"] for track in tracks], dtype=np.int64)
    if np.unique(pair_nos).shape[0] != pair_nos.shape[0]:
        raise ValueError("Legacy pair_No values are not unique in the selected cohort.")
    composite_followers = [
        (int(track["recording_id"]), int(track["follower_id"])) for track in tracks
    ]
    if len(set(composite_followers)) != len(composite_followers):
        raise ValueError("(recording_id, follower_id) is not unique in the selected cohort.")

    lengths = np.asarray([track["length"] for track in tracks], dtype=np.int64)
    offsets = np.concatenate((np.asarray([0], dtype=np.int64), np.cumsum(lengths)))
    cohort_arrays = {
        "physical_context": np.concatenate(
            [track["physical_context"] for track in tracks], axis=0
        ),
        "acceleration": np.concatenate(
            [track["acceleration"] for track in tracks], axis=0
        ),
        "offsets": offsets,
        "frames": np.concatenate([track["frames"] for track in tracks], axis=0),
        "leader_ids": np.concatenate(
            [track["leader_ids"] for track in tracks], axis=0
        ),
        "trajectory_ids": np.asarray(
            [track["trajectory_id"] for track in tracks], dtype=np.str_
        ),
        "recording_ids": np.asarray(
            [track["recording_id"] for track in tracks], dtype=np.int64
        ),
        "follower_ids": np.asarray(
            [track["follower_id"] for track in tracks], dtype=np.int64
        ),
        "pair_nos": pair_nos,
        "lengths": lengths,
        "follower_classes": np.asarray(
            [track["follower_class"] for track in tracks], dtype=np.str_
        ),
    }
    array_manifest = {
        key: {
            "dtype": str(np.asarray(value).dtype),
            "shape": list(np.asarray(value).shape),
            "sha256": stable_array_sha256(np.asarray(value)),
        }
        for key, value in cohort_arrays.items()
    }
    cohort_order_digest = sha256_bytes(
        _json_bytes(
            [
                {
                    "trajectory_id": track["trajectory_id"],
                    "payload_sha256": track["trajectory_payload_sha256"],
                }
                for track in tracks
            ]
        )
    )
    return {
        "arrays": cohort_arrays,
        "array_manifest": array_manifest,
        "tracks": tracks,
        "cohort_order_sha256": cohort_order_digest,
    }


def _historical_comparison_view(
    located_pairs: Sequence[LocatedTrackPair],
) -> list[dict[str, Any]]:
    return [
        {
            "recording_id": canonical["recording_id"],
            "follower_id": canonical["follower_id"],
            "pair_no": canonical["pair_no"],
            "length": canonical["length"],
            "trajectory_payload_sha256": canonical["trajectory_payload_sha256"],
        }
        for canonical in (canonicalize_track_pair(pair) for pair in located_pairs)
    ]


def compare_with_historical_cache(
    repo_root: Path,
    current_pairs: Sequence[LocatedTrackPair],
    historical_commit: str,
) -> dict[str, Any]:
    recordings = sorted({pair.recording_id for pair in current_pairs})
    historical_located: list[LocatedTrackPair] = []
    source_blobs: list[dict[str, Any]] = []
    for recording_id in recordings:
        pairs, blob = load_historical_preprocessing_cache(
            repo_root, historical_commit, recording_id
        )
        source_blobs.append(
            {
                "recording_id": recording_id,
                "relative_path": (
                    f"data/cache/track_pair_list_temp{recording_id:02d}"
                    "_MinLength50_freq_5.pkl"
                ),
                "size_bytes": len(blob),
                "sha256": sha256_bytes(blob),
            }
        )
        historical_located.extend(
            LocatedTrackPair(recording_id=recording_id, payload=pair) for pair in pairs
        )
    historical_selected = historical_located[: len(current_pairs)]
    current_view = _historical_comparison_view(current_pairs)
    historical_view = _historical_comparison_view(historical_selected)
    return {
        "historical_commit": historical_commit,
        "source_blobs": source_blobs,
        "current_selected_count": len(current_view),
        "historical_selected_count": len(historical_view),
        "canonical_trajectory_views_match": current_view == historical_view,
        "current_view_sha256": sha256_bytes(_json_bytes(current_view)),
        "historical_view_sha256": sha256_bytes(_json_bytes(historical_view)),
        "mismatches": [
            {
                "cohort_index": index,
                "current": current_item,
                "historical": historical_item,
            }
            for index, (current_item, historical_item) in enumerate(
                zip(current_view, historical_view)
            )
            if current_item != historical_item
        ],
    }


def parse_legacy_cache_name(name: str) -> dict[str, Any] | None:
    match = LEGACY_CACHE_PATTERN.fullmatch(name)
    if match is None:
        return None
    parsed: dict[str, Any] = {
        "n_pairs": int(match.group("n_pairs")),
        "k_b": int(match.group("k_b")),
        "k_s": int(match.group("k_s")),
        "slice_width_log_scale": float(match.group("slice_width")),
        "legacy_alpha_argument": int(match.group("alpha")),
        "inverse_gamma_shape": int(match.group("gamma_shape")),
        "inverse_gamma_scale": int(match.group("gamma_scale")),
        "filename_version": int(match.group("version")),
        "model_token": "FHMM",
        "sampler_token": "Slice",
    }
    parsed["joint_state_count"] = parsed["k_b"] * parsed["k_s"]
    parsed["transition_per_cell_concentration"] = (
        parsed["legacy_alpha_argument"] / parsed["k_b"]
    )
    parsed["transition_row_total_concentration"] = (
        parsed["joint_state_count"]
        * parsed["transition_per_cell_concentration"]
    )
    return parsed


def _shape(value: Any) -> list[int] | None:
    shape = getattr(value, "shape", None)
    return list(shape) if shape is not None else None


def inspect_legacy_cache_payload(
    path: Path,
    cohort_pair_nos: np.ndarray | None = None,
    cohort_lengths: np.ndarray | None = None,
) -> dict[str, Any]:
    """Load one legacy object cache and retain only bounded structural metadata."""

    raw = np.load(path, allow_pickle=True)
    try:
        if not isinstance(raw, np.ndarray) or raw.shape != (6,):
            raise ValueError(
                f"Legacy cache {path} has unexpected top-level shape {getattr(raw, 'shape', None)}."
            )
        values = raw.tolist()
        names = (
            "chains_idm_params",
            "chains_states",
            "chains_transition",
            "chains_sigma2",
            "chains_scenario_mean",
            "chains_scenario_covariance",
        )
        summary: dict[str, Any] = {
            "top_level_shape": list(raw.shape),
            "components": {
                name: {
                    "type": type(value).__name__,
                    "shape": _shape(value),
                }
                for name, value in zip(names, values)
            },
        }
        states = values[1]
        state_entries = list(states)
        pair_nos: list[int] = []
        lengths: list[int] = []
        state_shapes: list[list[int]] = []
        for index, entry in enumerate(state_entries):
            if not isinstance(entry, Mapping) or "chains" not in entry or "pair_No" not in entry:
                raise ValueError(f"Malformed state entry {index} in {path}.")
            chains = np.asarray(entry["chains"])
            if chains.ndim != 4 or chains.shape[-1] != 2:
                raise ValueError(
                    f"State entry {index} has unexpected chain shape {chains.shape}."
                )
            pair_nos.append(int(np.asarray(entry["pair_No"]).item()))
            lengths.append(int(chains.shape[2]))
            state_shapes.append(list(chains.shape))
        summary["states"] = {
            "n_trajectories": len(state_entries),
            "pair_nos": pair_nos,
            "trajectory_lengths": lengths,
            "unique_state_chain_shapes": sorted(
                {tuple(shape) for shape in state_shapes}
            ),
        }
        if cohort_pair_nos is not None and cohort_lengths is not None:
            summary["cohort_compatibility"] = {
                "pair_nos_match": pair_nos == cohort_pair_nos.astype(int).tolist(),
                "trajectory_lengths_match": lengths == cohort_lengths.astype(int).tolist(),
                "matches_pair_numbers_and_lengths": (
                    pair_nos == cohort_pair_nos.astype(int).tolist()
                    and lengths == cohort_lengths.astype(int).tolist()
                ),
            }
        return summary
    finally:
        del raw
        gc.collect()


def inventory_expected_legacy_caches(
    repo_root: Path,
    *,
    calculate_sha256: bool,
    inspect_payloads: bool,
    cohort_pair_nos: np.ndarray,
    cohort_lengths: np.ndarray,
) -> list[dict[str, Any]]:
    cache_root = repo_root / "MCMC" / "cache"
    inventory: list[dict[str, Any]] = []
    for arm, k_b, k_s in EXPECTED_LEGACY_ARMS:
        name = (
            f"FHMM_Slice_100pairs_KB{k_b}_KS{k_s}_w0.1_alpha1000"
            "_gamma100_1v0.npy"
        )
        path = cache_root / name
        item: dict[str, Any] = {
            "arm": arm,
            "expected_name": name,
            "path": path.relative_to(repo_root).as_posix(),
            "present": path.is_file(),
            "filename_config": parse_legacy_cache_name(name),
            "seed": "UNRECOVERABLE_NOT_STORED",
            "embedded_git_commit": "UNRECOVERABLE_NOT_STORED",
            "emission_configuration": {
                "filename_encoded": False,
                "status": "INFERRED_NOT_EMBEDDED",
                "evidence": (
                    "Historical code surrounding the cache mtimes evaluates the joint "
                    "IDM acceleration and Gaussian [v,s,dv] emission unconditionally; "
                    "Analysis-HMM.ipynb explicitly requests IDM-MvNormal for B5S5."
                ),
            },
        }
        if path.is_file():
            stat = path.stat()
            item.update(
                {
                    "size_bytes": stat.st_size,
                    "mtime_utc": datetime.fromtimestamp(
                        stat.st_mtime, tz=timezone.utc
                    ).isoformat(),
                }
            )
            if calculate_sha256:
                print(f"Hashing legacy cache {name} ({stat.st_size} bytes)...", flush=True)
                item["sha256"] = sha256_file(path)
            if inspect_payloads:
                print(f"Inspecting legacy cache payload {name}...", flush=True)
                item["payload"] = inspect_legacy_cache_payload(
                    path,
                    cohort_pair_nos=cohort_pair_nos,
                    cohort_lengths=cohort_lengths,
                )
        inventory.append(item)
    return inventory


def _source_file_inventory(
    repo_root: Path,
    recording_ids: Iterable[int],
    *,
    calculate_sha256: bool,
) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for recording_id in sorted(set(int(value) for value in recording_ids)):
        for suffix in ("_tracks.csv", "_tracksMeta.csv", "_recordingMeta.csv"):
            path = repo_root / "data" / "highD" / f"{recording_id:02d}{suffix}"
            item: dict[str, Any] = {
                "recording_id": recording_id,
                "path": path.relative_to(repo_root).as_posix(),
                "present": path.is_file(),
            }
            if path.is_file():
                stat = path.stat()
                item.update(
                    {
                        "size_bytes": stat.st_size,
                        "mtime_utc": datetime.fromtimestamp(
                            stat.st_mtime, tz=timezone.utc
                        ).isoformat(),
                    }
                )
                if calculate_sha256:
                    print(f"Hashing source file {path.name}...", flush=True)
                    item["sha256"] = sha256_file(path)
            entries.append(item)
    return entries


def _track_manifest_row(track: Mapping[str, Any], cohort_index: int) -> dict[str, Any]:
    return {
        "cohort_index_zero_based": cohort_index,
        "trajectory_id": track["trajectory_id"],
        "recording_id": track["recording_id"],
        "follower_id": track["follower_id"],
        "pair_no": track["pair_no"],
        "follower_class": track["follower_class"],
        "length_5hz": track["length"],
        "duration_between_first_and_last_sample_seconds": (
            (track["length"] - 1) / 5.0
        ),
        "first_frame": int(track["frames"][0]),
        "last_frame": int(track["frames"][-1]),
        "unique_leader_ids": sorted(
            int(value) for value in np.unique(track["leader_ids"])
        ),
        "hashes": track["hashes"],
        "trajectory_payload_sha256": track["trajectory_payload_sha256"],
        "audit": track["audit"],
    }


def _aggregate_domain_audit(tracks: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    count_fields = (
        "nonfinite_context_values",
        "nonfinite_acceleration_values",
        "negative_speed_values",
        "nonpositive_gap_values",
        "frame_step_not_five",
    )
    totals = {
        field: sum(int(track["audit"][field]) for track in tracks)
        for field in count_fields
    }
    return {
        "totals": totals,
        "status": "PASS" if not any(totals.values()) else "REVIEW_REQUIRED",
        "policy": (
            "No values are removed or repaired by this forensic exporter. Any nonzero "
            "count must be reviewed before formal inference."
        ),
    }


def build_legacy_continuity_decision(
    tracks: Sequence[Mapping[str, Any]],
    *,
    n_observations: int,
    target_input_fingerprint_sha256: str,
    case_indices: Sequence[int] = MANUSCRIPT_CASE_INDICES,
) -> dict[str, Any]:
    """Record the approved decision to preserve the exact legacy observations."""

    if n_observations <= 0:
        raise ValueError("n_observations must be positive.")
    negative_counts = [
        int(track["audit"]["negative_speed_values"]) for track in tracks
    ]
    negative_speed_values = sum(negative_counts)
    affected_indices = [
        index for index, count in enumerate(negative_counts) if count > 0
    ]
    affected_ids = [str(tracks[index]["trajectory_id"]) for index in affected_indices]
    unaffected_cases = [
        int(index)
        for index in case_indices
        if 0 <= index < len(tracks) and negative_counts[index] == 0
    ]
    minimum_observed_speed = min(
        float(track["audit"]["speed_min"]) for track in tracks
    )
    return {
        "decision_label": "scheme-b-frozen-cohort-continuity",
        "decision_source": "scheme_b_scope_and_implementation_disclosure",
        "allow_observed_negative_speed": True,
        "minimum_observed_speed_mps": minimum_observed_speed,
        "minimum_allowed_speed_mps": minimum_observed_speed,
        "negative_speed_values": negative_speed_values,
        "negative_speed_fraction": negative_speed_values / n_observations,
        "affected_trajectory_count": len(affected_indices),
        "affected_trajectory_indices_zero_based": affected_indices,
        "affected_trajectory_ids": affected_ids,
        "unaffected_case_indices_zero_based": unaffected_cases,
        "all_fixed_manuscript_cases_unaffected": (
            unaffected_cases == [int(index) for index in case_indices]
        ),
        "raw_observations_preserved": True,
        "speed_clipped": False,
        "observations_deleted": False,
        "relative_speed_recomputed": False,
        "target_input_fingerprint_unchanged": True,
        "target_input_fingerprint_sha256": target_input_fingerprint_sha256,
        "justification": (
            "Scheme B preserves the frozen historical cohort byte-for-byte so that "
            "inference corrections are isolated from data cleaning. Small negative "
            "speeds near standstill remain raw highD observations; no clipping, "
            "deletion, or delta-v recomputation is performed."
        ),
    }


def create_output_directory(output_root: Path) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    output = output_root / f"{stamp}_legacy_cohort_forensics"
    output.mkdir(parents=True, exist_ok=False)
    return output


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def run_forensics(
    repo_root: Path,
    output_root: Path,
    *,
    historical_commit: str = DEFAULT_HISTORICAL_COMMIT,
    cohort_limit: int = 100,
    calculate_cache_sha256: bool = True,
    calculate_source_sha256: bool = True,
    inspect_cache_payloads: bool = True,
) -> Path:
    repo_root = repo_root.resolve()
    output_directory = create_output_directory(output_root.resolve())
    located_pairs, preprocessing_inventory = load_located_pairs(
        repo_root, DEFAULT_RECORDINGS, limit=cohort_limit
    )
    if len(located_pairs) != cohort_limit:
        raise RuntimeError(
            f"Legacy selection produced {len(located_pairs)} pairs, expected {cohort_limit}."
        )
    cohort = build_cohort_arrays(located_pairs)
    arrays = cohort["arrays"]
    historical = compare_with_historical_cache(
        repo_root, located_pairs, historical_commit
    )
    cache_inventory = inventory_expected_legacy_caches(
        repo_root,
        calculate_sha256=calculate_cache_sha256,
        inspect_payloads=inspect_cache_payloads,
        cohort_pair_nos=arrays["pair_nos"],
        cohort_lengths=arrays["lengths"],
    )
    source_inventory = _source_file_inventory(
        repo_root,
        arrays["recording_ids"],
        calculate_sha256=calculate_source_sha256,
    )

    npz_path = output_directory / "cohort_data.npz"
    np.savez_compressed(npz_path, **arrays)
    trajectory_rows = [
        _track_manifest_row(track, index)
        for index, track in enumerate(cohort["tracks"])
    ]
    case_rows = [trajectory_rows[index] for index in MANUSCRIPT_CASE_INDICES]
    recording_counts = {
        str(recording_id): int((arrays["recording_ids"] == recording_id).sum())
        for recording_id in np.unique(arrays["recording_ids"])
    }
    class_counts = {
        str(follower_class): int((arrays["follower_classes"] == follower_class).sum())
        for follower_class in np.unique(arrays["follower_classes"])
    }
    target_inputs = {
        "cohort_order_sha256": cohort["cohort_order_sha256"],
        "context_columns": list(CONTEXT_COLUMNS),
        "frame_rate_hz": 5,
        "manuscript_case_indices_zero_based": list(MANUSCRIPT_CASE_INDICES),
        "historical_canonical_match": historical[
            "canonical_trajectory_views_match"
        ],
    }
    target_input_fingerprint = {
        "inputs": target_inputs,
        "sha256": sha256_bytes(_json_bytes(target_inputs)),
    }
    domain_audit = _aggregate_domain_audit(cohort["tracks"])
    domain_audit["legacy_continuity_decision"] = build_legacy_continuity_decision(
        cohort["tracks"],
        n_observations=int(arrays["acceleration"].shape[0]),
        target_input_fingerprint_sha256=target_input_fingerprint["sha256"],
    )
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "repo_root": str(repo_root),
        "selection_rule": {
            "source_code": "MCMC/MCMC_main.py:save_or_load_samples",
            "dataset_list": [1, 60],
            "dataset_list_semantics": "inclusive ascending recording IDs 1 through 60",
            "minimum_raw_valid_frames": 50 * 25,
            "minimum_nominal_seconds": 50,
            "take_first_n_after_recording_order_concatenation": cohort_limit,
            "lane_change_followers_excluded": True,
            "leader_follower_consistency_checked": True,
        },
        "downsampling": {
            "source_rate_hz": 25,
            "output_rate_hz": 5,
            "stride_frames": 5,
            "current_sample_python_slice": "[:-6:5]",
            "one_step_ahead_python_slice": "[5:-1:5]",
            "inference_context_order": list(CONTEXT_COLUMNS),
            "delta_v_definition": "v_follower - v_leader",
            "acceleration_role": "observed follower acceleration at current sample",
        },
        "cohort": {
            "n_trajectories": len(trajectory_rows),
            "n_observations_5hz": int(arrays["acceleration"].shape[0]),
            "recording_counts": recording_counts,
            "class_counts": class_counts,
            "unique_composite_followers": len(
                set(zip(arrays["recording_ids"], arrays["follower_ids"]))
            ),
            "unique_raw_follower_ids": int(np.unique(arrays["follower_ids"]).size),
            "unique_pair_nos": int(np.unique(arrays["pair_nos"]).size),
            "cohort_order_sha256": cohort["cohort_order_sha256"],
            "array_manifest": cohort["array_manifest"],
            "domain_audit": domain_audit,
            "trajectories": trajectory_rows,
        },
        "manuscript_cases": {
            "source_notebook": "MCMC/Evaluation-HMM.ipynb",
            "source_expression": "for driver_id in [2,3,6,9]",
            "indexing": "zero-based index into the legacy first-100 cohort; pair_No is matched to driver_id",
            "cases": case_rows,
        },
        "preprocessing_cache_inventory_until_selection_complete": preprocessing_inventory,
        "historical_preprocessing_comparison": historical,
        "legacy_posterior_caches": cache_inventory,
        "raw_source_inventory_for_selected_recordings": source_inventory,
        "artifacts": {
            "cohort_data_npz": {
                "path": npz_path.name,
                "size_bytes": npz_path.stat().st_size,
                "sha256": sha256_file(npz_path),
            }
        },
        "target_input_fingerprint": target_input_fingerprint,
        "known_provenance_limits": [
            "Legacy posterior caches do not embed a random seed.",
            "Legacy posterior caches do not embed a Git commit or dirty-worktree hash.",
            "The legacy filename does not encode emission_func.",
            "B5S1 and B1S5 legacy posterior caches are absent if marked present=false.",
            "File mtimes are filesystem metadata, not cryptographic run receipts.",
        ],
    }
    _write_json(output_directory / "manifest.json", manifest)
    print(f"Forensic output written to {output_directory}", flush=True)
    return output_directory


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    default_root = Path(__file__).resolve().parents[2]
    parser.add_argument("--repo-root", type=Path, default=default_root)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(__file__).resolve().parent / "forensic_outputs",
    )
    parser.add_argument(
        "--historical-commit", default=DEFAULT_HISTORICAL_COMMIT
    )
    parser.add_argument("--cohort-limit", type=int, default=100)
    parser.add_argument("--skip-cache-sha256", action="store_true")
    parser.add_argument("--skip-source-sha256", action="store_true")
    parser.add_argument("--skip-cache-payload-inspection", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    run_forensics(
        args.repo_root,
        args.output_root,
        historical_commit=args.historical_commit,
        cohort_limit=args.cohort_limit,
        calculate_cache_sha256=not args.skip_cache_sha256,
        calculate_source_sha256=not args.skip_source_sha256,
        inspect_cache_payloads=not args.skip_cache_payload_inspection,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
