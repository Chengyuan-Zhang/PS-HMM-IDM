"""Deterministic HighD car-following segment construction.

This module reads the original HighD CSV files directly.  It deliberately does
not import the legacy loader or read/write any legacy pickle/NumPy caches.  All
vehicle metadata are joined by the explicit HighD vehicle ``id``.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd


DEFAULT_DURATION_THRESHOLDS_S: Tuple[float, ...] = (20.0, 30.0, 40.0, 50.0)

TRACK_REQUIRED_COLUMNS: Tuple[str, ...] = (
    "id",
    "frame",
    "x",
    "width",
    "xVelocity",
    "xAcceleration",
    "precedingId",
    "laneId",
)
TRACK_OPTIONAL_COLUMNS: Tuple[str, ...] = ("followingId", "dhw")
TRACK_META_REQUIRED_COLUMNS: Tuple[str, ...] = (
    "id",
    "class",
    "drivingDirection",
    "numLaneChanges",
)
RECORDING_META_REQUIRED_COLUMNS: Tuple[str, ...] = (
    "id",
    "frameRate",
    "locationId",
)

MANIFEST_COLUMNS: Tuple[str, ...] = (
    "trajectory_id",
    "recording_id",
    "location_id",
    "follower_id",
    "leader_id",
    "follower_class",
    "driving_direction",
    "lane_id",
    "start_frame",
    "end_frame",
    "duration_s",
    "n_samples_raw",
    "n_samples",
    "source_frame_rate_hz",
    "output_frame_rate_hz",
)


@dataclass(frozen=True)
class PipelineConfig:
    """Configuration whose values affect segment membership or sample values."""

    source_frame_rate_hz: float = 25.0
    output_frame_rate_hz: float = 5.0
    duration_thresholds_s: Tuple[float, ...] = DEFAULT_DURATION_THRESHOLDS_S
    require_reciprocal_following: bool = True
    minimum_gap_m: float = 0.0

    def __post_init__(self) -> None:
        if self.source_frame_rate_hz <= 0.0 or self.output_frame_rate_hz <= 0.0:
            raise ValueError("Frame rates must be positive.")
        ratio = self.source_frame_rate_hz / self.output_frame_rate_hz
        if not math.isclose(ratio, round(ratio), rel_tol=0.0, abs_tol=1e-12):
            raise ValueError(
                "source_frame_rate_hz must be an integer multiple of "
                "output_frame_rate_hz."
            )
        if self.output_frame_rate_hz > self.source_frame_rate_hz:
            raise ValueError("Upsampling is not supported.")
        if self.minimum_gap_m < 0.0:
            raise ValueError("minimum_gap_m cannot be negative.")
        if not self.duration_thresholds_s:
            raise ValueError("At least one duration threshold is required.")
        if any(value < 0.0 for value in self.duration_thresholds_s):
            raise ValueError("Duration thresholds cannot be negative.")
        if tuple(sorted(set(self.duration_thresholds_s))) != self.duration_thresholds_s:
            raise ValueError("Duration thresholds must be unique and increasing.")

    @property
    def downsample_step(self) -> int:
        """Integer number of source frames between retained observations."""

        return int(round(self.source_frame_rate_hz / self.output_frame_rate_hz))


@dataclass(frozen=True)
class HighDRecordingPaths:
    """Paths to the three original CSV files required for one recording."""

    recording_id: int
    tracks_csv: Path
    tracks_meta_csv: Path
    recording_meta_csv: Path

    @classmethod
    def from_root(
        cls, root: Union[str, Path], recording_id: int
    ) -> "HighDRecordingPaths":
        base = Path(root)
        prefix = f"{int(recording_id):02d}"
        return cls(
            recording_id=int(recording_id),
            tracks_csv=base / f"{prefix}_tracks.csv",
            tracks_meta_csv=base / f"{prefix}_tracksMeta.csv",
            recording_meta_csv=base / f"{prefix}_recordingMeta.csv",
        )

    def validate(self) -> None:
        missing = [
            str(path)
            for path in (
                self.tracks_csv,
                self.tracks_meta_csv,
                self.recording_meta_csv,
            )
            if not path.is_file()
        ]
        if missing:
            raise FileNotFoundError("Missing HighD CSV file(s): " + ", ".join(missing))


@dataclass(frozen=True)
class SegmentRejection:
    """Audit record for a candidate follower or segment that was excluded."""

    recording_id: int
    follower_id: int
    reason: str
    leader_id: Optional[int] = None
    start_frame: Optional[int] = None
    end_frame: Optional[int] = None


@dataclass
class TrajectorySegment:
    """One accepted, maximal constant-leader and constant-lane segment."""

    manifest: Dict[str, object]
    samples: pd.DataFrame

    @property
    def trajectory_id(self) -> str:
        return str(self.manifest["trajectory_id"])


@dataclass
class RecordingBuildResult:
    """Accepted segments and auditable exclusions for one recording."""

    recording_id: int
    location_id: int
    segments: List[TrajectorySegment] = field(default_factory=list)
    rejections: List[SegmentRejection] = field(default_factory=list)

    @property
    def manifest(self) -> pd.DataFrame:
        return _manifest_frame(segment.manifest for segment in self.segments)

    @property
    def samples(self) -> pd.DataFrame:
        if not self.segments:
            return pd.DataFrame()
        return pd.concat(
            [segment.samples for segment in self.segments],
            ignore_index=True,
        ).sort_values(
            ["recording_id", "follower_id", "frame", "trajectory_id"],
            kind="mergesort",
            ignore_index=True,
        )


@dataclass
class DatasetBuildResult:
    """Combined deterministic output across a sorted recording list."""

    config: PipelineConfig
    recordings: List[RecordingBuildResult]

    @property
    def manifest(self) -> pd.DataFrame:
        return _manifest_frame(
            segment.manifest
            for recording in self.recordings
            for segment in recording.segments
        )

    @property
    def samples(self) -> pd.DataFrame:
        frames = [recording.samples for recording in self.recordings]
        frames = [frame for frame in frames if not frame.empty]
        if not frames:
            return pd.DataFrame()
        return pd.concat(frames, ignore_index=True).sort_values(
            ["recording_id", "follower_id", "frame", "trajectory_id"],
            kind="mergesort",
            ignore_index=True,
        )

    @property
    def rejections(self) -> List[SegmentRejection]:
        return [
            rejection
            for recording in self.recordings
            for rejection in recording.rejections
        ]


@dataclass(frozen=True)
class ManifestPaths:
    """Files created by :func:`write_manifest`."""

    csv_path: Path
    json_path: Path
    rejection_csv_path: Path


def _require_columns(frame: pd.DataFrame, required: Sequence[str], name: str) -> None:
    missing = sorted(set(required).difference(frame.columns))
    if missing:
        raise ValueError(f"{name} is missing required columns: {missing}")


def _coerce_integral_columns(
    frame: pd.DataFrame, columns: Sequence[str], name: str
) -> pd.DataFrame:
    result = frame.copy()
    for column in columns:
        values = pd.to_numeric(result[column], errors="coerce")
        if values.isna().any() or not np.allclose(values, np.round(values)):
            raise ValueError(f"{name}.{column} must contain finite integer values.")
        result[column] = values.astype(np.int64)
    return result


def _manifest_frame(rows: Iterable[Mapping[str, object]]) -> pd.DataFrame:
    data = list(rows)
    if not data:
        return pd.DataFrame(columns=MANIFEST_COLUMNS)
    frame = pd.DataFrame(data)
    frame = frame.loc[:, MANIFEST_COLUMNS]
    if frame["trajectory_id"].duplicated().any():
        duplicates = frame.loc[
            frame["trajectory_id"].duplicated(keep=False), "trajectory_id"
        ].tolist()
        raise ValueError(f"Duplicate trajectory IDs: {duplicates}")
    return frame.sort_values(
        ["recording_id", "follower_id", "start_frame", "leader_id", "lane_id"],
        kind="mergesort",
        ignore_index=True,
    )


def _trajectory_id(
    recording_id: int,
    follower_id: int,
    leader_id: int,
    start_frame: int,
    end_frame: int,
) -> str:
    return (
        f"r{recording_id:02d}_f{follower_id:04d}_l{leader_id:04d}_"
        f"s{start_frame:06d}_e{end_frame:06d}"
    )


def _travel_sign_by_vehicle(tracks: pd.DataFrame) -> Dict[int, int]:
    """Infer HighD travel orientation from each vehicle's signed velocity."""

    signs: Dict[int, int] = {}
    for vehicle_id, rows in tracks.groupby("id", sort=True):
        velocities = pd.to_numeric(rows["xVelocity"], errors="coerce").to_numpy()
        finite = velocities[np.isfinite(velocities)]
        nonzero = finite[~np.isclose(finite, 0.0)]
        if nonzero.size == 0:
            signs[int(vehicle_id)] = 0
            continue
        signs[int(vehicle_id)] = -1 if float(np.median(nonzero)) < 0.0 else 1
    return signs


def _normalise_longitudinal(
    x: np.ndarray,
    width: np.ndarray,
    velocity: np.ndarray,
    acceleration: np.ndarray,
    travel_sign: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if travel_sign == 1:
        return x.copy(), velocity.copy(), acceleration.copy()
    if travel_sign == -1:
        return -x - width, -velocity, -acceleration
    raise ValueError("Travel direction is undefined for a stationary vehicle track.")


def discover_recording_ids(root: Union[str, Path]) -> List[int]:
    """Return sorted recording IDs with all three required original CSVs."""

    base = Path(root)
    pattern = re.compile(r"^(\d{2})_tracks\.csv$")
    recording_ids: List[int] = []
    for tracks_path in sorted(base.glob("*_tracks.csv")):
        match = pattern.match(tracks_path.name)
        if match is None:
            continue
        recording_id = int(match.group(1))
        paths = HighDRecordingPaths.from_root(base, recording_id)
        if paths.tracks_meta_csv.is_file() and paths.recording_meta_csv.is_file():
            recording_ids.append(recording_id)
    return recording_ids


def read_recording(
    paths: HighDRecordingPaths,
    config: Optional[PipelineConfig] = None,
) -> RecordingBuildResult:
    """Read one recording's raw CSVs and construct accepted segments."""

    paths.validate()
    selected_track_columns = set(TRACK_REQUIRED_COLUMNS + TRACK_OPTIONAL_COLUMNS)
    tracks = pd.read_csv(
        paths.tracks_csv,
        usecols=lambda column: column in selected_track_columns,
    )
    track_meta = pd.read_csv(paths.tracks_meta_csv)
    recording_meta = pd.read_csv(paths.recording_meta_csv)
    return build_recording_from_frames(
        tracks=tracks,
        track_meta=track_meta,
        recording_meta=recording_meta,
        recording_id=paths.recording_id,
        config=config,
    )


def build_recording_from_frames(
    tracks: pd.DataFrame,
    track_meta: pd.DataFrame,
    recording_meta: pd.DataFrame,
    recording_id: int,
    config: Optional[PipelineConfig] = None,
) -> RecordingBuildResult:
    """Construct segments from in-memory HighD-shaped frames.

    This public in-memory entry point makes the transformations testable without
    reading the private full HighD dataset.
    """

    cfg = config or PipelineConfig()
    _require_columns(tracks, TRACK_REQUIRED_COLUMNS, "tracks")
    _require_columns(track_meta, TRACK_META_REQUIRED_COLUMNS, "tracksMeta")
    _require_columns(recording_meta, RECORDING_META_REQUIRED_COLUMNS, "recordingMeta")

    if len(recording_meta) != 1:
        raise ValueError("recordingMeta must contain exactly one row.")
    meta_recording_id = int(recording_meta.iloc[0]["id"])
    if meta_recording_id != int(recording_id):
        raise ValueError(
            f"Requested recording {recording_id}, but recordingMeta reports "
            f"{meta_recording_id}."
        )
    source_frame_rate = float(recording_meta.iloc[0]["frameRate"])
    if not math.isclose(
        source_frame_rate,
        cfg.source_frame_rate_hz,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError(
            f"Recording frame rate {source_frame_rate} does not match configured "
            f"rate {cfg.source_frame_rate_hz}."
        )
    location_id = int(recording_meta.iloc[0]["locationId"])

    tracks_clean = _coerce_integral_columns(
        tracks,
        ("id", "frame", "precedingId", "laneId"),
        "tracks",
    )
    if "followingId" in tracks_clean.columns:
        tracks_clean = _coerce_integral_columns(
            tracks_clean,
            ("followingId",),
            "tracks",
        )
    meta_clean = _coerce_integral_columns(
        track_meta,
        ("id", "drivingDirection", "numLaneChanges"),
        "tracksMeta",
    )
    if meta_clean["id"].duplicated().any():
        raise ValueError("tracksMeta.id must be unique within a recording.")
    classes = meta_clean["class"].astype("string").str.strip()
    if classes.isna().any() or (classes == "").any():
        raise ValueError("tracksMeta.class cannot be missing or empty.")
    meta_clean["class"] = classes.astype(str)
    if tracks_clean.duplicated(["id", "frame"]).any():
        raise ValueError("tracks must contain at most one row per vehicle and frame.")

    numeric_columns = ("x", "width", "xVelocity", "xAcceleration")
    for column in numeric_columns:
        tracks_clean[column] = pd.to_numeric(tracks_clean[column], errors="coerce")

    joined = tracks_clean.merge(
        meta_clean.loc[
            :, ("id", "class", "drivingDirection", "numLaneChanges")
        ],
        on="id",
        how="left",
        validate="many_to_one",
        indicator="_metadata_join",
    )
    if (joined["_metadata_join"] != "both").any():
        missing_ids = sorted(
            joined.loc[joined["_metadata_join"] != "both", "id"].unique().tolist()
        )
        raise ValueError(f"Track rows lack metadata for vehicle IDs: {missing_ids}")
    joined = joined.drop(columns="_metadata_join").sort_values(
        ["id", "frame"], kind="mergesort", ignore_index=True
    )

    signs = _travel_sign_by_vehicle(joined)
    leader_columns = [
        "id",
        "frame",
        "x",
        "width",
        "xVelocity",
        "xAcceleration",
    ]
    if "followingId" in joined.columns:
        leader_columns.append("followingId")
    leader_table = joined.loc[:, leader_columns].rename(
        columns={
            "id": "leader_id",
            "x": "leader_x_raw",
            "width": "leader_width_m",
            "xVelocity": "leader_speed_raw",
            "xAcceleration": "leader_acceleration_raw",
            "followingId": "leader_following_id",
        }
    )

    result = RecordingBuildResult(
        recording_id=int(recording_id),
        location_id=location_id,
    )
    observed_vehicle_ids = set(int(value) for value in joined["id"].unique())
    lane_changing_ids = set(
        int(value)
        for value in meta_clean.loc[
            meta_clean["numLaneChanges"] > 0, "id"
        ].tolist()
    )
    for follower_id in sorted(lane_changing_ids):
        if follower_id in observed_vehicle_ids:
            result.rejections.append(
                SegmentRejection(
                    recording_id=int(recording_id),
                    follower_id=follower_id,
                    reason="follower_numLaneChanges_gt_zero",
                )
            )

    # Align leaders once for the whole recording.  Rebuilding the same
    # million-row merge inside every follower segment made a single large
    # recording take minutes and scales approximately with the number of
    # candidate segments.  The global many-to-one join is mathematically
    # identical because (leader_id, frame) is unique by validation above.
    aligned_joined = joined.merge(
        leader_table,
        left_on=["precedingId", "frame"],
        right_on=["leader_id", "frame"],
        how="left",
        validate="many_to_one",
        indicator="_leader_join",
    )
    eligible = aligned_joined.loc[
        (~aligned_joined["id"].isin(lane_changing_ids))
        & (aligned_joined["precedingId"] > 0)
    ].copy()

    for follower_id, follower_rows in eligible.groupby("id", sort=True):
        follower_rows = follower_rows.sort_values("frame", kind="mergesort").copy()
        run_start = (
            follower_rows["frame"].diff().ne(1)
            | follower_rows["precedingId"].ne(follower_rows["precedingId"].shift())
            | follower_rows["laneId"].ne(follower_rows["laneId"].shift())
        )
        follower_rows["_run"] = run_start.cumsum()

        for _, candidate in follower_rows.groupby("_run", sort=True):
            candidate = candidate.drop(columns="_run").sort_values(
                "frame", kind="mergesort", ignore_index=True
            )
            leader_id = int(candidate.iloc[0]["precedingId"])
            lane_id = int(candidate.iloc[0]["laneId"])
            start_frame = int(candidate.iloc[0]["frame"])
            end_frame = int(candidate.iloc[-1]["frame"])
            rejection_kwargs = {
                "recording_id": int(recording_id),
                "follower_id": int(follower_id),
                "leader_id": leader_id,
                "start_frame": start_frame,
                "end_frame": end_frame,
            }

            if candidate["precedingId"].nunique() != 1:
                result.rejections.append(
                    SegmentRejection(reason="leader_changes_within_segment", **rejection_kwargs)
                )
                continue
            if candidate["laneId"].nunique() != 1:
                result.rejections.append(
                    SegmentRejection(reason="lane_changes_within_segment", **rejection_kwargs)
                )
                continue
            expected_frames = np.arange(start_frame, end_frame + 1, dtype=np.int64)
            if not np.array_equal(candidate["frame"].to_numpy(), expected_frames):
                result.rejections.append(
                    SegmentRejection(reason="noncontiguous_follower_frames", **rejection_kwargs)
                )
                continue

            aligned = candidate
            if (aligned["_leader_join"] != "both").any():
                result.rejections.append(
                    SegmentRejection(reason="missing_leader_frame", **rejection_kwargs)
                )
                continue
            if (
                cfg.require_reciprocal_following
                and "leader_following_id" in aligned.columns
                and (aligned["leader_following_id"] != int(follower_id)).any()
            ):
                result.rejections.append(
                    SegmentRejection(reason="nonreciprocal_leader_alignment", **rejection_kwargs)
                )
                continue

            follower_sign = signs.get(int(follower_id), 0)
            leader_sign = signs.get(leader_id, 0)
            if follower_sign == 0 or leader_sign == 0:
                result.rejections.append(
                    SegmentRejection(reason="undefined_travel_direction", **rejection_kwargs)
                )
                continue
            if follower_sign != leader_sign:
                result.rejections.append(
                    SegmentRejection(reason="opposing_travel_directions", **rejection_kwargs)
                )
                continue

            follower_values = aligned.loc[
                :, ("x", "width", "xVelocity", "xAcceleration")
            ].to_numpy(dtype=float)
            leader_values = aligned.loc[
                :,
                (
                    "leader_x_raw",
                    "leader_width_m",
                    "leader_speed_raw",
                    "leader_acceleration_raw",
                ),
            ].to_numpy(dtype=float)
            if not np.isfinite(follower_values).all() or not np.isfinite(
                leader_values
            ).all():
                result.rejections.append(
                    SegmentRejection(reason="nonfinite_kinematics", **rejection_kwargs)
                )
                continue
            if "dhw" in aligned.columns:
                reported_dhw = pd.to_numeric(aligned["dhw"], errors="coerce").to_numpy(
                    dtype=float
                )
                if not np.isfinite(reported_dhw).all():
                    result.rejections.append(
                        SegmentRejection(
                            reason="nonfinite_reported_dhw", **rejection_kwargs
                        )
                    )
                    continue
            if (follower_values[:, 1] <= 0.0).any() or (
                leader_values[:, 1] <= 0.0
            ).any():
                result.rejections.append(
                    SegmentRejection(reason="nonpositive_vehicle_width", **rejection_kwargs)
                )
                continue

            follower_x, follower_speed, follower_acceleration = _normalise_longitudinal(
                x=follower_values[:, 0],
                width=follower_values[:, 1],
                velocity=follower_values[:, 2],
                acceleration=follower_values[:, 3],
                travel_sign=follower_sign,
            )
            leader_x, leader_speed, leader_acceleration = _normalise_longitudinal(
                x=leader_values[:, 0],
                width=leader_values[:, 1],
                velocity=leader_values[:, 2],
                acceleration=leader_values[:, 3],
                travel_sign=leader_sign,
            )
            gap = leader_x - follower_x - follower_values[:, 1]
            if not np.isfinite(gap).all() or (gap <= cfg.minimum_gap_m).any():
                result.rejections.append(
                    SegmentRejection(reason="nonpositive_gap", **rejection_kwargs)
                )
                continue

            trajectory_id = _trajectory_id(
                recording_id=int(recording_id),
                follower_id=int(follower_id),
                leader_id=leader_id,
                start_frame=start_frame,
                end_frame=end_frame,
            )
            raw_samples = pd.DataFrame(
                {
                    "trajectory_id": trajectory_id,
                    "recording_id": int(recording_id),
                    "location_id": location_id,
                    "follower_id": int(follower_id),
                    "leader_id": leader_id,
                    "frame": aligned["frame"].to_numpy(dtype=np.int64),
                    "time_s": (aligned["frame"].to_numpy(dtype=float) - start_frame)
                    / cfg.source_frame_rate_hz,
                    "lane_id": lane_id,
                    "follower_class": str(aligned.iloc[0]["class"]),
                    "driving_direction": int(aligned.iloc[0]["drivingDirection"]),
                    "follower_x_m": follower_x,
                    "follower_speed_mps": follower_speed,
                    "follower_acceleration_mps2": follower_acceleration,
                    "leader_x_m": leader_x,
                    "leader_speed_mps": leader_speed,
                    "leader_acceleration_mps2": leader_acceleration,
                    "delta_v_mps": follower_speed - leader_speed,
                    "gap_m": gap,
                }
            )
            if "dhw" in aligned.columns:
                raw_samples["reported_dhw_m"] = reported_dhw
            samples = raw_samples.iloc[:: cfg.downsample_step].reset_index(drop=True)
            duration_s = (end_frame - start_frame) / cfg.source_frame_rate_hz
            manifest = {
                "trajectory_id": trajectory_id,
                "recording_id": int(recording_id),
                "location_id": location_id,
                "follower_id": int(follower_id),
                "leader_id": leader_id,
                "follower_class": str(aligned.iloc[0]["class"]),
                "driving_direction": int(aligned.iloc[0]["drivingDirection"]),
                "lane_id": lane_id,
                "start_frame": start_frame,
                "end_frame": end_frame,
                "duration_s": float(duration_s),
                "n_samples_raw": int(len(raw_samples)),
                "n_samples": int(len(samples)),
                "source_frame_rate_hz": float(cfg.source_frame_rate_hz),
                "output_frame_rate_hz": float(cfg.output_frame_rate_hz),
            }
            result.segments.append(TrajectorySegment(manifest=manifest, samples=samples))

    result.segments.sort(
        key=lambda segment: (
            int(segment.manifest["follower_id"]),
            int(segment.manifest["start_frame"]),
            int(segment.manifest["leader_id"]),
            int(segment.manifest["lane_id"]),
        )
    )
    result.rejections.sort(
        key=lambda rejection: (
            rejection.follower_id,
            -1 if rejection.start_frame is None else rejection.start_frame,
            rejection.reason,
        )
    )
    _validate_manifest_frame_nonoverlap(result.manifest)
    return result


def build_dataset(
    root: Union[str, Path],
    recording_ids: Sequence[int],
    config: Optional[PipelineConfig] = None,
) -> DatasetBuildResult:
    """Build a deterministic dataset in ascending recording-ID order."""

    cfg = config or PipelineConfig()
    unique_ids = sorted(set(int(value) for value in recording_ids))
    if not unique_ids:
        raise ValueError("recording_ids cannot be empty.")
    recordings = [
        read_recording(HighDRecordingPaths.from_root(root, recording_id), cfg)
        for recording_id in unique_ids
    ]
    result = DatasetBuildResult(config=cfg, recordings=recordings)
    _validate_manifest_frame_nonoverlap(result.manifest)
    return result


def _validate_manifest_frame_nonoverlap(manifest: pd.DataFrame) -> None:
    """Reject overlapping accepted intervals for the same recorded follower."""

    if manifest.empty:
        return
    _require_columns(
        manifest,
        ("recording_id", "follower_id", "start_frame", "end_frame"),
        "manifest",
    )
    ordered = manifest.sort_values(
        ["recording_id", "follower_id", "start_frame", "end_frame"],
        kind="mergesort",
    )
    for _, rows in ordered.groupby(["recording_id", "follower_id"], sort=True):
        previous_end: Optional[int] = None
        for row in rows.itertuples(index=False):
            start = int(row.start_frame)
            end = int(row.end_frame)
            if start > end:
                raise ValueError("Manifest start_frame cannot exceed end_frame.")
            if previous_end is not None and start <= previous_end:
                raise ValueError(
                    "Accepted segments overlap in frame for the same recorded follower."
                )
            previous_end = end


def filter_manifest_by_duration(
    manifest: pd.DataFrame, threshold_s: float
) -> pd.DataFrame:
    """Return a stable copy containing segments at least ``threshold_s`` long."""

    if threshold_s < 0.0:
        raise ValueError("threshold_s cannot be negative.")
    _require_columns(manifest, MANIFEST_COLUMNS, "manifest")
    return manifest.loc[manifest["duration_s"] >= threshold_s].sort_values(
        ["recording_id", "follower_id", "start_frame", "trajectory_id"],
        kind="mergesort",
        ignore_index=True,
    )


def build_duration_inventory(
    manifest: pd.DataFrame,
    thresholds_s: Sequence[float] = DEFAULT_DURATION_THRESHOLDS_S,
) -> pd.DataFrame:
    """Count eligible segments, drivers, recordings, locations, and samples."""

    _require_columns(manifest, MANIFEST_COLUMNS, "manifest")
    rows: List[Dict[str, object]] = []
    for threshold in sorted(set(float(value) for value in thresholds_s)):
        eligible = filter_manifest_by_duration(manifest, threshold)
        driver_keys = eligible.loc[:, ["recording_id", "follower_id"]].drop_duplicates()
        rows.append(
            {
                "threshold_s": threshold,
                "n_segments": int(len(eligible)),
                "n_drivers": int(len(driver_keys)),
                "n_recordings": int(eligible["recording_id"].nunique()),
                "n_locations": int(eligible["location_id"].nunique()),
                "n_samples": int(eligible["n_samples"].sum()),
                "n_samples_raw": int(eligible["n_samples_raw"].sum()),
            }
        )
    return pd.DataFrame(rows)


def write_manifest(
    result: DatasetBuildResult,
    output_directory: Union[str, Path],
    stem: str = "highd_segments",
) -> ManifestPaths:
    """Write stable CSV/JSON manifests and an exclusion audit.

    The function refuses to overwrite an existing manifest set.  Use a new run
    directory for every inventory build.
    """

    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    paths = ManifestPaths(
        csv_path=output / f"{stem}.csv",
        json_path=output / f"{stem}.json",
        rejection_csv_path=output / f"{stem}_rejections.csv",
    )
    existing = [
        str(path)
        for path in (paths.csv_path, paths.json_path, paths.rejection_csv_path)
        if path.exists()
    ]
    if existing:
        raise FileExistsError("Refusing to overwrite manifest file(s): " + ", ".join(existing))

    manifest = result.manifest
    inventory = build_duration_inventory(
        manifest,
        thresholds_s=result.config.duration_thresholds_s,
    )
    rejection_rows = [asdict(rejection) for rejection in result.rejections]
    rejection_frame = pd.DataFrame(
        rejection_rows,
        columns=(
            "recording_id",
            "follower_id",
            "reason",
            "leader_id",
            "start_frame",
            "end_frame",
        ),
    )
    manifest.to_csv(
        paths.csv_path,
        index=False,
        float_format="%.10g",
        line_terminator="\n",
    )
    rejection_frame.to_csv(
        paths.rejection_csv_path,
        index=False,
        float_format="%.10g",
        line_terminator="\n",
    )
    payload = {
        "schema_version": 1,
        "config": asdict(result.config),
        "recording_ids": sorted(recording.recording_id for recording in result.recordings),
        "manifest": manifest.to_dict(orient="records"),
        "duration_inventory": inventory.to_dict(orient="records"),
        "rejection_counts": dict(sorted(Counter(row.reason for row in result.rejections).items())),
    }
    paths.json_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return paths


def _partition_counts(n_groups: int, fractions: Sequence[float]) -> List[int]:
    raw = [n_groups * value for value in fractions]
    counts = [int(math.floor(value)) for value in raw]
    remaining = n_groups - sum(counts)
    remainder_order = sorted(
        range(len(fractions)),
        key=lambda index: (-(raw[index] - counts[index]), index),
    )
    for index in remainder_order[:remaining]:
        counts[index] += 1
    return counts


def assign_group_splits(
    manifest: pd.DataFrame,
    group_columns: Sequence[str],
    split_fractions: Optional[Mapping[str, float]] = None,
    seed: int = 0,
) -> pd.DataFrame:
    """Assign whole groups to deterministic hash-ordered splits."""

    fractions = split_fractions or {"train": 0.6, "validation": 0.2, "test": 0.2}
    if not fractions:
        raise ValueError("split_fractions cannot be empty.")
    names = list(fractions.keys())
    values = [float(fractions[name]) for name in names]
    if any(value < 0.0 for value in values) or not math.isclose(
        sum(values), 1.0, rel_tol=0.0, abs_tol=1e-12
    ):
        raise ValueError("Split fractions must be nonnegative and sum to one.")
    _require_columns(
        manifest,
        ("trajectory_id", *tuple(group_columns)),
        "manifest",
    )
    if manifest["trajectory_id"].duplicated().any():
        raise ValueError("manifest.trajectory_id must be unique.")

    groups = manifest.loc[:, list(group_columns)].drop_duplicates()
    group_records = [tuple(row) for row in groups.itertuples(index=False, name=None)]

    def digest(group: Tuple[object, ...]) -> str:
        message = json.dumps(
            {"seed": int(seed), "group": group},
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(message.encode("utf-8")).hexdigest()

    ordered_groups = sorted(group_records, key=lambda group: (digest(group), group))
    counts = _partition_counts(len(ordered_groups), values)
    group_to_split: Dict[Tuple[object, ...], str] = {}
    cursor = 0
    for split_name, count in zip(names, counts):
        for group in ordered_groups[cursor : cursor + count]:
            group_to_split[group] = split_name
        cursor += count

    assignments = manifest.loc[
        :, ["trajectory_id", *tuple(group_columns)]
    ].copy()
    assignments["split"] = [
        group_to_split[tuple(row)]
        for row in assignments.loc[:, list(group_columns)].itertuples(
            index=False, name=None
        )
    ]
    assignments = assignments.sort_values("trajectory_id", kind="mergesort", ignore_index=True)
    validate_split_leakage(manifest, assignments, group_columns)
    return assignments


def assign_driver_disjoint_splits(
    manifest: pd.DataFrame,
    split_fractions: Optional[Mapping[str, float]] = None,
    seed: int = 0,
) -> pd.DataFrame:
    """Assign every segment of a recorded follower to the same split."""

    return assign_group_splits(
        manifest,
        group_columns=("recording_id", "follower_id"),
        split_fractions=split_fractions,
        seed=seed,
    )


def assign_recording_disjoint_splits(
    manifest: pd.DataFrame,
    split_fractions: Optional[Mapping[str, float]] = None,
    seed: int = 0,
) -> pd.DataFrame:
    """Assign complete HighD recordings to deterministic splits."""

    return assign_group_splits(
        manifest,
        group_columns=("recording_id",),
        split_fractions=split_fractions,
        seed=seed,
    )


def validate_split_leakage(
    manifest: pd.DataFrame,
    assignments: pd.DataFrame,
    group_columns: Sequence[str],
) -> None:
    """Raise if a trajectory or protected group appears in multiple splits."""

    _require_columns(assignments, ("trajectory_id", "split"), "assignments")
    if assignments["trajectory_id"].duplicated().any():
        raise ValueError("A trajectory is assigned more than once.")
    if set(assignments["trajectory_id"]) != set(manifest["trajectory_id"]):
        raise ValueError("Assignments must cover exactly the manifest trajectories.")
    joined = manifest.loc[
        :, ["trajectory_id", *tuple(group_columns)]
    ].merge(
        assignments.loc[:, ["trajectory_id", "split"]],
        on="trajectory_id",
        how="left",
        validate="one_to_one",
    )
    group_split_counts = joined.groupby(list(group_columns), sort=True)["split"].nunique()
    if (group_split_counts > 1).any():
        raise ValueError("A protected group appears in more than one split.")
    _validate_manifest_frame_nonoverlap(manifest)


def blocked_temporal_folds(
    samples: pd.DataFrame,
    train_fraction: float = 0.5,
    gap_samples: int = 0,
    minimum_partition_samples: int = 1,
) -> pd.DataFrame:
    """Create early-to-late and late-to-early blocked folds per trajectory.

    ``gap_samples`` observations on each side of the split point are excluded
    from both partitions.  Reversing the fold changes which chronological block
    is training data; it never changes the underlying frame boundary.
    """

    required = ("trajectory_id", "recording_id", "follower_id", "frame")
    _require_columns(samples, required, "samples")
    if not 0.0 < train_fraction < 1.0:
        raise ValueError("train_fraction must lie strictly between zero and one.")
    if gap_samples < 0 or minimum_partition_samples < 1:
        raise ValueError("gap_samples and minimum_partition_samples are invalid.")
    if samples.duplicated(["trajectory_id", "frame"]).any():
        raise ValueError("Each trajectory/frame sample must be unique.")

    parts: List[pd.DataFrame] = []
    for trajectory_id, rows in samples.groupby("trajectory_id", sort=True):
        ordered = rows.sort_values("frame", kind="mergesort").reset_index(drop=True)
        split_index = int(math.floor(len(ordered) * train_fraction))
        early_end = split_index - gap_samples
        late_start = split_index + gap_samples
        early = ordered.iloc[:early_end].copy()
        late = ordered.iloc[late_start:].copy()
        if len(early) < minimum_partition_samples or len(late) < minimum_partition_samples:
            raise ValueError(
                f"Trajectory {trajectory_id} is too short for the requested temporal split."
            )

        for fold, training, testing in (
            ("early_to_late", early, late),
            ("late_to_early", late, early),
        ):
            train_rows = training.loc[:, required].copy()
            train_rows["fold"] = fold
            train_rows["partition"] = "train"
            test_rows = testing.loc[:, required].copy()
            test_rows["fold"] = fold
            test_rows["partition"] = "test"
            parts.extend((train_rows, test_rows))

    if not parts:
        return pd.DataFrame(columns=(*required, "fold", "partition"))
    result = pd.concat(parts, ignore_index=True).sort_values(
        ["fold", "trajectory_id", "partition", "frame"],
        kind="mergesort",
        ignore_index=True,
    )
    validate_temporal_folds(result)
    return result


def validate_temporal_folds(folds: pd.DataFrame) -> None:
    """Raise if any follower frame leaks between train and test within a fold."""

    required = (
        "trajectory_id",
        "recording_id",
        "follower_id",
        "frame",
        "fold",
        "partition",
    )
    _require_columns(folds, required, "folds")
    invalid_partitions = set(folds["partition"]).difference({"train", "test"})
    if invalid_partitions:
        raise ValueError(f"Unknown temporal partitions: {sorted(invalid_partitions)}")
    keys = ["recording_id", "follower_id", "frame", "fold"]
    leakage = folds.groupby(keys, sort=True)["partition"].nunique()
    if (leakage > 1).any():
        raise ValueError("A recorded follower frame appears in train and test.")
