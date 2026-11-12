"""Reconstruct and fingerprint the paper's 100-episode highD cohort."""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from experiments.manuscript_storyrev_20260808.forensics import (
    LocatedTrackPair,
    build_cohort_arrays,
)


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "configs" / "frozen_cohort_manifest.json"
RECORDING_ID = 25
PAPER_EPISODES = 100


def extract_legacy_pairs(
    tracks: Sequence[Mapping[str, Any]], lane_change_ids: Sequence[int], classes: Any
) -> list[dict[str, Any]]:
    """Apply the original selection and slicing rules to recording 25.

    These indexing rules intentionally match the historical preprocessing; the
    fingerprint check below rejects any change to the paper's actual cohort.
    """
    pairs: list[dict[str, Any]] = []
    for follower in tracks:
        if follower["id"] in lane_change_ids:
            continue
        preceding = np.asarray(follower["precedingId"])
        valid = np.nonzero(preceding)[0]
        ids, _, counts = np.unique(preceding, return_index=True, return_counts=True)
        # Preserve the historical string comparison and tie-breaking exactly.
        contains_string_zero = any(isinstance(value, str) and value == "0" for value in ids)
        use_longest = len(ids) > 2 if contains_string_zero else len(ids) > 1
        if use_longest:
            longest = ids[np.argmax(counts[1:]) + 1]
            valid = np.where(preceding == longest)[0]
        if len(valid) < 1250:
            continue

        follower_frames = np.asarray(follower["frame"])[valid]
        leader_ids = preceding[valid]
        leader_x = np.zeros(len(valid))
        leader_speed = np.zeros(len(valid))
        consistent = True
        for leader_id in np.unique(leader_ids):
            positions = np.where(leader_ids == leader_id)[0]
            leader = tracks[int(leader_id) - 1]
            if int(leader["id"]) != int(leader_id):
                raise ValueError("highD tracks are not in the historical ID order")
            _, _, leader_index = np.intersect1d(
                follower_frames[positions], leader["frame"], return_indices=True
            )
            if len(positions) != len(leader_index):
                raise ValueError("Follower and leader frames do not match")
            if np.unique(np.asarray(leader["followingId"])[leader_index])[0] != follower["id"]:
                consistent = False
                break
            leader_x[positions] = np.asarray(leader["bbox"])[leader_index, 0]
            leader_speed[positions] = np.asarray(leader["xVelocity"])[leader_index]
        if not consistent:
            continue

        follower_x = np.asarray(follower["bbox"])[valid, 0]
        follower_length = np.asarray(follower["bbox"])[valid, 2]
        follower_speed = np.asarray(follower["xVelocity"])[valid]
        sample = slice(None, -6, 5)
        pairs.append({
            "pair_No": len(pairs),
            "id_Foll": follower["id"],
            "class_foll": classes[follower["id"]],
            "lane_id": follower["laneId"],
            "frame_Foll": follower_frames[sample],
            "precedingId_Foll": leader_ids[sample],
            "aFollReal": np.asarray(follower["xAcceleration"])[valid][sample],
            "vFollReal": follower_speed[sample],
            "vLeaderReal": leader_speed[sample],
            "xFollReal": follower_x[sample],
            "sReal": (leader_x - follower_x - follower_length)[sample],
            "dvReal": (follower_speed - leader_speed)[sample],
        })
        if len(pairs) == PAPER_EPISODES:
            break
    return pairs


def load_pairs_from_highd(highd_root: Path) -> list[dict[str, Any]]:
    from data.read_csv import read_track_csv, read_track_meta_csv

    track_csv = highd_root / "25_tracks.csv"
    meta_csv = highd_root / "25_tracksMeta.csv"
    for path in (track_csv, meta_csv):
        if not path.is_file():
            raise FileNotFoundError(f"Missing licensed highD input: {path}")
    tracks = read_track_csv(str(track_csv))
    lane_change_ids, classes = read_track_meta_csv(str(meta_csv))
    return extract_legacy_pairs(tracks, lane_change_ids, classes)


def verify_and_write(pairs: Sequence[Mapping[str, Any]], output: Path) -> Path:
    if len(pairs) < PAPER_EPISODES:
        raise ValueError(f"Need {PAPER_EPISODES} eligible episodes, found {len(pairs)}")
    template = json.loads(MANIFEST.read_text(encoding="utf-8"))
    cohort = build_cohort_arrays(
        [LocatedTrackPair(RECORDING_ID, pair) for pair in pairs[:PAPER_EPISODES]]
    )
    expected = template["cohort"]["array_manifest"]
    mismatches = [name for name, metadata in cohort["array_manifest"].items()
                  if metadata != expected.get(name)]
    if mismatches or cohort["cohort_order_sha256"] != template["cohort"]["cohort_order_sha256"]:
        raise ValueError(
            "The reconstructed data do not match the paper's frozen cohort; "
            f"mismatched arrays: {mismatches}"
        )
    output.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(output / "cohort_data.npz", **cohort["arrays"])
    (output / "manifest.json").write_text(json.dumps(template, indent=2) + "\n", encoding="utf-8")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--highd-root", type=Path, help="Directory containing licensed recording 25 CSV files")
    source.add_argument("--trusted-pairs-cache", type=Path, help="Author's trusted legacy preprocessing pickle")
    parser.add_argument("--output-directory", type=Path, default=ROOT / "local_data" / "cohort")
    args = parser.parse_args()
    if args.highd_root is not None:
        pairs = load_pairs_from_highd(args.highd_root)
    else:
        # Pickle executes code: this option is only for a cache the user trusts.
        with args.trusted_pairs_cache.open("rb") as handle:
            pairs = pickle.load(handle)
        if not isinstance(pairs, list):
            raise TypeError("The trusted pairs cache must contain a list")
    output = verify_and_write(pairs, args.output_directory.resolve())
    print(f"Verified {PAPER_EPISODES} episodes against the paper's fingerprints: {output}")


if __name__ == "__main__":
    main()
