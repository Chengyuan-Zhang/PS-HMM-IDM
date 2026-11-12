"""Check the historically important selection and downsampling semantics."""

import importlib
import json
import sys
import types

import numpy as np
import pandas as pd
import pytest

from scripts.prepare_cohort import extract_legacy_pairs, verify_and_write


def _example_tracks():
    size = 1251
    frame = np.arange(size)
    box = np.column_stack((frame * 0.4, np.zeros(size), np.full(size, 4.0), np.ones(size)))
    leader = {"id": 1, "precedingId": np.zeros(size, dtype=int), "frame": frame,
              "followingId": np.full(size, 2), "bbox": box + np.array([50, 0, 0, 0]),
              "xVelocity": np.full(size, 11.0), "xAcceleration": np.zeros(size),
              "laneId": np.full(size, 3)}
    follower = {"id": 2, "precedingId": np.ones(size, dtype=int), "frame": frame,
                "followingId": np.zeros(size, dtype=int), "bbox": box,
                "xVelocity": np.full(size, 10.0), "xAcceleration": np.full(size, -0.2),
                "laneId": np.full(size, 3)}
    return [leader, follower], frame


def test_reciprocal_pair_and_historical_five_hz_slice(tmp_path):
    tracks, frame = _example_tracks()
    pair, = extract_legacy_pairs(tracks, [], ["unused", "Car", "Truck"])
    assert pair["pair_No"] == 0
    assert pair["class_foll"] == "Truck"  # Legacy class indexing, retained for matching.
    np.testing.assert_array_equal(pair["frame_Foll"], frame[:-6:5])
    np.testing.assert_array_equal(pair["sReal"], np.full(len(frame[:-6:5]), 46.0))
    np.testing.assert_array_equal(pair["dvReal"], np.full(len(frame[:-6:5]), -1.0))
    with pytest.raises(ValueError, match="100 eligible episodes"):
        verify_and_write([pair], tmp_path / "cohort")
    assert not (tmp_path / "cohort").exists()


def test_preprocessing_matches_original_pair_loader(monkeypatch):
    # The legacy loader imports an unrelated optional Kalman package eagerly.
    monkeypatch.setitem(sys.modules, "pykalman", types.SimpleNamespace(KalmanFilter=object))
    legacy = importlib.import_module("data.data_loader")
    monkeypatch.setattr(legacy, "pair_id", 0)
    tracks, _ = _example_tracks()
    classes = ["unused", "Car", "Truck"]
    expected, = legacy.load_CF_pairs(tracks, [], classes, min_traj_lenth=1250)
    actual, = extract_legacy_pairs(tracks, [], classes)
    for key in ("frame_Foll", "precedingId_Foll", "aFollReal", "vFollReal",
                "vLeaderReal", "sReal", "dvReal", "xFollReal", "lane_id"):
        np.testing.assert_array_equal(actual[key], expected[key])
    for key in ("pair_No", "id_Foll", "class_foll"):
        assert actual[key] == expected[key]


@pytest.mark.parametrize("direction", [1, -1])
def test_highd_csv_to_cohort_matches_legacy_on_synthetic_tracks(tmp_path, monkeypatch, direction):
    from data.read_csv import read_track_csv, read_track_meta_csv
    from experiments.manuscript_storyrev_20260808.forensics import LocatedTrackPair, build_cohort_arrays
    from scripts import prepare_cohort

    size = 1251
    frame = np.arange(size)
    rows = []
    for vehicle, offset, speed, preceding, following in ((1, 50.0, 11.0, 0, 2), (2, 0.0, 10.0, 1, 0)):
        physical_x = frame * 0.4 + offset
        raw_x = physical_x if direction == 1 else -physical_x - 4.0
        data = {
            "frame": frame, "id": vehicle, "x": raw_x, "y": 0.0, "width": 4.0, "height": 2.0,
            "xVelocity": direction * speed, "yVelocity": 0.0,
            "xAcceleration": direction * (-0.2 if vehicle == 2 else 0.0), "yAcceleration": 0.0,
            "precedingId": preceding, "followingId": following, "laneId": 3,
        }
        for column in ("frontSightDistance", "backSightDistance", "thw", "ttc", "dhw",
                       "precedingXVelocity", "leftFollowingId", "leftAlongsideId", "leftPrecedingId",
                       "rightFollowingId", "rightAlongsideId", "rightPrecedingId"):
            data[column] = 0
        rows.append(pd.DataFrame(data))
    pd.concat(rows, ignore_index=True).to_csv(tmp_path / "25_tracks.csv", index=False)
    pd.DataFrame({"numLaneChanges": [0, 1, 0], "class": ["Car", "Car", "Truck"]}).to_csv(
        tmp_path / "25_tracksMeta.csv", index=False
    )

    monkeypatch.setitem(sys.modules, "pykalman", types.SimpleNamespace(KalmanFilter=object))
    legacy = importlib.import_module("data.data_loader")
    monkeypatch.setattr(legacy, "pair_id", 0)
    tracks = read_track_csv(str(tmp_path / "25_tracks.csv"))
    lane_changes, classes = read_track_meta_csv(str(tmp_path / "25_tracksMeta.csv"))
    expected_pair, = legacy.load_CF_pairs(tracks, lane_changes, classes, min_traj_lenth=1250)
    expected = build_cohort_arrays([LocatedTrackPair(25, expected_pair)])
    template = tmp_path / "reference.json"
    template.write_text(json.dumps({"cohort": {"array_manifest": expected["array_manifest"],
                                               "cohort_order_sha256": expected["cohort_order_sha256"]}}),
                        encoding="utf-8")
    monkeypatch.setattr(prepare_cohort, "PAPER_EPISODES", 1)
    monkeypatch.setattr(prepare_cohort, "MANIFEST", template)
    output = tmp_path / "cohort"
    monkeypatch.setattr(sys, "argv", ["prepare_cohort", "--highd-root", str(tmp_path),
                                      "--output-directory", str(output)])
    prepare_cohort.main()
    with np.load(output / "cohort_data.npz", allow_pickle=False) as actual:
        for name, array in expected["arrays"].items():
            np.testing.assert_array_equal(actual[name], array)
