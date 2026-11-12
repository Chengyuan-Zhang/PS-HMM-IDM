"""The collector must not silently mix duplicated or incomplete model arms."""

import json

import pytest

from scripts.postprocess_runs import find_completed_runs


def test_selects_one_completed_run_per_arm_and_rejects_duplicates(tmp_path):
    for name, arm in (("first", "A"), ("second", "B")):
        directory = tmp_path / name
        directory.mkdir()
        (directory / "COMPLETED.json").write_text(json.dumps({"arm_id": arm}))
    assert [p.name for p in find_completed_runs(tmp_path, ("B", "A"))] == ["second", "first"]
    with pytest.raises(ValueError, match="Missing completed formal arms"):
        find_completed_runs(tmp_path, ("A", "C"))
    duplicate = tmp_path / "duplicate"
    duplicate.mkdir()
    (duplicate / "COMPLETED.json").write_text(json.dumps({"arm_id": "A"}))
    with pytest.raises(ValueError, match="More than one completed run"):
        find_completed_runs(tmp_path, ("A", "B"))
