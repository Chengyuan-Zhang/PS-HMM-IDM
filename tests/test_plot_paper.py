"""Do not allow a postprocessing manifest to address files outside its directory."""

import json
import sys

import pytest

from scripts import plot_paper


def test_rejects_unsafe_artifact_path_before_loading_data(tmp_path, monkeypatch):
    postprocess = tmp_path / "postprocess"
    postprocess.mkdir()
    (postprocess / "artifact_manifest.json").write_text(json.dumps({"artifacts": {"../escape": {}}}))
    monkeypatch.setattr(sys, "argv", ["plot_paper", "--postprocess-directory", str(postprocess),
                                     "--cohort-npz", "unused.npz", "--cohort-manifest", "unused.json",
                                     "--output-directory", str(tmp_path / "output")])
    with pytest.raises(ValueError, match="Unsafe postprocess artifact filename"):
        plot_paper.main()
    assert not (tmp_path / "output").exists()
