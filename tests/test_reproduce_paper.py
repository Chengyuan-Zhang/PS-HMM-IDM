"""Exercise resume and output guards without running the formal chains."""

import hashlib
import json
from pathlib import Path
import sys

import pytest

from scripts.reproduce_paper import figures_complete, latest_checkpoint


def _write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")


def test_latest_checkpoint_uses_most_advanced_incomplete_run(tmp_path: Path) -> None:
    runs = tmp_path / "runs"
    for name, iteration in (("earlier", 500), ("resumed", 1000), ("completed", 1500)):
        run = runs / name
        _write_json(run / "RUNNING.json", {"arm_id": "B2", "scientific_target_fingerprint": "target"})
        _write_json(run / "checkpoints" / f"iter_{iteration:08}" / "receipt.json", {"completed_iterations": iteration})
    _write_json(runs / "completed" / "COMPLETED.json", {"arm_id": "B2"})
    assert latest_checkpoint(runs, "B2", "target").name == "iter_00001000"
    with pytest.raises(ValueError, match="different target"):
        latest_checkpoint(runs, "B2", "another-target")


def test_figures_guard_requires_all_files_and_correct_source(tmp_path: Path) -> None:
    output, postprocess = tmp_path / "figures", tmp_path / "postprocess"
    output.mkdir()
    postprocess.mkdir()
    source = postprocess / "artifact_manifest.json"
    source.write_text("{}", encoding="utf-8")
    manifest = {
        "postprocess_manifest_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "figures_include_time_space": False,
        "files_sha256": {},
    }
    _write_json(output / "reproduction_manifest.json", manifest)
    with pytest.raises(ValueError, match="Figure 9"):
        figures_complete(output, postprocess, require_time_space=True)
    with pytest.raises(ValueError, match="lacks expected outputs"):
        figures_complete(output, postprocess, require_time_space=False)


def test_workflow_runs_each_stage_once_and_reuses_completed_outputs(tmp_path: Path, monkeypatch) -> None:
    from experiments.manuscript_storyrev_20260808 import postprocess, run_exact_reconstruction
    from scripts import prepare_cohort, reproduce_paper, verify_release

    provenance = reproduce_paper.read_json(reproduce_paper.ROOT / "configs/paper_provenance.json")
    calls = {"prepare": 0, "sample": 0, "postprocess": 0, "plot": 0, "reuse": 0}

    def prepare(pairs, cohort):
        calls["prepare"] += 1
        cohort.mkdir()
        (cohort / "cohort_data.npz").write_bytes(b"cohort")
        (cohort / "manifest.json").write_text("{}", encoding="utf-8")

    def preflight(config, *args, **kwargs):
        return {"scientific_target_fingerprint": provenance["arms"][config.stem]["target_fingerprint"]}

    def run(config, cohort, manifest, runs_root, **kwargs):
        calls["sample"] += 1
        info = provenance["arms"][config.stem]
        directory = runs_root / config.stem
        directory.mkdir(parents=True)
        (directory / "config.json").write_bytes(config.read_bytes())
        _write_json(directory / "COMPLETED.json", {"arm_id": info["arm_id"]})
        _write_json(directory / "run_metadata.json", {
            "scientific_target_fingerprint": info["target_fingerprint"],
            "source_sha256": provenance["scientific_source_sha256"],
        })
        return directory

    def process(runs, cohort, manifest, config, output, **kwargs):
        calls["postprocess"] += 1
        directory = output / "completed"
        directory.mkdir(parents=True)
        (directory / "artifact_manifest.json").write_text("{}", encoding="utf-8")
        _write_json(directory / "COMPLETED.json", {})
        return directory

    def plot(command, **kwargs):
        calls["plot"] += 1
        output = Path(command[command.index("--output-directory") + 1])
        post = Path(command[command.index("--postprocess-directory") + 1])
        output.mkdir()
        names = ["figure_05_state_resolution.pdf", "figure_06_transitions.pdf", "figure_07_scenario_covariances.pdf",
                 "figure_08a_cooccurrence.pdf", "figure_08b_conditional_composition.pdf", "figure_10_case_2.pdf",
                 "figure_11_case_3.pdf", "figure_12_case_6.pdf", "figure_13_case_9.pdf",
                 "table_05_response_states.json", "table_06_scenarios.json", "table_07_reconstruction.csv"]
        for name in names:
            (output / name).write_bytes(b"test")
        _write_json(output / "reproduction_manifest.json", {
            "postprocess_manifest_sha256": hashlib.sha256((post / "artifact_manifest.json").read_bytes()).hexdigest(),
            "figures_include_time_space": False,
            "files_sha256": {name: hashlib.sha256(b"test").hexdigest() for name in names},
        })

    def reuse(*args):
        calls["reuse"] += 1

    monkeypatch.setattr(verify_release, "main", lambda: None)
    monkeypatch.setattr(prepare_cohort, "load_pairs_from_highd", lambda path: [])
    monkeypatch.setattr(prepare_cohort, "verify_and_write", prepare)
    monkeypatch.setattr(run_exact_reconstruction, "preflight_run", preflight)
    monkeypatch.setattr(run_exact_reconstruction, "execute_run", run)
    monkeypatch.setattr(postprocess, "execute_postprocess", process)
    monkeypatch.setattr(postprocess, "_verify_run_artifacts", lambda path: None)
    monkeypatch.setattr(reproduce_paper, "verify_postprocess", reuse)
    monkeypatch.setattr(reproduce_paper.subprocess, "run", plot)
    for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "MKL_DYNAMIC"):
        monkeypatch.setenv(variable, "1")
    monkeypatch.setattr(sys, "argv", ["reproduce_paper", "--highd-root", str(tmp_path / "highd"),
                                      "--output-root", str(tmp_path / "result"), "--skip-time-space"])
    reproduce_paper.main()
    reproduce_paper.main()
    assert calls == {"prepare": 1, "sample": 5, "postprocess": 1, "plot": 1, "reuse": 1}
