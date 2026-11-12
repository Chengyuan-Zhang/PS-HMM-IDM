"""Prepare highD, run the five paper models, and generate the main outputs."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import subprocess
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
CONFIG_ROOT = ROOT / "experiments/manuscript_storyrev_20260808/configs"
ARM_ORDER = ("b1s1", "b2s2", "b5s1", "b1s5", "b5s5")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def completed_by_arm(runs_root: Path, provenance: dict[str, Any]) -> dict[str, Path]:
    """Reuse only completed, unmodified runs of the frozen paper targets."""
    from experiments.manuscript_storyrev_20260808.postprocess import _verify_run_artifacts

    expected = {info["arm_id"]: name for name, info in provenance["arms"].items()}
    found: dict[str, Path] = {}
    for receipt in sorted(runs_root.glob("*/COMPLETED.json")):
        run = receipt.parent
        arm = read_json(receipt).get("arm_id")
        if arm not in expected:
            raise ValueError(f"Unexpected completed arm in {runs_root}: {arm}")
        if arm in found:
            raise ValueError(f"Multiple completed runs for {arm}; choose a clean output directory")
        key = expected[arm]
        _verify_run_artifacts(run)
        if read_json(run / "config.json") != read_json(CONFIG_ROOT / f"{key}.json"):
            raise ValueError(f"Completed {arm} does not use the frozen configuration")
        metadata = read_json(run / "run_metadata.json")
        if metadata.get("scientific_target_fingerprint") != provenance["arms"][key]["target_fingerprint"]:
            raise ValueError(f"Completed {arm} has a different scientific target")
        if metadata.get("source_sha256") != provenance["scientific_source_sha256"]:
            raise ValueError(f"Completed {arm} has missing or different source hashes")
        found[arm] = run
    return found


def latest_checkpoint(runs_root: Path, arm_id: str, expected_target: str) -> Path | None:
    candidates: list[tuple[int, str, Path]] = []
    for running in sorted(runs_root.glob("*/RUNNING.json")):
        if (running.parent / "COMPLETED.json").exists():
            continue
        info = read_json(running)
        if info.get("arm_id") != arm_id:
            continue
        if info.get("scientific_target_fingerprint") != expected_target:
            raise ValueError(f"Incomplete {arm_id} run has a different target: {running.parent}")
        for receipt in running.parent.glob("checkpoints/iter_*/receipt.json"):
            checkpoint = read_json(receipt)
            candidates.append((int(checkpoint["completed_iterations"]), str(receipt.parent), receipt.parent))
    return max(candidates)[2] if candidates else None


def verify_postprocess(postprocess: Path, runs: dict[str, Path], cohort_npz: Path, manifest: Path) -> None:
    """Confirm reusable summaries came from these exact completed inputs."""
    from experiments.manuscript_storyrev_20260808.plot_manuscript_results import _verify_postprocessed

    artifacts = read_json(postprocess / "artifact_manifest.json").get("artifacts", {})
    if any(Path(name).name != name for name in artifacts):
        raise ValueError("Unsafe filename in postprocessing manifest")
    _verify_postprocessed(postprocess)
    meta = read_json(postprocess / "postprocess_metadata.json")
    if meta.get("require_formal") is not True or set(meta.get("input_runs", {})) != set(runs):
        raise ValueError("Existing postprocessing does not contain the five formal runs")
    for arm, run in runs.items():
        recorded = meta["input_runs"][arm]
        if (recorded.get("chain_sha256") != sha256_file(run / "chain.npz") or
                recorded.get("artifact_manifest_sha256") != sha256_file(run / "artifact_manifest.json")):
            raise ValueError(f"Existing postprocessing uses a different {arm} run")
    cohort = read_json(postprocess / "posterior_summary.json")["cohort"]
    if (cohort.get("npz_sha256") != sha256_file(cohort_npz) or
            cohort.get("manifest_sha256") != sha256_file(manifest)):
        raise ValueError("Existing postprocessing uses a different cohort")


def figures_complete(output: Path, postprocess: Path, *, require_time_space: bool) -> bool:
    manifest_path = output / "reproduction_manifest.json"
    if not manifest_path.exists():
        return False
    manifest = read_json(manifest_path)
    digest = sha256_file(postprocess / "artifact_manifest.json")
    if manifest.get("postprocess_manifest_sha256") != digest:
        raise ValueError("Existing figures were generated from different postprocessing")
    if require_time_space and manifest.get("figures_include_time_space") is not True:
        raise ValueError("Existing figures omit Figure 9; use a new output directory to include it")
    files = manifest.get("files_sha256")
    required = {"figure_05_state_resolution.pdf", "figure_06_transitions.pdf",
                "figure_07_scenario_covariances.pdf", "figure_08a_cooccurrence.pdf",
                "figure_08b_conditional_composition.pdf", "figure_10_case_2.pdf",
                "figure_11_case_3.pdf", "figure_12_case_6.pdf", "figure_13_case_9.pdf",
                "table_05_response_states.json", "table_06_scenarios.json",
                "table_07_reconstruction.csv"}
    if require_time_space:
        required.update(("figure_09_time_space.pdf", "figure_09_time_space_segments.csv"))
    if not isinstance(files, dict) or not required.issubset(files):
        raise ValueError("Existing figure manifest lacks expected outputs")
    for name, expected in files.items():
        if Path(name).name != name:
            raise ValueError("Unsafe filename in existing figure manifest")
        path = output / name
        if not path.is_file() or sha256_file(path) != expected:
            raise ValueError(f"Existing figure artifact is missing or changed: {name}")
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--highd-root", type=Path, help="Directory with 25_tracks.csv and 25_tracksMeta.csv")
    source.add_argument("--trusted-pairs-cache", type=Path, help="Author's trusted legacy pair cache")
    parser.add_argument("--output-root", type=Path, required=True, help="Private directory for data and outputs")
    parser.add_argument("--prepare-only", action="store_true", help="Prepare cohort and check all five targets without MCMC")
    parser.add_argument("--skip-time-space", action="store_true", help="Skip Figure 9 if position/lane metadata are unavailable")
    parser.add_argument("--checkpoint-interval", type=int, default=500)
    args = parser.parse_args()
    if args.checkpoint_interval <= 0:
        raise ValueError("checkpoint-interval must be positive")

    # Set numerical threading before importing NumPy through the model modules.
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[name] = "1"
    os.environ["MKL_DYNAMIC"] = "FALSE"
    from experiments.manuscript_storyrev_20260808.postprocess import execute_postprocess
    from experiments.manuscript_storyrev_20260808.run_exact_reconstruction import execute_run, preflight_run
    from scripts.prepare_cohort import load_pairs_from_highd, verify_and_write
    from scripts.verify_release import main as verify_release

    verify_release()
    provenance = read_json(ROOT / "configs/paper_provenance.json")
    output = args.output_root.resolve()
    output.mkdir(parents=True, exist_ok=True)
    cohort = output / "cohort"
    cohort_npz, manifest = cohort / "cohort_data.npz", cohort / "manifest.json"
    if not cohort.exists():
        if args.highd_root:
            pairs = load_pairs_from_highd(args.highd_root.resolve())
        else:
            # This option is only for a pickle explicitly trusted by the user.
            with args.trusted_pairs_cache.open("rb") as handle:
                pairs = pickle.load(handle)
            if not isinstance(pairs, list):
                raise TypeError("The trusted pairs cache must contain a list")
        verify_and_write(pairs, cohort)
        print(f"Verified and prepared the paper cohort: {cohort}", flush=True)
    elif not cohort_npz.is_file() or not manifest.is_file():
        raise ValueError(f"Existing cohort directory is incomplete: {cohort}")

    for key in ARM_ORDER:
        report = preflight_run(CONFIG_ROOT / f"{key}.json", cohort_npz, manifest,
                               checkpoint_interval=args.checkpoint_interval)
        if report["scientific_target_fingerprint"] != provenance["arms"][key]["target_fingerprint"]:
            raise ValueError(f"{key} target does not match the original paper run")
        print(f"Preflight passed: {key}", flush=True)
    if args.prepare_only:
        print("Preparation complete; no sampling was run.")
        return

    runs_root = output / "paper_runs"
    completed = completed_by_arm(runs_root, provenance)
    for key in ARM_ORDER:
        arm = provenance["arms"][key]["arm_id"]
        if arm in completed:
            print(f"Reusing completed model: {arm}", flush=True)
            continue
        checkpoint = latest_checkpoint(runs_root, arm, provenance["arms"][key]["target_fingerprint"])
        if checkpoint:
            print(f"Resuming {arm} from {checkpoint}", flush=True)
        else:
            print(f"Starting {arm}", flush=True)
        completed[arm] = execute_run(
            CONFIG_ROOT / f"{key}.json", cohort_npz, manifest, runs_root,
            checkpoint_interval=args.checkpoint_interval, resume_checkpoint=checkpoint,
        )

    post_root = output / "postprocessed"
    completed_post = sorted(post_root.glob("*/COMPLETED.json"))
    if len(completed_post) > 1:
        raise ValueError("Multiple completed postprocessing directories; select a clean output directory")
    if completed_post:
        postprocess = completed_post[0].parent
        verify_postprocess(postprocess, completed, cohort_npz, manifest)
        print(f"Reusing postprocessing: {postprocess}", flush=True)
    else:
        postprocess = execute_postprocess(
            [completed[provenance["arms"][key]["arm_id"]] for key in ARM_ORDER],
            cohort_npz, manifest, CONFIG_ROOT / "postprocess.json", post_root, require_formal=True,
        )
        print(f"Postprocessing complete: {postprocess}", flush=True)
    figures = output / "figures"
    if figures_complete(figures, postprocess, require_time_space=not args.skip_time_space):
        print(f"Verified existing figures and tables: {figures}")
        return
    if figures.exists():
        raise FileExistsError(f"Figure directory is incomplete; move it before retrying: {figures}")
    command = [sys.executable, "-m", "scripts.plot_paper", "--postprocess-directory", str(postprocess),
               "--cohort-npz", str(cohort_npz), "--cohort-manifest", str(manifest),
               "--output-directory", str(figures)]
    if not args.skip_time_space:
        if args.highd_root:
            command.extend(("--highd-root", str(args.highd_root.resolve())))
        else:
            command.extend(("--trusted-pairs-cache", str(args.trusted_pairs_cache.resolve())))
    subprocess.run(command, cwd=ROOT, check=True)
    print(f"Paper reproduction outputs: {output}")


if __name__ == "__main__":
    main()
