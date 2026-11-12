"""Fit the paper's model to a user's own regularly sampled episode CSV."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from experiments.manuscript_storyrev_20260808.postprocess import (
    PostprocessConfig,
    align_posterior,
    load_posterior_run,
)
from experiments.manuscript_storyrev_20260808.run_exact_reconstruction import (
    _canonical_cohort_hash,
    _sha256_array,
    _sha256_file,
    execute_run,
    preflight_run,
)
from experiments.manuscript_storyrev_20260808.sampler import ReconstructionConfig
from experiments.manuscript_storyrev_20260808.target import original_idm_acceleration
from scripts.reproduce_paper import latest_checkpoint


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs/custom_example.json"
PAPER_POSTPROCESS = ROOT / "experiments/manuscript_storyrev_20260808/configs/postprocess.json"
COLUMNS = ("episode_id", "time_s", "speed_mps", "gap_m", "relative_speed_mps", "acceleration_mps2")
PARAMETERS = ("desired_speed", "minimum_gap", "time_headway", "maximum_acceleration", "comfortable_deceleration")


def read_episodes(path: Path, sampling_hz: float) -> pd.DataFrame:
    """Validate units and episode boundaries without modifying observations."""
    if not np.isfinite(sampling_hz) or sampling_hz <= 0:
        raise ValueError("sampling-hz must be positive and finite")
    frame = pd.read_csv(path, dtype={"episode_id": "string"}, keep_default_na=False)
    missing = [name for name in COLUMNS if name not in frame.columns]
    if missing:
        raise ValueError(f"CSV is missing required columns: {missing}")
    frame = frame.loc[:, list(COLUMNS)].copy()
    if frame.empty:
        raise ValueError("CSV contains no observations")
    frame["episode_id"] = frame["episode_id"].str.strip()
    if frame["episode_id"].isna().any() or (frame["episode_id"] == "").any():
        raise ValueError("episode_id must be non-empty")
    for name in COLUMNS[1:]:
        frame[name] = pd.to_numeric(frame[name], errors="raise")
    if not np.isfinite(frame.loc[:, list(COLUMNS[1:])].to_numpy(dtype=float)).all():
        raise ValueError("time and observation columns must be finite")
    if (frame["speed_mps"] < 0).any():
        raise ValueError("speed_mps must be non-negative; orient and clean your own data explicitly")
    if (frame["gap_m"] <= 0).any():
        raise ValueError("gap_m must be a strictly positive bumper-to-bumper net gap")
    frame.sort_values(["episode_id", "time_s"], kind="stable", inplace=True, ignore_index=True)
    if frame.duplicated(["episode_id", "time_s"]).any():
        raise ValueError("Each episode_id and time_s pair must be unique")
    for episode_id, episode in frame.groupby("episode_id", sort=False):
        time = episode["time_s"].to_numpy(dtype=float)
        if time.size < 2:
            raise ValueError(f"Episode {episode_id!r} needs at least two observations")
        if not np.allclose(np.diff(time), 1.0 / sampling_hz, rtol=1e-5, atol=1e-8):
            raise ValueError(
                f"Episode {episode_id!r} is not regularly sampled at {sampling_hz:g} Hz; "
                "split gaps into separate episodes or resample before fitting"
            )
    return frame


def make_arrays(frame: pd.DataFrame) -> dict[str, np.ndarray]:
    groups = list(frame.groupby("episode_id", sort=False))
    lengths = np.array([len(group) for _, group in groups], dtype=np.int64)
    offsets = np.r_[np.array([0], dtype=np.int64), np.cumsum(lengths)]
    # The original runner requires unique numeric follower IDs. These are
    # episode-local surrogate keys, not observed driver identities.
    episode_keys = np.arange(len(groups), dtype=np.int64)
    return {
        "physical_context": frame[["speed_mps", "gap_m", "relative_speed_mps"]].to_numpy(dtype=np.float64),
        "acceleration": frame["acceleration_mps2"].to_numpy(dtype=np.float64),
        "offsets": offsets,
        "frames": np.concatenate([np.arange(n, dtype=np.int64) for n in lengths]),
        "trajectory_ids": np.asarray([str(key) for key, _ in groups]),
        "recording_ids": np.zeros(len(groups), dtype=np.int64),
        "follower_ids": episode_keys,
        "pair_nos": episode_keys.copy(),
        "lengths": lengths,
    }


def make_config(template: Path, num_episodes: int, sampling_hz: float) -> dict[str, Any]:
    raw = json.loads(template.read_text(encoding="utf-8"))
    raw["data"]["expected_num_trajectories"] = num_episodes
    raw["data"]["sampling_frequency_hz"] = sampling_hz
    raw["data"]["observed_negative_speed_policy"] = {
        "allow_observed_negative_speed": False,
        "minimum_allowed_speed_mps": 0.0,
        "raw_observations_preserved": True,
    }
    if raw["chain"].get("store_latent_paths") is not True:
        raise ValueError("Custom summaries require chain.store_latent_paths=true")
    behavior = int(raw["states"]["behavior"])
    scenario = int(raw["states"]["scenario"])
    if behavior <= 0 or scenario <= 0:
        raise ValueError("Both state counts must be positive")
    if behavior * scenario > np.iinfo(np.int16).max:
        raise ValueError("The joint state count exceeds the stored-path integer range")
    row_total = float(raw["transition_prior"]["row_total_concentration"])
    if not np.isfinite(row_total) or row_total <= 0:
        raise ValueError("Transition row_total_concentration must be positive and finite")
    # The frozen engine expresses its symmetric prior through legacy_alpha.
    raw["transition_prior"]["legacy_alpha"] = row_total / scenario
    raw["transition_prior"]["per_cell_concentration"] = row_total / (behavior * scenario)
    raw["arm_id"] = f"CUSTOM_B{behavior}_S{scenario}"
    raw["purpose"] = "CUSTOM_DATA_RECONSTRUCTION"
    ReconstructionConfig.from_dict(raw)
    return raw


def _interval(values: np.ndarray) -> dict[str, Any]:
    return {
        "mean": np.mean(values, axis=0).tolist(),
        "median": np.median(values, axis=0).tolist(),
        "lower_95": np.quantile(values, 0.025, axis=0).tolist(),
        "upper_95": np.quantile(values, 0.975, axis=0).tolist(),
    }


def write_summaries(output: Path, frame: pd.DataFrame, run_directory: Path) -> None:
    run = load_posterior_run(run_directory, require_formal=False)
    context = frame[["speed_mps", "gap_m", "relative_speed_mps"]].to_numpy(dtype=float)
    order = np.argsort(context[:, 0], kind="stable")
    anchors = context[order[np.linspace(0, len(order) - 1, min(8, len(order)), dtype=int)]]
    config = replace(PostprocessConfig.from_json(PAPER_POSTPROCESS), response_anchors=anchors)
    aligned = align_posterior(run, config)
    response_paths = aligned.paths // run.num_scenario_states
    scenario_paths = aligned.paths % run.num_scenario_states
    response_probs = np.stack(
        [np.mean(response_paths == b, axis=0) for b in range(run.num_behavior_states)], axis=1
    )
    scenario_probs = np.stack(
        [np.mean(scenario_paths == s, axis=0) for s in range(run.num_scenario_states)], axis=1
    )
    reconstructed = np.zeros(len(frame), dtype=float)
    for draw in range(run.num_draws):
        for state in range(run.num_behavior_states):
            selected = response_paths[draw] == state
            if np.any(selected):
                reconstructed[selected] += original_idm_acceleration(
                    aligned.idm_parameters[draw, state], context[selected]
                )
    reconstructed /= run.num_draws
    result_frame = frame.copy()
    result_frame["reconstructed_acceleration_mps2"] = reconstructed
    for index in range(run.num_behavior_states):
        result_frame[f"p_response_{index + 1}"] = response_probs[:, index]
    for index in range(run.num_scenario_states):
        result_frame[f"p_scenario_{index + 1}"] = scenario_probs[:, index]
    result_frame.to_csv(output / "state_probabilities.csv", index=False)
    physical_scenario_mean = aligned.scenario_mean * run.standardizer_scale + run.standardizer_mean
    physical_scenario_covariance = (
        aligned.scenario_covariance
        * run.standardizer_scale[None, None, :, None]
        * run.standardizer_scale[None, None, None, :]
    )
    np.savez_compressed(
        output / "aligned_posterior.npz",
        idm_parameters=aligned.idm_parameters,
        residual_variance=aligned.residual_variance,
        scenario_mean_standardized=aligned.scenario_mean,
        scenario_covariance_standardized=aligned.scenario_covariance,
        transition_probability=aligned.transition_probability,
        initial_probability=aligned.initial_probability,
        joint_paths=aligned.paths,
        trajectory_offsets=run.trajectory_offsets,
        behavior_old_for_new=aligned.behavior_old_for_new,
        scenario_old_for_new=aligned.scenario_old_for_new,
        response_anchors=anchors,
        standardizer_mean=run.standardizer_mean,
        standardizer_scale=run.standardizer_scale,
    )
    summary = {
        "purpose": "Retrospective smoothing and in-sample conditional reconstruction",
        "num_episodes": int(frame["episode_id"].nunique()),
        "num_observations": len(frame),
        "num_retained_draws": run.num_draws,
        "response_states": run.num_behavior_states,
        "scenario_states": run.num_scenario_states,
        "conditional_reconstruction_rmse_mps2": float(np.sqrt(np.mean(
            (frame["acceleration_mps2"].to_numpy(dtype=float) - reconstructed) ** 2
        ))),
        "parameter_order": PARAMETERS,
        "idm_parameters": _interval(aligned.idm_parameters),
        "residual_standard_deviation_mps2": _interval(np.sqrt(aligned.residual_variance)),
        "scenario_mean_physical_order_speed_gap_relative_speed": _interval(physical_scenario_mean),
        "scenario_covariance_physical_posterior_mean": np.mean(physical_scenario_covariance, axis=0).tolist(),
        "transition_matrix_posterior_mean": np.mean(aligned.transition_probability, axis=0).tolist(),
        "transition_index": "zero-based joint index = response_index * scenario_states + scenario_index",
        "response_occupancy_posterior_mean": response_probs.mean(axis=0).tolist(),
        "scenario_occupancy_posterior_mean": scenario_probs.mean(axis=0).tolist(),
        "alignment": {
            "reference": "maximum marginal log-likelihood retained draw",
            "response_anchors": "up to eight observed contexts selected at evenly spaced speed ranks",
            "reference_draw_index_zero_based": aligned.reference_draw_index,
            "caution": "Labels are arbitrary and are not automatically comparable between independent fits.",
        },
        "run_directory": str(run_directory.relative_to(output)),
    }
    (output / "posterior_summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )


def fit(csv_path: Path, template: Path, sampling_hz: float, output: Path,
        checkpoint_interval: int, *, resume: bool = False) -> Path:
    frame = read_episodes(csv_path, sampling_hz)
    arrays = make_arrays(frame)
    raw_config = make_config(template, len(arrays["lengths"]), sampling_hz)
    if checkpoint_interval <= 0:
        raise ValueError("checkpoint-interval must be positive")
    inputs = output / "input"
    cohort_path = inputs / "cohort_data.npz"
    manifest_path = inputs / "manifest.json"
    config_path = inputs / "config.json"
    manifest = {
        "source_csv_sha256": _sha256_file(csv_path),
        "cohort_content_sha256": _canonical_cohort_hash(arrays),
        "cohort": {"array_manifest": {
            name: {"dtype": str(array.dtype), "shape": list(array.shape), "sha256": _sha256_array(array)}
            for name, array in arrays.items()
        }},
        "identifier_note": "Numeric follower IDs are synthetic episode keys, not driver identities.",
    }
    if output.exists():
        if not resume:
            raise FileExistsError(f"Output already exists; use --resume for the same input and settings: {output}")
        if not all(path.is_file() for path in (cohort_path, manifest_path, config_path)):
            raise ValueError("Existing output has incomplete input artifacts; choose a new output directory")
        if json.loads(manifest_path.read_text(encoding="utf-8")) != manifest:
            raise ValueError("CSV or prepared data differ from the original custom fit")
        if json.loads(config_path.read_text(encoding="utf-8")) != raw_config:
            raise ValueError("Custom config differs from the original fit")
    else:
        output.mkdir(parents=True, exist_ok=False)
        inputs.mkdir()
        np.savez_compressed(cohort_path, **arrays)
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        config_path.write_text(json.dumps(raw_config, indent=2) + "\n", encoding="utf-8")
    report = preflight_run(config_path, cohort_path, manifest_path, checkpoint_interval=checkpoint_interval)
    completed = list((output / "runs").glob("*/COMPLETED.json"))
    if len(completed) > 1:
        raise ValueError("More than one completed custom chain exists in this output directory")
    if completed:
        run = completed[0].parent
        loaded = load_posterior_run(run, require_formal=False)
        if loaded.config != raw_config or loaded.run_metadata.get("scientific_target_fingerprint") != report["scientific_target_fingerprint"]:
            raise ValueError("Completed custom chain does not match the supplied input and config")
    else:
        checkpoint = latest_checkpoint(output / "runs", raw_config["arm_id"], report["scientific_target_fingerprint"])
        run = execute_run(config_path, cohort_path, manifest_path, output / "runs",
                          checkpoint_interval=checkpoint_interval, resume_checkpoint=checkpoint)
    summary_paths = [output / name for name in ("state_probabilities.csv", "aligned_posterior.npz", "posterior_summary.json")]
    if all(path.is_file() for path in summary_paths):
        return output
    if any(path.exists() for path in summary_paths):
        raise FileExistsError("Summary output is incomplete; move the partial summary files before resuming")
    write_summaries(output, frame, run)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, required=True, help="Your episode CSV; see docs/ADAPTATION.md")
    parser.add_argument("--sampling-hz", type=float, required=True, help="Constant sampling frequency of each episode")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="Editable custom-model JSON template")
    parser.add_argument("--output-dir", type=Path, required=True, help="New output directory; never overwritten")
    parser.add_argument("--checkpoint-interval", type=int, default=100)
    parser.add_argument("--resume", action="store_true", help="Resume or reuse the same input and configuration")
    args = parser.parse_args()
    output = fit(args.csv.resolve(), args.config.resolve(), args.sampling_hz,
                 args.output_dir.resolve(), args.checkpoint_interval, resume=args.resume)
    print(f"Custom fit complete: {output}")
    print("Summaries are retrospective, in-sample, and conditional on the supplied data.")


if __name__ == "__main__":
    main()
