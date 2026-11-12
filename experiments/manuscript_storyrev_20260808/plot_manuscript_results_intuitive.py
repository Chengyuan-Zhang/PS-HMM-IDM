"""Plot intuitive fixed-case summaries from the pinned formal postprocess receipt.

The posterior input is deliberately limited to verified postprocessed arrays.  The
formal MCMC chain is neither resolved nor opened by this module.  Observed
acceleration and physical context come from the byte-identical frozen cohort and
are independently checked against the cohort receipt in ``posterior_summary.json``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import patches
from numpy.typing import NDArray

from .postprocess import (
    CohortArrays,
    PostprocessConfig,
    _sha256_file,
    _write_json_new,
    load_cohort,
)


Array = NDArray[np.generic]
FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]

PINNED_POSTPROCESS_DIRECTORY_NAME = (
    "20260808T124529.501399Z_scheme_b_postprocess"
)
PINNED_POSTPROCESS_MANIFEST_SHA256 = (
    "9875307f539226c79e31fa0e3c60ca3a121052186eabce110edaf83b9456d09d"
)
PINNED_POSTERIOR_SUMMARY_SHA256 = (
    "dd48be7f0ddfec9f6e07591ccb1ddcd70a7d541ece1559c889d32173f9320f05"
)
PINNED_COMPLETED_SHA256 = (
    "78c86430694b1f0ea90aaa576b6907b067181c54e0589b1f4fbe38223501c857"
)

CASE_FILENAMES: Mapping[int, str] = {
    2: "case_I_posterior_probabilities.pdf",
    3: "case_II_posterior_probabilities.pdf",
    6: "case_III_posterior_probabilities.pdf",
    9: "case_IV_posterior_probabilities.pdf",
}
CASE_ORDINALS: Mapping[int, str] = {2: "I", 3: "II", 6: "III", 9: "IV"}

RESPONSE_LABELS: Tuple[str, ...] = (
    "Cautious",
    "Assertive A",
    "Assertive B",
    "Defensive",
    "Aggressive",
)
CONTEXT_LABELS: Tuple[str, ...] = tuple(f"Context {index}" for index in range(1, 6))
RESPONSE_COLORS: Tuple[str, ...] = (
    "#4C78A8",
    "#F2CF5B",
    "#E45756",
    "#72B7B2",
    "#B279A2",
)
CONTEXT_COLORS: Tuple[str, ...] = (
    "#66C2A5",
    "#FC8D62",
    "#8DA0CB",
    "#E78AC3",
    "#A6D854",
)

MAIN_REQUIRED_ARRAYS = (
    "behavior_probability",
    "scenario_probability",
    "trajectory_offsets",
    "expected_acceleration",
)
POOLED_REQUIRED_ARRAYS = (
    "trajectory_offsets",
    "expected_acceleration",
)


@dataclass(frozen=True)
class VerifiedPostprocessBundle:
    """Minimal verified inputs needed by the intuitive case figures."""

    directory: Path
    config: PostprocessConfig
    summary: Mapping[str, Any]
    main_arrays: Mapping[str, Array]
    pooled_arrays: Mapping[str, Array]
    manifest_sha256: str
    completed_sha256: str
    posterior_summary_sha256: str


def _configure_matplotlib() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 8.5,
            "axes.labelsize": 8.5,
            "axes.titlesize": 10.0,
            "legend.fontsize": 7.0,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.bbox": "tight",
        }
    )


def _load_json_object(path: Path) -> Mapping[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, Mapping):
        raise ValueError(f"Expected a JSON object in {path}.")
    return value


def _verify_manifest_artifacts(
    directory: Path, manifest: Mapping[str, Any]
) -> None:
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise ValueError("Postprocess artifact manifest is malformed.")
    for name, metadata in artifacts.items():
        if not isinstance(name, str) or Path(name).name != name:
            raise ValueError(f"Unsafe postprocess artifact name: {name!r}.")
        if not isinstance(metadata, Mapping):
            raise ValueError(f"Postprocess manifest entry {name!r} is malformed.")
        path = directory / name
        if not path.is_file():
            raise ValueError(f"Postprocess artifact {name!r} is missing.")
        if path.stat().st_size != int(metadata.get("bytes", -1)):
            raise ValueError(f"Postprocess artifact size mismatch for {name}.")
        expected_sha256 = str(metadata.get("sha256", "")).lower()
        if _sha256_file(path) != expected_sha256:
            raise ValueError(f"Postprocess artifact hash mismatch for {name}.")


def _load_required_npz_arrays(path: Path, names: Sequence[str]) -> Dict[str, Array]:
    with np.load(path, allow_pickle=False) as loaded:
        missing = [name for name in names if name not in loaded.files]
        if missing:
            raise ValueError(f"Postprocessed array file {path.name} lacks {missing}.")
        arrays = {name: loaded[name].copy() for name in names}
    if any(value.dtype.hasobject for value in arrays.values()):
        raise TypeError(f"Postprocessed array file {path.name} contains object data.")
    return arrays


def load_verified_postprocess(
    directory: Path,
    *,
    expected_manifest_sha256: str = PINNED_POSTPROCESS_MANIFEST_SHA256,
    expected_summary_sha256: str = PINNED_POSTERIOR_SUMMARY_SHA256,
    expected_completed_sha256: str = PINNED_COMPLETED_SHA256,
    require_pinned_directory_name: bool = True,
) -> VerifiedPostprocessBundle:
    """Fail closed unless ``directory`` is the complete pinned formal receipt."""

    directory = directory.resolve()
    if require_pinned_directory_name and directory.name != PINNED_POSTPROCESS_DIRECTORY_NAME:
        raise ValueError("Only the pinned formal postprocess directory is accepted.")
    required_files = (
        "COMPLETED.json",
        "artifact_manifest.json",
        "posterior_summary.json",
        "postprocess_metadata.json",
        "story_gate.json",
        "config.json",
    )
    missing = [name for name in required_files if not (directory / name).is_file()]
    if missing:
        raise ValueError(f"Postprocess receipt lacks required files: {missing}.")

    manifest_path = directory / "artifact_manifest.json"
    summary_path = directory / "posterior_summary.json"
    completed_path = directory / "COMPLETED.json"
    manifest_sha256 = _sha256_file(manifest_path)
    summary_sha256 = _sha256_file(summary_path)
    completed_sha256 = _sha256_file(completed_path)
    if manifest_sha256 != expected_manifest_sha256.lower():
        raise ValueError("Pinned postprocess manifest SHA-256 mismatch.")
    if summary_sha256 != expected_summary_sha256.lower():
        raise ValueError("Pinned posterior summary SHA-256 mismatch.")
    if completed_sha256 != expected_completed_sha256.lower():
        raise ValueError("Pinned COMPLETED receipt SHA-256 mismatch.")

    manifest = _load_json_object(manifest_path)
    _verify_manifest_artifacts(directory, manifest)
    completed = _load_json_object(completed_path)
    summary = _load_json_object(summary_path)
    metadata = _load_json_object(directory / "postprocess_metadata.json")
    story_gate = _load_json_object(directory / "story_gate.json")
    if completed.get("posterior_summary_sha256") != summary_sha256:
        raise ValueError("COMPLETED receipt does not identify posterior_summary.json.")
    if completed.get("story_gate_status") != "PASS" or story_gate.get("status") != "PASS":
        raise ValueError("Formal story gate is not PASS.")
    if metadata.get("require_formal") is not True:
        raise ValueError("Postprocess receipt is not marked as formal.")

    config = PostprocessConfig.from_json(directory / "config.json")
    arm_files = summary.get("arm_files")
    arms = summary.get("arms")
    if not isinstance(arm_files, Mapping) or not isinstance(arms, Mapping):
        raise ValueError("Posterior summary lacks arm mappings.")
    main_name = arm_files.get(config.manuscript_arm_id)
    pooled_name = arm_files.get(config.pooled_arm_id)
    if main_name != "arm_b5s5.npz" or pooled_name != "arm_b1s1.npz":
        raise ValueError("Posterior summary does not map the frozen main and pooled arms.")
    main_states = arms.get(config.manuscript_arm_id, {}).get("states")
    pooled_states = arms.get(config.pooled_arm_id, {}).get("states")
    if main_states != {"behavior": 5, "scenario": 5, "joint": 25}:
        raise ValueError("Main postprocessed arm does not have the frozen 5-by-5 state shape.")
    if pooled_states != {"behavior": 1, "scenario": 1, "joint": 1}:
        raise ValueError("Pooled postprocessed arm does not have the frozen state shape.")

    main_arrays = _load_required_npz_arrays(
        directory / str(main_name), MAIN_REQUIRED_ARRAYS
    )
    pooled_arrays = _load_required_npz_arrays(
        directory / str(pooled_name), POOLED_REQUIRED_ARRAYS
    )
    return VerifiedPostprocessBundle(
        directory=directory,
        config=config,
        summary=summary,
        main_arrays=main_arrays,
        pooled_arrays=pooled_arrays,
        manifest_sha256=manifest_sha256,
        completed_sha256=completed_sha256,
        posterior_summary_sha256=summary_sha256,
    )


def _validate_probability_matrix(
    probability: Array, observations: int, states: int, label: str
) -> FloatArray:
    values = np.asarray(probability, dtype=float)
    if values.shape != (observations, states):
        raise ValueError(f"{label} has shape {values.shape}, not {(observations, states)}.")
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{label} contains non-finite values.")
    if np.any(values < -1.0e-7) or np.any(values > 1.0 + 1.0e-7):
        raise ValueError(f"{label} contains values outside [0, 1].")
    if not np.allclose(values.sum(axis=1), 1.0, atol=1.0e-6, rtol=0.0):
        raise ValueError(f"{label} rows do not sum to one.")
    return np.clip(values, 0.0, 1.0)


def validate_case_inputs(
    bundle: VerifiedPostprocessBundle, cohort: CohortArrays
) -> Tuple[FloatArray, FloatArray, FloatArray, FloatArray]:
    """Validate only the posterior fields used by the intuitive case figures."""

    receipt = bundle.summary.get("cohort")
    if not isinstance(receipt, Mapping):
        raise ValueError("Posterior summary lacks its frozen cohort receipt.")
    expected_hashes = {
        "content_sha256": cohort.content_sha256,
        "npz_sha256": cohort.npz_sha256,
        "manifest_sha256": cohort.manifest_sha256,
    }
    for key, actual in expected_hashes.items():
        if str(receipt.get(key, "")).lower() != actual.lower():
            raise ValueError(f"Frozen cohort {key} differs from the postprocess receipt.")
    if int(receipt.get("num_observations", -1)) != cohort.num_observations:
        raise ValueError("Frozen cohort observation count differs from the receipt.")
    if int(receipt.get("num_trajectories", -1)) != cohort.num_trajectories:
        raise ValueError("Frozen cohort trajectory count differs from the receipt.")

    observations = cohort.num_observations
    behavior_probability = _validate_probability_matrix(
        bundle.main_arrays["behavior_probability"], observations, 5, "Response probabilities"
    )
    scenario_probability = _validate_probability_matrix(
        bundle.main_arrays["scenario_probability"], observations, 5, "Context probabilities"
    )
    main_expected = np.asarray(
        bundle.main_arrays["expected_acceleration"], dtype=float
    )
    pooled_expected = np.asarray(
        bundle.pooled_arrays["expected_acceleration"], dtype=float
    )
    if main_expected.shape != (observations,) or not np.all(np.isfinite(main_expected)):
        raise ValueError("Main posterior-mean conditional response is invalid.")
    if pooled_expected.shape != (observations,) or not np.all(np.isfinite(pooled_expected)):
        raise ValueError("Pooled posterior-mean conditional response is invalid.")

    expected_offsets = np.asarray(cohort.offsets, dtype=np.int64)
    for label, arrays in (
        ("main", bundle.main_arrays),
        ("pooled", bundle.pooled_arrays),
    ):
        offsets = np.asarray(arrays["trajectory_offsets"], dtype=np.int64)
        if not np.array_equal(offsets, expected_offsets):
            raise ValueError(f"The {label} posterior trajectory offsets differ from the cohort.")
    if cohort.physical_context.shape != (observations, 3):
        raise ValueError("Cohort physical context must use [v, s, delta-v].")
    if not np.all(np.isfinite(cohort.physical_context)) or not np.all(
        np.isfinite(cohort.acceleration)
    ):
        raise ValueError("Observed case inputs contain non-finite values.")
    for case_index in bundle.config.fixed_case_indices:
        if case_index not in CASE_FILENAMES or case_index >= cohort.num_trajectories:
            raise ValueError(f"Frozen case index {case_index} is unavailable.")
    return behavior_probability, scenario_probability, main_expected, pooled_expected


def _probability_legend_handles(
    labels: Sequence[str], colors: Sequence[str]
) -> Sequence[patches.Patch]:
    return [
        patches.Patch(facecolor=color, edgecolor="white", linewidth=0.3, label=label)
        for label, color in zip(labels, colors)
    ]


def plot_intuitive_case(
    case_index: int,
    *,
    cohort: CohortArrays,
    sampling_frequency_hz: float,
    behavior_probability: FloatArray,
    scenario_probability: FloatArray,
    main_expected_acceleration: FloatArray,
    pooled_expected_acceleration: FloatArray,
) -> plt.Figure:
    """Plot timewise marginal state probabilities and conditional mean responses."""

    if case_index not in CASE_ORDINALS:
        raise ValueError(f"Case index {case_index} is not one of the four frozen cases.")
    if not np.isfinite(sampling_frequency_hz) or sampling_frequency_hz <= 0.0:
        raise ValueError("sampling_frequency_hz must be finite and positive.")
    start, stop = (int(value) for value in cohort.offsets[case_index : case_index + 2])
    time_values = np.arange(stop - start, dtype=float) / sampling_frequency_hz
    acceleration = np.asarray(cohort.acceleration[start:stop], dtype=float)
    context = np.asarray(cohort.physical_context[start:stop], dtype=float)
    response_probability = np.asarray(behavior_probability[start:stop], dtype=float)
    context_probability = np.asarray(scenario_probability[start:stop], dtype=float)
    main_response = np.asarray(main_expected_acceleration[start:stop], dtype=float)
    pooled_response = np.asarray(pooled_expected_acceleration[start:stop], dtype=float)
    expected_length = time_values.size
    if any(
        value.shape[0] != expected_length
        for value in (
            acceleration,
            context,
            response_probability,
            context_probability,
            main_response,
            pooled_response,
        )
    ):
        raise ValueError(f"Case {case_index} arrays do not share one time axis.")

    figure = plt.figure(figsize=(7.2, 6.0))
    grid = figure.add_gridspec(
        4,
        1,
        height_ratios=(0.72, 1.62, 0.72, 1.62),
        hspace=0.10,
    )
    response_axis = figure.add_subplot(grid[0, 0])
    acceleration_axis = figure.add_subplot(grid[1, 0], sharex=response_axis)
    context_axis = figure.add_subplot(grid[2, 0], sharex=response_axis)
    physical_axis = figure.add_subplot(grid[3, 0], sharex=response_axis)

    response_axis.stackplot(
        time_values,
        *response_probability.T,
        colors=RESPONSE_COLORS,
        edgecolor="white",
        linewidth=0.25,
        alpha=0.92,
    )
    response_axis.set_ylim(0.0, 1.0)
    response_axis.set_yticks((0.0, 0.5, 1.0))
    response_axis.set_ylabel("Response\nprobability")
    response_axis.legend(
        handles=_probability_legend_handles(RESPONSE_LABELS, RESPONSE_COLORS),
        title="Post-hoc response pattern",
        loc="center",
        ncol=5,
        columnspacing=0.8,
        handlelength=1.25,
        framealpha=0.90,
        borderpad=0.35,
    )

    acceleration_axis.plot(
        time_values,
        pooled_response,
        color="0.42",
        linestyle="--",
        linewidth=1.15,
        label="Pooled IDM posterior mean",
        zorder=2,
    )
    acceleration_axis.plot(
        time_values,
        main_response,
        color="#C62828",
        linewidth=1.45,
        label="Main PS-HMM-IDM posterior mean",
        zorder=3,
    )
    acceleration_axis.plot(
        time_values,
        acceleration,
        color="black",
        linewidth=0.75,
        alpha=0.75,
        label="Observed acceleration",
        zorder=4,
    )
    acceleration_axis.set_ylabel(r"Acceleration (m/s$^2$)")
    acceleration_axis.grid(True, alpha=0.20, linewidth=0.5)
    acceleration_axis.legend(loc="upper right", ncol=1, framealpha=0.92)

    context_axis.stackplot(
        time_values,
        *context_probability.T,
        colors=CONTEXT_COLORS,
        edgecolor="white",
        linewidth=0.25,
        alpha=0.92,
    )
    context_axis.set_ylim(0.0, 1.0)
    context_axis.set_yticks((0.0, 0.5, 1.0))
    context_axis.set_ylabel("Context\nprobability")
    context_axis.legend(
        handles=_probability_legend_handles(CONTEXT_LABELS, CONTEXT_COLORS),
        title="Traffic context",
        loc="center",
        ncol=5,
        columnspacing=1.0,
        handlelength=1.25,
        framealpha=0.90,
        borderpad=0.35,
    )

    speed_line = physical_axis.plot(
        time_values,
        context[:, 0],
        color="#1F77B4",
        linewidth=1.25,
        label=r"Speed $v$",
    )[0]
    relative_speed_line = physical_axis.plot(
        time_values,
        context[:, 2],
        color="#D95F02",
        linewidth=1.05,
        label=r"Relative speed $\Delta v$",
    )[0]
    gap_axis = physical_axis.twinx()
    gap_line = gap_axis.plot(
        time_values,
        context[:, 1],
        color="#2E8B57",
        linewidth=1.20,
        label=r"Gap $s$",
    )[0]
    physical_axis.set_ylabel(r"$v,\Delta v$ (m/s)")
    gap_axis.set_ylabel(r"$s$ (m)")
    physical_axis.set_xlabel("Time (s)")
    physical_axis.grid(True, alpha=0.20, linewidth=0.5)
    physical_axis.legend(
        handles=(speed_line, relative_speed_line, gap_line),
        loc="upper right",
        ncol=3,
        framealpha=0.92,
    )

    for axis in (response_axis, acceleration_axis, context_axis):
        axis.tick_params(axis="x", which="both", labelbottom=False)
    for axis in (response_axis, acceleration_axis, context_axis, physical_axis):
        axis.margins(x=0.0)
    gap_axis.margins(x=0.0)
    figure.suptitle(
        f"Case {CASE_ORDINALS[case_index]}: observed-covariate conditional reconstruction",
        y=0.995,
    )
    figure.text(
        0.5,
        0.957,
        (
            "Stacked areas are timewise marginal posterior probabilities, "
            "not a fixed, sampled, or Viterbi state path."
        ),
        ha="center",
        va="top",
        fontsize=7.2,
        color="0.25",
    )
    figure.subplots_adjust(left=0.105, right=0.895, bottom=0.075, top=0.915)
    return figure


def _save_pdf_new(figure: plt.Figure, path: Path) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite case figure {path}.")
    temporary = path.with_name(f".{path.name}.tmp")
    if temporary.exists():
        raise FileExistsError(f"Stale temporary figure exists: {temporary}.")
    figure.savefig(
        temporary,
        format="pdf",
        metadata={
            "Creator": "plot_manuscript_results_intuitive.py",
            "Title": path.stem.replace("_", " "),
        },
    )
    plt.close(figure)
    os.replace(temporary, path)


def _new_timestamped_directory(output_root: Path) -> Path:
    output_root.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    output = output_root / f"{timestamp}_intuitive_case_figures"
    output.mkdir(parents=False, exist_ok=False)
    return output


def execute_intuitive_case_plotting(
    postprocess_directory: Path,
    cohort_npz: Path,
    cohort_manifest: Path,
    output_root: Path,
) -> Path:
    """Verify frozen inputs, then write four figures to a new append-only directory."""

    started = time.perf_counter()
    bundle = load_verified_postprocess(postprocess_directory)
    cohort = load_cohort(cohort_npz.resolve(), cohort_manifest.resolve())
    behavior_probability, scenario_probability, main_expected, pooled_expected = (
        validate_case_inputs(bundle, cohort)
    )
    output = _new_timestamped_directory(output_root.resolve())
    _configure_matplotlib()
    try:
        for case_index in bundle.config.fixed_case_indices:
            figure = plot_intuitive_case(
                case_index,
                cohort=cohort,
                sampling_frequency_hz=bundle.config.sampling_frequency_hz,
                behavior_probability=behavior_probability,
                scenario_probability=scenario_probability,
                main_expected_acceleration=main_expected,
                pooled_expected_acceleration=pooled_expected,
            )
            _save_pdf_new(figure, output / CASE_FILENAMES[case_index])

        figure_artifacts = {
            name: {
                "bytes": (output / name).stat().st_size,
                "sha256": _sha256_file(output / name),
            }
            for name in CASE_FILENAMES.values()
        }
        _write_json_new(
            output / "figure_artifact_manifest.json",
            {"artifacts": figure_artifacts},
        )
        metadata = {
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            "runtime_seconds": time.perf_counter() - started,
            "source_script": str(Path(__file__).resolve()),
            "source_script_sha256": _sha256_file(Path(__file__)),
            "formal_postprocess_directory": str(bundle.directory),
            "formal_postprocess_manifest_sha256": bundle.manifest_sha256,
            "formal_completed_sha256": bundle.completed_sha256,
            "formal_posterior_summary_sha256": bundle.posterior_summary_sha256,
            "cohort_npz": str(cohort_npz.resolve()),
            "cohort_npz_sha256": cohort.npz_sha256,
            "cohort_manifest": str(cohort_manifest.resolve()),
            "cohort_manifest_sha256": cohort.manifest_sha256,
            "cohort_content_sha256": cohort.content_sha256,
            "fixed_case_indices_zero_based": list(bundle.config.fixed_case_indices),
            "case_files": {
                str(index): name for index, name in CASE_FILENAMES.items()
            },
            "posterior_fields_used": {
                "main": list(MAIN_REQUIRED_ARRAYS),
                "pooled": list(POOLED_REQUIRED_ARRAYS),
            },
            "response_labels": list(RESPONSE_LABELS),
            "context_labels": list(CONTEXT_LABELS),
            "curve_definition": (
                "The red and gray curves are the stored posterior-mean expected_acceleration "
                "summaries for the main PS-HMM-IDM and pooled IDM, respectively."
            ),
            "state_definition": (
                "Stacked areas are timewise marginal posterior state probabilities; they are "
                "not a fixed state, one sampled path, or a Viterbi path."
            ),
            "interpretation_boundary": (
                "In-sample observed-covariate conditional reconstruction, not open-loop "
                "simulation or out-of-sample validation."
            ),
            "formal_chain_accessed": False,
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "matplotlib": matplotlib.__version__,
        }
        _write_json_new(output / "figure_metadata.json", metadata)
        _write_json_new(
            output / "COMPLETED.json",
            {
                "completed_at_utc": datetime.now(timezone.utc).isoformat(),
                "figure_artifact_manifest_sha256": _sha256_file(
                    output / "figure_artifact_manifest.json"
                ),
                "figure_metadata_sha256": _sha256_file(output / "figure_metadata.json"),
                "num_figures": len(CASE_FILENAMES),
            },
        )
    except Exception as error:
        if not (output / "FAILED.json").exists():
            _write_json_new(
                output / "FAILED.json",
                {
                    "failed_at_utc": datetime.now(timezone.utc).isoformat(),
                    "error_type": type(error).__name__,
                    "error": str(error),
                },
            )
        raise
    return output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--postprocess-directory", type=Path, required=True)
    parser.add_argument("--cohort-npz", type=Path, required=True)
    parser.add_argument("--cohort-manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    output = execute_intuitive_case_plotting(
        args.postprocess_directory,
        args.cohort_npz,
        args.cohort_manifest,
        args.output_root,
    )
    print(json.dumps({"status": "COMPLETED", "output": str(output)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
