"""Regenerate the four frozen manuscript cases in the original figure grammar.

The layout is a structural adaptation of the submitted evaluation plot: two
stacked time-series panels, cumulative posterior-probability ribbons on the
primary axes, and observed/model curves on overlaid axes.  Only verified frozen
postprocessed arrays and the byte-identified cohort receipt are accepted.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

from .plot_e3_interpretation import CONTEXT_COLORS, RESPONSE_COLORS, RESPONSE_NAMES
from .plot_manuscript_results_intuitive import load_verified_postprocess
from .postprocess import CohortArrays, _sha256_file, _write_json_new, load_cohort


Array = NDArray[np.generic]
FloatArray = NDArray[np.float64]

CASE_FILENAMES: Mapping[int, str] = {
    2: "alpha1000_KB5_KS5_case2.pdf",
    3: "alpha1000_KB5_KS5_case3.pdf",
    6: "alpha1000_KB5_KS5_case6.pdf",
    9: "alpha1000_KB5_KS5_case9.pdf",
}
CASE_ORDINALS: Mapping[int, str] = {2: "I", 3: "II", 6: "III", 9: "IV"}

MAIN_REQUIRED_ARRAYS: Tuple[str, ...] = (
    "behavior_probability",
    "scenario_probability",
    "trajectory_offsets",
    "expected_acceleration",
    "case_2_prediction_median",
    "case_2_prediction_lower_95",
    "case_2_prediction_upper_95",
    "case_3_prediction_median",
    "case_3_prediction_lower_95",
    "case_3_prediction_upper_95",
    "case_6_prediction_median",
    "case_6_prediction_lower_95",
    "case_6_prediction_upper_95",
    "case_9_prediction_median",
    "case_9_prediction_lower_95",
    "case_9_prediction_upper_95",
)
POOLED_REQUIRED_ARRAYS: Tuple[str, ...] = (
    "trajectory_offsets",
    "expected_acceleration",
    "case_2_prediction_median",
    "case_2_prediction_lower_95",
    "case_2_prediction_upper_95",
    "case_3_prediction_median",
    "case_3_prediction_lower_95",
    "case_3_prediction_upper_95",
    "case_6_prediction_median",
    "case_6_prediction_lower_95",
    "case_6_prediction_upper_95",
    "case_9_prediction_median",
    "case_9_prediction_lower_95",
    "case_9_prediction_upper_95",
)

MAIN_COLOR = "#B95C59"
POOLED_COLOR = "#68737D"
OBSERVED_COLOR = "#15191D"
SPEED_COLOR = "#3B6FB6"
GAP_COLOR = "#4F9A8A"
RELATIVE_SPEED_COLOR = "#B95C59"


def configure_matplotlib() -> None:
    """Keep the submitted plot proportions while improving print readability."""

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "font.size": 7.0,
            "axes.labelsize": 7.5,
            "axes.linewidth": 0.75,
            "legend.fontsize": 6.3,
            "xtick.labelsize": 6.5,
            "ytick.labelsize": 6.5,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.03,
        }
    )


def _load_npz_arrays(path: Path, names: Sequence[str]) -> Dict[str, Array]:
    with np.load(path, allow_pickle=False) as loaded:
        missing = [name for name in names if name not in loaded.files]
        if missing:
            raise ValueError(f"Postprocessed array file {path.name} lacks {missing}.")
        arrays = {name: loaded[name].copy() for name in names}
    if any(array.dtype.hasobject for array in arrays.values()):
        raise TypeError(f"Postprocessed array file {path.name} contains object data.")
    return arrays


def cumulative_probabilities(probability: Array, states: int) -> FloatArray:
    """Validate posterior probabilities and add the zero cumulative baseline."""

    values = np.asarray(probability, dtype=float)
    if values.ndim != 2 or values.shape[1] != states:
        raise ValueError(f"Probability matrix has shape {values.shape}, not (*, {states}).")
    if not np.all(np.isfinite(values)):
        raise ValueError("Probability matrix contains non-finite values.")
    if np.any(values < -1.0e-7) or np.any(values > 1.0 + 1.0e-7):
        raise ValueError("Probability matrix contains values outside [0, 1].")
    if not np.allclose(values.sum(axis=1), 1.0, atol=1.0e-6, rtol=0.0):
        raise ValueError("Probability rows do not sum to one.")
    clipped = np.clip(values, 0.0, 1.0)
    cumulative = np.column_stack((np.zeros(values.shape[0]), np.cumsum(clipped, axis=1)))
    cumulative[:, -1] = 1.0
    return cumulative.T


def _validate_series(name: str, value: Array, length: int) -> FloatArray:
    series = np.asarray(value, dtype=float)
    if series.shape != (length,) or not np.all(np.isfinite(series)):
        raise ValueError(f"{name} must be a finite series with length {length}.")
    return series


def _style_probability_axis(axis: plt.Axes, *, label: str) -> None:
    axis.set_ylim(0.0, 1.0)
    axis.set_yticks((0.0, 0.5, 1.0))
    axis.set_ylabel(label)
    axis.yaxis.tick_right()
    axis.yaxis.set_label_position("right")
    axis.tick_params(axis="y", colors="#5F676E", pad=2)
    axis.spines["left"].set_visible(False)
    axis.spines["right"].set_color("#AEB5BB")
    axis.spines["top"].set_visible(False)


def _stack_probabilities(
    axis: plt.Axes,
    time_values: FloatArray,
    probability: Array,
    colors: Sequence[str],
    labels: Sequence[str],
) -> None:
    cumulative = cumulative_probabilities(probability, len(colors))
    for state, (color, label) in enumerate(zip(colors, labels)):
        axis.fill_between(
            time_values,
            cumulative[state],
            cumulative[state + 1],
            color=color,
            alpha=0.24,
            linewidth=0.0,
            label=label,
            zorder=0,
        )


def plot_case_original_style(
    case_index: int,
    *,
    cohort: CohortArrays,
    sampling_frequency_hz: float,
    main_arrays: Mapping[str, Array],
    pooled_arrays: Mapping[str, Array],
) -> plt.Figure:
    """Plot one frozen case using the submitted cumulative-stack grammar."""

    if case_index not in CASE_FILENAMES:
        raise ValueError(f"Case {case_index} is not one of the four frozen cases.")
    if not np.isfinite(sampling_frequency_hz) or sampling_frequency_hz <= 0.0:
        raise ValueError("sampling_frequency_hz must be finite and positive.")
    start, stop = (int(value) for value in cohort.offsets[case_index : case_index + 2])
    length = stop - start
    if length < 2:
        raise ValueError(f"Case {case_index} is too short to plot.")
    time_values = np.arange(length, dtype=float) / sampling_frequency_hz
    acceleration = _validate_series("Observed acceleration", cohort.acceleration[start:stop], length)
    context = np.asarray(cohort.physical_context[start:stop], dtype=float)
    if context.shape != (length, 3) or not np.all(np.isfinite(context)):
        raise ValueError("Physical context must be finite [v, s, delta-v] values.")
    main_probability = np.asarray(main_arrays["behavior_probability"])[start:stop]
    context_probability = np.asarray(main_arrays["scenario_probability"])[start:stop]
    main_median = _validate_series(
        "PS-HMM-IDM median", main_arrays[f"case_{case_index}_prediction_median"], length
    )
    main_lower = _validate_series(
        "PS-HMM-IDM lower interval", main_arrays[f"case_{case_index}_prediction_lower_95"], length
    )
    main_upper = _validate_series(
        "PS-HMM-IDM upper interval", main_arrays[f"case_{case_index}_prediction_upper_95"], length
    )
    pooled_median = _validate_series(
        "Pooled IDM median", pooled_arrays[f"case_{case_index}_prediction_median"], length
    )
    pooled_lower = _validate_series(
        "Pooled IDM lower interval", pooled_arrays[f"case_{case_index}_prediction_lower_95"], length
    )
    pooled_upper = _validate_series(
        "Pooled IDM upper interval", pooled_arrays[f"case_{case_index}_prediction_upper_95"], length
    )
    if np.any(main_lower > main_upper) or np.any(pooled_lower > pooled_upper):
        raise ValueError("A posterior interval has a lower bound above its upper bound.")

    figure, probability_axes = plt.subplots(
        2,
        1,
        figsize=(7.2, 4.25),
        sharex=True,
        gridspec_kw={"height_ratios": (1.0, 1.0), "hspace": 0.14},
    )
    response_axis = probability_axes[0].twinx()
    kinematics_axis = probability_axes[1].twinx()
    gap_axis = kinematics_axis.twinx()
    gap_axis.spines["right"].set_position(("axes", 1.105))

    response_labels = tuple(f"R{i + 1} {name}" for i, name in enumerate(RESPONSE_NAMES))
    context_labels = tuple(f"C{i + 1}" for i in range(len(CONTEXT_COLORS)))
    _stack_probabilities(
        probability_axes[0], time_values, main_probability, RESPONSE_COLORS, response_labels
    )
    _stack_probabilities(
        probability_axes[1], time_values, context_probability, CONTEXT_COLORS, context_labels
    )
    _style_probability_axis(probability_axes[0], label="Posterior probability")
    _style_probability_axis(probability_axes[1], label="Posterior probability")

    response_axis.fill_between(
        time_values, pooled_lower, pooled_upper, color=POOLED_COLOR, alpha=0.10, linewidth=0.0
    )
    pooled_line = response_axis.plot(
        time_values,
        pooled_median,
        color=POOLED_COLOR,
        linestyle=(0, (4.0, 2.2)),
        linewidth=1.25,
        label="Pooled IDM",
        zorder=4,
    )[0]
    response_axis.fill_between(
        time_values, main_lower, main_upper, color=MAIN_COLOR, alpha=0.14, linewidth=0.0
    )
    main_line = response_axis.plot(
        time_values,
        main_median,
        color=MAIN_COLOR,
        linewidth=1.55,
        label="PS-HMM-IDM",
        zorder=5,
    )[0]
    observed_line = response_axis.plot(
        time_values,
        acceleration,
        color=OBSERVED_COLOR,
        linewidth=1.35,
        label="Observed acceleration",
        zorder=6,
    )[0]
    response_axis.axhline(0.0, color="#FFFFFF", linewidth=1.0, linestyle="--", zorder=1)
    response_axis.set_ylabel(r"Acceleration (m/s$^2$)")
    response_axis.yaxis.tick_left()
    response_axis.yaxis.set_label_position("left")
    response_axis.spines["right"].set_visible(False)
    response_axis.spines["top"].set_visible(False)
    response_axis.grid(True, color="#AAB1B7", linestyle=(0, (2.0, 2.5)), linewidth=0.45, alpha=0.65)
    curve_legend = response_axis.legend(
        handles=(pooled_line, observed_line, main_line),
        loc="upper right",
        ncol=3,
        frameon=True,
        facecolor="white",
        edgecolor="#9DA5AB",
        framealpha=0.94,
        handlelength=2.2,
        columnspacing=0.9,
        borderpad=0.35,
    )
    curve_legend.set_zorder(20)
    response_legend = probability_axes[0].legend(
        loc="lower right",
        ncol=5,
        frameon=True,
        facecolor="white",
        edgecolor="#B5BCC1",
        framealpha=0.90,
        handlelength=1.2,
        handletextpad=0.35,
        columnspacing=0.7,
        borderpad=0.30,
    )
    response_legend.set_zorder(20)

    speed_line = kinematics_axis.plot(
        time_values, context[:, 0], color=SPEED_COLOR, linewidth=1.45, label=r"$v$", zorder=5
    )[0]
    relative_line = kinematics_axis.plot(
        time_values,
        context[:, 2],
        color=RELATIVE_SPEED_COLOR,
        linewidth=1.25,
        label=r"$\Delta v$",
        zorder=5,
    )[0]
    gap_line = gap_axis.plot(
        time_values, context[:, 1], color=GAP_COLOR, linewidth=1.45, label=r"$s$", zorder=5
    )[0]
    kinematics_axis.axhline(0.0, color="#FFFFFF", linewidth=1.0, linestyle="--", zorder=1)
    kinematics_axis.set_ylabel(r"$v,\Delta v$ (m/s)")
    gap_axis.set_ylabel(r"$s$ (m)", color=GAP_COLOR)
    gap_axis.tick_params(axis="y", colors=GAP_COLOR, pad=2)
    gap_axis.spines["right"].set_color(GAP_COLOR)
    kinematics_axis.yaxis.tick_left()
    kinematics_axis.yaxis.set_label_position("left")
    kinematics_axis.spines["right"].set_visible(False)
    kinematics_axis.spines["top"].set_visible(False)
    gap_axis.spines["top"].set_visible(False)
    kinematics_axis.grid(
        True, color="#AAB1B7", linestyle=(0, (2.0, 2.5)), linewidth=0.45, alpha=0.65
    )
    kinematic_legend = kinematics_axis.legend(
        handles=(speed_line, gap_line, relative_line),
        loc="upper right",
        ncol=3,
        frameon=True,
        facecolor="white",
        edgecolor="#9DA5AB",
        framealpha=0.94,
        handlelength=1.8,
        columnspacing=0.9,
        borderpad=0.35,
    )
    kinematic_legend.set_zorder(20)
    context_legend = probability_axes[1].legend(
        loc="lower right",
        ncol=5,
        frameon=True,
        facecolor="white",
        edgecolor="#B5BCC1",
        framealpha=0.90,
        handlelength=1.2,
        handletextpad=0.35,
        columnspacing=0.8,
        borderpad=0.30,
    )
    context_legend.set_zorder(20)

    probability_axes[1].set_xlabel("Time (s)")
    probability_axes[1].set_xlim(time_values[0], time_values[-1])
    probability_axes[0].tick_params(axis="x", labelbottom=False)
    figure.subplots_adjust(left=0.10, right=0.865, top=0.975, bottom=0.105)
    return figure


def _save_pdf_new(figure: plt.Figure, path: Path) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite existing figure {path}.")
    temporary = path.with_name(f".{path.name}.tmp")
    figure.savefig(
        temporary,
        format="pdf",
        metadata={
            "Creator": "plot_case_study_original_style.py",
            "Title": path.stem.replace("_", " "),
        },
    )
    plt.close(figure)
    os.replace(temporary, path)


def _source_code_sha256() -> str:
    source = Path(__file__).resolve()
    return hashlib.sha256(source.read_bytes()).hexdigest()


def execute_plotting(
    postprocess_directory: Path,
    cohort_npz: Path,
    cohort_manifest: Path,
    output_root: Path,
) -> Path:
    """Verify the pinned receipt, then export four append-only vector PDFs."""

    started = time.perf_counter()
    bundle = load_verified_postprocess(postprocess_directory)
    cohort = load_cohort(cohort_npz.resolve(), cohort_manifest.resolve())
    receipt = bundle.summary.get("cohort")
    if not isinstance(receipt, Mapping):
        raise ValueError("Pinned posterior summary lacks its cohort receipt.")
    expected = {
        "content_sha256": cohort.content_sha256,
        "npz_sha256": cohort.npz_sha256,
        "manifest_sha256": cohort.manifest_sha256,
    }
    for key, value in expected.items():
        if str(receipt.get(key, "")).lower() != value.lower():
            raise ValueError(f"Frozen cohort {key} differs from the postprocess receipt.")
    if tuple(bundle.config.fixed_case_indices) != tuple(CASE_FILENAMES):
        raise ValueError("The pinned case indices differ from the manuscript cases.")

    arm_files = bundle.summary.get("arm_files")
    if not isinstance(arm_files, Mapping):
        raise ValueError("Pinned posterior summary lacks arm filenames.")
    main_path = bundle.directory / str(arm_files[bundle.config.manuscript_arm_id])
    pooled_path = bundle.directory / str(arm_files[bundle.config.pooled_arm_id])
    main_arrays = _load_npz_arrays(main_path, MAIN_REQUIRED_ARRAYS)
    pooled_arrays = _load_npz_arrays(pooled_path, POOLED_REQUIRED_ARRAYS)
    expected_offsets = np.asarray(cohort.offsets, dtype=np.int64)
    for label, arrays in (("main", main_arrays), ("pooled", pooled_arrays)):
        offsets = np.asarray(arrays["trajectory_offsets"], dtype=np.int64)
        if not np.array_equal(offsets, expected_offsets):
            raise ValueError(f"The {label} trajectory offsets differ from the cohort.")

    output_root = output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    output = output_root / f"{timestamp}_original_style_case_figures"
    output.mkdir(parents=False, exist_ok=False)
    configure_matplotlib()
    for case_index, filename in CASE_FILENAMES.items():
        figure = plot_case_original_style(
            case_index,
            cohort=cohort,
            sampling_frequency_hz=bundle.config.sampling_frequency_hz,
            main_arrays=main_arrays,
            pooled_arrays=pooled_arrays,
        )
        _save_pdf_new(figure, output / filename)

    artifacts = {
        filename: {
            "bytes": (output / filename).stat().st_size,
            "sha256": _sha256_file(output / filename),
        }
        for filename in CASE_FILENAMES.values()
    }
    _write_json_new(output / "figure_artifact_manifest.json", {"artifacts": artifacts})
    metadata: Dict[str, Any] = {
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "runtime_seconds": time.perf_counter() - started,
        "source_script": str(Path(__file__).resolve()),
        "source_script_sha256": _source_code_sha256(),
        "adaptation_level": "structural adaptation",
        "preserved_visual_grammar": (
            "Two stacked time-series panels with cumulative posterior-probability "
            "ribbons and overlaid response/kinematic curves."
        ),
        "postprocess_directory": str(bundle.directory),
        "postprocess_manifest_sha256": bundle.manifest_sha256,
        "posterior_summary_sha256": bundle.posterior_summary_sha256,
        "cohort_npz": str(cohort_npz.resolve()),
        "cohort_npz_sha256": cohort.npz_sha256,
        "cohort_manifest": str(cohort_manifest.resolve()),
        "cohort_manifest_sha256": cohort.manifest_sha256,
        "cohort_content_sha256": cohort.content_sha256,
        "fixed_case_indices_zero_based": list(CASE_FILENAMES),
        "sampling_frequency_hz": bundle.config.sampling_frequency_hz,
        "response_labels": list(RESPONSE_NAMES),
        "context_labels": [f"C{index}" for index in range(1, 6)],
        "state_display": (
            "Cumulative stacks are timewise marginal posterior probabilities, not modal "
            "background blocks or one sampled/Viterbi path."
        ),
        "curve_display": (
            "Solid terracotta is the main posterior median and its pointwise 95% interval; "
            "dashed slate is the pooled posterior median and interval; black is observed acceleration."
        ),
        "interpretation_boundary": (
            "In-sample observed-covariate conditional reconstruction, not open-loop "
            "simulation or out-of-sample validation."
        ),
        "python": sys.version,
        "numpy": np.__version__,
        "matplotlib": matplotlib.__version__,
        "platform": platform.platform(),
        "figure_artifacts": artifacts,
    }
    _write_json_new(output / "figure_metadata.json", metadata)
    _write_json_new(
        output / "COMPLETED.json",
        {
            "status": "COMPLETED",
            "completed_at_utc": metadata["completed_at_utc"],
            "figure_artifact_manifest_sha256": _sha256_file(
                output / "figure_artifact_manifest.json"
            ),
            "figure_metadata_sha256": _sha256_file(output / "figure_metadata.json"),
        },
    )
    return output


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--postprocess-directory", type=Path, required=True)
    parser.add_argument("--cohort-npz", type=Path, required=True)
    parser.add_argument("--cohort-manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    output = execute_plotting(
        args.postprocess_directory,
        args.cohort_npz,
        args.cohort_manifest,
        args.output_root,
    )
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
