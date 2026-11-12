"""Generate the fixed Scheme B manuscript figures from formal posterior artifacts."""

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
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import lines
from matplotlib.collections import LineCollection
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import Patch
from numpy.typing import NDArray

from .postprocess import (
    CohortArrays,
    PostprocessConfig,
    _sha256_file,
    _write_json_new,
    load_cohort,
)


FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]

FIGURE_FILENAMES = (
    "transition_mat.pdf",
    "cov_corr_mat.pdf",
    "states_stats.pdf",
    "time_space_3by2_state_speed.pdf",
    "alpha1000_KB5_KS5_case2.pdf",
    "alpha1000_KB5_KS5_case3.pdf",
    "alpha1000_KB5_KS5_case6.pdf",
    "alpha1000_KB5_KS5_case9.pdf",
)


def _configure_matplotlib() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 9.0,
            "axes.labelsize": 9.0,
            "axes.titlesize": 10.0,
            "legend.fontsize": 7.5,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.bbox": "tight",
        }
    )


def _verify_postprocessed(postprocess_directory: Path) -> None:
    for name in (
        "COMPLETED.json",
        "posterior_summary.json",
        "story_gate.json",
        "postprocess_metadata.json",
        "artifact_manifest.json",
    ):
        if not (postprocess_directory / name).is_file():
            raise ValueError(f"Postprocess artifact lacks {name}.")
    with (postprocess_directory / "artifact_manifest.json").open(
        "r", encoding="utf-8"
    ) as handle:
        manifest = json.load(handle)
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise ValueError("Postprocess artifact manifest is malformed.")
    for name, metadata in artifacts.items():
        path = postprocess_directory / name
        if not path.is_file() or not isinstance(metadata, Mapping):
            raise ValueError(f"Postprocess artifact {name!r} is missing.")
        if path.stat().st_size != int(metadata.get("bytes", -1)):
            raise ValueError(f"Postprocess artifact size mismatch for {name}.")
        if _sha256_file(path) != str(metadata.get("sha256", "")).lower():
            raise ValueError(f"Postprocess artifact hash mismatch for {name}.")


def _load_json(path: Path) -> Mapping[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, Mapping):
        raise ValueError(f"Expected a JSON object in {path}.")
    return value


def _load_npz(path: Path) -> Dict[str, NDArray[np.generic]]:
    with np.load(path, allow_pickle=False) as loaded:
        return {name: loaded[name].copy() for name in loaded.files}


def _time_space_metadata_sha256(
    trajectories: Sequence[Mapping[str, Any]],
) -> str:
    """Hash every trajectory identity, frame, position, and lane used in the plot."""

    digest = hashlib.sha256()
    for index, trajectory in enumerate(trajectories):
        identity = {
            "cohort_index_zero_based": index,
            "pair_no": int(trajectory["pair_No"]),
            "follower_id": int(trajectory["id_Foll"]),
        }
        digest.update(
            json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("utf-8")
        )
        for name in ("frame_Foll", "xFollReal", "lane_id"):
            values = np.ascontiguousarray(np.asarray(trajectory[name]))
            if values.dtype.hasobject:
                raise TypeError(f"Time-space metadata {name!r} cannot use object dtype.")
            canonical_dtype = values.dtype.newbyteorder("<")
            values = values.astype(canonical_dtype, copy=False)
            header = {
                "name": name,
                "shape": list(values.shape),
                "dtype": values.dtype.str,
            }
            digest.update(
                json.dumps(header, sort_keys=True, separators=(",", ":")).encode(
                    "utf-8"
                )
            )
            digest.update(values.tobytes(order="C"))
    return digest.hexdigest()


def _save_pdf_new(fig: plt.Figure, path: Path) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite manuscript figure {path}.")
    temporary = path.with_name(f".{path.name}.tmp")
    if temporary.exists():
        raise FileExistsError(f"Stale temporary figure exists: {temporary}.")
    fig.savefig(temporary, format="pdf")
    plt.close(fig)
    os.replace(temporary, path)


def plot_transition_matrix(
    arrays: Mapping[str, NDArray[np.generic]], behavior: int, scenario: int
) -> plt.Figure:
    transition = np.asarray(arrays["transition_mean"], dtype=float)
    joint = behavior * scenario
    if transition.shape != (joint, joint):
        raise ValueError("Transition summary has the wrong shape.")
    if not np.allclose(transition.sum(axis=1), 1.0, atol=1.0e-8):
        raise ValueError("Transition summary rows are not stochastic.")
    fig, ax = plt.subplots(figsize=(6.2, 5.2))
    image = ax.imshow(transition, cmap="magma_r", vmin=0.0, aspect="equal")
    labels = [
        f"({b + 1},{s + 1})" for b in range(behavior) for s in range(scenario)
    ]
    step = 1 if joint <= 10 else 2
    ticks = np.arange(0, joint, step)
    ax.set_xticks(ticks, [labels[index] for index in ticks], rotation=90)
    ax.set_yticks(ticks, [labels[index] for index in ticks])
    ax.set_xlabel(r"Next joint state $(z^{[\mathrm{B}]},z^{[\mathrm{S}]})$")
    ax.set_ylabel(r"Current joint state $(z^{[\mathrm{B}]},z^{[\mathrm{S}]})$")
    ax.set_title("Posterior mean full joint transition matrix")
    colorbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    colorbar.set_label("Transition probability")
    fig.tight_layout()
    return fig


def _covariance_to_correlation(covariance: FloatArray) -> FloatArray:
    standard_deviation = np.sqrt(np.diag(covariance))
    denominator = np.outer(standard_deviation, standard_deviation)
    correlation = covariance / denominator
    np.fill_diagonal(correlation, 1.0)
    return correlation


def plot_context_covariance_correlation(
    arrays: Mapping[str, NDArray[np.generic]], scenario: int
) -> plt.Figure:
    covariance = np.asarray(
        arrays["scenario_covariance_physical_mean"], dtype=float
    )
    means = np.asarray(arrays["scenario_mean_physical_median"], dtype=float)
    if covariance.shape != (scenario, 3, 3) or means.shape != (scenario, 3):
        raise ValueError("Physical scenario summary has the wrong shape.")
    fig, axes = plt.subplots(2, scenario, figsize=(2.25 * scenario, 4.5), squeeze=False)
    labels = [r"$v$ (m/s)", r"$s$ (m)", r"$\Delta v$ (m/s)"]
    covariance_scale = float(np.max(np.abs(covariance)))
    if not np.isfinite(covariance_scale) or covariance_scale <= 0.0:
        raise ValueError("Physical scenario covariances have an invalid color scale.")
    for state in range(scenario):
        cov = covariance[state]
        if np.any(np.linalg.eigvalsh(cov) <= 0.0):
            raise ValueError("A plotted scenario covariance is not positive definite.")
        cov_image = axes[0, state].imshow(
            cov,
            cmap="coolwarm",
            vmin=-covariance_scale,
            vmax=covariance_scale,
        )
        corr = _covariance_to_correlation(cov)
        corr_image = axes[1, state].imshow(
            corr, cmap="coolwarm", vmin=-1.0, vmax=1.0
        )
        for row in range(3):
            for column in range(3):
                axes[0, state].text(
                    column,
                    row,
                    f"{cov[row, column]:.2f}",
                    ha="center",
                    va="center",
                    fontsize=6.5,
                )
                axes[1, state].text(
                    column,
                    row,
                    f"{corr[row, column]:.2f}",
                    ha="center",
                    va="center",
                    fontsize=6.5,
                )
        axes[0, state].set_title(
            f"Context #{state + 1}\n"
            + r"$\mu$="
            + f"({means[state, 0]:.1f}, {means[state, 1]:.1f}, {means[state, 2]:.1f})"
        )
        for axis in axes[:, state]:
            axis.set_xticks(range(3), labels, rotation=55, ha="right")
            axis.set_yticks(range(3), labels if state == 0 else [])
        if state == 0:
            axes[0, state].set_ylabel("Physical-scale covariance")
            axes[1, state].set_ylabel("Correlation")
    # Finalize the panel geometry before Matplotlib allocates colorbar axes.
    # Re-expanding the panels after ``fig.colorbar`` places the fifth panel
    # underneath the colorbars in the five-context manuscript layout.
    fig.subplots_adjust(
        left=0.08,
        right=0.93,
        top=0.88,
        bottom=0.18,
        wspace=0.35,
        hspace=0.55,
    )
    covariance_colorbar = fig.colorbar(
        cov_image, ax=axes[0, :].tolist(), fraction=0.018, pad=0.025
    )
    covariance_colorbar.set_label("Covariance")
    correlation_colorbar = fig.colorbar(
        corr_image, ax=axes[1, :].tolist(), fraction=0.018, pad=0.025
    )
    correlation_colorbar.set_label("Correlation")
    return fig


def plot_state_cooccurrence(
    arrays: Mapping[str, NDArray[np.generic]], behavior: int, scenario: int
) -> plt.Figure:
    cooccurrence = np.asarray(arrays["cooccurrence_mean"], dtype=float)
    if cooccurrence.shape != (behavior, scenario) or not np.isclose(
        cooccurrence.sum(), 1.0, atol=1.0e-8
    ):
        raise ValueError("Posterior co-occurrence matrix is invalid.")
    percentages = 100.0 * cooccurrence
    fig, ax = plt.subplots(figsize=(5.6, 4.4))
    image = ax.imshow(percentages, cmap="YlGnBu", vmin=0.0)
    for row in range(behavior):
        for column in range(scenario):
            ax.text(
                column,
                row,
                f"{percentages[row, column]:.1f}",
                ha="center",
                va="center",
                color="black",
                fontsize=8,
            )
    ax.set_xticks(range(scenario), [f"#{index + 1}" for index in range(scenario)])
    ax.set_yticks(range(behavior), [f"#{index + 1}" for index in range(behavior)])
    ax.set_xlabel("Kinematic-context state")
    ax.set_ylabel("Response-archetype state")
    ax.set_title("Posterior-expected state co-occurrence")
    colorbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    colorbar.set_label("Share of retained latent assignments (%)")
    fig.tight_layout()
    return fig


def load_legacy_time_space_metadata(
    highd_root: Path, cohort: CohortArrays
) -> Tuple[Mapping[str, Any], ...]:
    """Reload positions/lanes and verify them against the frozen cohort arrays."""

    import data.data_loader as legacy_loader

    legacy_loader.pair_id = 0
    base_path = str(highd_root.resolve())
    if not base_path.endswith(("/", "\\")):
        base_path += os.sep
    trajectories = legacy_loader.read_training_data(
        base_path=base_path,
        dataset_list=list(range(1, 61)),
        min_traj_lenth=50 * 25,
    )[: cohort.num_trajectories]
    if len(trajectories) != cohort.num_trajectories:
        raise ValueError("Legacy loader did not reproduce 100 frozen trajectories.")
    verified = []
    for index, trajectory in enumerate(trajectories):
        start, stop = cohort.offsets[index : index + 2]
        speed = np.asarray(trajectory["vFollReal"], dtype=float)
        gap = np.asarray(trajectory["sReal"], dtype=float)
        relative_speed = np.asarray(trajectory["dvReal"], dtype=float)
        acceleration = np.asarray(trajectory["aFollReal"], dtype=float)
        frames = np.asarray(trajectory["frame_Foll"], dtype=np.int64)
        physical = np.column_stack((speed, gap, relative_speed))
        if int(trajectory["pair_No"]) != int(cohort.pair_nos[index]):
            raise ValueError(f"Pair index mismatch at frozen trajectory {index}.")
        if int(trajectory["id_Foll"]) != int(cohort.follower_ids[index]):
            raise ValueError(f"Follower ID mismatch at frozen trajectory {index}.")
        if not np.array_equal(frames, cohort.frames[start:stop]):
            raise ValueError(f"Frame mismatch at frozen trajectory {index}.")
        if not np.array_equal(physical, cohort.physical_context[start:stop]):
            raise ValueError(f"Physical-context mismatch at frozen trajectory {index}.")
        if not np.array_equal(acceleration, cohort.acceleration[start:stop]):
            raise ValueError(f"Acceleration mismatch at frozen trajectory {index}.")
        position = np.asarray(trajectory["xFollReal"], dtype=float)
        if position.shape != speed.shape:
            raise ValueError(f"Position length mismatch at frozen trajectory {index}.")
        verified.append(trajectory)
    return tuple(verified)


def _categorical_line_collection(
    x: FloatArray,
    y: FloatArray,
    state: IntArray,
    colors: Sequence[Tuple[float, float, float, float]],
) -> LineCollection:
    points = np.column_stack((x, y)).reshape(-1, 1, 2)
    segments = np.concatenate((points[:-1], points[1:]), axis=1)
    cmap = ListedColormap(colors)
    norm = BoundaryNorm(np.arange(len(colors) + 1) - 0.5, len(colors))
    collection = LineCollection(segments, cmap=cmap, norm=norm)
    collection.set_array(state[:-1])
    collection.set_linewidth(1.5)
    collection.set_alpha(0.82)
    return collection


def plot_time_space(
    arrays: Mapping[str, NDArray[np.generic]],
    cohort: CohortArrays,
    trajectories: Sequence[Mapping[str, Any]],
    config: PostprocessConfig,
    behavior: int,
    scenario: int,
) -> plt.Figure:
    behavior_modal = np.asarray(arrays["behavior_modal"], dtype=np.int64)
    scenario_modal = np.asarray(arrays["scenario_modal"], dtype=np.int64)
    if behavior_modal.shape != (cohort.num_observations,) or scenario_modal.shape != (
        cohort.num_observations,
    ):
        raise ValueError("Modal state arrays do not align with the cohort.")
    lanes = config.time_space_lane_ids
    if len(lanes) != 2:
        raise ValueError("The manuscript time-space interface requires exactly two lanes.")
    fig, axes = plt.subplots(3, 2, figsize=(9.2, 7.0), squeeze=False)
    behavior_colors = [plt.get_cmap("Paired")(index / max(behavior - 1, 1)) for index in range(behavior)]
    scenario_colors = [plt.get_cmap("Set2")(index / max(scenario - 1, 1)) for index in range(scenario)]
    speed_values = []
    plotted = 0
    for index, trajectory in enumerate(
        trajectories[: config.time_space_max_trajectories]
    ):
        lane = int(np.asarray(trajectory["lane_id"])[0])
        if lane not in lanes:
            continue
        column = lanes.index(lane)
        start, stop = cohort.offsets[index : index + 2]
        frame = np.asarray(trajectory["frame_Foll"], dtype=float)
        position = np.asarray(trajectory["xFollReal"], dtype=float)
        speed = cohort.physical_context[start:stop, 0]
        if frame.size < 2:
            continue
        axes[0, column].add_collection(
            _categorical_line_collection(
                frame, position, behavior_modal[start:stop], behavior_colors
            )
        )
        axes[1, column].add_collection(
            _categorical_line_collection(
                frame, position, scenario_modal[start:stop], scenario_colors
            )
        )
        points = np.column_stack((frame, position)).reshape(-1, 1, 2)
        segments = np.concatenate((points[:-1], points[1:]), axis=1)
        speed_collection = LineCollection(segments, cmap="turbo_r")
        speed_collection.set_array((speed[:-1] + speed[1:]) / 2.0)
        speed_collection.set_linewidth(1.5)
        axes[2, column].add_collection(speed_collection)
        speed_values.extend(speed.tolist())
        plotted += 1
    if plotted == 0 or not speed_values:
        raise ValueError("No Lane 3/4 trajectories were available for the time-space plot.")
    speed_min, speed_max = np.quantile(np.asarray(speed_values), (0.01, 0.99))
    for collection in axes[2, 0].collections + axes[2, 1].collections:
        collection.set_clim(speed_min, speed_max)
    for column, lane in enumerate(lanes):
        axes[0, column].set_title(f"Lane {lane}")
        for row in range(3):
            axes[row, column].autoscale()
            axes[row, column].grid(True, alpha=0.18)
            axes[row, column].set_ylabel("Longitudinal position (m)")
        axes[2, column].set_xlabel("Frame (25 Hz)")
    behavior_handles = [
        lines.Line2D([], [], color=behavior_colors[index], lw=2, label=f"Regime #{index + 1}")
        for index in range(behavior)
    ]
    scenario_handles = [
        lines.Line2D([], [], color=scenario_colors[index], lw=2, label=f"Context #{index + 1}")
        for index in range(scenario)
    ]
    axes[0, 1].legend(handles=behavior_handles, ncol=2, loc="best", frameon=True)
    axes[1, 1].legend(handles=scenario_handles, ncol=2, loc="best", frameon=True)
    scalar = plt.cm.ScalarMappable(cmap="turbo_r", norm=plt.Normalize(speed_min, speed_max))
    axes[0, 0].text(
        -0.19, 0.5, "Response archetype", transform=axes[0, 0].transAxes,
        rotation=90, va="center", ha="center", fontweight="bold"
    )
    axes[1, 0].text(
        -0.19, 0.5, "Kinematic context", transform=axes[1, 0].transAxes,
        rotation=90, va="center", ha="center", fontweight="bold"
    )
    axes[2, 0].text(
        -0.19, 0.5, "Observed speed", transform=axes[2, 0].transAxes,
        rotation=90, va="center", ha="center", fontweight="bold"
    )
    # Reserve panel geometry first, then let the shared colorbar shrink the
    # bottom row. Calling ``subplots_adjust`` after ``fig.colorbar`` expands the
    # right-hand speed panel back underneath the colorbar.
    fig.subplots_adjust(
        left=0.13,
        right=0.92,
        top=0.95,
        bottom=0.08,
        wspace=0.22,
        hspace=0.27,
    )
    colorbar = fig.colorbar(
        scalar, ax=axes[2, :].tolist(), fraction=0.025, pad=0.025
    )
    colorbar.set_label("Observed speed (m/s)")
    return fig


def _shade_modal_segments(
    ax: plt.Axes,
    time_values: FloatArray,
    states: IntArray,
    colors: Sequence[Tuple[float, float, float, float]],
) -> None:
    if time_values.size != states.size:
        raise ValueError("Time and modal-state arrays do not align.")
    start = 0
    half_step = 0.5 * np.median(np.diff(time_values)) if time_values.size > 1 else 0.1
    for index in range(1, states.size + 1):
        if index == states.size or states[index] != states[start]:
            left = time_values[start] - half_step
            right = time_values[index - 1] + half_step
            ax.axvspan(left, right, color=colors[int(states[start])], alpha=0.16, lw=0)
            start = index


def _fixed_case_ordinal(case_index: int, fixed_cases: Sequence[int]) -> str:
    """Map frozen zero-based trajectory indices to manuscript Case I--IV."""

    try:
        position = tuple(int(value) for value in fixed_cases).index(int(case_index))
    except ValueError as error:
        raise ValueError("Case index is not part of the frozen manuscript cases.") from error
    numerals = ("I", "II", "III", "IV")
    if position >= len(numerals):
        raise ValueError("No manuscript ordinal is defined for this fixed case.")
    return numerals[position]


def _categorical_state_handles(
    states: IntArray,
    colors: Sequence[Tuple[float, float, float, float]],
    *,
    label_prefix: str,
) -> Sequence[Patch]:
    present = np.unique(np.asarray(states, dtype=np.int64))
    if np.any(present < 0) or np.any(present >= len(colors)):
        raise ValueError("A modal state is outside the categorical color palette.")
    return [
        Patch(
            facecolor=colors[int(state)],
            edgecolor="none",
            alpha=0.32,
            label=f"{label_prefix} #{int(state) + 1}",
        )
        for state in present
    ]


def plot_case(
    case_index: int,
    manuscript_arrays: Mapping[str, NDArray[np.generic]],
    pooled_arrays: Mapping[str, NDArray[np.generic]],
    cohort: CohortArrays,
    config: PostprocessConfig,
    behavior: int,
    scenario: int,
) -> plt.Figure:
    start, stop = cohort.offsets[case_index : case_index + 2]
    acceleration = cohort.acceleration[start:stop]
    context = cohort.physical_context[start:stop]
    time_values = np.arange(stop - start, dtype=float) / config.sampling_frequency_hz
    behavior_modal = np.asarray(manuscript_arrays["behavior_modal"], dtype=np.int64)[
        start:stop
    ]
    scenario_modal = np.asarray(manuscript_arrays["scenario_modal"], dtype=np.int64)[
        start:stop
    ]
    full_median = np.asarray(
        manuscript_arrays[f"case_{case_index}_prediction_median"], dtype=float
    )
    full_lower = np.asarray(
        manuscript_arrays[f"case_{case_index}_prediction_lower_95"], dtype=float
    )
    full_upper = np.asarray(
        manuscript_arrays[f"case_{case_index}_prediction_upper_95"], dtype=float
    )
    pooled_median = np.asarray(
        pooled_arrays[f"case_{case_index}_prediction_median"], dtype=float
    )
    pooled_lower = np.asarray(
        pooled_arrays[f"case_{case_index}_prediction_lower_95"], dtype=float
    )
    pooled_upper = np.asarray(
        pooled_arrays[f"case_{case_index}_prediction_upper_95"], dtype=float
    )
    expected_shape = acceleration.shape
    for value in (
        full_median,
        full_lower,
        full_upper,
        pooled_median,
        pooled_lower,
        pooled_upper,
    ):
        if value.shape != expected_shape or not np.all(np.isfinite(value)):
            raise ValueError(f"Case {case_index} prediction summary is invalid.")
    behavior_colors = [plt.get_cmap("Paired")(index / max(behavior - 1, 1)) for index in range(behavior)]
    scenario_colors = [plt.get_cmap("Set2")(index / max(scenario - 1, 1)) for index in range(scenario)]
    fig, axes = plt.subplots(2, 1, figsize=(7.2, 5.1), sharex=True)
    _shade_modal_segments(axes[0], time_values, behavior_modal, behavior_colors)
    axes[0].fill_between(
        time_values, pooled_lower, pooled_upper, color="0.65", alpha=0.18, lw=0
    )
    axes[0].plot(
        time_values,
        pooled_median,
        color="0.35",
        linestyle="--",
        linewidth=1.3,
        label="Pooled B1S1 posterior median",
    )
    axes[0].fill_between(
        time_values, full_lower, full_upper, color="#D62728", alpha=0.18, lw=0
    )
    axes[0].plot(
        time_values,
        full_median,
        color="#D62728",
        linewidth=1.5,
        label="B5S5 conditional posterior median",
    )
    axes[0].plot(
        time_values,
        acceleration,
        color="black",
        linewidth=1.25,
        label="Observed acceleration",
    )
    axes[0].set_ylabel(r"Acceleration (m/s$^2$)")
    response_curve_legend = axes[0].legend(loc="upper left", ncol=1)
    axes[0].add_artist(response_curve_legend)
    axes[0].legend(
        handles=_categorical_state_handles(
            behavior_modal,
            behavior_colors,
            label_prefix="Regime",
        ),
        title="Modal response state",
        loc="upper right",
        ncol=min(3, int(np.unique(behavior_modal).size)),
    )
    axes[0].grid(True, alpha=0.2)
    _shade_modal_segments(axes[1], time_values, scenario_modal, scenario_colors)
    speed_line = axes[1].plot(
        time_values, context[:, 0], color="#1F77B4", lw=1.4, label=r"$v$"
    )[0]
    relative_line = axes[1].plot(
        time_values,
        context[:, 2],
        color="#D62728",
        lw=1.2,
        label=r"$\Delta v$",
    )[0]
    gap_axis = axes[1].twinx()
    gap_line = gap_axis.plot(
        time_values, context[:, 1], color="#2CA02C", lw=1.3, label=r"$s$"
    )[0]
    axes[1].set_ylabel(r"$v,\Delta v$ (m/s)")
    gap_axis.set_ylabel(r"$s$ (m)")
    axes[1].set_xlabel("Time (s)")
    axes[1].grid(True, alpha=0.2)
    context_curve_legend = axes[1].legend(
        handles=[speed_line, relative_line, gap_line], loc="upper left", ncol=3
    )
    axes[1].add_artist(context_curve_legend)
    axes[1].legend(
        handles=_categorical_state_handles(
            scenario_modal,
            scenario_colors,
            label_prefix="Context",
        ),
        title="Modal context state",
        loc="upper right",
        ncol=min(3, int(np.unique(scenario_modal).size)),
    )
    fig.suptitle(
        f"Case {_fixed_case_ordinal(case_index, config.fixed_case_indices)}: "
        "observed-covariate conditional reconstruction",
        y=0.995,
    )
    fig.tight_layout()
    return fig


def execute_plotting(
    postprocess_directory: Path,
    cohort_npz: Path,
    cohort_manifest: Path,
    highd_root: Path,
    figure_directory: Path,
    *,
    allow_gate_failure: bool = False,
) -> Path:
    """Generate all eight fixed-name vector PDFs without overwriting any figure."""

    started = time.perf_counter()
    postprocess_directory = postprocess_directory.resolve()
    _verify_postprocessed(postprocess_directory)
    metadata = _load_json(postprocess_directory / "postprocess_metadata.json")
    if not bool(metadata.get("require_formal")):
        raise ValueError("Manuscript figures require formal, not smoke, posterior runs.")
    summary = _load_json(postprocess_directory / "posterior_summary.json")
    gate = _load_json(postprocess_directory / "story_gate.json")
    if str(gate.get("status")) != "PASS" and not allow_gate_failure:
        raise ValueError(
            "Story gate did not pass; refusing manuscript figures before results rewrite."
        )
    config = PostprocessConfig.from_json(postprocess_directory / "config.json")
    cohort = load_cohort(cohort_npz, cohort_manifest)
    cohort_receipt = summary["cohort"]
    if str(cohort_receipt["content_sha256"]) != cohort.content_sha256:
        raise ValueError("Plot cohort differs from the postprocessed cohort.")
    arm_files = summary["arm_files"]
    manuscript_arrays = _load_npz(
        postprocess_directory / str(arm_files[config.manuscript_arm_id])
    )
    pooled_arrays = _load_npz(
        postprocess_directory / str(arm_files[config.pooled_arm_id])
    )
    manuscript_summary = summary["arms"][config.manuscript_arm_id]
    behavior = int(manuscript_summary["states"]["behavior"])
    scenario = int(manuscript_summary["states"]["scenario"])
    figure_directory.mkdir(parents=True, exist_ok=True)
    conflicts = [name for name in FIGURE_FILENAMES if (figure_directory / name).exists()]
    if conflicts:
        raise FileExistsError(f"Refusing to overwrite existing manuscript figures: {conflicts}.")
    _configure_matplotlib()
    _save_pdf_new(
        plot_transition_matrix(manuscript_arrays, behavior, scenario),
        figure_directory / "transition_mat.pdf",
    )
    _save_pdf_new(
        plot_context_covariance_correlation(manuscript_arrays, scenario),
        figure_directory / "cov_corr_mat.pdf",
    )
    _save_pdf_new(
        plot_state_cooccurrence(manuscript_arrays, behavior, scenario),
        figure_directory / "states_stats.pdf",
    )
    trajectories = load_legacy_time_space_metadata(highd_root, cohort)
    time_space_metadata_sha256 = _time_space_metadata_sha256(trajectories)
    _save_pdf_new(
        plot_time_space(
            manuscript_arrays,
            cohort,
            trajectories,
            config,
            behavior,
            scenario,
        ),
        figure_directory / "time_space_3by2_state_speed.pdf",
    )
    for case_index in config.fixed_case_indices:
        _save_pdf_new(
            plot_case(
                case_index,
                manuscript_arrays,
                pooled_arrays,
                cohort,
                config,
                behavior,
                scenario,
            ),
            figure_directory / f"alpha1000_KB5_KS5_case{case_index}.pdf",
        )
    figure_hashes = {
        name: {
            "sha256": _sha256_file(figure_directory / name),
            "bytes": (figure_directory / name).stat().st_size,
        }
        for name in FIGURE_FILENAMES
    }
    repository_root = Path(__file__).resolve().parents[2]
    receipt = {
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "runtime_seconds": time.perf_counter() - started,
        "postprocess_directory": str(postprocess_directory),
        "postprocess_manifest_sha256": _sha256_file(
            postprocess_directory / "artifact_manifest.json"
        ),
        "story_gate_status": gate["status"],
        "allow_gate_failure": allow_gate_failure,
        "cohort_content_sha256": cohort.content_sha256,
        "cohort_npz_sha256": _sha256_file(cohort_npz),
        "cohort_manifest_sha256": _sha256_file(cohort_manifest),
        "highd_root": str(highd_root.resolve()),
        "time_space_metadata_sha256": time_space_metadata_sha256,
        "time_space_metadata_trajectory_count": len(trajectories),
        "figure_directory": str(figure_directory.resolve()),
        "figure_artifacts": figure_hashes,
        "source_sha256": {
            "plot_manuscript_results.py": _sha256_file(Path(__file__)),
            "postprocess.py": _sha256_file(Path(__file__).with_name("postprocess.py")),
            "data/data_loader.py": _sha256_file(repository_root / "data" / "data_loader.py"),
            "data/read_csv.py": _sha256_file(repository_root / "data" / "read_csv.py"),
            "config.py": _sha256_file(repository_root / "config.py"),
        },
        "python": sys.version,
        "platform": platform.platform(),
        "matplotlib": matplotlib.__version__,
        "interpretation_boundary": (
            "Figures show in-sample posterior summaries and observed-covariate "
            "conditional reconstructions; case panels are not open-loop simulations."
        ),
    }
    _write_json_new(figure_directory / "figure_metadata.json", receipt)
    _write_json_new(
        figure_directory / "figure_artifact_manifest.json",
        {"artifacts": figure_hashes},
    )
    return figure_directory


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--postprocess-directory", type=Path, required=True)
    parser.add_argument("--cohort-npz", type=Path, required=True)
    parser.add_argument("--cohort-manifest", type=Path, required=True)
    parser.add_argument("--highd-root", type=Path, required=True)
    parser.add_argument("--figure-directory", type=Path, required=True)
    parser.add_argument(
        "--allow-gate-failure",
        action="store_true",
        help="Diagnostic use only; do not use figures in the manuscript.",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    output = execute_plotting(
        args.postprocess_directory,
        args.cohort_npz,
        args.cohort_manifest,
        args.highd_root,
        args.figure_directory,
        allow_gate_failure=args.allow_gate_failure,
    )
    print(json.dumps({"status": "COMPLETED", "figure_directory": str(output)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
