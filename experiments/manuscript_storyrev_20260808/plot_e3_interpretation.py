"""Generate the E3 interpretation figures from frozen formal posterior artifacts.

This module is intentionally limited to the three figures used in the
"Interpretable Driving Regimes and Traffic Scenarios" subsection.  It reads the
verified postprocessed posterior and cohort artifacts, but it never rewrites
them and it never regenerates the manuscript case-study figures.
"""

from __future__ import annotations

import argparse
import csv
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
from matplotlib import colors, lines
from matplotlib.collections import LineCollection
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from numpy.typing import NDArray

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "Helvetica", "DejaVu Sans", "Liberation Sans"]
plt.rcParams["svg.fonttype"] = "none"

from .plot_manuscript_results import (
    _load_json,
    _load_npz,
    _time_space_metadata_sha256,
    _verify_postprocessed,
    load_legacy_time_space_metadata,
)
from .postprocess import (
    CohortArrays,
    PostprocessConfig,
    _sha256_file,
    _write_json_new,
    load_cohort,
)


FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]

FIGURE_BASENAMES = (
    "fig_joint_dynamics",
    "fig_context_coupling",
    "fig_time_space_states",
)
EXPORT_FORMATS = ("svg", "pdf", "png")
EXPORT_SUFFIXES = (".svg", ".pdf", ".png")
SOURCE_DATA_FILENAMES = (
    "source_joint_transition.csv",
    "source_context_statistics.csv",
    "source_response_scenario_cooccurrence.csv",
    "source_time_space_segments.csv",
)

RESPONSE_NAMES = (
    "Cautious",
    "Assertive A",
    "Assertive B",
    "Defensive",
    "Aggressive",
)
CONTEXT_SHORT_NAMES = (
    "Very slow / tight / opening",
    "Slow / closing",
    "Moderate / short / opening",
    "Faster / near-steady",
    "Moderate / very long / opening",
)
VARIABLE_NAMES = ("Speed", "Gap", "Relative speed")
VARIABLE_SYMBOLS = ("v", "s", "Delta v")
VARIABLE_UNITS = ("m/s", "m", "m/s")

RESPONSE_COLORS = (
    "#3B6FB6",
    "#D9A441",
    "#E7C66A",
    "#4F9A8A",
    "#B95C59",
)
CONTEXT_COLORS = (
    "#5B7FA3",
    "#C58C52",
    "#5C9A86",
    "#8B6FA8",
    "#8A8175",
)
NEUTRAL_DARK = "#3F454B"
NEUTRAL_MID = "#747B82"
NEUTRAL_LIGHT = "#D8DDE2"


def configure_matplotlib() -> None:
    """Apply an editable, journal-width, sans-serif figure style."""

    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = [
        "Arial",
        "Helvetica",
        "DejaVu Sans",
        "Liberation Sans",
    ]
    plt.rcParams["svg.fonttype"] = "none"
    plt.rcParams.update(
        {
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
            "ps.fonttype": 42,
            "font.size": 8.0,
            "axes.labelsize": 8.0,
            "axes.titlesize": 8.5,
            "axes.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.labelsize": 7.0,
            "ytick.labelsize": 7.0,
            "legend.fontsize": 7.0,
            "legend.frameon": False,
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
        }
    )


def covariance_components(covariance: FloatArray) -> Tuple[FloatArray, FloatArray]:
    """Return physical standard deviations and correlations for each context."""

    covariance = np.asarray(covariance, dtype=float)
    if covariance.ndim != 3 or covariance.shape[1:] != (3, 3):
        raise ValueError("Context covariance must have shape (K, 3, 3).")
    if not np.all(np.isfinite(covariance)):
        raise ValueError("Context covariance contains non-finite values.")
    if not np.allclose(covariance, np.swapaxes(covariance, 1, 2), atol=1.0e-10):
        raise ValueError("Context covariance matrices are not symmetric.")
    eigenvalues = np.linalg.eigvalsh(covariance)
    if np.any(eigenvalues <= 0.0):
        raise ValueError("Every context covariance must be positive definite.")
    standard_deviation = np.sqrt(
        np.diagonal(covariance, axis1=1, axis2=2)
    )
    denominator = standard_deviation[:, :, None] * standard_deviation[:, None, :]
    correlation = covariance / denominator
    for state in range(correlation.shape[0]):
        np.fill_diagonal(correlation[state], 1.0)
    reconstructed = correlation * denominator
    if not np.allclose(reconstructed, covariance, atol=1.0e-10):
        raise ValueError("Standard-deviation/correlation decomposition lost covariance data.")
    return standard_deviation, correlation


def cooccurrence_percentages(cooccurrence: FloatArray) -> Tuple[FloatArray, FloatArray, FloatArray]:
    """Validate posterior co-occurrence and return percentages with marginals."""

    cooccurrence = np.asarray(cooccurrence, dtype=float)
    if cooccurrence.ndim != 2 or not np.all(np.isfinite(cooccurrence)):
        raise ValueError("Posterior co-occurrence must be a finite matrix.")
    if np.any(cooccurrence < 0.0) or not np.isclose(
        cooccurrence.sum(), 1.0, atol=1.0e-8
    ):
        raise ValueError("Posterior co-occurrence must be non-negative and sum to one.")
    percentages = 100.0 * cooccurrence
    return percentages, percentages.sum(axis=1), percentages.sum(axis=0)


def flatten_joint_index(response_index: int, context_index: int, context_count: int) -> int:
    """Return the zero-based joint index with context varying fastest."""

    if response_index < 0 or context_index < 0 or context_count <= 0:
        raise ValueError("Joint-state indices and context count must be non-negative.")
    if context_index >= context_count:
        raise ValueError("Context index exceeds the configured context count.")
    return response_index * context_count + context_index


def _add_panel_label(ax: plt.Axes, label: str, *, x: float = -0.12, y: float = 1.04) -> None:
    ax.text(
        x,
        y,
        label,
        transform=ax.transAxes,
        fontsize=9.0,
        fontweight="bold",
        ha="left",
        va="bottom",
    )


def _heatmap_text_color(cmap: colors.Colormap, norm: colors.Normalize, value: float) -> str:
    red, green, blue, _ = cmap(norm(value))
    luminance = 0.299 * red + 0.587 * green + 0.114 * blue
    return "white" if luminance < 0.48 else "black"


def _remove_heatmap_frame(ax: plt.Axes) -> None:
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(which="both", length=0)


def plot_joint_dynamics(
    arrays: Mapping[str, NDArray[np.generic]],
    response_count: int,
    context_count: int,
) -> plt.Figure:
    """Plot product-state organization and the unrestricted joint transition."""

    transition = np.asarray(arrays["transition_mean"], dtype=float)
    joint_count = response_count * context_count
    if transition.shape != (joint_count, joint_count):
        raise ValueError("Transition summary has the wrong shape.")
    if np.any(transition < 0.0) or not np.allclose(
        transition.sum(axis=1), 1.0, atol=1.0e-8
    ):
        raise ValueError("Transition rows must be non-negative and stochastic.")

    fig = plt.figure(figsize=(7.15, 3.35))
    grid = GridSpec(1, 2, figure=fig, width_ratios=(0.88, 1.65), wspace=0.34)

    ax_grid = fig.add_subplot(grid[0, 0])
    product_ids = np.arange(1, joint_count + 1).reshape(response_count, context_count)
    tile_colors = np.empty((response_count, context_count, 4), dtype=float)
    for response in range(response_count):
        base = colors.to_rgb(RESPONSE_COLORS[response])
        for context in range(context_count):
            blend = 0.08 + 0.10 * context
            tile_colors[response, context, :3] = (
                (1.0 - blend) * np.asarray(base) + blend * np.ones(3)
            )
            tile_colors[response, context, 3] = 1.0
    ax_grid.imshow(tile_colors, aspect="equal")
    for response in range(response_count):
        for context in range(context_count):
            ax_grid.text(
                context,
                response,
                str(product_ids[response, context]),
                ha="center",
                va="center",
                fontsize=7.0,
                color=NEUTRAL_DARK,
            )
    ax_grid.set_xticks(range(context_count), [f"C{i}" for i in range(1, context_count + 1)])
    ax_grid.set_yticks(range(response_count), [f"R{i}" for i in range(1, response_count + 1)])
    ax_grid.set_xlabel("Traffic-scenario coordinate")
    ax_grid.set_ylabel("Driver-action coordinate")
    ax_grid.set_title("Product-state indexing")
    ax_grid.set_xticks(np.arange(-0.5, context_count, 1), minor=True)
    ax_grid.set_yticks(np.arange(-0.5, response_count, 1), minor=True)
    ax_grid.grid(which="minor", color="white", linewidth=1.0)
    ax_grid.tick_params(which="minor", bottom=False, left=False)
    _remove_heatmap_frame(ax_grid)
    ax_grid.text(
        0.5,
        -0.30,
        "Context varies fastest within each response block",
        transform=ax_grid.transAxes,
        ha="center",
        va="top",
        color=NEUTRAL_MID,
        fontsize=7.0,
    )
    _add_panel_label(ax_grid, "a", x=-0.24, y=1.05)

    ax_transition = fig.add_subplot(grid[0, 1])
    cmap = plt.get_cmap("cividis")
    norm = colors.Normalize(vmin=0.0, vmax=float(np.max(transition)))
    image = ax_transition.imshow(
        transition, cmap=cmap, norm=norm, aspect="equal", interpolation="nearest"
    )
    block_centres = np.arange(context_count // 2, joint_count, context_count)
    major_labels = [f"R{i}" for i in range(1, response_count + 1)]
    ax_transition.set_xticks(
        np.arange(joint_count),
        [str(index % context_count + 1) for index in range(joint_count)],
    )
    ax_transition.set_yticks(block_centres, major_labels)
    ax_transition.tick_params(axis="x", labelsize=5.5, pad=1)
    response_axis = ax_transition.secondary_xaxis("top")
    response_axis.set_xticks(block_centres, major_labels)
    response_axis.tick_params(axis="x", length=0, labelsize=7.0, pad=2)
    response_axis.spines["top"].set_visible(False)
    ax_transition.set_xlabel("Next joint state: context within response block")
    ax_transition.set_ylabel("Current joint state")
    ax_transition.set_title("Posterior mean joint transition probability", pad=22)
    for boundary in np.arange(context_count - 0.5, joint_count - 0.5, context_count):
        ax_transition.axhline(boundary, color="white", linewidth=0.75, alpha=0.95)
        ax_transition.axvline(boundary, color="white", linewidth=0.75, alpha=0.95)
    for joint_index, value in enumerate(np.diag(transition)):
        if value >= 0.25:
            ax_transition.text(
                joint_index,
                joint_index,
                f"{value:.2f}",
                ha="center",
                va="center",
                fontsize=5.5,
                color=_heatmap_text_color(cmap, norm, float(value)),
            )

    _remove_heatmap_frame(ax_transition)
    colorbar = fig.colorbar(image, ax=ax_transition, fraction=0.047, pad=0.025)
    colorbar.set_label("Probability")
    colorbar.ax.tick_params(labelsize=6.5)
    _add_panel_label(ax_transition, "b", x=-0.20, y=1.16)

    fig.subplots_adjust(left=0.10, right=0.97, top=0.84, bottom=0.25)
    return fig


def plot_context_coupling(
    arrays: Mapping[str, NDArray[np.generic]],
    response_count: int,
    context_count: int,
) -> plt.Figure:
    """Combine physical context portraits and response-scenario coupling."""

    covariance = np.asarray(arrays["scenario_covariance_physical_mean"], dtype=float)
    means = np.asarray(arrays["scenario_mean_physical_median"], dtype=float)
    if means.shape != (context_count, 3):
        raise ValueError("Physical context means have the wrong shape.")
    standard_deviation, correlation = covariance_components(covariance)
    percentages, response_marginal, context_marginal = cooccurrence_percentages(
        np.asarray(arrays["cooccurrence_mean"], dtype=float)
    )
    if percentages.shape != (response_count, context_count):
        raise ValueError("Posterior co-occurrence has the wrong shape.")

    fig = plt.figure(figsize=(7.15, 6.65))
    outer = GridSpec(
        3,
        1,
        figure=fig,
        height_ratios=(1.05, 1.45, 2.35),
        hspace=0.44,
    )

    top = GridSpecFromSubplotSpec(1, 3, subplot_spec=outer[0], wspace=0.31)
    x = np.arange(context_count)
    for variable in range(3):
        ax = fig.add_subplot(top[0, variable])
        bars = ax.bar(
            x,
            standard_deviation[:, variable],
            width=0.70,
            color=CONTEXT_COLORS,
            edgecolor="white",
            linewidth=0.6,
        )
        ax.set_xticks(x, [f"C{i}" for i in range(1, context_count + 1)])
        ax.set_ylabel(f"SD ({VARIABLE_UNITS[variable]})")
        ax.set_title(VARIABLE_NAMES[variable])
        ax.set_ylim(0.0, 1.18 * float(np.max(standard_deviation[:, variable])))
        ax.tick_params(axis="x", length=0)
        for bar, value in zip(bars, standard_deviation[:, variable]):
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                value + 0.03 * float(np.max(standard_deviation[:, variable])),
                f"{value:.2f}",
                ha="center",
                va="bottom",
                fontsize=6.0,
                color=NEUTRAL_DARK,
            )
        if variable == 0:
            _add_panel_label(ax, "a", x=-0.30, y=1.08)

    middle = GridSpecFromSubplotSpec(
        1, context_count + 1, subplot_spec=outer[1], width_ratios=(1, 1, 1, 1, 1, 0.12), wspace=0.22
    )
    corr_cmap = plt.get_cmap("RdBu_r")
    corr_norm = colors.Normalize(vmin=-1.0, vmax=1.0)
    corr_axes = []
    for context in range(context_count):
        ax = fig.add_subplot(middle[0, context])
        corr_axes.append(ax)
        image_corr = ax.imshow(
            correlation[context],
            cmap=corr_cmap,
            norm=corr_norm,
            aspect="equal",
            interpolation="nearest",
        )
        for row in range(3):
            for column in range(3):
                value = float(correlation[context, row, column])
                ax.text(
                    column,
                    row,
                    f"{value:.2f}",
                    ha="center",
                    va="center",
                    fontsize=5.7,
                    color=_heatmap_text_color(corr_cmap, corr_norm, value),
                )
        ax.set_xticks(range(3), VARIABLE_SYMBOLS)
        ax.set_yticks(range(3), VARIABLE_SYMBOLS if context == 0 else [])
        ax.set_title(
            f"C{context + 1}: ({means[context, 0]:.1f}, {means[context, 1]:.1f}, {means[context, 2]:.1f})",
            fontsize=6.8,
        )
        _remove_heatmap_frame(ax)
        if context == 0:
            _add_panel_label(ax, "b", x=-0.43, y=1.11)
    cax_corr = fig.add_subplot(middle[0, -1])
    colorbar_corr = fig.colorbar(image_corr, cax=cax_corr)
    colorbar_corr.set_label("Correlation")
    colorbar_corr.set_ticks((-1.0, 0.0, 1.0))
    colorbar_corr.ax.tick_params(labelsize=6.0)

    bottom = GridSpecFromSubplotSpec(
        2,
        3,
        subplot_spec=outer[2],
        height_ratios=(0.28, 1.0),
        width_ratios=(1.0, 0.24, 0.08),
        hspace=0.05,
        wspace=0.08,
    )
    ax_context_marginal = fig.add_subplot(bottom[0, 0])
    ax_cooccurrence = fig.add_subplot(bottom[1, 0])
    ax_response_marginal = fig.add_subplot(bottom[1, 1], sharey=ax_cooccurrence)
    cax_cooccurrence = fig.add_subplot(bottom[1, 2])

    co_cmap = plt.get_cmap("cividis")
    co_norm = colors.Normalize(vmin=0.0, vmax=float(np.max(percentages)))
    image_co = ax_cooccurrence.imshow(
        percentages,
        cmap=co_cmap,
        norm=co_norm,
        aspect="auto",
        interpolation="nearest",
    )
    for row in range(response_count):
        for column in range(context_count):
            value = float(percentages[row, column])
            ax_cooccurrence.text(
                column,
                row,
                f"{value:.1f}",
                ha="center",
                va="center",
                fontsize=7.0,
                color=_heatmap_text_color(co_cmap, co_norm, value),
            )
    ax_cooccurrence.set_xticks(
        range(context_count), [f"C{i}" for i in range(1, context_count + 1)]
    )
    ax_cooccurrence.set_yticks(range(response_count), RESPONSE_NAMES)
    ax_cooccurrence.set_xlabel("External traffic scenario")
    ax_cooccurrence.set_ylabel("Intrinsic driver action")
    _remove_heatmap_frame(ax_cooccurrence)
    _add_panel_label(ax_cooccurrence, "c", x=-0.24, y=1.12)

    ax_context_marginal.bar(
        x,
        context_marginal,
        width=0.82,
        color=CONTEXT_COLORS,
        edgecolor="white",
        linewidth=0.6,
    )
    ax_context_marginal.set_xlim(-0.5, context_count - 0.5)
    ax_context_marginal.set_ylim(0.0, 1.18 * float(np.max(context_marginal)))
    ax_context_marginal.set_ylabel("Context share (%)", fontsize=6.5)
    ax_context_marginal.set_xticks([])
    ax_context_marginal.tick_params(axis="y", labelsize=6.0)
    for index, value in enumerate(context_marginal):
        ax_context_marginal.text(
            index,
            value + 0.6,
            f"{value:.1f}",
            ha="center",
            va="bottom",
            fontsize=6.0,
        )

    ax_response_marginal.barh(
        np.arange(response_count),
        response_marginal,
        height=0.82,
        color=RESPONSE_COLORS,
        edgecolor="white",
        linewidth=0.6,
    )
    ax_response_marginal.set_xlim(0.0, 1.15 * float(np.max(response_marginal)))
    ax_response_marginal.set_xlabel("Response\nshare (%)", fontsize=6.5)
    ax_response_marginal.tick_params(axis="y", left=False, labelleft=False)
    ax_response_marginal.tick_params(axis="x", labelsize=6.0)
    for index, value in enumerate(response_marginal):
        ax_response_marginal.text(
            value + 0.8,
            index,
            f"{value:.1f}",
            ha="left",
            va="center",
            fontsize=6.0,
        )
    colorbar_co = fig.colorbar(image_co, cax=cax_cooccurrence)
    colorbar_co.set_label("Joint share (%)")
    colorbar_co.ax.tick_params(labelsize=6.0)

    fig.subplots_adjust(left=0.15, right=0.96, top=0.93, bottom=0.08)
    return fig


def _categorical_line_collection(
    x: FloatArray,
    y: FloatArray,
    state: IntArray,
    palette: Sequence[str],
) -> LineCollection:
    if x.size != y.size or x.size != state.size:
        raise ValueError("Time-space coordinates and states must have equal lengths.")
    points = np.column_stack((x, y)).reshape(-1, 1, 2)
    segments = np.concatenate((points[:-1], points[1:]), axis=1)
    cmap = ListedColormap(palette)
    norm = BoundaryNorm(np.arange(len(palette) + 1) - 0.5, len(palette))
    collection = LineCollection(segments, cmap=cmap, norm=norm)
    collection.set_array(state[:-1])
    collection.set_linewidth(1.25)
    collection.set_alpha(0.94)
    return collection


def plot_time_space_states(
    arrays: Mapping[str, NDArray[np.generic]],
    cohort: CohortArrays,
    trajectories: Sequence[Mapping[str, Any]],
    config: PostprocessConfig,
    response_count: int,
    context_count: int,
) -> Tuple[plt.Figure, Tuple[Mapping[str, Any], ...]]:
    """Plot response, context, and speed on aligned Lane 3/4 trajectories."""

    response_modal = np.asarray(arrays["behavior_modal"], dtype=np.int64)
    context_modal = np.asarray(arrays["scenario_modal"], dtype=np.int64)
    if response_modal.shape != (cohort.num_observations,) or context_modal.shape != (
        cohort.num_observations,
    ):
        raise ValueError("Modal state arrays do not align with the frozen cohort.")
    if response_modal.min() < 0 or response_modal.max() >= response_count:
        raise ValueError("Response modal states are out of bounds.")
    if context_modal.min() < 0 or context_modal.max() >= context_count:
        raise ValueError("Context modal states are out of bounds.")
    lanes = tuple(int(value) for value in config.time_space_lane_ids)
    if len(lanes) != 2:
        raise ValueError("The E3 time-space figure requires exactly two lanes.")

    fig, axes = plt.subplots(3, 2, figsize=(7.15, 5.25), squeeze=False, sharex="col")
    speed_values = []
    plotted_records = []
    plotted = 0
    for trajectory_index, trajectory in enumerate(
        trajectories[: config.time_space_max_trajectories]
    ):
        lane = int(np.asarray(trajectory["lane_id"])[0])
        if lane not in lanes:
            continue
        column = lanes.index(lane)
        start, stop = cohort.offsets[trajectory_index : trajectory_index + 2]
        frame = np.asarray(trajectory["frame_Foll"], dtype=float)
        position = np.asarray(trajectory["xFollReal"], dtype=float)
        speed = cohort.physical_context[start:stop, 0]
        if frame.size < 2:
            continue
        axes[0, column].add_collection(
            _categorical_line_collection(
                frame, position, response_modal[start:stop], RESPONSE_COLORS
            )
        )
        axes[1, column].add_collection(
            _categorical_line_collection(
                frame, position, context_modal[start:stop], CONTEXT_COLORS
            )
        )
        points = np.column_stack((frame, position)).reshape(-1, 1, 2)
        segments = np.concatenate((points[:-1], points[1:]), axis=1)
        speed_collection = LineCollection(segments, cmap="cividis")
        speed_collection.set_array((speed[:-1] + speed[1:]) / 2.0)
        speed_collection.set_linewidth(1.25)
        axes[2, column].add_collection(speed_collection)
        speed_values.extend(speed.tolist())
        for local_index in range(frame.size - 1):
            plotted_records.append(
                {
                    "trajectory_index_zero_based": int(trajectory_index),
                    "pair_no": int(trajectory["pair_No"]),
                    "follower_id": int(trajectory["id_Foll"]),
                    "lane": lane,
                    "frame_start": int(frame[local_index]),
                    "frame_end": int(frame[local_index + 1]),
                    "position_start_m": float(position[local_index]),
                    "position_end_m": float(position[local_index + 1]),
                    "response_state": int(response_modal[start + local_index]) + 1,
                    "context_state": int(context_modal[start + local_index]) + 1,
                    "speed_mps": float(
                        0.5 * (speed[local_index] + speed[local_index + 1])
                    ),
                }
            )
        plotted += 1
    if plotted == 0 or not speed_values:
        raise ValueError("No Lane 3/4 trajectories were available for the time-space plot.")

    speed_min, speed_max = np.quantile(np.asarray(speed_values), (0.01, 0.99))
    for collection in axes[2, 0].collections + axes[2, 1].collections:
        collection.set_clim(float(speed_min), float(speed_max))
    row_labels = (
        "Intrinsic driver action",
        "External traffic scenario",
        "Observed speed",
    )
    for column, lane in enumerate(lanes):
        axes[0, column].set_title(f"Lane {lane}")
        for row in range(3):
            axes[row, column].autoscale()
            axes[row, column].grid(False)
            axes[row, column].set_ylabel("Position (m)")
            axes[row, column].tick_params(direction="out", length=2.5, width=0.7)
        axes[2, column].set_xlabel("Original highD frame (25 Hz)")
    for row, label in enumerate(row_labels):
        axes[row, 0].text(
            -0.25,
            0.5,
            label,
            transform=axes[row, 0].transAxes,
            rotation=90,
            rotation_mode="anchor",
            ha="center",
            va="center",
            fontweight="bold",
            fontsize=7.5,
        )

    response_handles = [
        lines.Line2D([], [], color=RESPONSE_COLORS[index], lw=2.0, label=name)
        for index, name in enumerate(RESPONSE_NAMES)
    ]
    context_handles = [
        lines.Line2D(
            [], [], color=CONTEXT_COLORS[index], lw=2.0, label=f"C{index + 1}"
        )
        for index in range(context_count)
    ]
    axes[0, 1].legend(
        handles=response_handles,
        ncol=3,
        loc="upper right",
        columnspacing=0.8,
        handlelength=1.5,
        borderaxespad=0.3,
    )
    axes[1, 1].legend(
        handles=context_handles,
        ncol=5,
        loc="upper right",
        columnspacing=0.65,
        handlelength=1.4,
        borderaxespad=0.3,
    )
    scalar = plt.cm.ScalarMappable(
        cmap="cividis", norm=plt.Normalize(float(speed_min), float(speed_max))
    )
    colorbar = fig.colorbar(
        scalar, ax=axes[2, :].tolist(), fraction=0.025, pad=0.018
    )
    colorbar.set_label("Speed (m/s)")
    colorbar.ax.tick_params(labelsize=6.5)
    _add_panel_label(axes[0, 0], "a", x=-0.29, y=1.07)
    _add_panel_label(axes[1, 0], "b", x=-0.29, y=1.07)
    _add_panel_label(axes[2, 0], "c", x=-0.29, y=1.07)
    fig.subplots_adjust(
        left=0.16,
        right=0.91,
        top=0.95,
        bottom=0.09,
        wspace=0.18,
        hspace=0.22,
    )
    return fig, tuple(plotted_records)


def _save_figure_bundle_new(fig: plt.Figure, output_base: Path) -> Dict[str, Mapping[str, Any]]:
    """Save editable vector, PDF, and raster preview without overwriting."""

    outputs = {extension: output_base.with_suffix(f".{extension}") for extension in EXPORT_FORMATS}
    conflicts = [str(path) for path in outputs.values() if path.exists()]
    if conflicts:
        raise FileExistsError(f"Refusing to overwrite E3 figure outputs: {conflicts}")
    temporaries = {
        extension: path.with_name(f".{path.stem}.tmp{path.suffix}")
        for extension, path in outputs.items()
    }
    stale = [str(path) for path in temporaries.values() if path.exists()]
    if stale:
        raise FileExistsError(f"Stale E3 temporary outputs exist: {stale}")
    try:
        fig.savefig(temporaries["svg"], format="svg")
        fig.savefig(temporaries["pdf"], format="pdf")
        fig.savefig(temporaries["png"], format="png", dpi=300)
        for extension in EXPORT_FORMATS:
            os.replace(temporaries[extension], outputs[extension])
    finally:
        plt.close(fig)
    return {
        path.name: {"sha256": _sha256_file(path), "bytes": path.stat().st_size}
        for path in outputs.values()
    }


def _write_csv_new(path: Path, fieldnames: Sequence[str], rows: Sequence[Mapping[str, Any]]) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite E3 source data {path}.")
    temporary = path.with_name(f".{path.name}.tmp")
    if temporary.exists():
        raise FileExistsError(f"Stale E3 source-data temporary exists: {temporary}.")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def _write_source_data(
    output_directory: Path,
    arrays: Mapping[str, NDArray[np.generic]],
    time_space_records: Sequence[Mapping[str, Any]],
    response_count: int,
    context_count: int,
) -> Dict[str, Mapping[str, Any]]:
    transition = np.asarray(arrays["transition_mean"], dtype=float)
    joint_rows = []
    for current in range(transition.shape[0]):
        for nxt in range(transition.shape[1]):
            joint_rows.append(
                {
                    "current_joint_state": current + 1,
                    "current_response_state": current // context_count + 1,
                    "current_context_state": current % context_count + 1,
                    "next_joint_state": nxt + 1,
                    "next_response_state": nxt // context_count + 1,
                    "next_context_state": nxt % context_count + 1,
                    "posterior_mean_transition_probability": float(transition[current, nxt]),
                }
            )
    _write_csv_new(
        output_directory / "source_joint_transition.csv",
        tuple(joint_rows[0].keys()),
        joint_rows,
    )

    covariance = np.asarray(arrays["scenario_covariance_physical_mean"], dtype=float)
    means = np.asarray(arrays["scenario_mean_physical_median"], dtype=float)
    standard_deviation, correlation = covariance_components(covariance)
    context_rows = []
    for context in range(context_count):
        row: Dict[str, Any] = {
            "context_state": context + 1,
            "context_name": CONTEXT_SHORT_NAMES[context],
        }
        for variable, symbol in enumerate(VARIABLE_SYMBOLS):
            row[f"median_{symbol}"] = float(means[context, variable])
            row[f"sd_{symbol}"] = float(standard_deviation[context, variable])
        row["corr_v_s"] = float(correlation[context, 0, 1])
        row["corr_v_delta_v"] = float(correlation[context, 0, 2])
        row["corr_s_delta_v"] = float(correlation[context, 1, 2])
        context_rows.append(row)
    _write_csv_new(
        output_directory / "source_context_statistics.csv",
        tuple(context_rows[0].keys()),
        context_rows,
    )

    percentages, response_marginal, context_marginal = cooccurrence_percentages(
        np.asarray(arrays["cooccurrence_mean"], dtype=float)
    )
    coupling_rows = []
    for response in range(response_count):
        for context in range(context_count):
            coupling_rows.append(
                {
                    "response_state": response + 1,
                    "response_name": RESPONSE_NAMES[response],
                    "context_state": context + 1,
                    "context_name": CONTEXT_SHORT_NAMES[context],
                    "joint_share_percent": float(percentages[response, context]),
                    "response_marginal_percent": float(response_marginal[response]),
                    "context_marginal_percent": float(context_marginal[context]),
                }
            )
    _write_csv_new(
        output_directory / "source_response_scenario_cooccurrence.csv",
        tuple(coupling_rows[0].keys()),
        coupling_rows,
    )

    _write_csv_new(
        output_directory / "source_time_space_segments.csv",
        tuple(time_space_records[0].keys()),
        time_space_records,
    )
    return {
        name: {
            "sha256": _sha256_file(output_directory / name),
            "bytes": (output_directory / name).stat().st_size,
        }
        for name in SOURCE_DATA_FILENAMES
    }


def execute_e3_plotting(
    postprocess_directory: Path,
    cohort_npz: Path,
    cohort_manifest: Path,
    highd_root: Path,
    output_directory: Path,
) -> Path:
    """Generate the E3-only figure bundle and provenance receipt."""

    started = time.perf_counter()
    postprocess_directory = postprocess_directory.resolve()
    _verify_postprocessed(postprocess_directory)
    metadata = _load_json(postprocess_directory / "postprocess_metadata.json")
    if not bool(metadata.get("require_formal")):
        raise ValueError("E3 manuscript figures require formal posterior runs.")
    gate = _load_json(postprocess_directory / "story_gate.json")
    if str(gate.get("status")) != "PASS":
        raise ValueError("Story gate did not pass; refusing E3 manuscript figures.")
    summary = _load_json(postprocess_directory / "posterior_summary.json")
    config = PostprocessConfig.from_json(postprocess_directory / "config.json")
    cohort = load_cohort(cohort_npz, cohort_manifest)
    cohort_receipt = summary["cohort"]
    if str(cohort_receipt["content_sha256"]) != cohort.content_sha256:
        raise ValueError("E3 cohort differs from the postprocessed cohort.")
    arm_files = summary["arm_files"]
    arrays = _load_npz(postprocess_directory / str(arm_files[config.manuscript_arm_id]))
    arm_summary = summary["arms"][config.manuscript_arm_id]
    response_count = int(arm_summary["states"]["behavior"])
    context_count = int(arm_summary["states"]["scenario"])
    if (response_count, context_count) != (5, 5):
        raise ValueError("The approved E3 figure contract requires the frozen 5 x 5 model.")

    if output_directory.exists():
        raise FileExistsError(f"Refusing to reuse E3 output directory {output_directory}.")
    output_directory.mkdir(parents=True)
    configure_matplotlib()

    figure_artifacts: Dict[str, Mapping[str, Any]] = {}
    figure_artifacts.update(
        _save_figure_bundle_new(
            plot_joint_dynamics(arrays, response_count, context_count),
            output_directory / "fig_joint_dynamics",
        )
    )
    figure_artifacts.update(
        _save_figure_bundle_new(
            plot_context_coupling(arrays, response_count, context_count),
            output_directory / "fig_context_coupling",
        )
    )
    trajectories = load_legacy_time_space_metadata(highd_root, cohort)
    time_space_metadata_sha256 = _time_space_metadata_sha256(trajectories)
    time_space_figure, time_space_records = plot_time_space_states(
        arrays,
        cohort,
        trajectories,
        config,
        response_count,
        context_count,
    )
    figure_artifacts.update(
        _save_figure_bundle_new(
            time_space_figure,
            output_directory / "fig_time_space_states",
        )
    )
    source_data_artifacts = _write_source_data(
        output_directory,
        arrays,
        time_space_records,
        response_count,
        context_count,
    )

    repository_root = Path(__file__).resolve().parents[2]
    receipt = {
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "runtime_seconds": time.perf_counter() - started,
        "figure_contract": {
            "core_conclusion": (
                "The response coordinate identifies recurring intrinsic driver-action "
                "archetypes, while the scenario coordinate organizes the external traffic "
                "conditions across which those archetypes recur."
            ),
            "archetype": "asymmetric mixed-modality figure set",
            "backend": "Python/matplotlib",
            "final_width_in": 7.15,
            "minimum_source_font_pt": 5.5,
            "exports": list(EXPORT_FORMATS),
        },
        "postprocess_directory": str(postprocess_directory),
        "postprocess_manifest_sha256": _sha256_file(
            postprocess_directory / "artifact_manifest.json"
        ),
        "story_gate_status": gate["status"],
        "manuscript_arm_id": config.manuscript_arm_id,
        "cohort_content_sha256": cohort.content_sha256,
        "cohort_npz_sha256": _sha256_file(cohort_npz),
        "cohort_manifest_sha256": _sha256_file(cohort_manifest),
        "time_space_metadata_sha256": time_space_metadata_sha256,
        "time_space_metadata_trajectory_count": len(trajectories),
        "plotted_time_space_segment_count": len(time_space_records),
        "output_directory": str(output_directory.resolve()),
        "figure_artifacts": figure_artifacts,
        "source_data_artifacts": source_data_artifacts,
        "source_sha256": {
            "plot_e3_interpretation.py": _sha256_file(Path(__file__)),
            "plot_manuscript_results.py": _sha256_file(
                Path(__file__).with_name("plot_manuscript_results.py")
            ),
            "postprocess.py": _sha256_file(Path(__file__).with_name("postprocess.py")),
            "data/data_loader.py": _sha256_file(repository_root / "data" / "data_loader.py"),
            "data/read_csv.py": _sha256_file(repository_root / "data" / "read_csv.py"),
            "config.py": _sha256_file(repository_root / "config.py"),
        },
        "python": sys.version,
        "platform": platform.platform(),
        "matplotlib": matplotlib.__version__,
        "data_exclusions": [],
        "transformations": {
            "context_covariance": (
                "Exact decomposition of each physical-scale posterior-mean covariance "
                "matrix into standard deviations and correlations; reconstruction checked."
            ),
            "cooccurrence": "Posterior mean joint proportions multiplied by 100.",
            "time_space": (
                "Same prespecified Lane 3/4 and first-85-trajectory display scope as the "
                "frozen manuscript plotting configuration; no stochastic downsampling."
            ),
        },
        "interpretation_boundary": (
            "Figures summarize in-sample, label-aligned posterior quantities and pointwise "
            "modal state assignments; they do not identify causal traffic phases or stable "
            "psychological driver traits."
        ),
    }
    _write_json_new(output_directory / "figure_metadata.json", receipt)
    _write_json_new(
        output_directory / "figure_artifact_manifest.json",
        {
            "figure_artifacts": figure_artifacts,
            "source_data_artifacts": source_data_artifacts,
        },
    )
    return output_directory


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--postprocess-directory", type=Path, required=True)
    parser.add_argument("--cohort-npz", type=Path, required=True)
    parser.add_argument("--cohort-manifest", type=Path, required=True)
    parser.add_argument("--highd-root", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    output = execute_e3_plotting(
        args.postprocess_directory,
        args.cohort_npz,
        args.cohort_manifest,
        args.highd_root,
        args.output_directory,
    )
    print(json.dumps({"status": "COMPLETED", "output_directory": str(output)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
