"""Polish the original E3 figure families using the frozen formal posterior.

The script preserves the manuscript's original figure organization: a separate
joint-transition heatmap paired with a TeX indexing diagram, a 2 x 5
covariance/correlation grid, grouped response-composition bars, and a 3 x 2
time-space plot.  It adds one standalone response-scenario co-occurrence figure
with marginals.  No sampler is run and no frozen input artifact is rewritten.
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
from matplotlib.gridspec import GridSpec
from numpy.typing import NDArray

from .plot_e3_interpretation import (
    CONTEXT_COLORS,
    CONTEXT_SHORT_NAMES,
    NEUTRAL_DARK,
    NEUTRAL_LIGHT,
    NEUTRAL_MID,
    RESPONSE_COLORS,
    RESPONSE_NAMES,
    VARIABLE_NAMES,
    VARIABLE_SYMBOLS,
    VARIABLE_UNITS,
    _heatmap_text_color,
    _remove_heatmap_frame,
    _write_csv_new,
    configure_matplotlib,
    covariance_components,
    cooccurrence_percentages,
    plot_time_space_states,
)
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

# A restrained manuscript palette shared by the two matrix-led E3 figures.
# Negative/positive values keep the conventional cool/warm distinction, while
# the joint-share map remains strictly sequential from low to high occupancy.
E3_DIVERGING_CMAP = colors.LinearSegmentedColormap.from_list(
    "e3_diverging",
    ("#294E63", "#9DB4BE", "#F5F1E9", "#D9A084", "#9A5147"),
)
E3_JOINT_CMAP = colors.LinearSegmentedColormap.from_list(
    "e3_joint",
    ("#F7F4ED", "#D7E2DD", "#9BBDB7", "#4F8790", "#19384F"),
)
E3_SCENARIO_MARGINAL_COLOR = "#718B99"
E3_RESPONSE_MARGINAL_COLOR = "#B17460"
E3_CELL_SEPARATOR_COLOR = "#FCFAF6"

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "Liberation Sans"],
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)

FIGURE_BASENAMES = (
    "transition_mat",
    "cov_corr_mat",
    "states_stats",
    "response_scenario_cooccurrence",
    "time_space_3by2_state_speed",
)
SOURCE_DATA_FILENAMES = (
    "source_joint_transition.csv",
    "source_context_statistics.csv",
    "source_response_scenario_coupling.csv",
    "source_time_space_segments.csv",
)
EXPORT_FORMATS = ("svg", "pdf", "png")


def _save_polished_figure_bundle(
    figure: plt.Figure, output_base: Path
) -> Dict[str, Mapping[str, Any]]:
    """Save SVG, PDF, and a 300-dpi PNG preview without overwriting."""

    outputs = {
        "svg": output_base.with_suffix(".svg"),
        "pdf": output_base.with_suffix(".pdf"),
        "png": output_base.with_suffix(".png"),
    }
    conflicts = [str(path) for path in outputs.values() if path.exists()]
    if conflicts:
        raise FileExistsError(f"Refusing to overwrite polished figures: {conflicts}")
    temporaries = {
        extension: path.with_name(f".{path.stem}.tmp{path.suffix}")
        for extension, path in outputs.items()
    }
    stale = [str(path) for path in temporaries.values() if path.exists()]
    if stale:
        raise FileExistsError(f"Stale polished-figure temporaries exist: {stale}")
    try:
        figure.savefig(temporaries["svg"], format="svg")
        figure.savefig(temporaries["pdf"], format="pdf")
        figure.savefig(temporaries["png"], format="png", dpi=300)
        for extension in EXPORT_FORMATS:
            os.replace(temporaries[extension], outputs[extension])
    finally:
        plt.close(figure)
    return {
        path.name: {"sha256": _sha256_file(path), "bytes": path.stat().st_size}
        for path in outputs.values()
    }


def conditional_response_percentages(cooccurrence: FloatArray) -> FloatArray:
    """Return 100 p(response | context), with each context column summing to 100."""

    cooccurrence = np.asarray(cooccurrence, dtype=float)
    cooccurrence_percentages(cooccurrence)
    context_mass = cooccurrence.sum(axis=0)
    if np.any(context_mass <= 0.0):
        raise ValueError("Every context requires positive posterior mass.")
    conditional = 100.0 * cooccurrence / context_mass[None, :]
    if not np.allclose(conditional.sum(axis=0), 100.0, atol=1.0e-8):
        raise ValueError("Conditional response percentages do not sum to 100 by context.")
    return conditional


def _validate_transition(
    arrays: Mapping[str, NDArray[np.generic]],
    response_count: int,
    context_count: int,
) -> FloatArray:
    transition = np.asarray(arrays["transition_mean"], dtype=float)
    joint_count = response_count * context_count
    if transition.shape != (joint_count, joint_count):
        raise ValueError("Transition summary has the wrong shape.")
    if not np.all(np.isfinite(transition)) or np.any(transition < 0.0):
        raise ValueError("Transition summary must be finite and non-negative.")
    if not np.allclose(transition.sum(axis=1), 1.0, atol=1.0e-8):
        raise ValueError("Transition rows must be stochastic.")
    return transition


def plot_transition_matrix_original_style(
    arrays: Mapping[str, NDArray[np.generic]],
    response_count: int,
    context_count: int,
) -> plt.Figure:
    """Preserve the original standalone 25 x 25 transition heatmap."""

    transition = _validate_transition(arrays, response_count, context_count)
    joint_count = response_count * context_count
    labels = [
        f"({response + 1},{context + 1})"
        for response in range(response_count)
        for context in range(context_count)
    ]

    fig, ax = plt.subplots(figsize=(4.45, 4.20))
    cmap = plt.get_cmap("cividis")
    norm = colors.Normalize(vmin=0.0, vmax=float(np.max(transition)))
    image = ax.imshow(
        transition,
        cmap=cmap,
        norm=norm,
        aspect="equal",
        interpolation="nearest",
    )
    ticks = np.arange(joint_count)
    ax.set_xticks(ticks, labels, rotation=90, rotation_mode="anchor", ha="left")
    ax.set_yticks(ticks, labels)
    ax.tick_params(axis="x", top=True, labeltop=True, bottom=False, labelbottom=False)
    ax.tick_params(axis="both", labelsize=5.2, length=1.8, width=0.55, pad=1.0)
    ax.set_xlabel("Next joint state (response, context)")
    ax.set_ylabel("Current joint state (response, context)")
    for boundary in np.arange(context_count - 0.5, joint_count - 0.5, context_count):
        ax.axhline(boundary, color="white", linewidth=0.55, alpha=0.78)
        ax.axvline(boundary, color="white", linewidth=0.55, alpha=0.78)
    for spine in ax.spines.values():
        spine.set_linewidth(0.7)

    colorbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.035)
    colorbar.set_label("Transition probability", fontsize=7.0)
    colorbar.ax.tick_params(labelsize=6.0, length=2.0)
    fig.subplots_adjust(left=0.18, right=0.88, top=0.82, bottom=0.13)
    return fig


def plot_context_covariance_correlation_original_style(
    arrays: Mapping[str, NDArray[np.generic]],
    context_count: int,
) -> plt.Figure:
    """Preserve the original 2 x K covariance/correlation matrix grid."""

    covariance = np.asarray(arrays["scenario_covariance_physical_mean"], dtype=float)
    means = np.asarray(arrays["scenario_mean_physical_median"], dtype=float)
    if covariance.shape != (context_count, 3, 3) or means.shape != (context_count, 3):
        raise ValueError("Physical context summaries have the wrong shape.")
    _, correlation = covariance_components(covariance)
    covariance_scale = float(np.max(np.abs(covariance)))
    if covariance_scale <= 0.0 or not np.isfinite(covariance_scale):
        raise ValueError("Physical context covariances have an invalid scale.")

    fig, axes = plt.subplots(
        2,
        context_count,
        figsize=(7.15, 3.35),
        squeeze=False,
    )
    covariance_cmap = E3_DIVERGING_CMAP
    covariance_norm = colors.Normalize(vmin=-covariance_scale, vmax=covariance_scale)
    correlation_cmap = E3_DIVERGING_CMAP
    correlation_norm = colors.Normalize(vmin=-1.0, vmax=1.0)
    tick_labels = [r"$v$", r"$s$", r"$\Delta v$"]

    for context in range(context_count):
        covariance_image = axes[0, context].imshow(
            covariance[context],
            cmap=covariance_cmap,
            norm=covariance_norm,
            aspect="equal",
            interpolation="nearest",
        )
        correlation_image = axes[1, context].imshow(
            correlation[context],
            cmap=correlation_cmap,
            norm=correlation_norm,
            aspect="equal",
            interpolation="nearest",
        )
        for row in range(3):
            for column in range(3):
                covariance_value = float(covariance[context, row, column])
                correlation_value = float(correlation[context, row, column])
                axes[0, context].text(
                    column,
                    row,
                    f"{covariance_value:.1f}",
                    ha="center",
                    va="center",
                    fontsize=5.1,
                    color=_heatmap_text_color(
                        covariance_cmap, covariance_norm, covariance_value
                    ),
                )
                axes[1, context].text(
                    column,
                    row,
                    f"{correlation_value:.2f}",
                    ha="center",
                    va="center",
                    fontsize=5.1,
                    color=_heatmap_text_color(
                        correlation_cmap, correlation_norm, correlation_value
                    ),
                )
        axes[0, context].set_title(
            f"C{context + 1}\n"
            + f"median=({means[context, 0]:.1f}, {means[context, 1]:.1f}, "
            + f"{means[context, 2]:.1f})",
            fontsize=6.4,
            pad=3.0,
        )
        for row in range(2):
            axis = axes[row, context]
            axis.set_xticks(range(3), tick_labels)
            axis.set_yticks(range(3), tick_labels if context == 0 else [])
            axis.set_xticks(np.arange(-0.5, 3.0, 1.0), minor=True)
            axis.set_yticks(np.arange(-0.5, 3.0, 1.0), minor=True)
            axis.grid(
                which="minor",
                color=E3_CELL_SEPARATOR_COLOR,
                linewidth=0.65,
            )
            axis.tick_params(axis="both", labelsize=5.6, length=0, pad=1.0)
            axis.tick_params(which="minor", bottom=False, left=False)
            _remove_heatmap_frame(axis)
    axes[0, 0].set_ylabel("Covariance", fontsize=7.0)
    axes[1, 0].set_ylabel("Correlation", fontsize=7.0)

    fig.subplots_adjust(
        left=0.075,
        right=0.925,
        top=0.84,
        bottom=0.12,
        wspace=0.22,
        hspace=0.42,
    )
    covariance_colorbar = fig.colorbar(
        covariance_image,
        ax=axes[0, :].tolist(),
        fraction=0.018,
        pad=0.018,
    )
    covariance_colorbar.set_label("Physical covariance", fontsize=6.5)
    covariance_colorbar.ax.tick_params(labelsize=5.5, length=1.8)
    covariance_colorbar.outline.set_edgecolor(NEUTRAL_MID)
    covariance_colorbar.outline.set_linewidth(0.5)
    correlation_colorbar = fig.colorbar(
        correlation_image,
        ax=axes[1, :].tolist(),
        fraction=0.018,
        pad=0.018,
    )
    correlation_colorbar.set_label("Correlation", fontsize=6.5)
    correlation_colorbar.set_ticks((-1.0, 0.0, 1.0))
    correlation_colorbar.ax.tick_params(labelsize=5.5, length=1.8)
    correlation_colorbar.outline.set_edgecolor(NEUTRAL_MID)
    correlation_colorbar.outline.set_linewidth(0.5)
    return fig


def plot_response_composition_original_style(
    arrays: Mapping[str, NDArray[np.generic]],
    response_count: int,
    context_count: int,
) -> plt.Figure:
    """Preserve grouped bars while reporting p(response | context)."""

    cooccurrence = np.asarray(arrays["cooccurrence_mean"], dtype=float)
    conditional = conditional_response_percentages(cooccurrence)
    if conditional.shape != (response_count, context_count):
        raise ValueError("Conditional response composition has the wrong shape.")

    fig, ax = plt.subplots(figsize=(7.15, 3.25))
    x = np.arange(context_count, dtype=float)
    group_width = 0.82
    bar_width = group_width / response_count
    offsets = (
        np.arange(response_count, dtype=float) - 0.5 * (response_count - 1)
    ) * bar_width
    for response in range(response_count):
        ax.bar(
            x + offsets[response],
            conditional[response],
            width=0.92 * bar_width,
            color=RESPONSE_COLORS[response],
            edgecolor="white",
            linewidth=0.45,
            label=RESPONSE_NAMES[response],
            zorder=3,
        )
    ax.set_xticks(x, [f"C{context + 1}" for context in range(context_count)])
    ax.set_xlabel("External traffic scenario")
    ax.set_ylabel(r"Response composition within scenario, $p(b\mid s)$ (%)")
    ax.set_ylim(0.0, 100.0)
    ax.set_xlim(-0.55, context_count - 0.45)
    ax.set_axisbelow(True)
    ax.grid(axis="y", color=NEUTRAL_LIGHT, linewidth=0.65, alpha=0.85)
    ax.legend(
        ncol=5,
        loc="upper right",
        columnspacing=0.8,
        handlelength=1.25,
        borderaxespad=0.25,
    )
    fig.subplots_adjust(left=0.12, right=0.985, top=0.94, bottom=0.20)
    return fig


def plot_response_scenario_cooccurrence(
    arrays: Mapping[str, NDArray[np.generic]],
    response_count: int,
    context_count: int,
) -> plt.Figure:
    """Plot the approved joint-share heatmap with both marginal distributions."""

    percentages, response_marginal, context_marginal = cooccurrence_percentages(
        np.asarray(arrays["cooccurrence_mean"], dtype=float)
    )
    if percentages.shape != (response_count, context_count):
        raise ValueError("Posterior co-occurrence has the wrong shape.")

    fig = plt.figure(figsize=(6.55, 4.35))
    grid = GridSpec(
        2,
        3,
        figure=fig,
        height_ratios=(0.28, 1.0),
        width_ratios=(1.0, 0.24, 0.055),
        hspace=0.04,
        wspace=0.08,
    )
    ax_context = fig.add_subplot(grid[0, 0])
    ax_heatmap = fig.add_subplot(grid[1, 0])
    ax_response = fig.add_subplot(grid[1, 1], sharey=ax_heatmap)
    cax = fig.add_subplot(grid[1, 2])

    cmap = E3_JOINT_CMAP
    norm = colors.Normalize(vmin=0.0, vmax=float(np.max(percentages)))
    image = ax_heatmap.imshow(
        percentages,
        cmap=cmap,
        norm=norm,
        aspect="auto",
        interpolation="nearest",
    )
    for response in range(response_count):
        for context in range(context_count):
            value = float(percentages[response, context])
            ax_heatmap.text(
                context,
                response,
                f"{value:.1f}",
                ha="center",
                va="center",
                fontsize=7.0,
                color=_heatmap_text_color(cmap, norm, value),
            )
    ax_heatmap.set_xticks(
        range(context_count), [f"C{context + 1}" for context in range(context_count)]
    )
    ax_heatmap.set_yticks(range(response_count), RESPONSE_NAMES)
    ax_heatmap.set_xlabel("External traffic scenario")
    ax_heatmap.set_ylabel("Intrinsic driver action")
    ax_heatmap.set_xticks(np.arange(-0.5, context_count, 1.0), minor=True)
    ax_heatmap.set_yticks(np.arange(-0.5, response_count, 1.0), minor=True)
    ax_heatmap.grid(
        which="minor",
        color=E3_CELL_SEPARATOR_COLOR,
        linewidth=0.75,
    )
    ax_heatmap.tick_params(which="minor", bottom=False, left=False)
    _remove_heatmap_frame(ax_heatmap)

    x = np.arange(context_count)
    ax_context.bar(
        x,
        context_marginal,
        width=0.82,
        color=E3_SCENARIO_MARGINAL_COLOR,
        edgecolor=E3_CELL_SEPARATOR_COLOR,
        linewidth=0.7,
    )
    ax_context.set_xlim(-0.5, context_count - 0.5)
    ax_context.set_ylim(0.0, 1.22 * float(np.max(context_marginal)))
    ax_context.set_ylabel("Scenario share (%)", fontsize=6.5)
    ax_context.set_xticks([])
    ax_context.tick_params(axis="y", labelsize=6.0, length=2.0)
    for context, value in enumerate(context_marginal):
        ax_context.text(
            context,
            value + 0.45,
            f"{value:.1f}",
            ha="center",
            va="bottom",
            fontsize=6.0,
            color=NEUTRAL_DARK,
        )

    ax_response.barh(
        np.arange(response_count),
        response_marginal,
        height=0.82,
        color=E3_RESPONSE_MARGINAL_COLOR,
        edgecolor=E3_CELL_SEPARATOR_COLOR,
        linewidth=0.7,
    )
    ax_response.set_xlim(0.0, 1.20 * float(np.max(response_marginal)))
    ax_response.set_xlabel("Response\nshare (%)", fontsize=6.5)
    ax_response.tick_params(axis="y", left=False, labelleft=False)
    ax_response.tick_params(axis="x", labelsize=6.0, length=2.0)
    for response, value in enumerate(response_marginal):
        ax_response.text(
            value + 0.55,
            response,
            f"{value:.1f}",
            ha="left",
            va="center",
            fontsize=6.0,
            color=NEUTRAL_DARK,
        )
    for axis in (ax_context, ax_response):
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.spines["left"].set_color(NEUTRAL_MID)
        axis.spines["bottom"].set_color(NEUTRAL_MID)
        axis.spines["left"].set_linewidth(0.6)
        axis.spines["bottom"].set_linewidth(0.6)

    colorbar = fig.colorbar(image, cax=cax)
    colorbar.set_label("Joint share (%)", fontsize=6.5)
    colorbar.ax.tick_params(labelsize=6.0, length=2.0)
    colorbar.outline.set_edgecolor(NEUTRAL_MID)
    colorbar.outline.set_linewidth(0.5)
    fig.subplots_adjust(left=0.17, right=0.95, top=0.96, bottom=0.14)
    return fig

def polish_time_space_legends(
    figure: plt.Figure, response_count: int, context_count: int
) -> None:
    """Keep the original in-panel legends readable above dense trajectories."""

    if len(figure.axes) < 6:
        raise ValueError("The 3 x 2 time-space figure is missing plotting axes.")
    response_axis = figure.axes[1]
    context_axis = figure.axes[3]
    original_response_legend = response_axis.get_legend()
    original_context_legend = context_axis.get_legend()
    if original_response_legend is not None:
        original_response_legend.remove()
    if original_context_legend is not None:
        original_context_legend.remove()

    response_handles = [
        lines.Line2D([], [], color=RESPONSE_COLORS[index], lw=2.0)
        for index in range(response_count)
    ]
    # Matplotlib fills a three-column, two-row legend column-wise. This input
    # order yields the intended visual sequence R1, R2, R3 / R4, R5.
    display_order = (0, 3, 1, 4, 2)
    response_legend = response_axis.legend(
        handles=[response_handles[index] for index in display_order],
        labels=[RESPONSE_NAMES[index] for index in display_order],
        ncol=3,
        loc="upper right",
        columnspacing=0.75,
        handlelength=1.4,
        borderaxespad=0.25,
        frameon=True,
    )
    context_handles = [
        lines.Line2D([], [], color=CONTEXT_COLORS[index], lw=2.0)
        for index in range(context_count)
    ]
    context_legend = context_axis.legend(
        handles=context_handles,
        labels=[f"C{index + 1}" for index in range(context_count)],
        ncol=5,
        loc="upper right",
        columnspacing=0.65,
        handlelength=1.35,
        borderaxespad=0.25,
        frameon=True,
    )
    for legend in (response_legend, context_legend):
        frame = legend.get_frame()
        frame.set_facecolor("white")
        frame.set_edgecolor(NEUTRAL_LIGHT)
        frame.set_linewidth(0.5)
        frame.set_alpha(0.92)


def _write_source_data(
    output_directory: Path,
    arrays: Mapping[str, NDArray[np.generic]],
    time_space_records: Sequence[Mapping[str, Any]],
    response_count: int,
    context_count: int,
) -> Dict[str, Mapping[str, Any]]:
    transition = _validate_transition(arrays, response_count, context_count)
    transition_rows = []
    for current in range(transition.shape[0]):
        for nxt in range(transition.shape[1]):
            transition_rows.append(
                {
                    "current_joint_state": current + 1,
                    "current_response_state": current // context_count + 1,
                    "current_context_state": current % context_count + 1,
                    "next_joint_state": nxt + 1,
                    "next_response_state": nxt // context_count + 1,
                    "next_context_state": nxt % context_count + 1,
                    "posterior_mean_transition_probability": float(
                        transition[current, nxt]
                    ),
                }
            )
    _write_csv_new(
        output_directory / SOURCE_DATA_FILENAMES[0],
        tuple(transition_rows[0].keys()),
        transition_rows,
    )

    covariance = np.asarray(arrays["scenario_covariance_physical_mean"], dtype=float)
    means = np.asarray(arrays["scenario_mean_physical_median"], dtype=float)
    standard_deviation, correlation = covariance_components(covariance)
    context_rows = []
    for context in range(context_count):
        for row in range(3):
            for column in range(3):
                context_rows.append(
                    {
                        "context_state": context + 1,
                        "context_name": CONTEXT_SHORT_NAMES[context],
                        "row_variable": VARIABLE_NAMES[row],
                        "row_unit": VARIABLE_UNITS[row],
                        "column_variable": VARIABLE_NAMES[column],
                        "column_unit": VARIABLE_UNITS[column],
                        "posterior_median_row_variable": float(means[context, row]),
                        "posterior_mean_covariance": float(
                            covariance[context, row, column]
                        ),
                        "derived_correlation": float(correlation[context, row, column]),
                        "posterior_implied_sd_row_variable": float(
                            standard_deviation[context, row]
                        ),
                    }
                )
    _write_csv_new(
        output_directory / SOURCE_DATA_FILENAMES[1],
        tuple(context_rows[0].keys()),
        context_rows,
    )

    cooccurrence = np.asarray(arrays["cooccurrence_mean"], dtype=float)
    joint_percentages, response_marginal, context_marginal = cooccurrence_percentages(
        cooccurrence
    )
    conditional = conditional_response_percentages(cooccurrence)
    coupling_rows = []
    for response in range(response_count):
        for context in range(context_count):
            coupling_rows.append(
                {
                    "response_state": response + 1,
                    "response_name": RESPONSE_NAMES[response],
                    "context_state": context + 1,
                    "context_name": CONTEXT_SHORT_NAMES[context],
                    "joint_share_percent": float(joint_percentages[response, context]),
                    "response_given_context_percent": float(
                        conditional[response, context]
                    ),
                    "response_marginal_percent": float(response_marginal[response]),
                    "context_marginal_percent": float(context_marginal[context]),
                }
            )
    _write_csv_new(
        output_directory / SOURCE_DATA_FILENAMES[2],
        tuple(coupling_rows[0].keys()),
        coupling_rows,
    )

    if not time_space_records:
        raise ValueError("The time-space source-data table cannot be empty.")
    _write_csv_new(
        output_directory / SOURCE_DATA_FILENAMES[3],
        tuple(time_space_records[0].keys()),
        time_space_records,
    )
    return {
        filename: {
            "sha256": _sha256_file(output_directory / filename),
            "bytes": (output_directory / filename).stat().st_size,
        }
        for filename in SOURCE_DATA_FILENAMES
    }


def execute_plotting(
    postprocess_directory: Path,
    cohort_npz: Path,
    cohort_manifest: Path,
    highd_root: Path,
    output_directory: Path,
) -> Path:
    """Generate the approved original-style E3 bundle and provenance receipt."""

    started = time.perf_counter()
    postprocess_directory = postprocess_directory.resolve()
    cohort_npz = cohort_npz.resolve()
    cohort_manifest = cohort_manifest.resolve()
    highd_root = highd_root.resolve()
    output_directory = output_directory.resolve()

    _verify_postprocessed(postprocess_directory)
    postprocess_metadata = _load_json(postprocess_directory / "postprocess_metadata.json")
    if not bool(postprocess_metadata.get("require_formal")):
        raise ValueError("Manuscript figures require formal posterior runs.")
    story_gate = _load_json(postprocess_directory / "story_gate.json")
    if str(story_gate.get("status")) != "PASS":
        raise ValueError("Story gate did not pass; refusing manuscript figures.")
    summary = _load_json(postprocess_directory / "posterior_summary.json")
    config = PostprocessConfig.from_json(postprocess_directory / "config.json")
    cohort = load_cohort(cohort_npz, cohort_manifest)
    if str(summary["cohort"]["content_sha256"]) != cohort.content_sha256:
        raise ValueError("Figure cohort differs from the postprocessed cohort.")

    arrays = _load_npz(
        postprocess_directory / str(summary["arm_files"][config.manuscript_arm_id])
    )
    arm_summary = summary["arms"][config.manuscript_arm_id]
    response_count = int(arm_summary["states"]["behavior"])
    context_count = int(arm_summary["states"]["scenario"])
    if (response_count, context_count) != (5, 5):
        raise ValueError("The approved E3 figure contract requires the frozen 5 x 5 arm.")
    if output_directory.exists():
        raise FileExistsError(f"Refusing to reuse figure output directory {output_directory}.")
    output_directory.mkdir(parents=True)
    configure_matplotlib()

    figure_artifacts: Dict[str, Mapping[str, Any]] = {}
    figure_artifacts.update(
        _save_polished_figure_bundle(
            plot_transition_matrix_original_style(
                arrays, response_count, context_count
            ),
            output_directory / FIGURE_BASENAMES[0],
        )
    )
    figure_artifacts.update(
        _save_polished_figure_bundle(
            plot_context_covariance_correlation_original_style(
                arrays, context_count
            ),
            output_directory / FIGURE_BASENAMES[1],
        )
    )
    figure_artifacts.update(
        _save_polished_figure_bundle(
            plot_response_composition_original_style(
                arrays, response_count, context_count
            ),
            output_directory / FIGURE_BASENAMES[2],
        )
    )
    figure_artifacts.update(
        _save_polished_figure_bundle(
            plot_response_scenario_cooccurrence(
                arrays, response_count, context_count
            ),
            output_directory / FIGURE_BASENAMES[3],
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
    polish_time_space_legends(time_space_figure, response_count, context_count)
    figure_artifacts.update(
        _save_polished_figure_bundle(
            time_space_figure,
            output_directory / FIGURE_BASENAMES[4],
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
    manuscript_figures = repository_root / "64596614d5f2a5878df47536" / "Figs"
    original_references = (
        manuscript_figures / "transition_mat.pdf",
        manuscript_figures / "cov_corr_mat.pdf",
        manuscript_figures / "states_stats.pdf",
        manuscript_figures / "time_space_3by2_state_speed.pdf",
        manuscript_figures / "storyrev" / "indexing.tex",
    )
    for reference in original_references:
        if not reference.is_file():
            raise FileNotFoundError(f"Original figure reference is missing: {reference}")

    receipt = {
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "runtime_seconds": time.perf_counter() - started,
        "figure_contract": {
            "core_conclusion": (
                "Response archetypes recur across multiple external scenarios, and each "
                "scenario admits multiple response actions, while their joint state "
                "evolves through an unrestricted transition matrix."
            ),
            "evidence_sequence": [
                "TeX product-state indexing plus posterior mean joint transition matrix",
                "original 2 x 5 physical covariance/correlation grid",
                "standalone posterior-expected joint co-occurrence with marginals",
                "original 3 x 2 time-space organization",
            ],
            "generated_supporting_artifact": (
                "The original grouped bars reporting p(response | context) are retained "
                "as a reproducible source artifact but are not required in the main-text "
                "evidence sequence."
            ),
            "backend": "Python/matplotlib for quantitative figures; TeX/TikZ retained for indexing",
            "final_width_in": 7.15,
            "minimum_source_font_pt": 5.1,
            "exports": ["svg", "pdf", "png"],
        },
        "postprocess_directory": str(postprocess_directory),
        "postprocess_manifest_sha256": _sha256_file(
            postprocess_directory / "artifact_manifest.json"
        ),
        "story_gate_status": story_gate["status"],
        "manuscript_arm_id": config.manuscript_arm_id,
        "cohort_content_sha256": cohort.content_sha256,
        "cohort_npz_sha256": _sha256_file(cohort_npz),
        "cohort_manifest_sha256": _sha256_file(cohort_manifest),
        "time_space_metadata_sha256": time_space_metadata_sha256,
        "time_space_metadata_trajectory_count": len(trajectories),
        "plotted_time_space_segment_count": len(time_space_records),
        "output_directory": str(output_directory),
        "figure_artifacts": figure_artifacts,
        "source_data_artifacts": source_data_artifacts,
        "source_sha256": {
            "plot_e3_polished_originals.py": _sha256_file(Path(__file__)),
            "plot_e3_interpretation.py": _sha256_file(
                Path(__file__).with_name("plot_e3_interpretation.py")
            ),
            "plot_manuscript_results.py": _sha256_file(
                Path(__file__).with_name("plot_manuscript_results.py")
            ),
            "postprocess.py": _sha256_file(Path(__file__).with_name("postprocess.py")),
            "data/data_loader.py": _sha256_file(repository_root / "data" / "data_loader.py"),
            "data/read_csv.py": _sha256_file(repository_root / "data" / "read_csv.py"),
            "config.py": _sha256_file(repository_root / "config.py"),
        },
        "original_figure_reference_sha256": {
            str(reference.relative_to(repository_root)): _sha256_file(reference)
            for reference in original_references
        },
        "python": sys.version,
        "platform": platform.platform(),
        "matplotlib": matplotlib.__version__,
        "data_exclusions": [],
        "transformations": {
            "transition": "Posterior mean row-stochastic joint transition matrix; no thresholding.",
            "context_covariance": (
                "Physical-scale posterior mean covariance and its exact correlation "
                "decomposition; positive definiteness and reconstruction verified."
            ),
            "grouped_bars": (
                "Posterior co-occurrence normalized within each context so every "
                "scenario group sums to 100 percent."
            ),
            "cooccurrence": "Posterior mean joint proportions multiplied by 100, with exact marginals.",
            "time_space": (
                "Same prespecified Lane 3/4 and first-85-trajectory display scope as "
                "the frozen configuration; no stochastic downsampling."
            ),
        },
        "interpretation_boundary": (
            "Figures summarize in-sample, label-aligned posterior quantities and "
            "pointwise modal state assignments; they do not identify causal traffic "
            "phases or stable psychological driver traits."
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
    output = execute_plotting(
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
