"""Plot posterior-soft state reorganization across the 1, 2, and 5-state arms.

The two Sankey panels use the same retained observation-time points in three
independently fitted models.  Ribbon weights are products of pointwise
posterior marginals, not temporal transition probabilities or deterministic
state-nesting counts.
"""

from __future__ import annotations

import argparse
import csv
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
from matplotlib import colors
from matplotlib.path import Path as MplPath
from matplotlib.patches import PathPatch, Rectangle
from numpy.typing import NDArray

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans"],
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
    }
)

from .postprocess import _sha256_file, _write_json_new


FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]

ARM_FILENAMES = {1: "arm_b1s1.npz", 2: "arm_b2s2.npz", 5: "arm_b5s5.npz"}
FIGURE_BASENAME = "state_resolution_sankey"
EXPORT_FORMATS = ("svg", "pdf", "png", "tiff")
SOURCE_LINKS_FILENAME = "source_state_resolution_sankey_links.csv"
SOURCE_NODES_FILENAME = "source_state_resolution_sankey_nodes.csv"
FIGURE_WIDTH_IN = 7.15
FIGURE_HEIGHT_IN = 4.80
PANEL_TITLE_Y = 0.985
RESOLUTION_LABEL_Y = 0.855
RESOLUTION_ARROW_Y = 0.873

RESPONSE_NAMES = (
    "Cautious",
    "Assertive A",
    "Assertive B",
    "Defensive",
    "Aggressive",
)
RESPONSE_COLORS = (
    "#3B6FB6",
    "#D9A441",
    "#E7C66A",
    "#4F9A8A",
    "#B95C59",
)
SCENARIO_NAMES = (
    "Very slow / tight / opening",
    "Slow / closing",
    "Moderate / short / opening",
    "Faster / near-steady",
    "Moderate / very long / opening",
)
SCENARIO_COLORS = (
    "#5B7FA3",
    "#C58C52",
    "#5C9A86",
    "#8B6FA8",
    "#8A8175",
)

NEUTRAL_DARK = "#454C52"
NEUTRAL_MID = "#707980"
NEUTRAL_NODE = "#F5F6F7"
NEUTRAL_LINK_1 = "#CCD2D7"
NEUTRAL_LINK_2 = "#E3E6E9"
WHITE = "#FFFFFF"


@dataclass(frozen=True)
class PanelSpec:
    """Display identity for one independently fitted latent coordinate."""

    key: str
    title: str
    probability_key: str
    labels: Mapping[int, Tuple[str, ...]]
    visual_orders: Mapping[int, Tuple[int, ...]]
    terminal_colors: Tuple[str, ...]
    terminal_prefix: str
    resolution_symbol: str


@dataclass(frozen=True)
class PanelData:
    """Validated node marginals and adjacent-resolution overlap matrices."""

    spec: PanelSpec
    probabilities: Mapping[int, FloatArray]
    marginals: Mapping[int, FloatArray]
    links: Mapping[Tuple[int, int], FloatArray]
    observation_count: int


@dataclass(frozen=True)
class NodeInterval:
    """Vertical extent for one state node in panel coordinates."""

    bottom: float
    top: float


RESPONSE_SPEC = PanelSpec(
    key="response",
    title="Response coordinate",
    probability_key="behavior_probability",
    labels={
        1: ("Pooled response",),
        2: ("Assertive", "Cautious"),
        5: RESPONSE_NAMES,
    },
    visual_orders={1: (0,), 2: (1, 0), 5: (0, 1, 2, 3, 4)},
    terminal_colors=RESPONSE_COLORS,
    terminal_prefix="R",
    resolution_symbol=r"K^{[\mathrm{B}]}",
)

SCENARIO_SPEC = PanelSpec(
    key="scenario",
    title="Scenario coordinate",
    probability_key="scenario_probability",
    labels={
        1: ("Pooled scenario",),
        2: (
            "Lower-speed / short-gap opening",
            "Higher-speed / long-gap closing",
        ),
        5: SCENARIO_NAMES,
    },
    visual_orders={1: (0,), 2: (0, 1), 5: (0, 1, 2, 3, 4)},
    terminal_colors=SCENARIO_COLORS,
    terminal_prefix="C",
    resolution_symbol=r"K^{[\mathrm{S}]}",
)


def configure_matplotlib() -> None:
    """Apply the journal-width style while preserving editable vector text."""

    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans", "Liberation Sans"]
    plt.rcParams["svg.fonttype"] = "none"
    plt.rcParams.update(
        {
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "font.size": 7.0,
            "figure.dpi": 150,
            "savefig.dpi": 600,
            "savefig.bbox": None,
        }
    )


def validate_probability_matrix(
    values: NDArray[np.generic], expected_states: int, label: str
) -> FloatArray:
    """Return a finite row-stochastic posterior probability matrix."""

    matrix = np.asarray(values, dtype=float)
    if matrix.ndim != 2 or matrix.shape[1] != expected_states:
        raise ValueError(
            f"{label} must have shape (N, {expected_states}); got {matrix.shape}."
        )
    if matrix.shape[0] == 0 or not np.all(np.isfinite(matrix)):
        raise ValueError(f"{label} must contain finite observations.")
    if np.any(matrix < -1.0e-8):
        raise ValueError(f"{label} contains negative posterior probabilities.")
    if not np.allclose(matrix.sum(axis=1), 1.0, atol=2.0e-6):
        raise ValueError(f"{label} rows must sum to one.")
    matrix = np.clip(matrix, 0.0, 1.0)
    return matrix / matrix.sum(axis=1, keepdims=True)


def posterior_soft_overlap(left: FloatArray, right: FloatArray) -> FloatArray:
    """Compute average products of pointwise posterior marginals."""

    left = np.asarray(left, dtype=float)
    right = np.asarray(right, dtype=float)
    if left.ndim != 2 or right.ndim != 2 or left.shape[0] != right.shape[0]:
        raise ValueError("Overlap inputs must be 2D and share the observation dimension.")
    if left.shape[0] == 0:
        raise ValueError("Overlap requires at least one observation.")
    overlap = left.T @ right / float(left.shape[0])
    if np.any(overlap < -1.0e-12) or not np.isclose(overlap.sum(), 1.0, atol=2.0e-6):
        raise ValueError("Posterior-soft overlap must be non-negative and sum to one.")
    return np.clip(overlap, 0.0, None)


def build_panel_data(
    probabilities: Mapping[int, NDArray[np.generic]], spec: PanelSpec
) -> PanelData:
    """Validate three arms and construct the two adjacent overlap matrices."""

    validated: Dict[int, FloatArray] = {}
    observation_count: Optional[int] = None
    for state_count in (1, 2, 5):
        if state_count not in probabilities:
            raise ValueError(f"Missing K={state_count} probabilities for {spec.key}.")
        matrix = validate_probability_matrix(
            probabilities[state_count], state_count, f"{spec.title} K={state_count}"
        )
        if observation_count is None:
            observation_count = matrix.shape[0]
        elif matrix.shape[0] != observation_count:
            raise ValueError(f"{spec.title} arms do not share the same observations.")
        validated[state_count] = matrix
    if observation_count is None:
        raise AssertionError("The fixed state-count loop cannot be empty.")

    marginals = {state_count: matrix.mean(axis=0) for state_count, matrix in validated.items()}
    links = {
        (1, 2): posterior_soft_overlap(validated[1], validated[2]),
        (2, 5): posterior_soft_overlap(validated[2], validated[5]),
    }
    np.testing.assert_allclose(links[(1, 2)].sum(axis=1), marginals[1], atol=2.0e-6)
    np.testing.assert_allclose(links[(1, 2)].sum(axis=0), marginals[2], atol=2.0e-6)
    np.testing.assert_allclose(links[(2, 5)].sum(axis=1), marginals[2], atol=2.0e-6)
    np.testing.assert_allclose(links[(2, 5)].sum(axis=0), marginals[5], atol=2.0e-6)
    return PanelData(
        spec=spec,
        probabilities=validated,
        marginals=marginals,
        links=links,
        observation_count=observation_count,
    )


def _load_npz(path: Path) -> Dict[str, NDArray[np.generic]]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing canonical postprocessed arm: {path}")
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def load_canonical_panels(
    postprocess_directory: Path,
) -> Tuple[PanelData, PanelData, Mapping[int, Path], int]:
    """Load the three frozen arms and verify shared trajectory indexing."""

    directory = postprocess_directory.resolve()
    arm_paths = {k: directory / filename for k, filename in ARM_FILENAMES.items()}
    arrays = {k: _load_npz(path) for k, path in arm_paths.items()}
    offsets: Dict[int, IntArray] = {}
    for state_count, arm in arrays.items():
        if "trajectory_offsets" not in arm:
            raise ValueError(f"K={state_count} arm lacks trajectory_offsets.")
        current = np.asarray(arm["trajectory_offsets"], dtype=np.int64)
        if current.ndim != 1 or current.size < 2 or current[0] != 0:
            raise ValueError(f"K={state_count} trajectory_offsets are invalid.")
        if np.any(np.diff(current) <= 0):
            raise ValueError(f"K={state_count} trajectory_offsets must be strictly increasing.")
        offsets[state_count] = current
    for state_count in (2, 5):
        if not np.array_equal(offsets[1], offsets[state_count]):
            raise ValueError("Canonical arms do not share identical trajectory offsets.")

    response = build_panel_data(
        {k: arrays[k][RESPONSE_SPEC.probability_key] for k in (1, 2, 5)},
        RESPONSE_SPEC,
    )
    scenario = build_panel_data(
        {k: arrays[k][SCENARIO_SPEC.probability_key] for k in (1, 2, 5)},
        SCENARIO_SPEC,
    )
    if response.observation_count != int(offsets[1][-1]):
        raise ValueError("Probability rows disagree with trajectory offsets.")
    if scenario.observation_count != response.observation_count:
        raise ValueError("Response and scenario panels use different observations.")
    return response, scenario, arm_paths, int(offsets[1].size - 1)


def blend_with_white(hex_color: str, white_fraction: float = 0.58) -> str:
    """Return a precomposed opaque pastel of a categorical terminal color."""

    if not 0.0 <= white_fraction <= 1.0:
        raise ValueError("white_fraction must lie in [0, 1].")
    rgb = np.asarray(colors.to_rgb(hex_color), dtype=float)
    blended = (1.0 - white_fraction) * rgb + white_fraction * np.ones(3)
    return colors.to_hex(blended, keep_alpha=False)


def _stack_intervals(
    marginals: FloatArray,
    visual_order: Sequence[int],
    *,
    center: float = 0.47,
    flow_height: float = 0.70,
    gap: float = 0.018,
) -> Mapping[int, NodeInterval]:
    """Place nodes top-to-bottom using exact marginal-proportional heights."""

    if tuple(sorted(visual_order)) != tuple(range(marginals.size)):
        raise ValueError("Visual order must be a permutation of native state indices.")
    total_extent = flow_height * float(marginals.sum()) + gap * (marginals.size - 1)
    cursor = center + total_extent / 2.0
    intervals: Dict[int, NodeInterval] = {}
    for native_index in visual_order:
        height = flow_height * float(marginals[native_index])
        intervals[native_index] = NodeInterval(bottom=cursor - height, top=cursor)
        cursor -= height + gap
    return intervals


def _ribbon_patch(
    source_x: float,
    target_x: float,
    source_bottom: float,
    source_top: float,
    target_bottom: float,
    target_top: float,
    *,
    facecolor: str,
) -> PathPatch:
    """Create one opaque cubic-Bezier alluvial ribbon."""

    control_left = source_x + 0.43 * (target_x - source_x)
    control_right = target_x - 0.43 * (target_x - source_x)
    vertices = [
        (source_x, source_top),
        (control_left, source_top),
        (control_right, target_top),
        (target_x, target_top),
        (target_x, target_bottom),
        (control_right, target_bottom),
        (control_left, source_bottom),
        (source_x, source_bottom),
        (source_x, source_top),
    ]
    codes = [
        MplPath.MOVETO,
        MplPath.CURVE4,
        MplPath.CURVE4,
        MplPath.CURVE4,
        MplPath.LINETO,
        MplPath.CURVE4,
        MplPath.CURVE4,
        MplPath.CURVE4,
        MplPath.CLOSEPOLY,
    ]
    return PathPatch(
        MplPath(vertices, codes),
        facecolor=facecolor,
        edgecolor=WHITE,
        linewidth=0.28,
        alpha=1.0,
        joinstyle="round",
        zorder=1,
    )


def _draw_link_stage(
    ax: plt.Axes,
    panel: PanelData,
    source_k: int,
    target_k: int,
    source_intervals: Mapping[int, NodeInterval],
    target_intervals: Mapping[int, NodeInterval],
    source_x: float,
    target_x: float,
    *,
    flow_height: float,
) -> None:
    matrix = panel.links[(source_k, target_k)]
    source_order = panel.spec.visual_orders[source_k]
    target_order = panel.spec.visual_orders[target_k]
    source_cursor = {state: source_intervals[state].top for state in source_order}
    target_cursor = {state: target_intervals[state].top for state in target_order}
    for source in source_order:
        for target in target_order:
            height = flow_height * float(matrix[source, target])
            source_top = source_cursor[source]
            source_bottom = source_top - height
            target_top = target_cursor[target]
            target_bottom = target_top - height
            if target_k == 2:
                fill = NEUTRAL_LINK_1 if target_order.index(target) == 0 else NEUTRAL_LINK_2
            else:
                fill = blend_with_white(panel.spec.terminal_colors[target])
            ax.add_patch(
                _ribbon_patch(
                    source_x,
                    target_x,
                    source_bottom,
                    source_top,
                    target_bottom,
                    target_top,
                    facecolor=fill,
                )
            )
            source_cursor[source] = source_bottom
            target_cursor[target] = target_bottom
    for state in source_order:
        if not np.isclose(
            source_cursor[state], source_intervals[state].bottom, atol=2.0e-6
        ):
            raise ValueError("Rendered outgoing ribbons do not conserve source mass.")
    for state in target_order:
        if not np.isclose(
            target_cursor[state], target_intervals[state].bottom, atol=2.0e-6
        ):
            raise ValueError("Rendered incoming ribbons do not conserve target mass.")


def _format_percent(value: float) -> str:
    return f"{100.0 * value:.1f}%"


def _node_display_label(panel: PanelData, state_count: int, native_index: int) -> str:
    """Wrap only card-internal labels; source-data labels remain canonical."""

    label = panel.spec.labels[state_count][native_index]
    if state_count == 1:
        return label.replace(" ", "\n", 1)
    if state_count == 2 and panel.spec.key == "scenario":
        if native_index == 0:
            return "Lower-speed /\nshort-gap opening"
        return "Higher-speed /\nlong-gap closing"
    return label


def _draw_panel(ax: plt.Axes, panel: PanelData, panel_label: str) -> None:
    """Draw one complete 1-to-2-to-5 alluvial panel."""

    flow_height = 0.74
    x_left, x_middle, x_right = 0.015, 0.405, 0.785
    widths = {1: 0.120, 2: 0.140, 5: 0.026}
    gaps = {1: 0.0, 2: 0.030, 5: 0.014}
    intervals = {
        state_count: _stack_intervals(
            panel.marginals[state_count],
            panel.spec.visual_orders[state_count],
            center=0.455,
            flow_height=flow_height,
            gap=gaps[state_count],
        )
        for state_count in (1, 2, 5)
    }

    _draw_link_stage(
        ax,
        panel,
        1,
        2,
        intervals[1],
        intervals[2],
        x_left + widths[1],
        x_middle,
        flow_height=flow_height,
    )
    _draw_link_stage(
        ax,
        panel,
        2,
        5,
        intervals[2],
        intervals[5],
        x_middle + widths[2],
        x_right,
        flow_height=flow_height,
    )

    for state_count, x in ((1, x_left), (2, x_middle), (5, x_right)):
        for native_index in panel.spec.visual_orders[state_count]:
            interval = intervals[state_count][native_index]
            if state_count == 1:
                fill, edge, linewidth = NEUTRAL_DARK, NEUTRAL_DARK, 0.7
            elif state_count == 2:
                fill, edge, linewidth = NEUTRAL_NODE, NEUTRAL_MID, 0.75
            else:
                fill, edge, linewidth = panel.spec.terminal_colors[native_index], WHITE, 0.55
            ax.add_patch(
                Rectangle(
                    (x, interval.bottom),
                    widths[state_count],
                    interval.top - interval.bottom,
                    facecolor=fill,
                    edgecolor=edge,
                    linewidth=linewidth,
                    zorder=3,
                )
            )

            share = panel.marginals[state_count][native_index]
            label = _node_display_label(panel, state_count, native_index)
            y = 0.5 * (interval.bottom + interval.top)
            if state_count == 1:
                ax.text(
                    x + widths[state_count] / 2.0,
                    y,
                    f"{label}\n{_format_percent(float(share))}",
                    ha="center",
                    va="center",
                    fontsize=6.4,
                    color=WHITE,
                    fontweight="bold",
                    linespacing=1.25,
                    zorder=4,
                )
            elif state_count == 2:
                ax.text(
                    x + widths[state_count] / 2.0,
                    y,
                    f"{label}\n{_format_percent(float(share))}",
                    ha="center",
                    va="center",
                    fontsize=6.6 if panel.spec.key == "response" else 5.8,
                    color=NEUTRAL_DARK,
                    fontweight="bold",
                    linespacing=1.22,
                    zorder=4,
                )
            else:
                terminal_label = f"{panel.spec.terminal_prefix}{native_index + 1}  {label}"
                ax.text(
                    x + widths[state_count] + 0.012,
                    y,
                    terminal_label,
                    ha="left",
                    va="center",
                    fontsize=6.25,
                    color=NEUTRAL_DARK,
                    zorder=4,
                )
                ax.text(
                    1.205,
                    y,
                    _format_percent(float(share)),
                    ha="right",
                    va="center",
                    fontsize=6.25,
                    color=NEUTRAL_DARK,
                    fontweight="bold",
                    zorder=4,
                )

    for x, width, state_count in (
        (x_left, widths[1], 1),
        (x_middle, widths[2], 2),
        (x_right, widths[5], 5),
    ):
        ax.text(
            x + width / 2.0,
            RESOLUTION_LABEL_Y,
            f"${panel.spec.resolution_symbol} = {state_count}$",
            ha="center",
            va="bottom",
            fontsize=7.2,
            fontweight="bold",
            color=NEUTRAL_DARK,
        )
    ax.text(
        1.205,
        RESOLUTION_LABEL_Y,
        "Posterior share",
        ha="right",
        va="bottom",
        fontsize=6.1,
        color=NEUTRAL_MID,
    )

    ax.text(
        -0.065,
        PANEL_TITLE_Y,
        panel_label,
        ha="left",
        va="top",
        fontsize=9.0,
        fontweight="bold",
        color="#222222",
    )
    ax.text(
        -0.022,
        PANEL_TITLE_Y,
        panel.spec.title,
        ha="left",
        va="top",
        fontsize=8.5,
        fontweight="bold",
        color="#222222",
    )
    for arrow_start, arrow_end in ((0.18, 0.34), (0.59, 0.73)):
        ax.annotate(
            "",
            xy=(arrow_end, RESOLUTION_ARROW_Y),
            xytext=(arrow_start, RESOLUTION_ARROW_Y),
            arrowprops={"arrowstyle": "-|>", "color": NEUTRAL_MID, "lw": 0.65},
        )
    ax.set_xlim(-0.075, 1.225)
    ax.set_ylim(0.015, 0.995)
    ax.axis("off")


def plot_state_resolution_sankey(
    response: PanelData, scenario: PanelData
) -> plt.Figure:
    """Create the stacked response and scenario Sankey figure."""

    if response.observation_count != scenario.observation_count:
        raise ValueError("The two Sankey panels must share the same observations.")
    configure_matplotlib()
    fig, axes = plt.subplots(2, 1, figsize=(7.15, FIGURE_HEIGHT_IN))
    _draw_panel(axes[0], response, "a")
    _draw_panel(axes[1], scenario, "b")
    fig.text(
        0.5,
        0.012,
        "Ribbon width represents posterior-soft overlap across independently fitted models "
        f"({response.observation_count:,} episode-time observations).",
        ha="center",
        va="bottom",
        fontsize=6.2,
        color=NEUTRAL_MID,
    )
    fig.subplots_adjust(left=0.018, right=0.992, top=0.995, bottom=0.058, hspace=0.015)
    return fig


def _save_figure_bundle(
    fig: plt.Figure, output_base: Path
) -> Dict[str, Mapping[str, Any]]:
    outputs = {
        "svg": Path(f"{output_base}.svg"),
        "pdf": Path(f"{output_base}.pdf"),
        "png": Path(f"{output_base}.png"),
        "tiff": Path(f"{output_base}.tiff"),
    }
    conflicts = [str(path) for path in outputs.values() if path.exists()]
    if conflicts:
        raise FileExistsError(f"Refusing to overwrite Sankey figure outputs: {conflicts}")
    temporaries = {
        extension: path.with_name(f".{path.stem}.tmp{path.suffix}")
        for extension, path in outputs.items()
    }
    try:
        fig.savefig(temporaries["svg"], format="svg")
        fig.savefig(temporaries["pdf"], format="pdf")
        fig.savefig(temporaries["png"], format="png", dpi=600)
        fig.savefig(
            temporaries["tiff"],
            format="tiff",
            dpi=600,
            pil_kwargs={"compression": "tiff_lzw"},
        )
        for extension in EXPORT_FORMATS:
            os.replace(temporaries[extension], outputs[extension])
    finally:
        plt.close(fig)
    return {
        path.name: {"sha256": _sha256_file(path), "bytes": path.stat().st_size}
        for path in outputs.values()
    }


def source_rows(panel: PanelData) -> Tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Return auditable node and link rows in native and visual coordinates."""

    node_rows: list[dict[str, Any]] = []
    link_rows: list[dict[str, Any]] = []
    for state_count in (1, 2, 5):
        order = panel.spec.visual_orders[state_count]
        for visual_position, native_index in enumerate(order, start=1):
            color = (
                panel.spec.terminal_colors[native_index]
                if state_count == 5
                else (NEUTRAL_DARK if state_count == 1 else NEUTRAL_NODE)
            )
            node_rows.append(
                {
                    "panel": panel.spec.key,
                    "state_count": state_count,
                    "native_state_index": native_index + 1,
                    "visual_position_top_to_bottom": visual_position,
                    "state_label": panel.spec.labels[state_count][native_index],
                    "posterior_expected_share": float(panel.marginals[state_count][native_index]),
                    "percent_of_all_observations": 100.0
                    * float(panel.marginals[state_count][native_index]),
                    "display_color": color,
                }
            )
    for source_k, target_k in ((1, 2), (2, 5)):
        matrix = panel.links[(source_k, target_k)]
        source_order = panel.spec.visual_orders[source_k]
        target_order = panel.spec.visual_orders[target_k]
        for source in source_order:
            for target in target_order:
                link_rows.append(
                    {
                        "panel": panel.spec.key,
                        "source_state_count": source_k,
                        "target_state_count": target_k,
                        "source_native_state_index": source + 1,
                        "source_visual_position": source_order.index(source) + 1,
                        "source_label": panel.spec.labels[source_k][source],
                        "target_native_state_index": target + 1,
                        "target_visual_position": target_order.index(target) + 1,
                        "target_label": panel.spec.labels[target_k][target],
                        "posterior_soft_overlap": float(matrix[source, target]),
                        "percent_of_all_observations": 100.0 * float(matrix[source, target]),
                    }
                )
    return node_rows, link_rows


def _write_csv_new(
    path: Path, fieldnames: Sequence[str], rows: Sequence[Mapping[str, Any]]
) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite Sankey source data: {path}")
    temporary = path.with_name(f".{path.name}.tmp")
    if temporary.exists():
        raise FileExistsError(f"Stale Sankey source-data temporary exists: {temporary}")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def execute_plotting(postprocess_directory: Path, output_directory: Path) -> Path:
    """Generate a non-overwriting review candidate and its provenance bundle."""

    started = time.perf_counter()
    output_directory = output_directory.resolve()
    if output_directory.exists():
        raise FileExistsError(f"Refusing to reuse Sankey output directory: {output_directory}")
    output_directory.mkdir(parents=True)
    response, scenario, arm_paths, trajectory_count = load_canonical_panels(
        postprocess_directory
    )
    figure_artifacts = _save_figure_bundle(
        plot_state_resolution_sankey(response, scenario),
        output_directory / FIGURE_BASENAME,
    )

    node_rows: list[dict[str, Any]] = []
    link_rows: list[dict[str, Any]] = []
    for panel in (response, scenario):
        panel_nodes, panel_links = source_rows(panel)
        node_rows.extend(panel_nodes)
        link_rows.extend(panel_links)
    _write_csv_new(output_directory / SOURCE_NODES_FILENAME, tuple(node_rows[0]), node_rows)
    _write_csv_new(output_directory / SOURCE_LINKS_FILENAME, tuple(link_rows[0]), link_rows)
    source_artifacts = {
        filename: {
            "sha256": _sha256_file(output_directory / filename),
            "bytes": (output_directory / filename).stat().st_size,
        }
        for filename in (SOURCE_NODES_FILENAME, SOURCE_LINKS_FILENAME)
    }

    metadata = {
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "runtime_seconds": time.perf_counter() - started,
        "status": "CANDIDATE_FOR_AUTHOR_REVIEW",
        "figure_contract": {
            "core_conclusion": (
                "Across the same retained observations, increasing model resolution produces "
                "a structured but non-nested reorganization of posterior response and scenario "
                "assignments."
            ),
            "archetype": "two-panel quantitative alluvial diagram",
            "backend": "Python/matplotlib",
            "final_width_in": FIGURE_WIDTH_IN,
            "final_height_in": FIGURE_HEIGHT_IN,
            "minimum_source_font_pt": 5.8,
            "exports": list(EXPORT_FORMATS),
        },
        "observation_count": response.observation_count,
        "trajectory_count": trajectory_count,
        "models": ["B1S1", "B2S2", "B5S5"],
        "model_relationship": (
            "The three models were fitted independently to the same retained observations; "
            "the displayed sequence changes response and scenario resolution jointly."
        ),
        "ribbon_definition": (
            "w_ij = mean_n p(z_n^(K)=i | D) p(z_n^(K')=j | D)"
        ),
        "interpretation_boundary": (
            "Ribbons are cross-model posterior-soft overlaps. They are not temporal transitions "
            "and do not assert deterministic nesting or identity across fitted models."
        ),
        "data_exclusions": [],
        "link_threshold": None,
        "all_links_retained": True,
        "visual_order": {
            "response_K2_native_indices_top_to_bottom": [2, 1],
            "scenario_K2_native_indices_top_to_bottom": [1, 2],
            "K5_native_indices_top_to_bottom": [1, 2, 3, 4, 5],
        },
        "color_encoding": (
            "K=1 and K=1-to-2 links are neutral; K=2-to-5 links use opaque, "
            "white-precomposed target-state colors; K=5 nodes use saturated target colors."
        ),
        "input_artifacts": {
            str(state_count): {
                "path": str(path.resolve()),
                "sha256": _sha256_file(path),
            }
            for state_count, path in arm_paths.items()
        },
        "source_sha256": {Path(__file__).name: _sha256_file(Path(__file__))},
        "python": sys.version,
        "platform": platform.platform(),
        "matplotlib": matplotlib.__version__,
    }
    _write_json_new(output_directory / "figure_metadata.json", metadata)
    manifest = {
        "figure_artifacts": figure_artifacts,
        "source_data_artifacts": source_artifacts,
        "metadata": {
            "figure_metadata.json": {
                "sha256": _sha256_file(output_directory / "figure_metadata.json"),
                "bytes": (output_directory / "figure_metadata.json").stat().st_size,
            }
        },
    }
    _write_json_new(output_directory / "figure_artifact_manifest.json", manifest)
    _write_json_new(
        output_directory / "COMPLETED.json",
        {
            "status": "CANDIDATE_FOR_AUTHOR_REVIEW",
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            "output_directory": str(output_directory),
            "manifest_sha256": _sha256_file(
                output_directory / "figure_artifact_manifest.json"
            ),
        },
    )
    return output_directory


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--postprocess-directory", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    output = execute_plotting(args.postprocess_directory, args.output_directory)
    print(json.dumps({"status": "COMPLETED", "output_directory": str(output)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
