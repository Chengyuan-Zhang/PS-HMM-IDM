"""Apply color-vision-accessible styling to an existing case-study figure.

This module changes visual encodings only. It never changes the plotted
posterior probabilities, observed data, or reconstructed model trajectories.
"""

from __future__ import annotations

from itertools import combinations
from typing import Any, Dict, Mapping, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.legend import Legend
from matplotlib.patches import Patch
from numpy.typing import NDArray

from .plot_case_study_original_style import RESPONSE_NAMES, cumulative_probabilities


plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans", "Liberation Sans"]
plt.rcParams["svg.fonttype"] = "none"


FloatArray = NDArray[np.float64]

LEGEND_ZORDER = 1000
STATE_EDGE_COLOR = "#70777D"
STATE_EDGE_WIDTH = 0.35
BOUNDARY_COLOR = "#FFFFFF"
BOUNDARY_WIDTH = 0.40

# These are fixed, precomposed display colors. They approximate the selected
# categorical bases mixed with white at an equivalent alpha of 0.38, but are
# exported as fully opaque colors so SVG, PDF, and PNG render identically.
RESPONSE_STATE_COLORS: Tuple[str, ...] = (
    "#9EC9E2",
    "#F6DB9E",
    "#ECCCDE",
    "#9EDACA",
    "#EFC29E",
)
CONTEXT_STATE_COLORS: Tuple[str, ...] = (
    "#B1ABD2",
    "#D2ECF9",
    "#B8DFD8",
    "#F2ECCB",
    "#ECC5CB",
)

LINE_COLORS: Mapping[str, str] = {
    "Pooled IDM": "#68737D",
    "Observed acceleration": "#15191D",
    "PS-HMM-IDM": "#D55E00",
    r"$v$": "#0072B2",
    r"$s$": "#009E73",
    r"$\Delta v$": "#D55E00",
}

CVD_MATRICES: Mapping[str, FloatArray] = {
    "protanopia": np.array(
        [
            [0.152286, 1.052583, -0.204868],
            [0.114503, 0.786281, 0.099216],
            [-0.003882, -0.048116, 1.051998],
        ],
        dtype=float,
    ),
    "deuteranopia": np.array(
        [
            [0.367322, 0.860646, -0.227968],
            [0.280085, 0.672501, 0.047413],
            [-0.011820, 0.042940, 0.968881],
        ],
        dtype=float,
    ),
    "tritanopia": np.array(
        [
            [1.255528, -0.076749, -0.178779],
            [-0.078411, 0.930809, 0.147602],
            [0.004733, 0.691367, 0.303900],
        ],
        dtype=float,
    ),
}


def _find_axis(figure: plt.Figure, ylabel: str, *, startswith: bool = False) -> plt.Axes:
    if startswith:
        matches = [axis for axis in figure.axes if axis.get_ylabel().startswith(ylabel)]
    else:
        matches = [axis for axis in figure.axes if axis.get_ylabel() == ylabel]
    if len(matches) != 1:
        raise ValueError(f"Expected one axis matching {ylabel!r}, found {len(matches)}.")
    return matches[0]


def _probability_axes(figure: plt.Figure) -> Tuple[plt.Axes, plt.Axes]:
    matches = [
        axis for axis in figure.axes if axis.get_ylabel() == "Posterior probability"
    ]
    if len(matches) != 2:
        raise ValueError(f"Expected two probability axes, found {len(matches)}.")
    upper, lower = sorted(matches, key=lambda axis: axis.get_position().y0, reverse=True)
    return upper, lower


def _line_by_label(figure: plt.Figure) -> Dict[str, Any]:
    found: Dict[str, Any] = {}
    for axis in figure.axes:
        for line in axis.lines:
            label = str(line.get_label())
            if label and not label.startswith("_"):
                if label in found:
                    raise ValueError(f"Duplicate labelled line {label!r}.")
                found[label] = line
    return found


def _snapshot_line_data(lines: Mapping[str, Any]) -> Dict[str, Tuple[FloatArray, FloatArray]]:
    return {
        label: (
            np.asarray(line.get_xdata(), dtype=float).copy(),
            np.asarray(line.get_ydata(), dtype=float).copy(),
        )
        for label, line in lines.items()
    }


def _assert_line_data_unchanged(
    before: Mapping[str, Tuple[FloatArray, FloatArray]], lines: Mapping[str, Any]
) -> None:
    if set(before) != set(lines):
        raise RuntimeError("Styling changed the set of labelled data lines.")
    for label, (x_before, y_before) in before.items():
        np.testing.assert_array_equal(np.asarray(lines[label].get_xdata()), x_before)
        np.testing.assert_array_equal(np.asarray(lines[label].get_ydata()), y_before)


def _remove_all_legends(figure: plt.Figure) -> None:
    for axis in figure.axes:
        for artist in list(axis.get_children()):
            if isinstance(artist, Legend):
                artist.remove()


def _style_legend(legend: Legend) -> Legend:
    legend.set_zorder(LEGEND_ZORDER)
    return legend


def _state_patches(colors: Sequence[str], labels: Sequence[str]) -> Tuple[Patch, ...]:
    return tuple(
        Patch(
            facecolor=color,
            edgecolor=STATE_EDGE_COLOR,
            linewidth=STATE_EDGE_WIDTH,
            label=label,
        )
        for color, label in zip(colors, labels)
    )


def _set_probability_fills(axis: plt.Axes, colors: Sequence[str]) -> None:
    collections = list(axis.collections)
    if len(collections) != len(colors):
        raise ValueError(
            f"Expected {len(colors)} state fills on a probability axis, found "
            f"{len(collections)}."
        )
    for collection, color in zip(collections, colors):
        collection.set_facecolor(color)
        collection.set_edgecolor("none")
        collection.set_linewidth(0.0)
        collection.set_alpha(1.0)


def _add_probability_boundaries(
    axis: plt.Axes,
    time_values: FloatArray,
    probability: FloatArray,
    *,
    prefix: str,
) -> int:
    cumulative = cumulative_probabilities(probability, probability.shape[1])
    for index in range(1, cumulative.shape[0] - 1):
        axis.plot(
            time_values,
            cumulative[index],
            color=BOUNDARY_COLOR,
            linewidth=BOUNDARY_WIDTH,
            alpha=1.0,
            solid_capstyle="butt",
            label=f"_{prefix}_state_boundary_{index}",
            zorder=1.2,
        )
    return cumulative.shape[0] - 2


def _variant_line_styles(variant: str) -> Mapping[str, Any]:
    if variant == "B_light":
        return {
            "Pooled IDM": (0, (4.0, 2.2)),
            "Observed acceleration": "-",
            "PS-HMM-IDM": "-",
            r"$v$": "-",
            r"$s$": (0, (5.0, 2.0)),
            r"$\Delta v$": (0, (3.2, 1.4, 0.9, 1.4)),
        }
    if variant == "B_strong":
        return {
            "Pooled IDM": (0, (1.3, 1.4)),
            "Observed acceleration": "-",
            "PS-HMM-IDM": (0, (5.5, 1.8)),
            r"$v$": "-",
            r"$s$": (0, (5.0, 2.0)),
            r"$\Delta v$": (0, (3.2, 1.4, 0.9, 1.4)),
        }
    raise ValueError("variant must be 'B_light' or 'B_strong'.")


def _rebuild_front_legends(
    figure: plt.Figure,
    lines: Mapping[str, Any],
    *,
    variant: str,
) -> Dict[str, Any]:
    response_axis = _find_axis(figure, "Acceleration", startswith=True)
    kinematics_axis = _find_axis(figure, r"$v,\Delta v$ (m/s)")
    gap_axis = _find_axis(figure, r"$s$ (m)")
    upper_probability, lower_probability = _probability_axes(figure)
    _remove_all_legends(figure)

    response_axis.set_zorder(max(response_axis.get_zorder(), upper_probability.get_zorder()) + 1)
    response_axis.patch.set_visible(False)
    gap_axis.set_zorder(
        max(
            gap_axis.get_zorder(),
            kinematics_axis.get_zorder(),
            lower_probability.get_zorder(),
        )
        + 1
    )
    gap_axis.patch.set_visible(False)

    curve_order = ("Pooled IDM", "Observed acceleration", "PS-HMM-IDM")
    lower_order = (r"$v$", r"$s$", r"$\Delta v$")
    handle_length = 2.3 if variant == "B_light" else 2.9
    curve_legend = _style_legend(
        response_axis.legend(
            handles=tuple(lines[label] for label in curve_order),
            labels=curve_order,
            loc="upper right",
            ncol=3,
            frameon=True,
            facecolor="white",
            edgecolor="#9DA5AB",
            framealpha=0.96,
            handlelength=handle_length,
            columnspacing=0.9,
            borderpad=0.35,
        )
    )
    response_axis.add_artist(curve_legend)
    response_labels = tuple(
        f"R{index + 1} {name}" for index, name in enumerate(RESPONSE_NAMES)
    )
    response_legend = _style_legend(
        response_axis.legend(
            handles=_state_patches(RESPONSE_STATE_COLORS, response_labels),
            labels=response_labels,
            loc="lower right",
            ncol=5,
            frameon=True,
            facecolor="white",
            edgecolor="#B5BCC1",
            framealpha=0.92,
            handlelength=1.2,
            handletextpad=0.35,
            columnspacing=0.7,
            borderpad=0.30,
        )
    )

    kinematic_legend = _style_legend(
        gap_axis.legend(
            handles=tuple(lines[label] for label in lower_order),
            labels=lower_order,
            loc="upper right",
            ncol=3,
            frameon=True,
            facecolor="white",
            edgecolor="#9DA5AB",
            framealpha=0.96,
            handlelength=2.6,
            columnspacing=0.9,
            borderpad=0.35,
        )
    )
    gap_axis.add_artist(kinematic_legend)
    context_labels = tuple(f"C{index + 1}" for index in range(5))
    context_legend = _style_legend(
        gap_axis.legend(
            handles=_state_patches(CONTEXT_STATE_COLORS, context_labels),
            labels=context_labels,
            loc="lower right",
            ncol=5,
            frameon=True,
            facecolor="white",
            edgecolor="#B5BCC1",
            framealpha=0.92,
            handlelength=1.2,
            handletextpad=0.35,
            columnspacing=0.8,
            borderpad=0.30,
        )
    )
    return {
        "upper_front_axis": response_axis.get_ylabel(),
        "lower_front_axis": gap_axis.get_ylabel(),
        "legend_zorder": LEGEND_ZORDER,
        "curve_legend_handlelength": handle_length,
        "upper_legends": [list(curve_order), list(response_labels)],
        "lower_legends": [list(lower_order), list(context_labels)],
        "legend_count": 4,
        "all_legends_frontmost": all(
            legend.get_zorder() == LEGEND_ZORDER
            for legend in (curve_legend, response_legend, kinematic_legend, context_legend)
        ),
    }


def apply_accessible_case_style(
    figure: plt.Figure,
    *,
    response_probability: FloatArray,
    context_probability: FloatArray,
    variant: str,
) -> Dict[str, Any]:
    """Apply the approved B-light or B-strong encoding without changing data."""

    response_values = np.asarray(response_probability, dtype=float)
    context_values = np.asarray(context_probability, dtype=float)
    if response_values.ndim != 2 or context_values.ndim != 2:
        raise ValueError("State probabilities must be two-dimensional arrays.")
    if response_values.shape != context_values.shape or response_values.shape[1] != 5:
        raise ValueError("Response and context probabilities must share shape (T, 5).")
    cumulative_probabilities(response_values, 5)
    cumulative_probabilities(context_values, 5)

    lines = _line_by_label(figure)
    required = set(LINE_COLORS)
    missing = required - set(lines)
    if missing:
        raise ValueError(f"Case figure lacks labelled lines {sorted(missing)}.")
    lines = {label: lines[label] for label in LINE_COLORS}
    before = _snapshot_line_data(lines)
    time_values = np.asarray(lines["PS-HMM-IDM"].get_xdata(), dtype=float)
    if time_values.shape != (response_values.shape[0],):
        raise ValueError("Probability arrays are not aligned with the case time axis.")

    upper_probability, lower_probability = _probability_axes(figure)
    _set_probability_fills(upper_probability, RESPONSE_STATE_COLORS)
    _set_probability_fills(lower_probability, CONTEXT_STATE_COLORS)
    response_boundaries = _add_probability_boundaries(
        upper_probability,
        time_values,
        response_values,
        prefix="response",
    )
    context_boundaries = _add_probability_boundaries(
        lower_probability,
        time_values,
        context_values,
        prefix="context",
    )

    styles = _variant_line_styles(variant)
    widths = {
        "Pooled IDM": 1.25,
        "Observed acceleration": 1.35,
        "PS-HMM-IDM": 1.65,
        r"$v$": 1.45,
        r"$s$": 1.45,
        r"$\Delta v$": 1.25,
    }
    for label, line in lines.items():
        line.set_color(LINE_COLORS[label])
        line.set_linestyle(styles[label])
        line.set_linewidth(widths[label])
        line.set_alpha(1.0)

    gap_axis = _find_axis(figure, r"$s$ (m)")
    gap_axis.spines["left"].set_color(LINE_COLORS[r"$s$"])
    gap_axis.tick_params(axis="y", colors=LINE_COLORS[r"$s$"], pad=2)
    gap_axis.set_ylabel(r"$s$ (m)", color=LINE_COLORS[r"$s$"], labelpad=2)
    legend_audit = _rebuild_front_legends(figure, lines, variant=variant)
    _assert_line_data_unchanged(before, lines)

    return {
        "variant": variant,
        "data_integrity": "PASS: all labelled x/y arrays are byte-for-byte unchanged",
        "state_fill_alpha": 1.0,
        "fixed_precomposed_state_colors": True,
        "response_state_colors": list(RESPONSE_STATE_COLORS),
        "context_state_colors": list(CONTEXT_STATE_COLORS),
        "line_colors": dict(LINE_COLORS),
        "line_style_design": {
            "B_light": "Main response remains solid; pooled response and lower-panel variables use redundant dashes.",
            "B_strong": "Both response models and all lower-panel variables use distinct redundant line patterns.",
        }[variant],
        "response_boundary_count": response_boundaries,
        "context_boundary_count": context_boundaries,
        "boundary_color": BOUNDARY_COLOR,
        "boundary_width_points": BOUNDARY_WIDTH,
        "legend_audit": legend_audit,
    }


def _hex_to_rgb(color: str) -> FloatArray:
    value = color.lstrip("#")
    if len(value) != 6:
        raise ValueError(f"Expected a six-digit hex color, received {color!r}.")
    return np.array([int(value[index : index + 2], 16) for index in (0, 2, 4)], dtype=float) / 255.0


def _srgb_to_linear(rgb: FloatArray) -> FloatArray:
    values = np.asarray(rgb, dtype=float)
    return np.where(values <= 0.04045, values / 12.92, ((values + 0.055) / 1.055) ** 2.4)


def _linear_to_srgb(rgb: FloatArray) -> FloatArray:
    values = np.clip(np.asarray(rgb, dtype=float), 0.0, 1.0)
    return np.where(values <= 0.0031308, 12.92 * values, 1.055 * values ** (1.0 / 2.4) - 0.055)


def simulate_color_vision(image: FloatArray, mode: str) -> FloatArray:
    """Return a deterministic diagnostic approximation for one vision mode."""

    values = np.asarray(image, dtype=float)
    if values.ndim < 2 or values.shape[-1] not in (3, 4):
        raise ValueError("Image must end with three RGB or four RGBA channels.")
    if not np.all(np.isfinite(values)) or np.any(values < 0.0) or np.any(values > 1.0):
        raise ValueError("Image values must be finite and lie in [0, 1].")
    rgb = values[..., :3]
    linear = _srgb_to_linear(rgb)
    if mode == "grayscale":
        luminance = np.sum(linear * np.array([0.2126, 0.7152, 0.0722]), axis=-1)
        transformed_linear = np.repeat(luminance[..., None], 3, axis=-1)
    elif mode in CVD_MATRICES:
        transformed_linear = linear @ CVD_MATRICES[mode].T
    else:
        raise ValueError(f"Unknown color-vision diagnostic mode {mode!r}.")
    transformed = _linear_to_srgb(transformed_linear)
    if values.shape[-1] == 4:
        transformed = np.concatenate((transformed, values[..., 3:4]), axis=-1)
    return np.clip(transformed, 0.0, 1.0)


def _rgb_to_lab(rgb: FloatArray) -> FloatArray:
    linear = _srgb_to_linear(np.asarray(rgb, dtype=float))
    xyz = linear @ np.array(
        [
            [0.4124564, 0.3575761, 0.1804375],
            [0.2126729, 0.7151522, 0.0721750],
            [0.0193339, 0.1191920, 0.9503041],
        ],
        dtype=float,
    ).T
    xyz = xyz / np.array([0.95047, 1.0, 1.08883], dtype=float)
    delta = 6.0 / 29.0
    f_xyz = np.where(xyz > delta**3, np.cbrt(xyz), xyz / (3.0 * delta**2) + 4.0 / 29.0)
    return np.column_stack(
        (
            116.0 * f_xyz[:, 1] - 16.0,
            500.0 * (f_xyz[:, 0] - f_xyz[:, 1]),
            200.0 * (f_xyz[:, 1] - f_xyz[:, 2]),
        )
    )


def palette_audit(colors: Sequence[str]) -> Dict[str, Any]:
    """Report normal/CVD pairwise Lab separation and luminance range."""

    rgb = np.vstack([_hex_to_rgb(color) for color in colors])
    modes = ("normal", "protanopia", "deuteranopia", "tritanopia", "grayscale")
    separation: Dict[str, Dict[str, Any]] = {}
    for mode in modes:
        transformed = rgb if mode == "normal" else simulate_color_vision(rgb, mode)
        lab = _rgb_to_lab(np.asarray(transformed)[..., :3])
        pairs = [
            {
                "indices_zero_based": [first, second],
                "delta_e_76": float(np.linalg.norm(lab[first] - lab[second])),
            }
            for first, second in combinations(range(len(colors)), 2)
        ]
        closest = min(pairs, key=lambda item: item["delta_e_76"])
        linear = _srgb_to_linear(np.asarray(transformed)[..., :3])
        luminance = linear @ np.array([0.2126, 0.7152, 0.0722], dtype=float)
        separation[mode] = {
            "minimum_pairwise_delta_e_76": closest["delta_e_76"],
            "closest_pair_indices_zero_based": closest["indices_zero_based"],
            "relative_luminance_minimum": float(luminance.min()),
            "relative_luminance_maximum": float(luminance.max()),
            "relative_luminance_range": float(np.ptp(luminance)),
        }
    return {"colors": list(colors), "diagnostic": separation}

