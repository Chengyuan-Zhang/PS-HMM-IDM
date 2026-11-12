"""Render every case-study legend above all line artists in its panel."""

from __future__ import annotations

from typing import Any, Dict, Iterable, Sequence

import matplotlib.pyplot as plt
from matplotlib.legend import Legend


LEGEND_ZORDER = 1000


def _find_axis(figure: plt.Figure, ylabel: str) -> plt.Axes:
    matches = [axis for axis in figure.axes if axis.get_ylabel() == ylabel]
    if len(matches) != 1:
        raise ValueError(f"Expected one axis labelled {ylabel!r}, found {len(matches)}.")
    return matches[0]


def _line_by_label(axes: Iterable[plt.Axes]) -> Dict[str, Any]:
    lines: Dict[str, Any] = {}
    for axis in axes:
        for line in axis.lines:
            label = str(line.get_label())
            if label and not label.startswith("_"):
                lines[label] = line
    return lines


def _remove_all_legends(figure: plt.Figure) -> None:
    for axis in figure.axes:
        for artist in list(axis.get_children()):
            if isinstance(artist, Legend):
                artist.remove()


def _style_legend(legend: Legend) -> Legend:
    legend.set_zorder(LEGEND_ZORDER)
    return legend


def _legend_labels(legend: Legend) -> Sequence[str]:
    return tuple(text.get_text() for text in legend.get_texts())


def raise_case_legends_to_front(figure: plt.Figure) -> Dict[str, Any]:
    """Rebuild all legends on the last-drawn twin axis of each panel.

    Matplotlib draws complete axes in layer order. A high-zorder legend on an
    earlier ``twinx`` axis can therefore still be crossed by a line belonging
    to a later twin axis. The upper legends are rebuilt on the response axis;
    the lower legends are rebuilt on the gap axis, which is the final twin axis
    in that panel. Their positions, entries, and styling remain unchanged.
    """

    response_axis = _find_axis(figure, r"Acceleration (m/s$^2$)")
    kinematics_axis = _find_axis(figure, r"$v,\Delta v$ (m/s)")
    gap_axis = _find_axis(figure, r"$s$ (m)")
    probability_axes = [
        axis for axis in figure.axes if axis.get_ylabel() == "Posterior probability"
    ]
    if len(probability_axes) != 2:
        raise ValueError(
            "Expected two posterior-probability axes, "
            f"found {len(probability_axes)}."
        )
    upper_probability, lower_probability = sorted(
        probability_axes, key=lambda axis: axis.get_position().y0, reverse=True
    )

    response_lines = _line_by_label((response_axis,))
    lower_lines = _line_by_label((kinematics_axis, gap_axis))
    required_response = ("Pooled IDM", "Observed acceleration", "PS-HMM-IDM")
    required_lower = (r"$v$", r"$s$", r"$\Delta v$")
    for label in required_response:
        if label not in response_lines:
            raise ValueError(f"Missing upper-panel line labelled {label!r}.")
    for label in required_lower:
        if label not in lower_lines:
            raise ValueError(f"Missing lower-panel line labelled {label!r}.")

    response_handles, response_labels = upper_probability.get_legend_handles_labels()
    context_handles, context_labels = lower_probability.get_legend_handles_labels()
    if len(response_handles) != 5 or len(context_handles) != 5:
        raise ValueError("Expected five response and five context probability handles.")

    _remove_all_legends(figure)

    response_axis.set_zorder(max(response_axis.get_zorder(), upper_probability.get_zorder()) + 1)
    response_axis.patch.set_visible(False)
    curve_legend = _style_legend(
        response_axis.legend(
            handles=tuple(response_lines[label] for label in required_response),
            labels=required_response,
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
    )
    response_axis.add_artist(curve_legend)
    response_legend = _style_legend(
        response_axis.legend(
            handles=response_handles,
            labels=response_labels,
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
    )

    gap_axis.set_zorder(
        max(
            gap_axis.get_zorder(),
            kinematics_axis.get_zorder(),
            lower_probability.get_zorder(),
        )
        + 1
    )
    gap_axis.patch.set_visible(False)
    kinematic_legend = _style_legend(
        gap_axis.legend(
            handles=tuple(lower_lines[label] for label in required_lower),
            labels=required_lower,
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
    )
    gap_axis.add_artist(kinematic_legend)
    context_legend = _style_legend(
        gap_axis.legend(
            handles=context_handles,
            labels=context_labels,
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
    )

    return {
        "upper_front_axis": response_axis.get_ylabel(),
        "lower_front_axis": gap_axis.get_ylabel(),
        "legend_zorder": LEGEND_ZORDER,
        "upper_legends": [
            list(_legend_labels(curve_legend)),
            list(_legend_labels(response_legend)),
        ],
        "lower_legends": [
            list(_legend_labels(kinematic_legend)),
            list(_legend_labels(context_legend)),
        ],
    }
