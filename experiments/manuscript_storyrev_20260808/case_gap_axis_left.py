"""Move the case-study gap axis to an outer-left position."""

from __future__ import annotations

from typing import Any, Dict

import matplotlib.pyplot as plt


GAP_SPINE_AXES_POSITION = -0.105
SUBPLOT_LEFT = 0.165
SUBPLOT_RIGHT = 0.935


def _find_axis(figure: plt.Figure, ylabel: str) -> plt.Axes:
    matches = [axis for axis in figure.axes if axis.get_ylabel() == ylabel]
    if len(matches) != 1:
        raise ValueError(f"Expected one axis labelled {ylabel!r}, found {len(matches)}.")
    return matches[0]


def move_gap_axis_to_outer_left(figure: plt.Figure) -> Dict[str, Any]:
    """Place the independent ``s`` scale outside the ``v, Δv`` left axis."""

    gap_axis = _find_axis(figure, r"$s$ (m)")
    kinematics_axis = _find_axis(figure, r"$v,\Delta v$ (m/s)")
    probability_axes = [
        axis for axis in figure.axes if axis.get_ylabel() == "Posterior probability"
    ]
    if len(probability_axes) != 2:
        raise ValueError(
            "Expected two posterior-probability axes, "
            f"found {len(probability_axes)}."
        )

    gap_axis.spines["right"].set_visible(False)
    gap_axis.spines["left"].set_visible(True)
    gap_axis.spines["left"].set_position(("axes", GAP_SPINE_AXES_POSITION))
    gap_axis.spines["left"].set_color("#4A9484")
    gap_axis.yaxis.set_ticks_position("left")
    gap_axis.yaxis.set_label_position("left")
    gap_axis.tick_params(
        axis="y",
        left=True,
        labelleft=True,
        right=False,
        labelright=False,
        colors="#4A9484",
        pad=2,
    )
    gap_axis.set_ylabel(r"$s$ (m)", color="#4A9484", labelpad=2)

    kinematics_axis.spines["left"].set_position(("axes", 0.0))
    kinematics_axis.yaxis.set_ticks_position("left")
    kinematics_axis.yaxis.set_label_position("left")
    figure.subplots_adjust(
        left=SUBPLOT_LEFT,
        right=SUBPLOT_RIGHT,
        top=0.975,
        bottom=0.105,
        hspace=0.14,
    )

    return {
        "gap_axis_side": "outer left",
        "gap_spine_axes_position": GAP_SPINE_AXES_POSITION,
        "kinematics_axis_side": "inner left",
        "posterior_probability_axis_side": "right",
        "subplot_left": figure.subplotpars.left,
        "subplot_right": figure.subplotpars.right,
        "gap_right_spine_visible": gap_axis.spines["right"].get_visible(),
        "gap_left_spine_visible": gap_axis.spines["left"].get_visible(),
    }
