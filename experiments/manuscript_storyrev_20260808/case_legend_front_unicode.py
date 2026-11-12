"""Unicode-label compatibility for the frontmost case-study legends."""

from __future__ import annotations

from typing import Any, Dict

import matplotlib.pyplot as plt

from .case_legend_front import raise_case_legends_to_front as _raise_case_legends_to_front


def raise_case_legends_to_front(figure: plt.Figure) -> Dict[str, Any]:
    """Apply the layer fix while preserving the exported Unicode unit label."""

    matches = [
        axis for axis in figure.axes if axis.get_ylabel().startswith("Acceleration")
    ]
    if len(matches) != 1:
        raise ValueError(f"Expected one acceleration axis, found {len(matches)}.")
    response_axis = matches[0]
    original_label = response_axis.get_ylabel()
    response_axis.set_ylabel(r"Acceleration (m/s$^2$)")
    try:
        audit = dict(_raise_case_legends_to_front(figure))
    finally:
        response_axis.set_ylabel(original_label)
    audit["upper_front_axis"] = original_label
    audit["unicode_acceleration_label_preserved"] = original_label == "Acceleration (m/s²)"
    return audit
