"""Corrected Intelligent Driver Model acceleration function."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Union

import numpy as np
from numpy.typing import ArrayLike, NDArray


FloatOrArray = Union[float, NDArray[np.float64]]


@dataclass(frozen=True)
class CorrectedIDM:
    """Intelligent Driver Model with a non-negative dynamic desired gap.

    ``relative_speed`` follows the repository convention
    ``follower_speed - leader_speed``.  All physical parameters and inputs use
    SI units.  The class is deterministic; acceleration residuals belong in an
    observation model and are not injected here.

    By default a non-positive observed gap raises an error because it denotes a
    collision or invalid record.  A caller handling an already-recorded failure
    may opt into numerical clipping with ``clip_nonpositive_gap=True``.  The
    opt-in makes that modeling decision visible instead of silently hiding it.
    """

    desired_speed: float
    minimum_gap: float
    time_headway: float
    max_acceleration: float
    comfortable_deceleration: float
    acceleration_exponent: float = 4.0
    gap_floor: float = 1.0e-3

    def __post_init__(self) -> None:
        parameter_values = np.asarray(
            [
                self.desired_speed,
                self.minimum_gap,
                self.time_headway,
                self.max_acceleration,
                self.comfortable_deceleration,
                self.acceleration_exponent,
                self.gap_floor,
            ],
            dtype=float,
        )
        if not np.all(np.isfinite(parameter_values)):
            raise ValueError("All IDM parameters must be finite.")
        if np.any(parameter_values <= 0.0):
            raise ValueError("All IDM parameters must be strictly positive.")

    def desired_gap(
        self,
        speed: ArrayLike,
        relative_speed: ArrayLike,
    ) -> FloatOrArray:
        """Return the corrected desired net gap in metres.

        The dynamic contribution is clipped below at zero:

        ``s0 + max(0, v*T + v*dv/(2*sqrt(a*b)))``.
        """

        speed_array, relative_speed_array = np.broadcast_arrays(
            np.asarray(speed, dtype=float),
            np.asarray(relative_speed, dtype=float),
        )
        self._require_finite(speed_array, relative_speed_array)
        if np.any(speed_array < 0.0):
            raise ValueError("speed must be non-negative.")

        dynamic_gap = (
            speed_array * self.time_headway
            + speed_array
            * relative_speed_array
            / (
                2.0
                * np.sqrt(
                    self.max_acceleration * self.comfortable_deceleration
                )
            )
        )
        result = self.minimum_gap + np.maximum(0.0, dynamic_gap)
        return self._scalar_or_array(result)

    def acceleration(
        self,
        speed: ArrayLike,
        gap: ArrayLike,
        relative_speed: ArrayLike,
        *,
        clip_nonpositive_gap: bool = False,
    ) -> FloatOrArray:
        """Return IDM acceleration in metres per second squared.

        Parameters broadcast according to NumPy rules.  ``gap`` is the net
        bumper-to-bumper spacing and must be positive unless explicit clipping
        is requested by the caller.
        """

        speed_array, gap_array, relative_speed_array = np.broadcast_arrays(
            np.asarray(speed, dtype=float),
            np.asarray(gap, dtype=float),
            np.asarray(relative_speed, dtype=float),
        )
        self._require_finite(speed_array, gap_array, relative_speed_array)
        if np.any(speed_array < 0.0):
            raise ValueError("speed must be non-negative.")
        if np.any(gap_array <= 0.0) and not clip_nonpositive_gap:
            raise ValueError(
                "gap must be strictly positive; pass "
                "clip_nonpositive_gap=True only for explicit failure handling."
            )

        effective_gap = np.maximum(gap_array, self.gap_floor)
        desired_gap = np.asarray(
            self.desired_gap(speed_array, relative_speed_array), dtype=float
        )
        acceleration = self.max_acceleration * (
            1.0
            - (speed_array / self.desired_speed) ** self.acceleration_exponent
            - (desired_gap / effective_gap) ** 2
        )
        return self._scalar_or_array(acceleration)

    @staticmethod
    def _require_finite(*arrays: NDArray[np.float64]) -> None:
        if not all(np.all(np.isfinite(array)) for array in arrays):
            raise ValueError("IDM inputs must be finite.")

    @staticmethod
    def _scalar_or_array(value: NDArray[np.float64]) -> FloatOrArray:
        if value.ndim == 0:
            return float(value)
        return value.astype(float, copy=False)
