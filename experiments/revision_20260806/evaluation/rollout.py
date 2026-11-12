"""Numerically consistent leader-conditioned follower rollouts.

The manuscript's acceleration residual is an observation residual.  This module
therefore performs deterministic rollouts by default.  A stochastic acceleration
policy may be supplied explicitly, but no residual scaling or transition-matrix
modification is applied implicitly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Tuple

import numpy as np


AccelerationPolicy = Callable[[int, float, float, float, np.random.Generator], float]


@dataclass(frozen=True)
class RolloutResult:
    """One leader-conditioned rollout on a fixed sampling grid."""

    time: np.ndarray
    follower_position: np.ndarray
    follower_speed: np.ndarray
    gap: np.ndarray
    relative_speed: np.ndarray
    acceleration: np.ndarray
    collision_index: Optional[int]

    @property
    def collided(self) -> bool:
        """Return whether a non-positive net gap occurred."""

        return self.collision_index is not None


@dataclass(frozen=True)
class RolloutMetrics:
    """Trajectory errors and physical-validity indicators."""

    n_evaluated: int
    position_rmse_m: float
    speed_rmse_mps: float
    gap_rmse_m: float
    final_position_error_m: float
    final_speed_error_mps: float
    final_gap_error_m: float
    collided: bool
    collision_time_s: Optional[float]


def integrate_constant_acceleration(
    position_m: float,
    speed_mps: float,
    acceleration_mps2: float,
    dt_s: float,
) -> Tuple[float, float]:
    """Advance one step while handling a stop inside the integration interval.

    If constant deceleration would make speed negative before the end of the
    interval, the vehicle travels only until its stopping time and remains at
    rest thereafter.  This keeps the non-negative speed constraint consistent
    with displacement.
    """

    values = np.asarray([position_m, speed_mps, acceleration_mps2, dt_s], dtype=float)
    if not np.all(np.isfinite(values)):
        raise ValueError("Kinematic inputs must be finite.")
    if dt_s <= 0.0:
        raise ValueError("dt_s must be positive.")
    if speed_mps < 0.0:
        raise ValueError("speed_mps must be non-negative.")

    proposed_speed = speed_mps + acceleration_mps2 * dt_s
    if acceleration_mps2 < 0.0 and proposed_speed < 0.0:
        stopping_time = -speed_mps / acceleration_mps2
        displacement = (
            speed_mps * stopping_time
            + 0.5 * acceleration_mps2 * stopping_time**2
        )
        return float(position_m + displacement), 0.0

    displacement = speed_mps * dt_s + 0.5 * acceleration_mps2 * dt_s**2
    return float(position_m + displacement), float(max(proposed_speed, 0.0))


def leader_conditioned_rollout(
    *,
    leader_position_m: np.ndarray,
    leader_speed_mps: np.ndarray,
    follower_length_m: np.ndarray,
    initial_follower_position_m: float,
    initial_follower_speed_mps: float,
    dt_s: float,
    acceleration_policy: AccelerationPolicy,
    seed: int,
    stop_on_collision: bool = True,
) -> RolloutResult:
    """Roll out a follower against an observed leader trajectory.

    Positions follow the repository's transformed highD convention, where the
    net gap is ``leader_position - follower_position - follower_length``.
    ``acceleration_policy`` sees only the current simulated follower state and
    current observed leader state.  Any latent-state filtering must therefore
    also be causal inside that policy.
    """

    leader_position = np.asarray(leader_position_m, dtype=float)
    leader_speed = np.asarray(leader_speed_mps, dtype=float)
    follower_length = np.asarray(follower_length_m, dtype=float)
    if leader_position.ndim != 1:
        raise ValueError("leader_position_m must be one-dimensional.")
    if not (leader_position.shape == leader_speed.shape == follower_length.shape):
        raise ValueError("Leader arrays and follower_length_m must have equal shapes.")
    if leader_position.size < 2:
        raise ValueError("At least two time points are required for a rollout.")
    if not np.all(np.isfinite(np.column_stack((leader_position, leader_speed, follower_length)))):
        raise ValueError("Leader trajectory and vehicle length must be finite.")
    if np.any(follower_length <= 0.0):
        raise ValueError("follower_length_m must be strictly positive.")
    if dt_s <= 0.0:
        raise ValueError("dt_s must be positive.")

    n_steps = leader_position.size
    position = np.full(n_steps, np.nan, dtype=float)
    speed = np.full(n_steps, np.nan, dtype=float)
    gap = np.full(n_steps, np.nan, dtype=float)
    relative_speed = np.full(n_steps, np.nan, dtype=float)
    acceleration = np.full(n_steps - 1, np.nan, dtype=float)
    time = np.arange(n_steps, dtype=float) * dt_s

    position[0] = float(initial_follower_position_m)
    speed[0] = float(initial_follower_speed_mps)
    if speed[0] < 0.0:
        raise ValueError("initial_follower_speed_mps must be non-negative.")
    gap[0] = leader_position[0] - position[0] - follower_length[0]
    relative_speed[0] = speed[0] - leader_speed[0]

    collision_index: Optional[int] = 0 if gap[0] <= 0.0 else None
    rng = np.random.default_rng(seed)
    if collision_index is None:
        for step in range(n_steps - 1):
            proposed_acceleration = float(
                acceleration_policy(
                    step,
                    speed[step],
                    gap[step],
                    relative_speed[step],
                    rng,
                )
            )
            if not np.isfinite(proposed_acceleration):
                raise ValueError(f"Acceleration policy returned a non-finite value at step {step}.")
            acceleration[step] = proposed_acceleration
            position[step + 1], speed[step + 1] = integrate_constant_acceleration(
                position[step], speed[step], proposed_acceleration, dt_s
            )
            gap[step + 1] = (
                leader_position[step + 1]
                - position[step + 1]
                - follower_length[step + 1]
            )
            relative_speed[step + 1] = speed[step + 1] - leader_speed[step + 1]
            if gap[step + 1] <= 0.0:
                collision_index = step + 1
                if stop_on_collision:
                    break

    return RolloutResult(
        time=time,
        follower_position=position,
        follower_speed=speed,
        gap=gap,
        relative_speed=relative_speed,
        acceleration=acceleration,
        collision_index=collision_index,
    )


def _rmse(prediction: np.ndarray, observation: np.ndarray) -> float:
    difference = prediction - observation
    return float(np.sqrt(np.mean(difference**2)))


def summarize_rollout(
    rollout: RolloutResult,
    *,
    observed_follower_position_m: np.ndarray,
    observed_follower_speed_mps: np.ndarray,
    observed_gap_m: np.ndarray,
    horizon_s: Optional[float] = None,
) -> RolloutMetrics:
    """Compute errors through a fixed horizon or the last finite rollout point."""

    observed_position = np.asarray(observed_follower_position_m, dtype=float)
    observed_speed = np.asarray(observed_follower_speed_mps, dtype=float)
    observed_gap = np.asarray(observed_gap_m, dtype=float)
    if not (
        observed_position.shape
        == observed_speed.shape
        == observed_gap.shape
        == rollout.time.shape
    ):
        raise ValueError("Observed trajectories must match the rollout time grid.")
    if horizon_s is not None and horizon_s <= 0.0:
        raise ValueError("horizon_s must be positive when provided.")

    mask = (
        np.isfinite(rollout.follower_position)
        & np.isfinite(rollout.follower_speed)
        & np.isfinite(rollout.gap)
        & np.isfinite(observed_position)
        & np.isfinite(observed_speed)
        & np.isfinite(observed_gap)
    )
    if horizon_s is not None:
        mask &= rollout.time <= horizon_s + np.finfo(float).eps
    indices = np.flatnonzero(mask)
    if indices.size == 0:
        raise ValueError("No finite rollout points are available for evaluation.")
    final_index = int(indices[-1])

    collision_time = (
        None
        if rollout.collision_index is None
        else float(rollout.time[rollout.collision_index])
    )
    return RolloutMetrics(
        n_evaluated=int(indices.size),
        position_rmse_m=_rmse(rollout.follower_position[indices], observed_position[indices]),
        speed_rmse_mps=_rmse(rollout.follower_speed[indices], observed_speed[indices]),
        gap_rmse_m=_rmse(rollout.gap[indices], observed_gap[indices]),
        final_position_error_m=float(
            rollout.follower_position[final_index] - observed_position[final_index]
        ),
        final_speed_error_mps=float(
            rollout.follower_speed[final_index] - observed_speed[final_index]
        ),
        final_gap_error_m=float(rollout.gap[final_index] - observed_gap[final_index]),
        collided=rollout.collided,
        collision_time_s=collision_time,
    )

