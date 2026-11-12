"""Joint behavior-scenario state indexing."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
from numpy.typing import ArrayLike, NDArray


@dataclass(frozen=True)
class JointStateSpace:
    """Cartesian state space with behavior-major indexing.

    The scenario coordinate varies fastest:

    ``joint_state = behavior_state * num_scenario_states + scenario_state``.

    This convention is used by transition matrices, emission likelihoods, and
    posterior paths throughout the revision code.
    """

    num_behavior_states: int
    num_scenario_states: int

    def __post_init__(self) -> None:
        if not isinstance(self.num_behavior_states, (int, np.integer)) or not isinstance(
            self.num_scenario_states, (int, np.integer)
        ):
            raise ValueError("State counts must be integers.")
        if self.num_behavior_states <= 0 or self.num_scenario_states <= 0:
            raise ValueError("State counts must be positive integers.")

    @property
    def num_joint_states(self) -> int:
        return self.num_behavior_states * self.num_scenario_states

    def encode(self, behavior_state: int, scenario_state: int) -> int:
        """Encode one behavior-scenario pair as a joint-state integer."""

        self._validate_behavior_state(behavior_state)
        self._validate_scenario_state(scenario_state)
        return behavior_state * self.num_scenario_states + scenario_state

    def decode(self, joint_state: int) -> Tuple[int, int]:
        """Decode one joint-state integer as ``(behavior, scenario)``."""

        self._validate_joint_state(joint_state)
        behavior_state, scenario_state = divmod(
            int(joint_state), self.num_scenario_states
        )
        return behavior_state, scenario_state

    def encode_array(
        self,
        behavior_states: ArrayLike,
        scenario_states: ArrayLike,
    ) -> NDArray[np.int64]:
        """Vectorized version of :meth:`encode`."""

        behavior, scenario = np.broadcast_arrays(
            np.asarray(behavior_states), np.asarray(scenario_states)
        )
        if not self._is_integer_array(behavior) or not self._is_integer_array(scenario):
            raise ValueError("State arrays must contain integer values.")
        behavior = behavior.astype(np.int64, copy=False)
        scenario = scenario.astype(np.int64, copy=False)
        if np.any((behavior < 0) | (behavior >= self.num_behavior_states)):
            raise ValueError("A behavior state is out of range.")
        if np.any((scenario < 0) | (scenario >= self.num_scenario_states)):
            raise ValueError("A scenario state is out of range.")
        return behavior * self.num_scenario_states + scenario

    def decode_array(
        self,
        joint_states: ArrayLike,
    ) -> Tuple[NDArray[np.int64], NDArray[np.int64]]:
        """Vectorized version of :meth:`decode`."""

        joint = np.asarray(joint_states)
        if not self._is_integer_array(joint):
            raise ValueError("Joint-state arrays must contain integer values.")
        joint = joint.astype(np.int64, copy=False)
        if np.any((joint < 0) | (joint >= self.num_joint_states)):
            raise ValueError("A joint state is out of range.")
        return (
            joint // self.num_scenario_states,
            joint % self.num_scenario_states,
        )

    def _validate_behavior_state(self, state: int) -> None:
        if not isinstance(state, (int, np.integer)) or not (
            0 <= int(state) < self.num_behavior_states
        ):
            raise ValueError("behavior_state is out of range.")

    def _validate_scenario_state(self, state: int) -> None:
        if not isinstance(state, (int, np.integer)) or not (
            0 <= int(state) < self.num_scenario_states
        ):
            raise ValueError("scenario_state is out of range.")

    def _validate_joint_state(self, state: int) -> None:
        if not isinstance(state, (int, np.integer)) or not (
            0 <= int(state) < self.num_joint_states
        ):
            raise ValueError("joint_state is out of range.")

    @staticmethod
    def _is_integer_array(array: NDArray[np.generic]) -> bool:
        if np.issubdtype(array.dtype, np.integer):
            return True
        if not np.issubdtype(array.dtype, np.floating):
            return False
        return bool(np.all(np.isfinite(array)) and np.all(array == np.floor(array)))
