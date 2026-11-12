"""Ownership maps for reduced IDM parameter layouts.

The emission model always receives one physical five-parameter IDM vector per
behavior state.  Some model specifications, however, assign selected
parameters to a single shared owner.  This module makes that distinction
explicit: the authoritative state is a vector of unique log-parameter owners,
while the expanded ``(behavior, parameter)`` array is a deterministic view.

No method in this module silently projects inconsistent shared values.  Such a
projection is necessarily lossy and belongs in an explicitly documented
initialization procedure, never in a recurrent Markov-chain transition.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Tuple, Union

import numpy as np
from numpy.typing import ArrayLike, NDArray


FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]
BoolArray = NDArray[np.bool_]
IDMOwnershipName = Literal[
    "behavior_specific", "shared_s0", "shared_v0_s0"
]

IDM_PARAMETER_NAMES: Tuple[str, ...] = ("v0", "s0", "T", "a", "b")
IDM_OWNERSHIP_NAMES: Tuple[IDMOwnershipName, ...] = (
    "behavior_specific",
    "shared_s0",
    "shared_v0_s0",
)


@dataclass(frozen=True)
class IDMOwnershipLayout:
    """Immutable map from expanded IDM cells to unique log-parameter owners.

    ``owner_index[b, j]`` identifies the unique owner used by behavior ``b``
    and IDM parameter ``j``.  Shared owners have
    ``owner_behavior_index == -1``.  Owner order is deterministic: shared
    parameters first in IDM parameter order, followed by behavior-specific
    parameters in behavior-major, parameter-minor order.
    """

    name: IDMOwnershipName
    num_behavior_states: int
    owner_index: IntArray
    owner_parameter_index: IntArray
    owner_behavior_index: IntArray
    shared_parameter_mask: BoolArray

    def __post_init__(self) -> None:
        if self.name not in IDM_OWNERSHIP_NAMES:
            raise ValueError(f"Unknown IDM ownership layout: {self.name!r}.")
        if (
            isinstance(self.num_behavior_states, (bool, np.bool_))
            or not isinstance(self.num_behavior_states, (int, np.integer))
            or self.num_behavior_states <= 0
        ):
            raise ValueError("num_behavior_states must be a positive integer.")

        behavior_count = int(self.num_behavior_states)
        owner_index = np.asarray(self.owner_index, dtype=np.int64)
        owner_parameter = np.asarray(
            self.owner_parameter_index, dtype=np.int64
        )
        owner_behavior = np.asarray(
            self.owner_behavior_index, dtype=np.int64
        )
        shared_mask = np.asarray(self.shared_parameter_mask, dtype=bool)
        if owner_index.shape != (behavior_count, 5):
            raise ValueError(
                "owner_index must have shape "
                f"({behavior_count}, 5)."
            )
        if owner_parameter.ndim != 1 or owner_parameter.size == 0:
            raise ValueError("owner_parameter_index must be non-empty and one-dimensional.")
        owner_count = int(owner_parameter.size)
        if owner_behavior.shape != (owner_count,):
            raise ValueError(
                "owner_behavior_index must match owner_parameter_index."
            )
        if shared_mask.shape != (5,):
            raise ValueError("shared_parameter_mask must have shape (5,).")
        if np.any(owner_index < 0) or np.any(owner_index >= owner_count):
            raise ValueError("owner_index contains an invalid owner identifier.")
        if not np.array_equal(np.unique(owner_index), np.arange(owner_count)):
            raise ValueError("Every owner must occur in owner_index exactly by identity.")
        if np.any(owner_parameter < 0) or np.any(owner_parameter >= 5):
            raise ValueError("owner_parameter_index contains an invalid IDM parameter.")
        if np.any(owner_behavior < -1) or np.any(owner_behavior >= behavior_count):
            raise ValueError("owner_behavior_index contains an invalid behavior.")

        for owner in range(owner_count):
            cells = np.argwhere(owner_index == owner)
            parameter_indices = cells[:, 1]
            if not np.all(parameter_indices == owner_parameter[owner]):
                raise ValueError("An owner cannot span different IDM parameters.")
            expected_behavior = int(owner_behavior[owner])
            if expected_behavior == -1:
                if cells.shape[0] != behavior_count:
                    raise ValueError(
                        "A shared owner must occur once for every behavior."
                    )
                if not shared_mask[owner_parameter[owner]]:
                    raise ValueError("Shared-owner metadata is inconsistent.")
            else:
                if cells.shape != (1, 2) or cells[0, 0] != expected_behavior:
                    raise ValueError(
                        "A behavior-specific owner must occur in exactly one behavior."
                    )
                if shared_mask[owner_parameter[owner]]:
                    raise ValueError("Behavior-owner metadata is inconsistent.")

        for parameter_index in range(5):
            owners = np.unique(owner_index[:, parameter_index])
            expected = 1 if shared_mask[parameter_index] else behavior_count
            if owners.size != expected:
                raise ValueError("shared_parameter_mask disagrees with owner_index.")

        owner_index = owner_index.copy()
        owner_parameter = owner_parameter.copy()
        owner_behavior = owner_behavior.copy()
        shared_mask = shared_mask.copy()
        owner_index.setflags(write=False)
        owner_parameter.setflags(write=False)
        owner_behavior.setflags(write=False)
        shared_mask.setflags(write=False)
        object.__setattr__(self, "num_behavior_states", behavior_count)
        object.__setattr__(self, "owner_index", owner_index)
        object.__setattr__(self, "owner_parameter_index", owner_parameter)
        object.__setattr__(self, "owner_behavior_index", owner_behavior)
        object.__setattr__(self, "shared_parameter_mask", shared_mask)

    @property
    def num_owned_parameters(self) -> int:
        """Number of unique authoritative log parameters."""

        return int(self.owner_parameter_index.size)

    def affected_behaviors(self, owner: int) -> IntArray:
        """Return behavior indices whose expanded row depends on ``owner``."""

        owner_index = _validate_owner(owner, self.num_owned_parameters)
        affected = np.flatnonzero(
            np.any(self.owner_index == owner_index, axis=1)
        ).astype(np.int64)
        return affected

    def expand_log(self, owned_log_parameters: ArrayLike) -> FloatArray:
        """Expand unique owner values to the emission-facing log-IDM array."""

        owned = self._validate_owned_log(owned_log_parameters)
        return np.asarray(owned[self.owner_index], dtype=float)

    def decode_physical(self, owned_log_parameters: ArrayLike) -> FloatArray:
        """Decode unique owner values to expanded positive physical IDM values."""

        expanded_log = self.expand_log(owned_log_parameters)
        with np.errstate(over="ignore", under="ignore", invalid="ignore"):
            physical = np.exp(expanded_log)
        if not np.all(np.isfinite(physical)) or np.any(physical <= 0.0):
            raise ValueError(
                "owned_log_parameters do not decode to finite positive values."
            )
        return np.asarray(physical, dtype=float)

    def collapse_log(self, expanded_log_parameters: ArrayLike) -> FloatArray:
        """Collapse an exactly ownership-consistent expanded log-IDM array.

        Shared cells must be bitwise equal.  Inconsistent values raise instead
        of being averaged, because averaging would be an implicit projection.
        """

        expanded = np.asarray(expanded_log_parameters, dtype=float)
        expected = (self.num_behavior_states, 5)
        if expanded.shape != expected or not np.all(np.isfinite(expanded)):
            raise ValueError(
                "expanded_log_parameters must be finite with shape "
                f"{expected}."
            )
        owned = np.empty(self.num_owned_parameters, dtype=float)
        for owner in range(self.num_owned_parameters):
            values = expanded[self.owner_index == owner]
            if values.size == 0:
                raise RuntimeError("The ownership map contains an unused owner.")
            if not np.all(values == values[0]):
                raise ValueError(
                    "Expanded log parameters are inconsistent for shared owner "
                    f"{owner}."
                )
            owned[owner] = values[0]
        return owned

    def collapse_physical(self, expanded_parameters: ArrayLike) -> FloatArray:
        """Return unique log owners from an exactly consistent physical array."""

        parameters = np.asarray(expanded_parameters, dtype=float)
        expected = (self.num_behavior_states, 5)
        if (
            parameters.shape != expected
            or not np.all(np.isfinite(parameters))
            or np.any(parameters <= 0.0)
        ):
            raise ValueError(
                "expanded_parameters must be finite and positive with shape "
                f"{expected}."
            )
        return self.collapse_log(np.log(parameters))

    def _validate_owned_log(self, owned_log_parameters: ArrayLike) -> FloatArray:
        owned = np.asarray(owned_log_parameters, dtype=float)
        expected = (self.num_owned_parameters,)
        if owned.shape != expected or not np.all(np.isfinite(owned)):
            raise ValueError(
                "owned_log_parameters must be finite with shape "
                f"{expected}."
            )
        return owned


def build_idm_ownership_layout(
    name: Union[IDMOwnershipName, str], num_behavior_states: int
) -> IDMOwnershipLayout:
    """Construct one of the three supported deterministic ownership layouts."""

    if name not in IDM_OWNERSHIP_NAMES:
        raise ValueError(
            f"Unknown IDM ownership layout {name!r}; expected one of "
            f"{IDM_OWNERSHIP_NAMES}."
        )
    if (
        isinstance(num_behavior_states, (bool, np.bool_))
        or not isinstance(num_behavior_states, (int, np.integer))
        or num_behavior_states <= 0
    ):
        raise ValueError("num_behavior_states must be a positive integer.")
    behavior_count = int(num_behavior_states)
    shared_indices = {
        "behavior_specific": (),
        "shared_s0": (1,),
        "shared_v0_s0": (0, 1),
    }[name]
    shared_mask = np.zeros(5, dtype=bool)
    shared_mask[list(shared_indices)] = True

    owner_index = np.empty((behavior_count, 5), dtype=np.int64)
    owner_parameter: list[int] = []
    owner_behavior: list[int] = []
    owner = 0
    for parameter_index in shared_indices:
        owner_index[:, parameter_index] = owner
        owner_parameter.append(parameter_index)
        owner_behavior.append(-1)
        owner += 1
    for behavior in range(behavior_count):
        for parameter_index in range(5):
            if shared_mask[parameter_index]:
                continue
            owner_index[behavior, parameter_index] = owner
            owner_parameter.append(parameter_index)
            owner_behavior.append(behavior)
            owner += 1

    return IDMOwnershipLayout(
        name=name,  # type: ignore[arg-type]
        num_behavior_states=behavior_count,
        owner_index=owner_index,
        owner_parameter_index=np.asarray(owner_parameter, dtype=np.int64),
        owner_behavior_index=np.asarray(owner_behavior, dtype=np.int64),
        shared_parameter_mask=shared_mask,
    )


def _validate_owner(owner: int, owner_count: int) -> int:
    if (
        isinstance(owner, (bool, np.bool_))
        or not isinstance(owner, (int, np.integer))
        or owner < 0
        or owner >= owner_count
    ):
        raise IndexError(f"owner must be an integer in [0, {owner_count}).")
    return int(owner)


__all__ = [
    "IDMOwnershipLayout",
    "IDMOwnershipName",
    "IDM_OWNERSHIP_NAMES",
    "IDM_PARAMETER_NAMES",
    "build_idm_ownership_layout",
]
