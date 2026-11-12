"""Shared numerical conventions for the revision model."""

from typing import Tuple


# ``relative_speed`` is follower speed minus leader speed.  Keeping this order
# in one importable constant prevents the historical [v, s, dv] / [v, dv, s]
# mismatch between likelihood, standardization, and plotting code.
COVARIATE_ORDER: Tuple[str, str, str] = (
    "speed",
    "gap",
    "relative_speed",
)
