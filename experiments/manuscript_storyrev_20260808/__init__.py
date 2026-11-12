"""Legacy-faithful exact reconstruction for the manuscript story revision.

This package is deliberately isolated from the historical :mod:`MCMC`
implementation and from the alternative model developed under
``revision_20260806``.  It reuses that revision's audited numerical kernels
for exact FFBS and conjugate updates while preserving the historical,
untruncated IDM emission used by the manuscript experiments.
"""

from .sampler import (
    ReconstructionConfig,
    ReconstructionResult,
    ReconstructionState,
    initialize_state,
    run_chain,
)
from .target import (
    COVARIATE_ORDER,
    ContextStandardizer,
    StoryTrajectory,
    original_idm_acceleration,
    scientific_target_fingerprint,
)

__all__ = [
    "COVARIATE_ORDER",
    "ContextStandardizer",
    "ReconstructionConfig",
    "ReconstructionResult",
    "ReconstructionState",
    "StoryTrajectory",
    "initialize_state",
    "original_idm_acceleration",
    "run_chain",
    "scientific_target_fingerprint",
]
