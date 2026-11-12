"""Corrected, testable model core for the 2026 manuscript revision.

The legacy implementation under :mod:`MCMC` is intentionally not imported
here.  This package provides a small numerical core whose state indexing,
transition priors, and latent-path sampling are explicit and independently
testable.
"""

from .constants import COVARIATE_ORDER
from .idm import CorrectedIDM
from .inference import (
    ForwardFilterResult,
    ffbs_sample_path,
    ffbs_sample_paths,
    log_forward_filter,
    log_forward_filter_sequences,
)
from .state_space import JointStateSpace
from .transitions import (
    ContextConditionedTransitions,
    FullJointTransitions,
    IndependentTransitions,
    build_context_conditioned_transition,
    build_full_joint_transition,
    build_independent_transition,
    count_context_conditioned_transitions,
    count_independent_transitions,
    count_initial_states,
    count_joint_transitions,
    context_conditioned_transition_parameter_count,
    dirichlet_prior_parameters,
    full_joint_transition_parameter_count,
    independent_transition_parameter_count,
    initial_distribution_parameter_count,
    sample_context_conditioned_transitions,
    sample_dirichlet_posterior_rows,
    sample_full_joint_transitions,
    sample_independent_transitions,
    sample_initial_distribution,
)

__all__ = [
    "COVARIATE_ORDER",
    "ContextConditionedTransitions",
    "CorrectedIDM",
    "ForwardFilterResult",
    "FullJointTransitions",
    "IndependentTransitions",
    "JointStateSpace",
    "build_context_conditioned_transition",
    "build_full_joint_transition",
    "build_independent_transition",
    "context_conditioned_transition_parameter_count",
    "count_context_conditioned_transitions",
    "count_independent_transitions",
    "count_initial_states",
    "count_joint_transitions",
    "dirichlet_prior_parameters",
    "ffbs_sample_path",
    "ffbs_sample_paths",
    "full_joint_transition_parameter_count",
    "independent_transition_parameter_count",
    "initial_distribution_parameter_count",
    "log_forward_filter",
    "log_forward_filter_sequences",
    "sample_context_conditioned_transitions",
    "sample_dirichlet_posterior_rows",
    "sample_full_joint_transitions",
    "sample_independent_transitions",
    "sample_initial_distribution",
]
