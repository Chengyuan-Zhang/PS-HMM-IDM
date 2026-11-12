"""Exact-FFBS Gibbs sampler with explicit IDM parameter-transition kernels."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Dict, Literal, Mapping, Optional, Sequence, Tuple

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ..model import (
    ContextConditionedTransitions,
    FullJointTransitions,
    IndependentTransitions,
    JointStateSpace,
    ffbs_sample_paths,
    log_forward_filter_sequences,
    sample_context_conditioned_transitions,
    sample_dirichlet_posterior_rows,
    sample_full_joint_transitions,
    sample_independent_transitions,
    sample_initial_distribution,
)
from .emissions import emission_log_probability_sequences
from .idm_block import (
    IDMBlockMetric,
    build_fixed_idm_block_metric,
    sample_idm_parameters_block_mh,
)
from .idm_owned_slice import (
    OwnedLogIDMSliceAudit,
    decode_supported_owned_log,
    full_bracket_owned_log_idm_slice_update,
)
from .idm_ownership import (
    IDM_OWNERSHIP_NAMES,
    IDMOwnershipLayout,
    build_idm_ownership_layout,
)
from .structures import (
    SamplerConfig,
    SamplerPriors,
    SamplerState,
    Trajectory,
    TransitionParameters,
)
from .updates import (
    sample_idm_parameters_mh,
    sample_inverse_gamma,
    sample_log_normal_wishart,
    sample_normal_inverse_wishart,
    sample_residual_variances,
    sample_scenario_emissions,
    log_normal_wishart_posterior,
    normal_inverse_wishart_posterior,
)


FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]
BoolArray = NDArray[np.bool_]
IDMParameterKernelName = Literal[
    "coordinate_random_walk", "owned_full_bracket_slice"
]
IDM_PARAMETER_KERNEL_NAMES: Tuple[IDMParameterKernelName, ...] = (
    "coordinate_random_walk",
    "owned_full_bracket_slice",
)
_IDM_PARAMETER_KERNEL_CODES = {
    "coordinate_random_walk": 0,
    "owned_full_bracket_slice": 1,
}
_IDM_OWNERSHIP_LAYOUT_CODES = {
    name: index for index, name in enumerate(IDM_OWNERSHIP_NAMES)
}


@dataclass(frozen=True)
class IDMParameterKernelConfig:
    """Select the IDM transition kernel without changing its target model.

    The end-to-end HMM currently supports only the original
    behavior-specific ownership layout.  Reduced ownership layouts are
    intentionally rejected here because they define different statistical
    models and require their own initialization and interpretation protocol.
    """

    name: IDMParameterKernelName = "coordinate_random_walk"
    ownership_layout: str = "behavior_specific"
    max_bracket_shrinks: int = 10_000

    def __post_init__(self) -> None:
        if self.name not in IDM_PARAMETER_KERNEL_NAMES:
            raise ValueError(
                f"Unknown IDM parameter kernel {self.name!r}; expected one of "
                f"{IDM_PARAMETER_KERNEL_NAMES}."
            )
        if self.ownership_layout not in IDM_OWNERSHIP_NAMES:
            raise ValueError(
                f"Unknown IDM ownership layout {self.ownership_layout!r}; "
                f"expected one of {IDM_OWNERSHIP_NAMES}."
            )
        if self.ownership_layout != "behavior_specific":
            raise ValueError(
                "The end-to-end HMM sampler supports only the "
                "behavior_specific IDM ownership layout; reduced layouts "
                "change the statistical model and fail closed here."
            )
        if (
            isinstance(self.max_bracket_shrinks, (bool, np.bool_))
            or not isinstance(self.max_bracket_shrinks, (int, np.integer))
            or self.max_bracket_shrinks < 0
        ):
            raise ValueError(
                "max_bracket_shrinks must be a non-negative integer."
            )
        object.__setattr__(
            self, "max_bracket_shrinks", int(self.max_bracket_shrinks)
        )

    def as_dict(self) -> Dict[str, object]:
        """Return explicit JSON-compatible provenance for this kernel."""

        return {
            "name": self.name,
            "implementation": (
                "experiments.revision_20260806.inference.idm_owned_slice."
                "full_bracket_owned_log_idm_slice_update"
                if self.name == "owned_full_bracket_slice"
                else (
                    "experiments.revision_20260806.inference.updates."
                    "sample_idm_parameters_mh"
                )
            ),
            "ownership_layout": self.ownership_layout,
            "authoritative_parameterization": (
                "unique_owned_log_parameters"
                if self.name == "owned_full_bracket_slice"
                else "expanded_physical_parameters"
            ),
            "global_support_brackets": self.name == "owned_full_bracket_slice",
            "max_bracket_shrinks": (
                self.max_bracket_shrinks
                if self.name == "owned_full_bracket_slice"
                else None
            ),
            "kernel_code": _IDM_PARAMETER_KERNEL_CODES[self.name],
            "ownership_layout_code": _IDM_OWNERSHIP_LAYOUT_CODES[
                self.ownership_layout
            ],
            "statistical_target_changed": False,
        }


@dataclass(frozen=True)
class ChainSamples:
    idm_parameters: FloatArray
    residual_variance: FloatArray
    scenario_mean: FloatArray
    scenario_covariance: FloatArray
    initial_probability: FloatArray
    joint_transition: FloatArray
    transition_components: Mapping[str, FloatArray]
    latent_paths: Optional[Tuple[IntArray, ...]]
    owned_log_parameters: Optional[FloatArray] = None
    idm_hierarchy_mean: Optional[FloatArray] = None
    idm_hierarchy_precision: Optional[FloatArray] = None

    def as_npz_payload(self) -> Dict[str, FloatArray]:
        """Return dense arrays suitable for ``np.savez_compressed``."""

        payload: Dict[str, FloatArray] = {
            "idm_parameters": self.idm_parameters,
            "residual_variance": self.residual_variance,
            "scenario_mean": self.scenario_mean,
            "scenario_covariance": self.scenario_covariance,
            "initial_probability": self.initial_probability,
            "joint_transition": self.joint_transition,
        }
        payload.update(
            {f"transition_{key}": value for key, value in self.transition_components.items()}
        )
        if self.latent_paths is not None:
            for trajectory_index, paths in enumerate(self.latent_paths):
                payload[f"latent_path_{trajectory_index}"] = paths
        if self.owned_log_parameters is not None:
            payload["owned_log_parameters"] = self.owned_log_parameters
        if (self.idm_hierarchy_mean is None) != (
            self.idm_hierarchy_precision is None
        ):
            raise ValueError(
                "IDM hierarchy mean and precision samples must be present together."
            )
        if self.idm_hierarchy_mean is not None:
            payload["idm_hierarchy_mean"] = self.idm_hierarchy_mean
            payload["idm_hierarchy_precision"] = (
                self.idm_hierarchy_precision
            )
        return payload


@dataclass(frozen=True)
class ChainDiagnostics:
    idm_parameter_kernel: str
    idm_ownership_layout: str
    coordinate_mh_accepted: NDArray[np.bool_]
    marginal_log_likelihood: FloatArray
    occupancy: IntArray
    proposal_standard_deviation_history: FloatArray
    final_proposal_standard_deviation: FloatArray
    proposal_adaptation_gain: FloatArray
    block_mh_accepted: NDArray[np.bool_]
    block_scale_history: FloatArray
    final_block_scale: FloatArray
    block_adaptation_gain: FloatArray
    block_metric_expansion_log_parameters: FloatArray
    block_metric_precision: FloatArray
    block_metric_covariance: FloatArray
    block_metric_cholesky: FloatArray
    block_metric_precision_eigenvalues_raw: FloatArray
    block_metric_precision_eigenvalues: FloatArray
    block_metric_covariance_eigenvalues: FloatArray
    block_metric_raw_condition_number: FloatArray
    block_metric_condition_number: FloatArray
    block_metric_num_observations: IntArray
    block_metric_likelihood_rank: IntArray
    block_metric_used_prior_only: NDArray[np.bool_]
    block_metric_regularized: NDArray[np.bool_]
    owned_slice_owner_index: IntArray
    owned_slice_owner_parameter_index: IntArray
    owned_slice_owner_behavior_index: IntArray
    owned_slice_owner_moved: BoolArray
    owned_slice_proposal_count: IntArray
    owned_slice_shrink_count: IntArray
    owned_slice_bound_rejections: IntArray
    owned_slice_nonfinite_rejections: IntArray
    owned_slice_accepted_within_original_bounds: BoolArray
    owned_slice_accepted_above_slice_height: BoolArray
    owned_slice_initial_state_adjusted: BoolArray
    owned_slice_initial_boundary_canonicalized: BoolArray
    owned_slice_final_within_original_bounds: BoolArray
    owned_slice_final_above_last_slice_height: BoolArray

    @property
    def mh_accepted(self) -> NDArray[np.bool_]:
        """Backward-compatible alias for coordinate-wise MH decisions."""

        return self.coordinate_mh_accepted

    @property
    def acceptance_rate(self) -> FloatArray:
        """Return coordinate-MH acceptance, which is undefined for slice."""

        if self.idm_parameter_kernel != "coordinate_random_walk":
            raise RuntimeError(
                "Coordinate-MH acceptance is unavailable for the owned slice kernel."
            )

        return self.coordinate_mh_accepted.mean(axis=0)

    @property
    def block_acceptance_rate(self) -> FloatArray:
        return self.block_mh_accepted.mean(axis=0)

    @property
    def owned_slice_move_rate(self) -> FloatArray:
        """Return per-owner move rates only for the owned slice kernel."""

        if self.idm_parameter_kernel != "owned_full_bracket_slice":
            raise RuntimeError(
                "Owned-slice move rates are unavailable for coordinate MH."
            )
        return self.owned_slice_owner_moved.mean(axis=0)


@dataclass(frozen=True)
class ChainResult:
    samples: ChainSamples
    diagnostics: ChainDiagnostics
    final_state: SamplerState
    seed: int

    def as_npz_payload(self) -> Dict[str, NDArray[np.generic]]:
        """Include both posterior draws and diagnostics in one save payload."""

        payload: Dict[str, NDArray[np.generic]] = dict(self.samples.as_npz_payload())
        payload.update(
            {
                "diagnostic_idm_parameter_kernel_code": np.asarray(
                    _IDM_PARAMETER_KERNEL_CODES[
                        self.diagnostics.idm_parameter_kernel
                    ],
                    dtype=np.int64,
                ),
                "diagnostic_idm_ownership_layout_code": np.asarray(
                    _IDM_OWNERSHIP_LAYOUT_CODES[
                        self.diagnostics.idm_ownership_layout
                    ],
                    dtype=np.int64,
                ),
                # Retain the historical key as a byte-identical alias while
                # making the two composed MH kernels explicit.
                "diagnostic_mh_accepted": (
                    self.diagnostics.coordinate_mh_accepted
                ),
                "diagnostic_coordinate_mh_accepted": (
                    self.diagnostics.coordinate_mh_accepted
                ),
                "diagnostic_marginal_log_likelihood": (
                    self.diagnostics.marginal_log_likelihood
                ),
                "diagnostic_occupancy": self.diagnostics.occupancy,
                "diagnostic_proposal_standard_deviation_history": (
                    self.diagnostics.proposal_standard_deviation_history
                ),
                "diagnostic_final_proposal_standard_deviation": (
                    self.diagnostics.final_proposal_standard_deviation
                ),
                "diagnostic_proposal_adaptation_gain": (
                    self.diagnostics.proposal_adaptation_gain
                ),
                "diagnostic_block_mh_accepted": (
                    self.diagnostics.block_mh_accepted
                ),
                "diagnostic_block_scale_history": (
                    self.diagnostics.block_scale_history
                ),
                "diagnostic_final_block_scale": (
                    self.diagnostics.final_block_scale
                ),
                "diagnostic_block_adaptation_gain": (
                    self.diagnostics.block_adaptation_gain
                ),
                "diagnostic_block_metric_expansion_log_parameters": (
                    self.diagnostics.block_metric_expansion_log_parameters
                ),
                "diagnostic_block_metric_precision": (
                    self.diagnostics.block_metric_precision
                ),
                "diagnostic_block_metric_covariance": (
                    self.diagnostics.block_metric_covariance
                ),
                "diagnostic_block_metric_cholesky": (
                    self.diagnostics.block_metric_cholesky
                ),
                "diagnostic_block_metric_precision_eigenvalues_raw": (
                    self.diagnostics.block_metric_precision_eigenvalues_raw
                ),
                "diagnostic_block_metric_precision_eigenvalues": (
                    self.diagnostics.block_metric_precision_eigenvalues
                ),
                "diagnostic_block_metric_covariance_eigenvalues": (
                    self.diagnostics.block_metric_covariance_eigenvalues
                ),
                "diagnostic_block_metric_raw_condition_number": (
                    self.diagnostics.block_metric_raw_condition_number
                ),
                "diagnostic_block_metric_condition_number": (
                    self.diagnostics.block_metric_condition_number
                ),
                "diagnostic_block_metric_num_observations": (
                    self.diagnostics.block_metric_num_observations
                ),
                "diagnostic_block_metric_likelihood_rank": (
                    self.diagnostics.block_metric_likelihood_rank
                ),
                "diagnostic_block_metric_used_prior_only": (
                    self.diagnostics.block_metric_used_prior_only
                ),
                "diagnostic_block_metric_regularized": (
                    self.diagnostics.block_metric_regularized
                ),
                "diagnostic_owned_slice_owner_index": (
                    self.diagnostics.owned_slice_owner_index
                ),
                "diagnostic_owned_slice_owner_parameter_index": (
                    self.diagnostics.owned_slice_owner_parameter_index
                ),
                "diagnostic_owned_slice_owner_behavior_index": (
                    self.diagnostics.owned_slice_owner_behavior_index
                ),
                "diagnostic_owned_slice_owner_moved": (
                    self.diagnostics.owned_slice_owner_moved
                ),
                "diagnostic_owned_slice_proposal_count": (
                    self.diagnostics.owned_slice_proposal_count
                ),
                "diagnostic_owned_slice_shrink_count": (
                    self.diagnostics.owned_slice_shrink_count
                ),
                "diagnostic_owned_slice_bound_rejections": (
                    self.diagnostics.owned_slice_bound_rejections
                ),
                "diagnostic_owned_slice_nonfinite_rejections": (
                    self.diagnostics.owned_slice_nonfinite_rejections
                ),
                "diagnostic_owned_slice_accepted_within_original_bounds": (
                    self.diagnostics.owned_slice_accepted_within_original_bounds
                ),
                "diagnostic_owned_slice_accepted_above_slice_height": (
                    self.diagnostics.owned_slice_accepted_above_slice_height
                ),
                "diagnostic_owned_slice_initial_state_adjusted": (
                    self.diagnostics.owned_slice_initial_state_adjusted
                ),
                "diagnostic_owned_slice_initial_boundary_canonicalized": (
                    self.diagnostics.owned_slice_initial_boundary_canonicalized
                ),
                "diagnostic_owned_slice_final_within_original_bounds": (
                    self.diagnostics.owned_slice_final_within_original_bounds
                ),
                "diagnostic_owned_slice_final_above_last_slice_height": (
                    self.diagnostics.owned_slice_final_above_last_slice_height
                ),
                "seed": np.asarray(self.seed, dtype=np.uint64),
            }
        )
        return payload


@dataclass(frozen=True)
class MultiChainResult:
    chains: Tuple[ChainResult, ...]
    master_seed: int


@dataclass(frozen=True)
class _GibbsStepResult:
    state: SamplerState
    coordinate_mh_accepted: BoolArray
    block_mh_accepted: BoolArray
    owned_slice_audit: Optional[OwnedLogIDMSliceAudit]


def _resolve_idm_parameter_kernel(
    value: Optional[IDMParameterKernelConfig],
) -> IDMParameterKernelConfig:
    if value is None:
        return IDMParameterKernelConfig()
    if not isinstance(value, IDMParameterKernelConfig):
        raise TypeError(
            "idm_parameter_kernel must be an IDMParameterKernelConfig."
        )
    return value


def _validate_idm_parameter_kernel_compatibility(
    config: SamplerConfig, kernel: IDMParameterKernelConfig
) -> None:
    if kernel.name != "owned_full_bracket_slice":
        return
    if config.adapt_idm_proposals:
        raise ValueError(
            "The owned full-bracket slice kernel cannot use coordinate-MH "
            "proposal adaptation."
        )
    if config.use_idm_block_proposal or config.adapt_idm_block_scale:
        raise ValueError(
            "The owned full-bracket slice kernel cannot be composed with the "
            "IDM block-MH proposal."
        )


def _validate_idm_prior_kernel_compatibility(
    config: SamplerConfig,
    priors: SamplerPriors,
) -> None:
    """Reject kernels whose target has not been derived for the hierarchy."""

    if (
        priors.idm_prior_family == "hierarchical_log_normal_wishart"
        and config.use_idm_block_proposal
    ):
        raise ValueError(
            "The fixed-metric IDM block-MH proposal currently implements the "
            "fixed independent prior only; hierarchical IDM runs must disable it."
        )


def _validate_idm_prior_state(
    state: SamplerState, priors: SamplerPriors
) -> None:
    has_hierarchy = state.idm_hierarchy_mean is not None
    if priors.idm_prior_family == "fixed_independent":
        if has_hierarchy:
            raise ValueError(
                "A fixed-independent IDM run cannot consume hierarchy state."
            )
        return
    if not has_hierarchy or state.idm_hierarchy_precision is None:
        raise ValueError(
            "A hierarchical IDM run requires current hierarchy mean and precision."
        )


def _initialize_missing_idm_hierarchy(
    state: SamplerState,
    priors: SamplerPriors,
    rng: np.random.Generator,
) -> SamplerState:
    """Attach a valid hierarchy draw to an external warm-start state."""

    if priors.idm_prior_family == "fixed_independent":
        _validate_idm_prior_state(state, priors)
        return state
    if state.idm_hierarchy_mean is not None:
        _validate_idm_prior_state(state, priors)
        return state
    posterior = log_normal_wishart_posterior(
        priors.idm_hierarchy, np.log(state.idm_parameters)
    )
    mean, precision = sample_log_normal_wishart(posterior, rng)
    return replace(
        state,
        idm_hierarchy_mean=mean,
        idm_hierarchy_precision=precision,
    )


def _owned_layout(
    kernel: IDMParameterKernelConfig, state_space: JointStateSpace
) -> IDMOwnershipLayout:
    layout = build_idm_ownership_layout(
        kernel.ownership_layout, state_space.num_behavior_states
    )
    if layout.name != "behavior_specific":
        raise ValueError(
            "The end-to-end HMM sampler received an unsupported reduced "
            "ownership layout."
        )
    return layout


def _validate_trajectories(trajectories: Sequence[Trajectory]) -> Tuple[Trajectory, ...]:
    result = tuple(trajectories)
    if not result:
        raise ValueError("At least one trajectory is required.")
    if len({trajectory.trajectory_id for trajectory in result}) != len(result):
        raise ValueError("trajectory_id values must be unique.")
    return result


def _sample_prior_transition(
    config: SamplerConfig,
    priors: SamplerPriors,
    state_space: JointStateSpace,
    rng: np.random.Generator,
) -> TransitionParameters:
    behavior_count = state_space.num_behavior_states
    scenario_count = state_space.num_scenario_states
    joint_count = state_space.num_joint_states
    transition_prior = priors.transition
    if config.transition_family == "independent":
        behavior = sample_dirichlet_posterior_rows(
            np.zeros((behavior_count, behavior_count)),
            transition_prior.behavior_total_concentration,
            rng,
        )
        scenario = sample_dirichlet_posterior_rows(
            np.zeros((scenario_count, scenario_count)),
            transition_prior.scenario_total_concentration,
            rng,
        )
        return IndependentTransitions(state_space, behavior, scenario)
    if config.transition_family == "context_conditioned":
        scenario = sample_dirichlet_posterior_rows(
            np.zeros((scenario_count, scenario_count)),
            transition_prior.scenario_total_concentration,
            rng,
        )
        behavior_by_context = np.stack(
            [
                sample_dirichlet_posterior_rows(
                    np.zeros((behavior_count, behavior_count)),
                    transition_prior.behavior_total_concentration,
                    rng,
                )
                for _ in range(scenario_count)
            ]
        )
        return ContextConditionedTransitions(
            state_space, scenario, behavior_by_context
        )
    joint = sample_dirichlet_posterior_rows(
        np.zeros((joint_count, joint_count)),
        transition_prior.full_joint_total_concentration,
        rng,
    )
    return FullJointTransitions(state_space, joint)


def initialize_state(
    trajectories: Sequence[Trajectory],
    config: SamplerConfig,
    priors: SamplerPriors,
    rng: np.random.Generator,
) -> SamplerState:
    trajectories = _validate_trajectories(trajectories)
    state_space = JointStateSpace(
        config.num_behavior_states, config.num_scenario_states
    )
    paths = tuple(
        rng.integers(0, state_space.num_joint_states, size=x.acceleration.size)
        for x in trajectories
    )
    initial_alpha = np.full(
        state_space.num_joint_states,
        priors.transition.initial_total_concentration / state_space.num_joint_states,
    )
    initial_probability = rng.dirichlet(initial_alpha)
    transition = _sample_prior_transition(config, priors, state_space, rng)

    idm_parameters = np.empty((state_space.num_behavior_states, 5))
    for behavior in range(state_space.num_behavior_states):
        for _ in range(10_000):
            log_parameters = rng.normal(
                priors.idm.log_mean, priors.idm.log_standard_deviation
            )
            parameters = np.exp(log_parameters)
            if np.all(parameters >= priors.idm.lower) and np.all(
                parameters <= priors.idm.upper
            ):
                idm_parameters[behavior] = parameters
                break
        else:
            raise RuntimeError("Could not initialize bounded IDM parameters.")

    residual_variance = np.array(
        [
            sample_inverse_gamma(
                priors.residual_variance.shape,
                priors.residual_variance.scale,
                rng,
            )
            for _ in range(state_space.num_behavior_states)
        ]
    )
    empty = np.empty((0, priors.scenario.mean.size))
    scenario_posterior = normal_inverse_wishart_posterior(priors.scenario, empty)
    scenario_draws = [
        sample_normal_inverse_wishart(scenario_posterior, rng)
        for _ in range(state_space.num_scenario_states)
    ]
    scenario_mean = np.stack([draw[0] for draw in scenario_draws])
    scenario_covariance = np.stack([draw[1] for draw in scenario_draws])
    hierarchy_mean: Optional[FloatArray] = None
    hierarchy_precision: Optional[FloatArray] = None
    if priors.idm_prior_family == "hierarchical_log_normal_wishart":
        hierarchy_posterior = log_normal_wishart_posterior(
            priors.idm_hierarchy, np.log(idm_parameters)
        )
        hierarchy_mean, hierarchy_precision = sample_log_normal_wishart(
            hierarchy_posterior, rng
        )
    return SamplerState(
        paths=paths,
        initial_probability=initial_probability,
        transition=transition,
        idm_parameters=idm_parameters,
        residual_variance=residual_variance,
        scenario_mean=scenario_mean,
        scenario_covariance=scenario_covariance,
        idm_hierarchy_mean=hierarchy_mean,
        idm_hierarchy_precision=hierarchy_precision,
    )


def sample_transition_posterior(
    paths: Sequence[IntArray],
    config: SamplerConfig,
    priors: SamplerPriors,
    state_space: JointStateSpace,
    rng: np.random.Generator,
) -> TransitionParameters:
    transition_prior = priors.transition
    if config.transition_family == "independent":
        return sample_independent_transitions(
            paths,
            state_space,
            transition_prior.behavior_total_concentration,
            transition_prior.scenario_total_concentration,
            rng,
        )
    if config.transition_family == "context_conditioned":
        return sample_context_conditioned_transitions(
            paths,
            state_space,
            transition_prior.behavior_total_concentration,
            transition_prior.scenario_total_concentration,
            rng,
        )
    return sample_full_joint_transitions(
        paths,
        state_space,
        transition_prior.full_joint_total_concentration,
        rng,
    )


def gibbs_step(
    trajectories: Sequence[Trajectory],
    state: SamplerState,
    config: SamplerConfig,
    priors: SamplerPriors,
    rng: np.random.Generator,
    idm_proposal_standard_deviation: Optional[ArrayLike] = None,
    idm_block_metric: Optional[IDMBlockMetric] = None,
    idm_block_scale: Optional[ArrayLike] = None,
    idm_parameter_kernel: Optional[IDMParameterKernelConfig] = None,
) -> Tuple[SamplerState, NDArray[np.bool_], NDArray[np.bool_]]:
    """Perform one exact-path Gibbs iteration with the selected IDM kernel.

    The three-value return remains backward compatible.  Detailed owned-slice
    diagnostics are collected by :func:`run_chain` through the internal result
    object rather than overloading coordinate-MH acceptance.
    """

    result = _gibbs_step_with_diagnostics(
        trajectories,
        state,
        config,
        priors,
        rng,
        idm_proposal_standard_deviation,
        idm_block_metric,
        idm_block_scale,
        _resolve_idm_parameter_kernel(idm_parameter_kernel),
    )
    return (
        result.state,
        result.coordinate_mh_accepted,
        result.block_mh_accepted,
    )


def _gibbs_step_with_diagnostics(
    trajectories: Sequence[Trajectory],
    state: SamplerState,
    config: SamplerConfig,
    priors: SamplerPriors,
    rng: np.random.Generator,
    idm_proposal_standard_deviation: Optional[ArrayLike],
    idm_block_metric: Optional[IDMBlockMetric],
    idm_block_scale: Optional[ArrayLike],
    idm_parameter_kernel: IDMParameterKernelConfig,
) -> _GibbsStepResult:
    """Return one Gibbs transition and kernel-specific audit information."""

    trajectories = _validate_trajectories(trajectories)
    _validate_idm_parameter_kernel_compatibility(
        config, idm_parameter_kernel
    )
    _validate_idm_prior_kernel_compatibility(config, priors)
    _validate_idm_prior_state(state, priors)
    if (
        idm_parameter_kernel.name == "coordinate_random_walk"
        and state.idm_owned_log_parameters is not None
    ):
        raise ValueError(
            "A coordinate-random-walk chain cannot consume an authoritative "
            "owned log-IDM state from another kernel."
        )
    if idm_parameter_kernel.name == "owned_full_bracket_slice" and (
        idm_proposal_standard_deviation is not None
        or idm_block_metric is not None
        or idm_block_scale is not None
    ):
        raise ValueError(
            "Coordinate-MH proposal controls cannot be supplied to the owned "
            "full-bracket slice kernel."
        )
    state_space = JointStateSpace(
        config.num_behavior_states, config.num_scenario_states
    )
    slice_layout: Optional[IDMOwnershipLayout] = None
    if idm_parameter_kernel.name == "owned_full_bracket_slice":
        slice_layout = _owned_layout(idm_parameter_kernel, state_space)
        if state.idm_owned_log_parameters is not None:
            decoded = decode_supported_owned_log(
                state.idm_owned_log_parameters, priors.idm, slice_layout
            )
            if decoded is None or not np.array_equal(
                decoded[1], state.idm_parameters
            ):
                raise ValueError(
                    "The authoritative owned log-IDM state is unsupported or "
                    "does not exactly match the expanded physical state."
                )
    transition_matrix = state.transition.joint_matrix()
    emissions = emission_log_probability_sequences(
        trajectories, state, state_space
    )
    paths = ffbs_sample_paths(
        emissions, state.initial_probability, transition_matrix, rng
    )
    initial_probability = sample_initial_distribution(
        paths,
        state_space.num_joint_states,
        priors.transition.initial_total_concentration,
        rng,
    )
    transition = sample_transition_posterior(
        paths, config, priors, state_space, rng
    )
    owned_slice_audit: Optional[OwnedLogIDMSliceAudit]
    if idm_parameter_kernel.name == "owned_full_bracket_slice":
        if slice_layout is None:
            raise RuntimeError("Owned-slice layout was not initialized.")
        slice_update = full_bracket_owned_log_idm_slice_update(
            state.idm_parameters,
            state.residual_variance,
            trajectories,
            paths,
            state_space,
            priors.idm,
            slice_layout,
            rng,
            hierarchy_mean=state.idm_hierarchy_mean,
            hierarchy_precision=state.idm_hierarchy_precision,
            current_owned_log_parameters=state.idm_owned_log_parameters,
            max_bracket_shrinks=(
                idm_parameter_kernel.max_bracket_shrinks
            ),
        )
        idm_parameters = slice_update.parameters
        owned_log_parameters: Optional[FloatArray] = (
            slice_update.owned_log_parameters
        )
        coordinate_accepted = np.empty((0, 0), dtype=bool)
        block_accepted = np.zeros(
            state_space.num_behavior_states, dtype=bool
        )
        owned_slice_audit = slice_update.audit
    else:
        idm_update = sample_idm_parameters_mh(
            state.idm_parameters,
            state.residual_variance,
            trajectories,
            paths,
            state_space,
            priors.idm,
            rng,
            idm_proposal_standard_deviation,
            hierarchy_mean=state.idm_hierarchy_mean,
            hierarchy_precision=state.idm_hierarchy_precision,
        )
        if config.use_idm_block_proposal:
            if idm_block_metric is None or idm_block_scale is None:
                raise ValueError(
                    "The enabled IDM block proposal needs a fixed metric and scale."
                )
            block_update = sample_idm_parameters_block_mh(
                idm_update.parameters,
                state.residual_variance,
                trajectories,
                paths,
                state_space,
                priors.idm,
                idm_block_metric,
                idm_block_scale,
                rng,
            )
            idm_parameters = block_update.parameters
            block_accepted = block_update.accepted
        else:
            if idm_block_metric is not None or idm_block_scale is not None:
                raise ValueError(
                    "IDM block metric/scale were supplied while the kernel is disabled."
                )
            idm_parameters = idm_update.parameters
            block_accepted = np.zeros(
                state_space.num_behavior_states, dtype=bool
            )
        coordinate_accepted = idm_update.accepted
        owned_log_parameters = None
        owned_slice_audit = None
    hierarchy_mean: Optional[FloatArray] = None
    hierarchy_precision: Optional[FloatArray] = None
    if priors.idm_prior_family == "hierarchical_log_normal_wishart":
        hierarchy_posterior = log_normal_wishart_posterior(
            priors.idm_hierarchy, np.log(idm_parameters)
        )
        hierarchy_mean, hierarchy_precision = sample_log_normal_wishart(
            hierarchy_posterior, rng
        )
    residual_variance = sample_residual_variances(
        idm_parameters,
        trajectories,
        paths,
        state_space,
        priors.residual_variance,
        rng,
    )
    scenario_mean, scenario_covariance = sample_scenario_emissions(
        trajectories, paths, state_space, priors.scenario, rng
    )
    return _GibbsStepResult(
        state=SamplerState(
            paths=paths,
            initial_probability=initial_probability,
            transition=transition,
            idm_parameters=idm_parameters,
            residual_variance=residual_variance,
            scenario_mean=scenario_mean,
            scenario_covariance=scenario_covariance,
            idm_owned_log_parameters=owned_log_parameters,
            idm_hierarchy_mean=hierarchy_mean,
            idm_hierarchy_precision=hierarchy_precision,
        ),
        coordinate_mh_accepted=coordinate_accepted,
        block_mh_accepted=block_accepted,
        owned_slice_audit=owned_slice_audit,
    )


def _marginal_log_likelihood(
    trajectories: Sequence[Trajectory],
    state: SamplerState,
    state_space: JointStateSpace,
) -> float:
    emissions = emission_log_probability_sequences(trajectories, state, state_space)
    filters = log_forward_filter_sequences(
        emissions, state.initial_probability, state.transition.joint_matrix()
    )
    return float(sum(result.log_likelihood for result in filters))


def _transition_component(transition: TransitionParameters) -> Dict[str, FloatArray]:
    if isinstance(transition, IndependentTransitions):
        return {"behavior": transition.behavior, "scenario": transition.scenario}
    if isinstance(transition, ContextConditionedTransitions):
        return {
            "behavior_by_context": transition.behavior_by_context,
            "scenario": transition.scenario,
        }
    return {"joint": transition.joint}


def adapt_idm_proposal_standard_deviation(
    current: ArrayLike,
    accepted: ArrayLike,
    config: SamplerConfig,
    iteration: int,
) -> Tuple[FloatArray, float]:
    """Apply one burn-in-only Robbins-Monro update in log-SD space."""

    expected = (config.num_behavior_states, 5)
    proposal = np.asarray(current, dtype=float)
    acceptance = np.asarray(accepted)
    if proposal.shape != expected or acceptance.shape != expected:
        raise ValueError(f"Proposal and acceptance matrices must have shape {expected}.")
    if not np.all(np.isfinite(proposal)) or np.any(proposal <= 0.0):
        raise ValueError("Proposal standard deviations must be finite and positive.")
    if iteration < 0 or iteration >= config.num_iterations:
        raise ValueError("iteration is outside the configured chain range.")
    if not config.adapt_idm_proposals or iteration >= config.burn_in:
        return proposal.copy(), 0.0
    gain = config.idm_adaptation_gain_scale / (
        iteration + 1.0 + config.idm_adaptation_gain_offset
    ) ** config.idm_adaptation_gain_exponent
    log_proposal = np.log(proposal)
    log_proposal += gain * (
        acceptance.astype(float) - config.idm_adaptation_target_acceptance
    )
    updated = np.exp(
        np.clip(
            log_proposal,
            np.log(config.idm_proposal_standard_deviation_minimum),
            np.log(config.idm_proposal_standard_deviation_maximum),
        )
    )
    return updated, float(gain)


def adapt_idm_block_scale(
    current: ArrayLike,
    accepted: ArrayLike,
    config: SamplerConfig,
    iteration: int,
) -> Tuple[FloatArray, float]:
    """Adapt one scalar block multiplier per behavior during burn-in only."""

    expected = (config.num_behavior_states,)
    scale = np.asarray(current, dtype=float)
    acceptance = np.asarray(accepted)
    if scale.shape != expected or acceptance.shape != expected:
        raise ValueError(
            f"IDM block scale and acceptance must have shape {expected}."
        )
    if not np.all(np.isfinite(scale)) or np.any(scale <= 0.0):
        raise ValueError("IDM block scales must be finite and positive.")
    if acceptance.dtype != np.bool_:
        raise ValueError("IDM block acceptance values must be boolean.")
    if iteration < 0 or iteration >= config.num_iterations:
        raise ValueError("iteration is outside the configured chain range.")
    if not config.adapt_idm_block_scale or iteration >= config.burn_in:
        return scale.copy(), 0.0
    gain = config.idm_block_adaptation_gain_scale / (
        iteration + 1.0 + config.idm_block_adaptation_gain_offset
    ) ** config.idm_block_adaptation_gain_exponent
    log_scale = np.log(scale)
    log_scale += gain * (
        acceptance.astype(float)
        - config.idm_block_adaptation_target_acceptance
    )
    updated = np.exp(
        np.clip(
            log_scale,
            np.log(config.idm_block_scale_minimum),
            np.log(config.idm_block_scale_maximum),
        )
    )
    return updated, float(gain)


def run_chain(
    trajectories: Sequence[Trajectory],
    config: SamplerConfig,
    *,
    seed: int,
    priors: Optional[SamplerPriors] = None,
    initial_state: Optional[SamplerState] = None,
    idm_parameter_kernel: Optional[IDMParameterKernelConfig] = None,
) -> ChainResult:
    """Run one chain without mutating global NumPy random state."""

    trajectories = _validate_trajectories(trajectories)
    priors = SamplerPriors() if priors is None else priors
    kernel = _resolve_idm_parameter_kernel(idm_parameter_kernel)
    _validate_idm_parameter_kernel_compatibility(config, kernel)
    _validate_idm_prior_kernel_compatibility(config, priors)
    rng = np.random.default_rng(seed)
    state_space = JointStateSpace(
        config.num_behavior_states, config.num_scenario_states
    )
    state = (
        initialize_state(trajectories, config, priors, rng)
        if initial_state is None
        else initial_state
    )
    state = _initialize_missing_idm_hierarchy(state, priors, rng)
    if (
        kernel.name == "coordinate_random_walk"
        and state.idm_owned_log_parameters is not None
    ):
        raise ValueError(
            "A coordinate-random-walk chain cannot start from an authoritative "
            "owned log-IDM state."
        )
    fixed_block_metric = None
    if config.use_idm_block_proposal:
        # This deterministic construction occurs exactly once at the supplied
        # or sampled initial chain state.  Its covariance is never adapted.
        fixed_block_metric = build_fixed_idm_block_metric(
            state.idm_parameters,
            state.residual_variance,
            trajectories,
            state.paths,
            state_space,
            priors.idm,
        )
    saved_states = []
    if kernel.name == "coordinate_random_walk":
        coordinate_accepted = np.zeros(
            (config.num_iterations, config.num_behavior_states, 5), dtype=bool
        )
    else:
        coordinate_accepted = np.empty(
            (config.num_iterations, 0, 0), dtype=bool
        )
    block_accepted = np.zeros(
        (config.num_iterations, config.num_behavior_states), dtype=bool
    )
    log_likelihood = np.empty(config.num_iterations)
    occupancy = np.zeros(
        (config.num_iterations, state_space.num_joint_states), dtype=np.int64
    )
    if kernel.name == "coordinate_random_walk":
        initial_proposal = np.broadcast_to(
            priors.idm.proposal_standard_deviation,
            (config.num_behavior_states, 5),
        ).copy()
        if config.adapt_idm_proposals:
            proposal_standard_deviation = np.clip(
                initial_proposal,
                config.idm_proposal_standard_deviation_minimum,
                config.idm_proposal_standard_deviation_maximum,
            )
        else:
            proposal_standard_deviation = initial_proposal
        proposal_history = np.empty(
            (config.num_iterations + 1, config.num_behavior_states, 5),
            dtype=float,
        )
        proposal_history[0] = proposal_standard_deviation
        adaptation_gain = np.zeros(config.num_iterations, dtype=float)
    else:
        proposal_standard_deviation = np.empty((0, 0), dtype=float)
        proposal_history = np.empty((0, 0, 0), dtype=float)
        adaptation_gain = np.empty(0, dtype=float)

    if kernel.name == "owned_full_bracket_slice":
        layout = _owned_layout(kernel, state_space)
        owner_count = layout.num_owned_parameters
        owned_slice_owner_index = layout.owner_index.copy()
        owned_slice_owner_parameter_index = layout.owner_parameter_index.copy()
        owned_slice_owner_behavior_index = layout.owner_behavior_index.copy()
        owned_slice_owner_moved = np.zeros(
            (config.num_iterations, owner_count), dtype=bool
        )
        owned_slice_proposal_count = np.zeros(
            (config.num_iterations, owner_count), dtype=np.int64
        )
        owned_slice_shrink_count = np.zeros_like(owned_slice_proposal_count)
        owned_slice_bound_rejections = np.zeros_like(
            owned_slice_proposal_count
        )
        owned_slice_nonfinite_rejections = np.zeros_like(
            owned_slice_proposal_count
        )
        owned_slice_accepted_within = np.zeros(
            (config.num_iterations, owner_count), dtype=bool
        )
        owned_slice_accepted_above = np.zeros_like(
            owned_slice_accepted_within
        )
        owned_slice_initial_state_adjusted = np.zeros(
            config.num_iterations, dtype=bool
        )
        owned_slice_initial_boundary_canonicalized = np.zeros(
            config.num_iterations, dtype=bool
        )
        owned_slice_final_within = np.zeros(config.num_iterations, dtype=bool)
        owned_slice_final_above = np.zeros(config.num_iterations, dtype=bool)
    else:
        owner_count = 0
        owned_slice_owner_index = np.empty((0, 0), dtype=np.int64)
        owned_slice_owner_parameter_index = np.empty(0, dtype=np.int64)
        owned_slice_owner_behavior_index = np.empty(0, dtype=np.int64)
        owned_slice_owner_moved = np.empty(
            (config.num_iterations, 0), dtype=bool
        )
        owned_slice_proposal_count = np.empty(
            (config.num_iterations, 0), dtype=np.int64
        )
        owned_slice_shrink_count = owned_slice_proposal_count.copy()
        owned_slice_bound_rejections = owned_slice_proposal_count.copy()
        owned_slice_nonfinite_rejections = owned_slice_proposal_count.copy()
        owned_slice_accepted_within = np.empty(
            (config.num_iterations, 0), dtype=bool
        )
        owned_slice_accepted_above = owned_slice_accepted_within.copy()
        owned_slice_initial_state_adjusted = np.empty(0, dtype=bool)
        owned_slice_initial_boundary_canonicalized = np.empty(0, dtype=bool)
        owned_slice_final_within = np.empty(0, dtype=bool)
        owned_slice_final_above = np.empty(0, dtype=bool)

    if config.use_idm_block_proposal:
        block_scale = np.full(
            config.num_behavior_states,
            config.idm_block_initial_scale,
            dtype=float,
        )
        block_scale_history = np.empty(
            (config.num_iterations + 1, config.num_behavior_states),
            dtype=float,
        )
        block_scale_history[0] = block_scale
        block_adaptation_gain = np.zeros(config.num_iterations, dtype=float)
    else:
        # Record the configured, fixed scalar even when the kernel is off.
        # Availability of the metric is represented separately by its empty
        # diagnostic arrays, so a disabled run cannot masquerade as fitted.
        block_scale = np.full(
            config.num_behavior_states,
            config.idm_block_initial_scale,
            dtype=float,
        )
        block_scale_history = np.broadcast_to(
            block_scale,
            (config.num_iterations + 1, config.num_behavior_states),
        ).copy()
        block_adaptation_gain = np.zeros(config.num_iterations, dtype=float)

    for iteration in range(config.num_iterations):
        if kernel.name == "owned_full_bracket_slice":
            step = _gibbs_step_with_diagnostics(
                trajectories,
                state,
                config,
                priors,
                rng,
                None,
                None,
                None,
                kernel,
            )
        else:
            step = _gibbs_step_with_diagnostics(
                trajectories,
                state,
                config,
                priors,
                rng,
                proposal_standard_deviation,
                fixed_block_metric if config.use_idm_block_proposal else None,
                block_scale if config.use_idm_block_proposal else None,
                kernel,
            )
        state = step.state
        block_accepted[iteration] = step.block_mh_accepted
        if kernel.name == "coordinate_random_walk":
            if step.owned_slice_audit is not None:
                raise RuntimeError(
                    "Coordinate MH unexpectedly returned an owned-slice audit."
                )
            coordinate_accepted[iteration] = step.coordinate_mh_accepted
            proposal_standard_deviation, adaptation_gain[iteration] = (
                adapt_idm_proposal_standard_deviation(
                    proposal_standard_deviation,
                    coordinate_accepted[iteration],
                    config,
                    iteration,
                )
            )
            proposal_history[iteration + 1] = proposal_standard_deviation
        else:
            audit = step.owned_slice_audit
            if audit is None or state.idm_owned_log_parameters is None:
                raise RuntimeError(
                    "Owned slice did not return its audit and authoritative state."
                )
            if (
                audit.layout_name != kernel.ownership_layout
                or audit.proposal_count.shape != (owner_count,)
            ):
                raise RuntimeError(
                    "Owned-slice audit layout disagrees with the configured HMM."
                )
            owner_order = np.arange(owner_count, dtype=np.int64)
            current_coordinate = (
                audit.coordinate_current_owned_log_parameters[
                    owner_order, owner_order
                ]
            )
            owned_slice_owner_moved[iteration] = (
                audit.accepted_coordinate_value != current_coordinate
            )
            owned_slice_proposal_count[iteration] = audit.proposal_count
            owned_slice_shrink_count[iteration] = audit.shrink_count
            owned_slice_bound_rejections[iteration] = audit.bound_rejections
            owned_slice_nonfinite_rejections[iteration] = (
                audit.nonfinite_rejections
            )
            owned_slice_accepted_within[iteration] = (
                audit.accepted_within_original_bounds
            )
            owned_slice_accepted_above[iteration] = (
                audit.accepted_above_slice_height
            )
            owned_slice_initial_state_adjusted[iteration] = (
                audit.initial_state_adjusted
            )
            owned_slice_initial_boundary_canonicalized[iteration] = (
                audit.initial_boundary_canonicalized
            )
            owned_slice_final_within[iteration] = (
                audit.final_within_original_bounds
            )
            owned_slice_final_above[iteration] = (
                audit.final_above_last_slice_height
            )
        if config.use_idm_block_proposal:
            if block_scale is None:
                raise RuntimeError("Enabled IDM block scale was not initialized.")
            block_scale, block_adaptation_gain[iteration] = (
                adapt_idm_block_scale(
                    block_scale,
                    block_accepted[iteration],
                    config,
                    iteration,
                )
            )
            block_scale_history[iteration + 1] = block_scale
        log_likelihood[iteration] = _marginal_log_likelihood(
            trajectories, state, state_space
        )
        occupancy[iteration] = np.bincount(
            np.concatenate(state.paths), minlength=state_space.num_joint_states
        )
        if iteration >= config.burn_in and (
            (iteration - config.burn_in) % config.thin == 0
        ):
            saved_states.append(state)

    components = {
        key: np.stack([_transition_component(x.transition)[key] for x in saved_states])
        for key in _transition_component(saved_states[0].transition)
    }
    latent_paths = None
    if config.store_latent_paths:
        latent_paths = tuple(
            np.stack([state_draw.paths[index] for state_draw in saved_states])
            for index in range(len(trajectories))
        )
    owned_log_samples: Optional[FloatArray] = None
    if kernel.name == "owned_full_bracket_slice":
        owned_draws = [
            state_draw.idm_owned_log_parameters for state_draw in saved_states
        ]
        if any(draw is None for draw in owned_draws):
            raise RuntimeError(
                "A saved owned-slice state lost its authoritative log parameters."
            )
        owned_log_samples = np.stack(
            [np.asarray(draw, dtype=float) for draw in owned_draws]
        )
    samples = ChainSamples(
        idm_parameters=np.stack([x.idm_parameters for x in saved_states]),
        residual_variance=np.stack([x.residual_variance for x in saved_states]),
        scenario_mean=np.stack([x.scenario_mean for x in saved_states]),
        scenario_covariance=np.stack([x.scenario_covariance for x in saved_states]),
        initial_probability=np.stack([x.initial_probability for x in saved_states]),
        joint_transition=np.stack(
            [x.transition.joint_matrix() for x in saved_states]
        ),
        transition_components=components,
        latent_paths=latent_paths,
        owned_log_parameters=owned_log_samples,
        idm_hierarchy_mean=(
            np.stack([x.idm_hierarchy_mean for x in saved_states])
            if priors.idm_prior_family
            == "hierarchical_log_normal_wishart"
            else None
        ),
        idm_hierarchy_precision=(
            np.stack([x.idm_hierarchy_precision for x in saved_states])
            if priors.idm_prior_family
            == "hierarchical_log_normal_wishart"
            else None
        ),
    )
    if fixed_block_metric is None:
        empty_matrix = np.empty((0, 5, 5), dtype=float)
        empty_eigenvalues = np.empty((0, 5), dtype=float)
        empty_float = np.empty(0, dtype=float)
        empty_integer = np.empty(0, dtype=np.int64)
        empty_boolean = np.empty(0, dtype=bool)
        block_metric_expansion = np.empty((0, 5), dtype=float)
        block_metric_precision = empty_matrix.copy()
        block_metric_covariance = empty_matrix.copy()
        block_metric_cholesky = empty_matrix.copy()
        block_metric_precision_eigenvalues_raw = empty_eigenvalues.copy()
        block_metric_precision_eigenvalues = empty_eigenvalues.copy()
        block_metric_covariance_eigenvalues = empty_eigenvalues.copy()
        block_metric_raw_condition_number = empty_float.copy()
        block_metric_condition_number = empty_float.copy()
        block_metric_num_observations = empty_integer.copy()
        block_metric_likelihood_rank = empty_integer.copy()
        block_metric_used_prior_only = empty_boolean.copy()
        block_metric_regularized = empty_boolean.copy()
        if block_scale is None:
            raise RuntimeError("Disabled IDM block scale was not initialized.")
        final_block_scale = block_scale.copy()
    else:
        block_metric_expansion = (
            fixed_block_metric.expansion_log_parameters.copy()
        )
        block_metric_precision = fixed_block_metric.precision.copy()
        block_metric_covariance = fixed_block_metric.covariance.copy()
        block_metric_cholesky = fixed_block_metric.cholesky.copy()
        block_metric_precision_eigenvalues_raw = (
            fixed_block_metric.precision_eigenvalues_raw.copy()
        )
        block_metric_precision_eigenvalues = (
            fixed_block_metric.precision_eigenvalues.copy()
        )
        block_metric_covariance_eigenvalues = (
            fixed_block_metric.covariance_eigenvalues.copy()
        )
        block_metric_raw_condition_number = (
            fixed_block_metric.raw_condition_number.copy()
        )
        block_metric_condition_number = (
            fixed_block_metric.condition_number.copy()
        )
        block_metric_num_observations = (
            fixed_block_metric.num_observations.copy()
        )
        block_metric_likelihood_rank = (
            fixed_block_metric.likelihood_rank.copy()
        )
        block_metric_used_prior_only = (
            fixed_block_metric.used_prior_only.copy()
        )
        block_metric_regularized = fixed_block_metric.regularized.copy()
        if block_scale is None:
            raise RuntimeError("Enabled IDM block scale was lost.")
        final_block_scale = block_scale.copy()

    diagnostics = ChainDiagnostics(
        idm_parameter_kernel=kernel.name,
        idm_ownership_layout=kernel.ownership_layout,
        coordinate_mh_accepted=coordinate_accepted,
        marginal_log_likelihood=log_likelihood,
        occupancy=occupancy,
        proposal_standard_deviation_history=proposal_history,
        final_proposal_standard_deviation=(
            proposal_standard_deviation.copy()
        ),
        proposal_adaptation_gain=adaptation_gain,
        block_mh_accepted=block_accepted,
        block_scale_history=block_scale_history,
        final_block_scale=final_block_scale,
        block_adaptation_gain=block_adaptation_gain,
        block_metric_expansion_log_parameters=block_metric_expansion,
        block_metric_precision=block_metric_precision,
        block_metric_covariance=block_metric_covariance,
        block_metric_cholesky=block_metric_cholesky,
        block_metric_precision_eigenvalues_raw=(
            block_metric_precision_eigenvalues_raw
        ),
        block_metric_precision_eigenvalues=(
            block_metric_precision_eigenvalues
        ),
        block_metric_covariance_eigenvalues=(
            block_metric_covariance_eigenvalues
        ),
        block_metric_raw_condition_number=(
            block_metric_raw_condition_number
        ),
        block_metric_condition_number=block_metric_condition_number,
        block_metric_num_observations=block_metric_num_observations,
        block_metric_likelihood_rank=block_metric_likelihood_rank,
        block_metric_used_prior_only=block_metric_used_prior_only,
        block_metric_regularized=block_metric_regularized,
        owned_slice_owner_index=owned_slice_owner_index,
        owned_slice_owner_parameter_index=(
            owned_slice_owner_parameter_index
        ),
        owned_slice_owner_behavior_index=(
            owned_slice_owner_behavior_index
        ),
        owned_slice_owner_moved=owned_slice_owner_moved,
        owned_slice_proposal_count=owned_slice_proposal_count,
        owned_slice_shrink_count=owned_slice_shrink_count,
        owned_slice_bound_rejections=owned_slice_bound_rejections,
        owned_slice_nonfinite_rejections=(
            owned_slice_nonfinite_rejections
        ),
        owned_slice_accepted_within_original_bounds=(
            owned_slice_accepted_within
        ),
        owned_slice_accepted_above_slice_height=(
            owned_slice_accepted_above
        ),
        owned_slice_initial_state_adjusted=(
            owned_slice_initial_state_adjusted
        ),
        owned_slice_initial_boundary_canonicalized=(
            owned_slice_initial_boundary_canonicalized
        ),
        owned_slice_final_within_original_bounds=(
            owned_slice_final_within
        ),
        owned_slice_final_above_last_slice_height=(
            owned_slice_final_above
        ),
    )
    return ChainResult(samples, diagnostics, state, int(seed))


def run_chains(
    trajectories: Sequence[Trajectory],
    config: SamplerConfig,
    *,
    num_chains: int,
    master_seed: int,
    priors: Optional[SamplerPriors] = None,
    initial_states: Optional[Sequence[SamplerState]] = None,
    idm_parameter_kernel: Optional[IDMParameterKernelConfig] = None,
) -> MultiChainResult:
    """Run reproducible independent chains from spawned random streams."""

    if num_chains <= 0:
        raise ValueError("num_chains must be positive.")
    resolved_initial_states = None
    if initial_states is not None:
        resolved_initial_states = tuple(initial_states)
        if len(resolved_initial_states) != num_chains:
            raise ValueError("initial_states must contain exactly one state per chain.")
    seed_sequence = np.random.SeedSequence(master_seed)
    child_sequences = seed_sequence.spawn(num_chains)
    child_seeds = [int(x.generate_state(1, dtype=np.uint64)[0]) for x in child_sequences]
    if resolved_initial_states is None:
        # Keep the historical prior-initialized path and RNG behavior unchanged.
        chains = tuple(
            run_chain(
                trajectories,
                config,
                seed=seed,
                priors=priors,
                idm_parameter_kernel=idm_parameter_kernel,
            )
            for seed in child_seeds
        )
    else:
        chains = tuple(
            run_chain(
                trajectories,
                config,
                seed=seed,
                priors=priors,
                initial_state=initial_state,
                idm_parameter_kernel=idm_parameter_kernel,
            )
            for seed, initial_state in zip(child_seeds, resolved_initial_states)
        )
    return MultiChainResult(chains, int(master_seed))
