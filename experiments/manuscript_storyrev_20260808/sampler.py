"""Exact Gibbs reconstruction under the frozen manuscript target."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np
from numpy.typing import ArrayLike, NDArray

from experiments.revision_20260806.inference.structures import (
    NormalInverseWishartPrior,
)
from experiments.revision_20260806.inference.updates import (
    normal_inverse_wishart_posterior,
    sample_inverse_gamma,
    sample_normal_inverse_wishart,
)
from experiments.revision_20260806.model import (
    JointStateSpace,
    count_initial_states,
    count_joint_transitions,
    ffbs_sample_path,
    log_forward_filter_sequences,
    sample_dirichlet_posterior_rows,
    sample_initial_distribution,
)

from .target import (
    COVARIATE_ORDER,
    ContextStandardizer,
    StoryTrajectory,
    emission_log_probability_sequences,
    normal_log_density,
    original_idm_acceleration,
    stable_cholesky,
)

if TYPE_CHECKING:
    from .checkpoint import RestoredCheckpoint


FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


def _tuple_of_floats(value: Any, length: int, name: str) -> Tuple[float, ...]:
    array = np.asarray(value, dtype=float)
    if array.shape != (length,) or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain {length} finite values.")
    return tuple(float(item) for item in array)


@dataclass(frozen=True)
class ReconstructionConfig:
    """Fully explicit scientific target and single-chain run configuration."""

    schema_version: int
    model_id: str
    arm_id: str
    expected_num_trajectories: int
    sampling_frequency_hz: float
    allow_observed_negative_speed: bool
    minimum_allowed_speed_mps: float
    expected_negative_speed_count: int
    expected_negative_speed_fraction: float
    expected_negative_speed_affected_trajectories: int
    negative_speed_unaffected_case_indices: Tuple[int, ...]
    preserve_raw_observations: bool
    negative_speed_justification: str
    num_behavior_states: int
    num_scenario_states: int
    master_seed: int
    seed_spawn_index: int
    num_iterations: int
    burn_in: int
    thin: int
    store_latent_paths: bool
    acceleration_exponent: float
    idm_log_mean: Tuple[float, ...]
    idm_log_standard_deviation: Tuple[float, ...]
    idm_lower: Tuple[float, ...]
    idm_upper: Tuple[float, ...]
    slice_width: Tuple[float, ...]
    slice_max_steps_out: int
    slice_max_shrink: int
    residual_variance_shape: float
    residual_variance_scale: float
    scenario_mean: Tuple[float, ...]
    scenario_mean_strength: float
    scenario_degrees_of_freedom: float
    scenario_scale_diagonal: Tuple[float, ...]
    transition_legacy_alpha: float
    transition_per_cell_concentration: float
    transition_row_total_concentration: float
    initial_total_concentration: float
    context_order: Tuple[str, ...]

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise ValueError("Unsupported reconstruction-config schema version.")
        if self.model_id != "ps_hmm_idm_original_idm_exact_ffbs_v1":
            raise ValueError("Unexpected model_id for the frozen reconstruction target.")
        if not self.arm_id:
            raise ValueError("arm_id must be non-empty.")
        if self.expected_num_trajectories <= 0:
            raise ValueError("expected_num_trajectories must be positive.")
        if not np.isfinite(self.sampling_frequency_hz) or self.sampling_frequency_hz <= 0:
            raise ValueError("sampling_frequency_hz must be finite and positive.")
        if not isinstance(self.allow_observed_negative_speed, bool) or not isinstance(
            self.preserve_raw_observations, bool
        ):
            raise ValueError("Observed-speed policy switches must be boolean.")
        if not np.isfinite(self.minimum_allowed_speed_mps):
            raise ValueError("minimum_allowed_speed_mps must be finite.")
        if self.expected_negative_speed_count < 0 or (
            self.expected_negative_speed_affected_trajectories < 0
        ):
            raise ValueError("Expected negative-speed counts must be non-negative.")
        if not 0.0 <= self.expected_negative_speed_fraction <= 1.0:
            raise ValueError("Expected negative-speed fraction must lie in [0, 1].")
        if any(index < 0 for index in self.negative_speed_unaffected_case_indices):
            raise ValueError("Case indices must be non-negative.")
        if self.allow_observed_negative_speed:
            if self.minimum_allowed_speed_mps >= 0.0:
                raise ValueError(
                    "Negative-speed opt-in requires a negative finite lower bound."
                )
            if (
                self.expected_negative_speed_count <= 0
                or self.expected_negative_speed_affected_trajectories <= 0
                or self.expected_negative_speed_fraction <= 0.0
                or not self.preserve_raw_observations
                or not self.negative_speed_justification.strip()
            ):
                raise ValueError(
                    "Negative-speed opt-in requires explicit audited expectations, "
                    "raw preservation, and a justification."
                )
        elif (
            self.minimum_allowed_speed_mps != 0.0
            or self.expected_negative_speed_count != 0
            or self.expected_negative_speed_fraction != 0.0
            or self.expected_negative_speed_affected_trajectories != 0
        ):
            raise ValueError("Strict speed policy must use zero-valued exceptions.")
        if self.num_behavior_states <= 0 or self.num_scenario_states <= 0:
            raise ValueError("State counts must be positive.")
        if self.master_seed < 0 or self.seed_spawn_index < 0:
            raise ValueError("Seed entropy and spawn index must be non-negative.")
        if (
            self.num_iterations <= 0
            or self.burn_in < 0
            or self.burn_in >= self.num_iterations
            or self.thin <= 0
        ):
            raise ValueError("Iteration, burn-in, or thinning configuration is invalid.")
        if not isinstance(self.store_latent_paths, bool):
            raise ValueError("store_latent_paths must be boolean.")
        if not np.isfinite(self.acceleration_exponent) or self.acceleration_exponent <= 0:
            raise ValueError("acceleration_exponent must be finite and positive.")
        if not np.isclose(self.acceleration_exponent, 4.0, rtol=0.0, atol=0.0):
            raise ValueError("The frozen historical IDM target requires exponent 4.")

        vector_fields = {
            "idm_log_mean": self.idm_log_mean,
            "idm_log_standard_deviation": self.idm_log_standard_deviation,
            "idm_lower": self.idm_lower,
            "idm_upper": self.idm_upper,
            "slice_width": self.slice_width,
        }
        arrays = {
            name: np.asarray(value, dtype=float)
            for name, value in vector_fields.items()
        }
        if any(array.shape != (5,) for array in arrays.values()):
            raise ValueError("Every IDM and slice vector must have length five.")
        if any(not np.all(np.isfinite(array)) for array in arrays.values()):
            raise ValueError("IDM and slice vectors must be finite.")
        if np.any(arrays["idm_log_standard_deviation"] <= 0.0):
            raise ValueError("IDM log prior standard deviations must be positive.")
        if np.any(arrays["idm_lower"] <= 0.0) or np.any(
            arrays["idm_upper"] <= arrays["idm_lower"]
        ):
            raise ValueError("IDM bounds must be positive and ordered.")
        prior_mean = np.exp(arrays["idm_log_mean"])
        if np.any(prior_mean < arrays["idm_lower"]) or np.any(
            prior_mean > arrays["idm_upper"]
        ):
            raise ValueError("IDM prior mean must lie inside the hard bounds.")
        if np.any(arrays["slice_width"] <= 0.0):
            raise ValueError("Slice widths must be positive.")
        if self.slice_max_steps_out <= 0 or self.slice_max_shrink <= 0:
            raise ValueError("Slice iteration limits must be positive.")

        scalar_positive = (
            self.residual_variance_shape,
            self.residual_variance_scale,
            self.scenario_mean_strength,
            self.scenario_degrees_of_freedom,
            self.transition_legacy_alpha,
            self.transition_per_cell_concentration,
            self.transition_row_total_concentration,
            self.initial_total_concentration,
        )
        if any(not np.isfinite(value) or value <= 0.0 for value in scalar_positive):
            raise ValueError("All prior hyperparameters must be finite and positive.")
        if self.scenario_degrees_of_freedom <= 2.0:
            raise ValueError("Scenario NIW degrees of freedom must exceed dimension - 1.")
        scenario_mean = np.asarray(self.scenario_mean, dtype=float)
        scenario_scale = np.asarray(self.scenario_scale_diagonal, dtype=float)
        if scenario_mean.shape != (3,) or scenario_scale.shape != (3,):
            raise ValueError("Scenario mean and scale diagonal must have length three.")
        if not np.all(np.isfinite(scenario_mean)) or np.any(scenario_scale <= 0.0):
            raise ValueError("Scenario prior values must be finite with positive scale.")
        if tuple(self.context_order) != COVARIATE_ORDER:
            raise ValueError("Canonical context order must be [speed, gap, relative_speed].")

        expected_per_cell = self.transition_legacy_alpha / self.num_behavior_states
        expected_row_total = (
            expected_per_cell * self.num_behavior_states * self.num_scenario_states
        )
        if not np.isclose(
            self.transition_per_cell_concentration,
            expected_per_cell,
            rtol=0.0,
            atol=1.0e-12,
        ):
            raise ValueError(
                "transition_per_cell_concentration does not equal "
                "legacy_alpha / num_behavior_states."
            )
        if not np.isclose(
            self.transition_row_total_concentration,
            expected_row_total,
            rtol=0.0,
            atol=1.0e-12,
        ):
            raise ValueError(
                "transition_row_total_concentration is inconsistent with the "
                "legacy effective cell concentration."
            )

    @property
    def num_joint_states(self) -> int:
        return self.num_behavior_states * self.num_scenario_states

    @property
    def seed(self) -> int:
        """A compact receipt for the configured NumPy SeedSequence child."""

        sequence = np.random.SeedSequence(
            self.master_seed, spawn_key=(self.seed_spawn_index,)
        )
        return int(sequence.generate_state(1, dtype=np.uint64)[0])

    def make_rng(self) -> np.random.Generator:
        return np.random.default_rng(
            np.random.SeedSequence(
                self.master_seed, spawn_key=(self.seed_spawn_index,)
            )
        )

    @property
    def num_retained_draws(self) -> int:
        return len(range(self.burn_in, self.num_iterations, self.thin))

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "ReconstructionConfig":
        state = raw["states"]
        data = raw["data"]
        speed_policy = data.get("observed_negative_speed_policy", {})
        chain = raw["chain"]
        idm = raw["idm"]
        residual = raw["residual_variance_prior"]
        scenario = raw["scenario_niw_prior"]
        transition = raw["transition_prior"]
        initial = raw["initial_distribution_prior"]
        if idm.get("variant") != "historical_untruncated_desired_gap":
            raise ValueError("Config must select the historical untruncated IDM.")
        if tuple(idm.get("parameter_order", ())) != (
            "desired_speed",
            "minimum_gap",
            "time_headway",
            "maximum_acceleration",
            "comfortable_deceleration",
        ):
            raise ValueError("Config IDM parameter order is not canonical.")
        if residual.get("family") != "inverse_gamma_shape_scale":
            raise ValueError("Residual prior must use shape-scale inverse-Gamma.")
        if transition.get("family") != "rowwise_symmetric_dirichlet":
            raise ValueError("Transition prior must be rowwise symmetric Dirichlet.")
        if initial.get("family") != "symmetric_dirichlet":
            raise ValueError("Initial-state prior must be symmetric Dirichlet.")
        return cls(
            schema_version=int(raw["schema_version"]),
            model_id=str(raw["model_id"]),
            arm_id=str(raw["arm_id"]),
            expected_num_trajectories=int(data["expected_num_trajectories"]),
            sampling_frequency_hz=float(data["sampling_frequency_hz"]),
            allow_observed_negative_speed=speed_policy.get(
                "allow_observed_negative_speed", False
            ),
            minimum_allowed_speed_mps=float(
                speed_policy.get("minimum_allowed_speed_mps", 0.0)
            ),
            expected_negative_speed_count=int(
                speed_policy.get("expected_negative_speed_count", 0)
            ),
            expected_negative_speed_fraction=float(
                speed_policy.get("expected_negative_speed_fraction", 0.0)
            ),
            expected_negative_speed_affected_trajectories=int(
                speed_policy.get("expected_affected_trajectory_count", 0)
            ),
            negative_speed_unaffected_case_indices=tuple(
                int(index)
                for index in speed_policy.get(
                    "unaffected_case_indices_zero_based", []
                )
            ),
            preserve_raw_observations=speed_policy.get(
                "raw_observations_preserved", False
            ),
            negative_speed_justification=str(
                speed_policy.get("justification", "")
            ),
            num_behavior_states=int(state["behavior"]),
            num_scenario_states=int(state["scenario"]),
            master_seed=int(chain["master_seed"]),
            seed_spawn_index=int(chain["seed_spawn_index"]),
            num_iterations=int(chain["num_iterations"]),
            burn_in=int(chain["burn_in"]),
            thin=int(chain["thin"]),
            store_latent_paths=bool(chain["store_latent_paths"]),
            acceleration_exponent=float(idm["acceleration_exponent"]),
            idm_log_mean=_tuple_of_floats(idm["log_prior_mean"], 5, "log_prior_mean"),
            idm_log_standard_deviation=_tuple_of_floats(
                idm["log_prior_standard_deviation"], 5, "log prior standard deviation"
            ),
            idm_lower=_tuple_of_floats(idm["lower"], 5, "IDM lower bounds"),
            idm_upper=_tuple_of_floats(idm["upper"], 5, "IDM upper bounds"),
            slice_width=_tuple_of_floats(idm["slice_width"], 5, "slice width"),
            slice_max_steps_out=int(idm["slice_max_steps_out"]),
            slice_max_shrink=int(idm["slice_max_shrink"]),
            residual_variance_shape=float(residual["shape"]),
            residual_variance_scale=float(residual["scale"]),
            scenario_mean=_tuple_of_floats(scenario["mean"], 3, "scenario mean"),
            scenario_mean_strength=float(scenario["mean_strength"]),
            scenario_degrees_of_freedom=float(scenario["degrees_of_freedom"]),
            scenario_scale_diagonal=_tuple_of_floats(
                scenario["scale_diagonal"], 3, "scenario scale diagonal"
            ),
            transition_legacy_alpha=float(transition["legacy_alpha"]),
            transition_per_cell_concentration=float(
                transition["per_cell_concentration"]
            ),
            transition_row_total_concentration=float(
                transition["row_total_concentration"]
            ),
            initial_total_concentration=float(initial["total_concentration"]),
            context_order=tuple(str(item) for item in raw["context_order"]),
        )

    @classmethod
    def from_json(cls, path: Path) -> "ReconstructionConfig":
        with path.open("r", encoding="utf-8") as handle:
            raw = json.load(handle)
        return cls.from_dict(raw)

    def scientific_target_payload(self) -> Dict[str, Any]:
        """Return only quantities defining the posterior target."""

        return {
            "model_id": self.model_id,
            "observation_design": {
                "expected_num_trajectories": self.expected_num_trajectories,
                "sampling_frequency_hz": self.sampling_frequency_hz,
                "observed_negative_speed_policy": {
                    "allow_observed_negative_speed": self.allow_observed_negative_speed,
                    "minimum_allowed_speed_mps": self.minimum_allowed_speed_mps,
                    "raw_observations_preserved": self.preserve_raw_observations,
                },
            },
            "states": {
                "behavior": self.num_behavior_states,
                "scenario": self.num_scenario_states,
                "joint": self.num_joint_states,
            },
            "transition_family": "full_joint_unrestricted",
            "context_order": list(self.context_order),
            "idm": {
                "variant": "historical_untruncated_desired_gap",
                "acceleration_exponent": self.acceleration_exponent,
                "log_prior_mean": list(self.idm_log_mean),
                "log_prior_standard_deviation": list(
                    self.idm_log_standard_deviation
                ),
                "lower": list(self.idm_lower),
                "upper": list(self.idm_upper),
            },
            "residual_variance_prior": {
                "family": "inverse_gamma_shape_scale",
                "shape": self.residual_variance_shape,
                "scale": self.residual_variance_scale,
            },
            "scenario_prior": {
                "family": "normal_inverse_wishart",
                "mean": list(self.scenario_mean),
                "mean_strength": self.scenario_mean_strength,
                "degrees_of_freedom": self.scenario_degrees_of_freedom,
                "scale_diagonal": list(self.scenario_scale_diagonal),
            },
            "transition_prior": {
                "family": "rowwise_symmetric_dirichlet",
                "legacy_alpha": self.transition_legacy_alpha,
                "per_cell_concentration": self.transition_per_cell_concentration,
                "row_total_concentration": self.transition_row_total_concentration,
            },
            "initial_distribution_prior": {
                "family": "symmetric_dirichlet",
                "total_concentration": self.initial_total_concentration,
            },
        }


@dataclass(frozen=True)
class ReconstructionState:
    paths: Tuple[IntArray, ...]
    initial_probability: FloatArray
    transition_probability: FloatArray
    idm_parameters: FloatArray
    residual_variance: FloatArray
    scenario_mean: FloatArray
    scenario_covariance: FloatArray


@dataclass(frozen=True)
class SliceDiagnostics:
    step_out_evaluations: NDArray[np.int64]
    shrink_evaluations: NDArray[np.int64]


@dataclass(frozen=True)
class ReconstructionResult:
    idm_parameters: FloatArray
    residual_variance: FloatArray
    scenario_mean: FloatArray
    scenario_covariance: FloatArray
    initial_probability: FloatArray
    transition_probability: FloatArray
    log_likelihood: FloatArray
    slice_step_out_evaluations: NDArray[np.int64]
    slice_shrink_evaluations: NDArray[np.int64]
    paths: Optional[NDArray[np.int16]]
    trajectory_offsets: NDArray[np.int64]
    final_state: ReconstructionState
    initialization_runtime_seconds: float
    iteration_runtime_seconds: FloatArray

    def as_npz_payload(self) -> Dict[str, NDArray[np.generic]]:
        payload: Dict[str, NDArray[np.generic]] = {
            "idm_parameters": self.idm_parameters,
            "residual_variance": self.residual_variance,
            "scenario_mean": self.scenario_mean,
            "scenario_covariance": self.scenario_covariance,
            "initial_probability": self.initial_probability,
            "transition_probability": self.transition_probability,
            "log_likelihood": self.log_likelihood,
            "slice_step_out_evaluations": self.slice_step_out_evaluations,
            "slice_shrink_evaluations": self.slice_shrink_evaluations,
            "trajectory_offsets": self.trajectory_offsets,
        }
        if self.paths is not None:
            payload["paths"] = self.paths
        return payload


@dataclass(frozen=True)
class ReconstructionCheckpointSnapshot:
    """Already-materialized chain progress at a complete Gibbs-step boundary."""

    completed_iterations: int
    retained_index: int
    state: ReconstructionState
    rng: np.random.Generator
    retained_history: Mapping[str, NDArray[np.generic]]
    trajectory_offsets: NDArray[np.int64]
    initialization_runtime_seconds: float
    iteration_runtime_seconds: FloatArray


CheckpointCallback = Callable[[ReconstructionCheckpointSnapshot], None]


def _validate_trajectories(
    trajectories: Sequence[StoryTrajectory],
) -> Tuple[StoryTrajectory, ...]:
    sequence = tuple(trajectories)
    if not sequence:
        raise ValueError("At least one trajectory is required.")
    identifiers = [trajectory.trajectory_id for trajectory in sequence]
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("trajectory_id values must be unique.")
    return sequence


def _rank_bins(values: FloatArray, num_bins: int) -> IntArray:
    if values.ndim != 1 or values.size < num_bins:
        raise ValueError("Rank initialization needs at least one value per state.")
    order = np.argsort(values, kind="stable")
    labels = np.empty(values.size, dtype=np.int64)
    labels[order] = np.minimum(
        (np.arange(values.size, dtype=np.int64) * num_bins) // values.size,
        num_bins - 1,
    )
    return labels


def _split_path(flat_path: IntArray, offsets: IntArray) -> Tuple[IntArray, ...]:
    return tuple(
        flat_path[offsets[index] : offsets[index + 1]].copy()
        for index in range(offsets.size - 1)
    )


def _trajectory_offsets(trajectories: Sequence[StoryTrajectory]) -> IntArray:
    lengths = np.asarray(
        [trajectory.acceleration.size for trajectory in trajectories], dtype=np.int64
    )
    return np.concatenate((np.array([0], dtype=np.int64), np.cumsum(lengths)))


def _scenario_prior(config: ReconstructionConfig) -> NormalInverseWishartPrior:
    return NormalInverseWishartPrior(
        mean=np.asarray(config.scenario_mean, dtype=float),
        mean_strength=config.scenario_mean_strength,
        degrees_of_freedom=config.scenario_degrees_of_freedom,
        scale=np.diag(np.asarray(config.scenario_scale_diagonal, dtype=float)),
    )


def _posterior_mean_transition(
    paths: Sequence[IntArray], config: ReconstructionConfig
) -> FloatArray:
    counts = count_joint_transitions(paths, config.num_joint_states).astype(float)
    parameters = counts + config.transition_per_cell_concentration
    return parameters / parameters.sum(axis=1, keepdims=True)


def _posterior_mean_initial(
    paths: Sequence[IntArray], config: ReconstructionConfig
) -> FloatArray:
    counts = count_initial_states(paths, config.num_joint_states).astype(float)
    parameters = counts + config.initial_total_concentration / config.num_joint_states
    return parameters / parameters.sum()


def initialize_state(
    trajectories: Sequence[StoryTrajectory],
    config: ReconstructionConfig,
) -> ReconstructionState:
    """Create deterministic data-informed values before seeded Gibbs updates."""

    sequence = _validate_trajectories(trajectories)
    state_space = JointStateSpace(
        config.num_behavior_states, config.num_scenario_states
    )
    offsets = _trajectory_offsets(sequence)
    acceleration = np.concatenate(
        [trajectory.acceleration for trajectory in sequence], axis=0
    )
    standardized_context = np.concatenate(
        [trajectory.standardized_context for trajectory in sequence], axis=0
    )
    behavior = _rank_bins(acceleration, config.num_behavior_states)
    scenario = _rank_bins(
        standardized_context[:, 0], config.num_scenario_states
    )
    flat_path = state_space.encode_array(behavior, scenario)
    paths = _split_path(flat_path, offsets)

    prior_mean = np.exp(np.asarray(config.idm_log_mean, dtype=float))
    idm_parameters = np.broadcast_to(
        prior_mean, (config.num_behavior_states, 5)
    ).copy()
    residual_variance = np.empty(config.num_behavior_states)
    physical_context = np.concatenate(
        [trajectory.physical_context for trajectory in sequence], axis=0
    )
    for behavior_state in range(config.num_behavior_states):
        selected = behavior == behavior_state
        predicted = original_idm_acceleration(
            idm_parameters[behavior_state],
            physical_context[selected],
            minimum_allowed_speed_mps=config.minimum_allowed_speed_mps,
        )
        residual = acceleration[selected] - predicted
        residual_variance[behavior_state] = max(
            float(np.mean(residual * residual)), 1.0e-6
        )

    prior = _scenario_prior(config)
    scenario_mean = np.empty((config.num_scenario_states, 3))
    scenario_covariance = np.empty((config.num_scenario_states, 3, 3))
    for scenario_state in range(config.num_scenario_states):
        posterior = normal_inverse_wishart_posterior(
            prior, standardized_context[scenario == scenario_state]
        )
        scenario_mean[scenario_state] = posterior.mean
        covariance_denominator = (
            posterior.degrees_of_freedom - posterior.mean.size - 1.0
        )
        if covariance_denominator <= 0.0:
            scenario_covariance[scenario_state] = posterior.scale.copy()
        else:
            scenario_covariance[scenario_state] = (
                posterior.scale / covariance_denominator
            )
        _, scenario_covariance[scenario_state] = stable_cholesky(
            scenario_covariance[scenario_state]
        )

    state = ReconstructionState(
        paths=paths,
        initial_probability=_posterior_mean_initial(paths, config),
        transition_probability=_posterior_mean_transition(paths, config),
        idm_parameters=idm_parameters,
        residual_variance=residual_variance,
        scenario_mean=scenario_mean,
        scenario_covariance=scenario_covariance,
    )
    validate_state(state, sequence, config)
    return state


def validate_state(
    state: ReconstructionState,
    trajectories: Sequence[StoryTrajectory],
    config: ReconstructionConfig,
) -> None:
    sequence = _validate_trajectories(trajectories)
    state_space = JointStateSpace(
        config.num_behavior_states, config.num_scenario_states
    )
    if len(state.paths) != len(sequence):
        raise ValueError("Every trajectory needs exactly one latent path.")
    for trajectory, path_like in zip(sequence, state.paths):
        if not np.isclose(
            trajectory.minimum_allowed_speed_mps,
            config.minimum_allowed_speed_mps,
            rtol=0.0,
            atol=1.0e-12,
        ):
            raise ValueError(
                "Trajectory speed support does not match the reconstruction config."
            )
        path = np.asarray(path_like)
        if path.shape != trajectory.acceleration.shape:
            raise ValueError("Latent path length does not match trajectory.")
        state_space.decode_array(path)

    initial = np.asarray(state.initial_probability, dtype=float)
    transition = np.asarray(state.transition_probability, dtype=float)
    if initial.shape != (config.num_joint_states,) or not np.allclose(
        initial.sum(), 1.0, atol=1.0e-12
    ):
        raise ValueError("Initial probability is not a valid joint-state vector.")
    if np.any(initial < 0.0) or not np.all(np.isfinite(initial)):
        raise ValueError("Initial probability must be finite and non-negative.")
    if transition.shape != (config.num_joint_states, config.num_joint_states):
        raise ValueError("Transition matrix has the wrong shape.")
    if (
        np.any(transition < 0.0)
        or not np.all(np.isfinite(transition))
        or not np.allclose(transition.sum(axis=1), 1.0, atol=1.0e-12)
    ):
        raise ValueError("Transition matrix must be row stochastic.")

    parameters = np.asarray(state.idm_parameters, dtype=float)
    lower = np.asarray(config.idm_lower, dtype=float)
    upper = np.asarray(config.idm_upper, dtype=float)
    if parameters.shape != (config.num_behavior_states, 5):
        raise ValueError("IDM parameter shape is invalid.")
    if (
        not np.all(np.isfinite(parameters))
        or np.any(parameters < lower)
        or np.any(parameters > upper)
    ):
        raise ValueError("IDM parameters violate finite hard support.")
    variance = np.asarray(state.residual_variance, dtype=float)
    if variance.shape != (config.num_behavior_states,) or (
        not np.all(np.isfinite(variance)) or np.any(variance <= 0.0)
    ):
        raise ValueError("Residual variances must be finite and positive.")
    means = np.asarray(state.scenario_mean, dtype=float)
    covariance = np.asarray(state.scenario_covariance, dtype=float)
    if means.shape != (config.num_scenario_states, 3) or not np.all(
        np.isfinite(means)
    ):
        raise ValueError("Scenario means have invalid shape or values.")
    if covariance.shape != (config.num_scenario_states, 3, 3):
        raise ValueError("Scenario covariance shape is invalid.")
    for matrix in covariance:
        stable_cholesky(matrix)


def _behavior_observations(
    trajectories: Sequence[StoryTrajectory],
    paths: Sequence[IntArray],
    state_space: JointStateSpace,
    behavior_state: int,
) -> Tuple[FloatArray, FloatArray]:
    context_parts = []
    acceleration_parts = []
    for trajectory, path in zip(trajectories, paths):
        behavior, _ = state_space.decode_array(path)
        selected = behavior == behavior_state
        context_parts.append(trajectory.physical_context[selected])
        acceleration_parts.append(trajectory.acceleration[selected])
    return (
        np.concatenate(context_parts, axis=0),
        np.concatenate(acceleration_parts, axis=0),
    )


def _log_idm_coordinate_target(
    log_parameters: FloatArray,
    context: FloatArray,
    acceleration: FloatArray,
    residual_variance: float,
    config: ReconstructionConfig,
) -> float:
    lower = np.log(np.asarray(config.idm_lower, dtype=float))
    upper = np.log(np.asarray(config.idm_upper, dtype=float))
    if np.any(log_parameters < lower) or np.any(log_parameters > upper):
        return -np.inf
    prior_mean = np.asarray(config.idm_log_mean, dtype=float)
    prior_sd = np.asarray(config.idm_log_standard_deviation, dtype=float)
    standardized = (log_parameters - prior_mean) / prior_sd
    log_prior = float(
        -0.5 * np.sum(standardized * standardized)
        - np.log(prior_sd).sum()
        - 0.5 * log_parameters.size * np.log(2.0 * np.pi)
    )
    if acceleration.size == 0:
        return log_prior
    predicted = original_idm_acceleration(
        np.exp(log_parameters),
        context,
        minimum_allowed_speed_mps=config.minimum_allowed_speed_mps,
    )
    log_likelihood = float(
        normal_log_density(acceleration, predicted, residual_variance).sum()
    )
    return log_prior + log_likelihood


def _bounded_coordinate_slice(
    current: FloatArray,
    coordinate: int,
    log_target: Any,
    lower: float,
    upper: float,
    width: float,
    max_steps_out: int,
    max_shrink: int,
    rng: np.random.Generator,
) -> Tuple[FloatArray, int, int]:
    current_value = float(current[coordinate])
    current_log_density = float(log_target(current))
    if not np.isfinite(current_log_density):
        raise FloatingPointError("Current IDM slice target is not finite.")
    log_height = current_log_density + np.log(rng.random())
    left = max(lower, current_value - width * rng.random())
    right = min(upper, left + width)
    steps_left = int(rng.integers(0, max_steps_out + 1))
    steps_right = max_steps_out - steps_left
    step_evaluations = 0

    candidate = current.copy()
    while steps_left > 0 and left > lower:
        candidate[coordinate] = left
        step_evaluations += 1
        if float(log_target(candidate)) <= log_height:
            break
        left = max(lower, left - width)
        steps_left -= 1
    while steps_right > 0 and right < upper:
        candidate[coordinate] = right
        step_evaluations += 1
        if float(log_target(candidate)) <= log_height:
            break
        right = min(upper, right + width)
        steps_right -= 1

    for shrink_evaluations in range(1, max_shrink + 1):
        proposal_value = rng.uniform(left, right)
        candidate = current.copy()
        candidate[coordinate] = proposal_value
        proposed_log_density = float(log_target(candidate))
        if np.isfinite(proposed_log_density) and proposed_log_density >= log_height:
            return candidate, step_evaluations, shrink_evaluations
        if proposal_value < current_value:
            left = proposal_value
        else:
            right = proposal_value
    raise RuntimeError("IDM coordinate slice sampler exceeded its shrink limit.")


def sample_idm_parameters_slice(
    state: ReconstructionState,
    trajectories: Sequence[StoryTrajectory],
    state_space: JointStateSpace,
    config: ReconstructionConfig,
    rng: np.random.Generator,
) -> Tuple[FloatArray, SliceDiagnostics]:
    parameters = np.asarray(state.idm_parameters, dtype=float).copy()
    step_out = np.zeros((config.num_behavior_states, 5), dtype=np.int64)
    shrink = np.zeros_like(step_out)
    log_lower = np.log(np.asarray(config.idm_lower, dtype=float))
    log_upper = np.log(np.asarray(config.idm_upper, dtype=float))
    width = np.asarray(config.slice_width, dtype=float)
    for behavior_state in range(config.num_behavior_states):
        context, acceleration = _behavior_observations(
            trajectories, state.paths, state_space, behavior_state
        )
        log_parameters = np.log(parameters[behavior_state])

        def log_target(candidate: FloatArray) -> float:
            return _log_idm_coordinate_target(
                candidate,
                context,
                acceleration,
                float(state.residual_variance[behavior_state]),
                config,
            )

        for coordinate in range(5):
            log_parameters, step_count, shrink_count = _bounded_coordinate_slice(
                log_parameters,
                coordinate,
                log_target,
                float(log_lower[coordinate]),
                float(log_upper[coordinate]),
                float(width[coordinate]),
                config.slice_max_steps_out,
                config.slice_max_shrink,
                rng,
            )
            step_out[behavior_state, coordinate] = step_count
            shrink[behavior_state, coordinate] = shrink_count
        parameters[behavior_state] = np.exp(log_parameters)
    return parameters, SliceDiagnostics(step_out, shrink)


def sample_residual_variances(
    idm_parameters: FloatArray,
    state: ReconstructionState,
    trajectories: Sequence[StoryTrajectory],
    state_space: JointStateSpace,
    config: ReconstructionConfig,
    rng: np.random.Generator,
) -> FloatArray:
    variances = np.empty(config.num_behavior_states)
    for behavior_state in range(config.num_behavior_states):
        context, acceleration = _behavior_observations(
            trajectories, state.paths, state_space, behavior_state
        )
        if acceleration.size:
            residual = acceleration - original_idm_acceleration(
                idm_parameters[behavior_state],
                context,
                minimum_allowed_speed_mps=config.minimum_allowed_speed_mps,
            )
            squared_error = float(residual @ residual)
        else:
            squared_error = 0.0
        posterior_shape = (
            config.residual_variance_shape + 0.5 * acceleration.size
        )
        posterior_scale = config.residual_variance_scale + 0.5 * squared_error
        variances[behavior_state] = sample_inverse_gamma(
            posterior_shape, posterior_scale, rng
        )
    return variances


def sample_scenario_emissions(
    state: ReconstructionState,
    trajectories: Sequence[StoryTrajectory],
    state_space: JointStateSpace,
    config: ReconstructionConfig,
    rng: np.random.Generator,
) -> Tuple[FloatArray, FloatArray]:
    prior = _scenario_prior(config)
    means = np.empty((config.num_scenario_states, 3))
    covariances = np.empty((config.num_scenario_states, 3, 3))
    scenario_paths = tuple(
        state_space.decode_array(path)[1] for path in state.paths
    )
    for scenario_state in range(config.num_scenario_states):
        observations = np.concatenate(
            [
                trajectory.standardized_context[path == scenario_state]
                for trajectory, path in zip(trajectories, scenario_paths)
            ],
            axis=0,
        )
        posterior = normal_inverse_wishart_posterior(prior, observations)
        means[scenario_state], covariances[scenario_state] = (
            sample_normal_inverse_wishart(posterior, rng)
        )
    return means, covariances


def _sample_paths_and_log_likelihood(
    state: ReconstructionState,
    trajectories: Sequence[StoryTrajectory],
    state_space: JointStateSpace,
    rng: np.random.Generator,
) -> Tuple[Tuple[IntArray, ...], float]:
    log_emissions = emission_log_probability_sequences(
        trajectories,
        state_space,
        state.idm_parameters,
        state.residual_variance,
        state.scenario_mean,
        state.scenario_covariance,
    )
    filtered = log_forward_filter_sequences(
        log_emissions,
        state.initial_probability,
        state.transition_probability,
    )
    paths = tuple(
        ffbs_sample_path(result, state.transition_probability, rng)
        for result in filtered
    )
    return paths, float(sum(result.log_likelihood for result in filtered))


def gibbs_step(
    state: ReconstructionState,
    trajectories: Sequence[StoryTrajectory],
    config: ReconstructionConfig,
    rng: np.random.Generator,
) -> Tuple[ReconstructionState, float, SliceDiagnostics]:
    sequence = _validate_trajectories(trajectories)
    validate_state(state, sequence, config)
    state_space = JointStateSpace(
        config.num_behavior_states, config.num_scenario_states
    )
    idm_parameters, slice_diagnostics = sample_idm_parameters_slice(
        state, sequence, state_space, config, rng
    )
    residual_variance = sample_residual_variances(
        idm_parameters, state, sequence, state_space, config, rng
    )
    scenario_mean, scenario_covariance = sample_scenario_emissions(
        state, sequence, state_space, config, rng
    )
    transition_counts = count_joint_transitions(
        state.paths, config.num_joint_states
    )
    transition_probability = sample_dirichlet_posterior_rows(
        transition_counts,
        config.transition_row_total_concentration,
        rng,
    )
    initial_probability = sample_initial_distribution(
        state.paths,
        config.num_joint_states,
        config.initial_total_concentration,
        rng,
    )
    parameter_state = ReconstructionState(
        paths=state.paths,
        initial_probability=initial_probability,
        transition_probability=transition_probability,
        idm_parameters=idm_parameters,
        residual_variance=residual_variance,
        scenario_mean=scenario_mean,
        scenario_covariance=scenario_covariance,
    )
    paths, marginal_log_likelihood = _sample_paths_and_log_likelihood(
        parameter_state, sequence, state_space, rng
    )
    updated = ReconstructionState(
        paths=paths,
        initial_probability=parameter_state.initial_probability,
        transition_probability=parameter_state.transition_probability,
        idm_parameters=parameter_state.idm_parameters,
        residual_variance=parameter_state.residual_variance,
        scenario_mean=parameter_state.scenario_mean,
        scenario_covariance=parameter_state.scenario_covariance,
    )
    validate_state(updated, sequence, config)
    return updated, marginal_log_likelihood, slice_diagnostics


def run_chain(
    trajectories: Sequence[StoryTrajectory],
    config: ReconstructionConfig,
    *,
    initial_state: Optional[ReconstructionState] = None,
    rng: Optional[np.random.Generator] = None,
    resume_checkpoint: Optional["RestoredCheckpoint"] = None,
    checkpoint_interval: Optional[int] = None,
    checkpoint_callback: Optional[CheckpointCallback] = None,
) -> ReconstructionResult:
    """Run one deterministic-seed exact chain; no convergence is implied."""

    sequence = _validate_trajectories(trajectories)
    if initial_state is not None and resume_checkpoint is not None:
        raise ValueError("initial_state and resume_checkpoint are mutually exclusive.")
    if checkpoint_callback is None and checkpoint_interval is not None:
        raise ValueError("checkpoint_interval requires a checkpoint_callback.")
    if checkpoint_callback is not None and (
        checkpoint_interval is None or checkpoint_interval <= 0
    ):
        raise ValueError("Checkpoint callbacks require a positive interval.")

    if resume_checkpoint is not None and rng is None:
        generator = resume_checkpoint.restore_generator()
    else:
        generator = config.make_rng() if rng is None else rng
    if not isinstance(generator.bit_generator, np.random.PCG64):
        raise ValueError("The frozen reconstruction requires the PCG64 bit generator.")
    if resume_checkpoint is not None and (
        generator.bit_generator.state != dict(resume_checkpoint.rng_state)
    ):
        raise ValueError("Resume generator does not match checkpoint PCG64 state.")
    if resume_checkpoint is None:
        initialization_started = time.perf_counter()
        state = (
            initialize_state(sequence, config)
            if initial_state is None
            else initial_state
        )
        validate_state(state, sequence, config)
        initialization_runtime_seconds = (
            time.perf_counter() - initialization_started
        )
        start_iteration = 0
        retained_index = 0
    else:
        start_iteration = int(resume_checkpoint.completed_iterations)
        retained_index = int(resume_checkpoint.retained_index)
        if start_iteration < 0 or start_iteration > config.num_iterations:
            raise ValueError(
                "Checkpoint completed_iterations is outside the configured schedule."
            )
        expected_retained = len(
            range(config.burn_in, start_iteration, config.thin)
        )
        if retained_index != expected_retained:
            raise ValueError(
                "Checkpoint retained_index disagrees with burn-in/thinning schedule."
            )
        if resume_checkpoint.initialization_runtime_seconds is None:
            raise ValueError("Checkpoint lacks initialization timing history.")
        if resume_checkpoint.iteration_runtime_seconds is None:
            raise ValueError("Checkpoint lacks iteration timing history.")
        initialization_runtime_seconds = float(
            resume_checkpoint.initialization_runtime_seconds
        )
        state = resume_checkpoint.state
        validate_state(state, sequence, config)
    retained_count = config.num_retained_draws
    joint_count = config.num_joint_states
    idm = np.empty((retained_count, config.num_behavior_states, 5))
    variance = np.empty((retained_count, config.num_behavior_states))
    scenario_mean = np.empty((retained_count, config.num_scenario_states, 3))
    scenario_covariance = np.empty(
        (retained_count, config.num_scenario_states, 3, 3)
    )
    initial = np.empty((retained_count, joint_count))
    transition = np.empty((retained_count, joint_count, joint_count))
    log_likelihood = np.empty(retained_count)
    slice_step_out = np.empty(
        (retained_count, config.num_behavior_states, 5), dtype=np.int64
    )
    slice_shrink = np.empty_like(slice_step_out)
    iteration_runtime_seconds = np.empty(config.num_iterations, dtype=np.float64)
    offsets = _trajectory_offsets(sequence)
    stored_paths = (
        np.empty((retained_count, int(offsets[-1])), dtype=np.int16)
        if config.store_latent_paths
        else None
    )
    history_arrays: Dict[str, NDArray[np.generic]] = {
        "idm_parameters": idm,
        "residual_variance": variance,
        "scenario_mean": scenario_mean,
        "scenario_covariance": scenario_covariance,
        "initial_probability": initial,
        "transition_probability": transition,
        "log_likelihood": log_likelihood,
        "slice_step_out_evaluations": slice_step_out,
        "slice_shrink_evaluations": slice_shrink,
    }
    if stored_paths is not None:
        history_arrays["paths"] = stored_paths
    if resume_checkpoint is not None:
        if not np.array_equal(resume_checkpoint.trajectory_offsets, offsets):
            raise ValueError("Checkpoint trajectory offsets disagree with this cohort.")
        restored_history = resume_checkpoint.retained_history
        if set(restored_history) != set(history_arrays):
            raise ValueError(
                "Checkpoint retained-history fields disagree with storage policy."
            )
        for name, destination in history_arrays.items():
            prefix = np.asarray(restored_history[name])
            if prefix.shape != destination[:retained_index].shape:
                raise ValueError(
                    f"Checkpoint retained history {name!r} has an invalid shape."
                )
            destination[:retained_index] = prefix
        timing_prefix = np.asarray(
            resume_checkpoint.iteration_runtime_seconds, dtype=np.float64
        )
        if timing_prefix.shape != (start_iteration,):
            raise ValueError("Checkpoint iteration timing prefix has an invalid shape.")
        iteration_runtime_seconds[:start_iteration] = timing_prefix

    for iteration in range(start_iteration, config.num_iterations):
        iteration_started = time.perf_counter()
        state, iteration_log_likelihood, slice_diagnostics = gibbs_step(
            state, sequence, config, generator
        )
        iteration_runtime_seconds[iteration] = time.perf_counter() - iteration_started
        should_retain = iteration >= config.burn_in and (
            iteration - config.burn_in
        ) % config.thin == 0
        if should_retain:
            idm[retained_index] = state.idm_parameters
            variance[retained_index] = state.residual_variance
            scenario_mean[retained_index] = state.scenario_mean
            scenario_covariance[retained_index] = state.scenario_covariance
            initial[retained_index] = state.initial_probability
            transition[retained_index] = state.transition_probability
            log_likelihood[retained_index] = iteration_log_likelihood
            slice_step_out[retained_index] = slice_diagnostics.step_out_evaluations
            slice_shrink[retained_index] = slice_diagnostics.shrink_evaluations
            if stored_paths is not None:
                stored_paths[retained_index] = np.concatenate(state.paths).astype(
                    np.int16, copy=False
                )
            retained_index += 1
        completed_iterations = iteration + 1
        if (
            checkpoint_callback is not None
            and completed_iterations % int(checkpoint_interval) == 0
        ):
            checkpoint_callback(
                ReconstructionCheckpointSnapshot(
                    completed_iterations=completed_iterations,
                    retained_index=retained_index,
                    state=state,
                    rng=generator,
                    retained_history={
                        name: value[:retained_index]
                        for name, value in history_arrays.items()
                    },
                    trajectory_offsets=offsets,
                    initialization_runtime_seconds=initialization_runtime_seconds,
                    iteration_runtime_seconds=iteration_runtime_seconds[
                        :completed_iterations
                    ],
                )
            )
    if retained_index != retained_count:
        raise AssertionError("Retained-draw allocation does not match chain schedule.")
    return ReconstructionResult(
        idm_parameters=idm,
        residual_variance=variance,
        scenario_mean=scenario_mean,
        scenario_covariance=scenario_covariance,
        initial_probability=initial,
        transition_probability=transition,
        log_likelihood=log_likelihood,
        slice_step_out_evaluations=slice_step_out,
        slice_shrink_evaluations=slice_shrink,
        paths=stored_paths,
        trajectory_offsets=offsets,
        final_state=state,
        initialization_runtime_seconds=initialization_runtime_seconds,
        iteration_runtime_seconds=iteration_runtime_seconds,
    )


def standardize_trajectories(
    acceleration_sequences: Sequence[ArrayLike],
    physical_context_sequences: Sequence[ArrayLike],
    trajectory_ids: Sequence[str],
    *,
    minimum_allowed_speed_mps: float = 0.0,
) -> Tuple[Tuple[StoryTrajectory, ...], ContextStandardizer]:
    if not (
        len(acceleration_sequences)
        == len(physical_context_sequences)
        == len(trajectory_ids)
    ):
        raise ValueError("Acceleration, context, and ID sequences must align.")
    standardizer = ContextStandardizer.fit(physical_context_sequences)
    trajectories = tuple(
        StoryTrajectory.from_physical(
            acceleration,
            context,
            standardizer,
            str(trajectory_id),
            minimum_allowed_speed_mps=minimum_allowed_speed_mps,
        )
        for acceleration, context, trajectory_id in zip(
            acceleration_sequences, physical_context_sequences, trajectory_ids
        )
    )
    return trajectories, standardizer
