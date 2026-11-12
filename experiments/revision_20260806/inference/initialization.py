"""Optional training-only initialization for the revised latent-state sampler.

These helpers choose a starting point only.  They do not alter any prior,
proposal, full conditional, Metropolis ratio, or FFBS operation in the sampler.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Sequence, Tuple

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ..evaluation.baselines import (
    AccelerationCalibrationData,
    calibrate_idm_acceleration,
)
from ..model import (
    ContextConditionedTransitions,
    FullJointTransitions,
    IndependentTransitions,
    JointStateSpace,
    count_context_conditioned_transitions,
    count_independent_transitions,
    count_initial_states,
    count_joint_transitions,
    dirichlet_prior_parameters,
    log_forward_filter_sequences,
)
from .emissions import emission_log_probability_sequences, idm_acceleration
from .structures import SamplerConfig, SamplerPriors, SamplerState, Trajectory


FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


@dataclass(frozen=True)
class KMeansDiagnostics:
    iterations: int
    inertia: float
    empty_cluster_repairs: int


@dataclass(frozen=True)
class InitializationDiagnostics:
    strategy: str
    seed: int
    num_trajectories: int
    num_observations: int
    scenario_counts: IntArray
    behavior_counts: IntArray
    initial_joint_counts: IntArray
    within_trajectory_transition_count: int
    excluded_boundary_count: int
    scenario_kmeans: KMeansDiagnostics
    behavior_kmeans: KMeansDiagnostics
    pooled_idm_parameters: FloatArray
    pooled_idm_rmse: float
    pooled_idm_success: bool
    pooled_idm_function_evaluations: int
    behavior_fit_success: Tuple[bool, ...]
    behavior_fit_function_evaluations: Tuple[int, ...]
    behavior_fit_fallback: Tuple[bool, ...]
    behavior_idm_parameters_before_perturbation: FloatArray
    initial_idm_parameters: FloatArray
    initial_residual_variance: FloatArray
    residual_variance_floor: float
    scenario_covariance_floor: float
    log_parameter_perturbation_sd: float
    realized_log_parameter_perturbation: FloatArray
    initial_marginal_log_likelihood: float

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class DataInformedInitialization:
    state: SamplerState
    diagnostics: InitializationDiagnostics


@dataclass(frozen=True)
class _KMeansResult:
    labels: IntArray
    centers: FloatArray
    diagnostics: KMeansDiagnostics


def _validate_trajectories(
    trajectories: Sequence[Trajectory],
) -> Tuple[Trajectory, ...]:
    validated = tuple(trajectories)
    if not validated:
        raise ValueError("At least one training trajectory is required.")
    if len({trajectory.trajectory_id for trajectory in validated}) != len(validated):
        raise ValueError("Training trajectory_id values must be unique.")
    return validated


def _repair_empty_clusters(
    labels: IntArray,
    squared_distance: FloatArray,
    num_clusters: int,
) -> Tuple[IntArray, int]:
    repaired = labels.copy()
    repairs = 0
    counts = np.bincount(repaired, minlength=num_clusters)
    for empty_cluster in np.flatnonzero(counts == 0):
        assigned_distance = squared_distance[
            np.arange(repaired.size), repaired
        ].copy()
        assigned_distance[counts[repaired] <= 1] = -np.inf
        point = int(np.argmax(assigned_distance))
        if not np.isfinite(assigned_distance[point]):
            raise ValueError("Cannot repair an empty cluster without emptying another.")
        old_cluster = int(repaired[point])
        repaired[point] = int(empty_cluster)
        counts[old_cluster] -= 1
        counts[empty_cluster] += 1
        repairs += 1
    return repaired, repairs


def _seeded_kmeans(
    observations: ArrayLike,
    num_clusters: int,
    rng: np.random.Generator,
    *,
    max_iterations: int,
) -> _KMeansResult:
    data = np.asarray(observations, dtype=float)
    if data.ndim != 2 or data.shape[0] == 0 or not np.all(np.isfinite(data)):
        raise ValueError("K-means observations must be a non-empty finite matrix.")
    if num_clusters <= 0 or data.shape[0] < num_clusters:
        raise ValueError("K-means requires at least one observation per cluster.")
    if max_iterations <= 0:
        raise ValueError("max_iterations must be positive.")

    centers = np.empty((num_clusters, data.shape[1]), dtype=float)
    first = int(rng.integers(data.shape[0]))
    centers[0] = data[first]
    closest_distance = np.sum((data - centers[0]) ** 2, axis=1)
    selected = {first}
    for cluster in range(1, num_clusters):
        total = float(closest_distance.sum())
        if total > 0.0 and np.isfinite(total):
            point = int(rng.choice(data.shape[0], p=closest_distance / total))
        else:
            point = next(index for index in range(data.shape[0]) if index not in selected)
        selected.add(point)
        centers[cluster] = data[point]
        distance = np.sum((data - centers[cluster]) ** 2, axis=1)
        closest_distance = np.minimum(closest_distance, distance)

    previous_labels = None
    total_repairs = 0
    for iteration in range(1, max_iterations + 1):
        squared_distance = np.sum(
            (data[:, np.newaxis, :] - centers[np.newaxis, :, :]) ** 2,
            axis=2,
        )
        labels = np.argmin(squared_distance, axis=1).astype(np.int64)
        labels, repairs = _repair_empty_clusters(
            labels, squared_distance, num_clusters
        )
        total_repairs += repairs
        new_centers = np.stack(
            [data[labels == cluster].mean(axis=0) for cluster in range(num_clusters)]
        )
        if previous_labels is not None and np.array_equal(labels, previous_labels):
            centers = new_centers
            break
        centers = new_centers
        previous_labels = labels.copy()
    else:
        iteration = max_iterations

    final_distance = np.sum(
        (data[:, np.newaxis, :] - centers[np.newaxis, :, :]) ** 2,
        axis=2,
    )
    inertia = float(final_distance[np.arange(data.shape[0]), labels].sum())
    return _KMeansResult(
        labels,
        centers,
        KMeansDiagnostics(iteration, inertia, total_repairs),
    )


def _canonicalize_1d_clusters(result: _KMeansResult) -> _KMeansResult:
    order = np.argsort(result.centers[:, 0], kind="stable")
    old_to_new = np.empty(order.size, dtype=np.int64)
    old_to_new[order] = np.arange(order.size)
    return _KMeansResult(
        old_to_new[result.labels], result.centers[order], result.diagnostics
    )


def _calibration_sequence(
    physical_context: FloatArray,
    acceleration: FloatArray,
    trajectory_id: str,
) -> AccelerationCalibrationData:
    return AccelerationCalibrationData(
        speed_mps=physical_context[:, 0],
        gap_m=physical_context[:, 1],
        relative_speed_mps=physical_context[:, 2],
        acceleration_mps2=acceleration,
        trajectory_id=trajectory_id,
    )


def _posterior_mean_rows(
    counts: ArrayLike, total_concentration: float
) -> FloatArray:
    count_array = np.asarray(counts, dtype=float)
    prior = dirichlet_prior_parameters(
        count_array.shape[0], count_array.shape[1], total_concentration
    )
    posterior = count_array + prior
    return posterior / posterior.sum(axis=1, keepdims=True)


def _transition_from_paths(
    paths: Tuple[IntArray, ...],
    config: SamplerConfig,
    priors: SamplerPriors,
    state_space: JointStateSpace,
):
    transition_prior = priors.transition
    if config.transition_family == "independent":
        behavior_counts, scenario_counts = count_independent_transitions(
            paths, state_space
        )
        return IndependentTransitions(
            state_space,
            _posterior_mean_rows(
                behavior_counts,
                transition_prior.behavior_total_concentration,
            ),
            _posterior_mean_rows(
                scenario_counts,
                transition_prior.scenario_total_concentration,
            ),
        )
    if config.transition_family == "context_conditioned":
        scenario_counts, behavior_counts = count_context_conditioned_transitions(
            paths, state_space
        )
        behavior = np.stack(
            [
                _posterior_mean_rows(
                    behavior_counts[scenario],
                    transition_prior.behavior_total_concentration,
                )
                for scenario in range(state_space.num_scenario_states)
            ]
        )
        return ContextConditionedTransitions(
            state_space,
            _posterior_mean_rows(
                scenario_counts,
                transition_prior.scenario_total_concentration,
            ),
            behavior,
        )
    joint_counts = count_joint_transitions(paths, state_space.num_joint_states)
    return FullJointTransitions(
        state_space,
        _posterior_mean_rows(
            joint_counts, transition_prior.full_joint_total_concentration
        ),
    )


def _initial_probability_from_paths(
    paths: Tuple[IntArray, ...],
    state_space: JointStateSpace,
    priors: SamplerPriors,
) -> Tuple[FloatArray, IntArray]:
    counts = count_initial_states(paths, state_space.num_joint_states)
    prior = np.full(
        state_space.num_joint_states,
        priors.transition.initial_total_concentration
        / state_space.num_joint_states,
    )
    posterior = counts.astype(float) + prior
    return posterior / posterior.sum(), counts


def marginal_log_likelihood_at_state(
    trajectories: Sequence[Trajectory],
    state: SamplerState,
    state_space: JointStateSpace,
) -> float:
    emissions = emission_log_probability_sequences(trajectories, state, state_space)
    filters = log_forward_filter_sequences(
        emissions, state.initial_probability, state.transition.joint_matrix()
    )
    return float(sum(result.log_likelihood for result in filters))


def initialize_data_informed(
    trajectories: Sequence[Trajectory],
    config: SamplerConfig,
    priors: SamplerPriors,
    *,
    seed: int,
    max_kmeans_iterations: int = 30,
    idm_max_function_evaluations: int = 200,
    residual_variance_floor: float = 1.0e-4,
    scenario_covariance_floor: float = 1.0e-4,
    log_parameter_perturbation_sd: float = 0.005,
) -> DataInformedInitialization:
    """Construct one seeded start using only the supplied training trajectories."""

    training = _validate_trajectories(trajectories)
    if seed < 0:
        raise ValueError("Initialization seed must be non-negative.")
    positive_settings = (
        idm_max_function_evaluations,
        residual_variance_floor,
        scenario_covariance_floor,
    )
    if any(value <= 0 for value in positive_settings):
        raise ValueError("Fit budget and variance/covariance floors must be positive.")
    if log_parameter_perturbation_sd < 0.0:
        raise ValueError("log_parameter_perturbation_sd cannot be negative.")
    rng = np.random.default_rng(seed)
    state_space = JointStateSpace(
        config.num_behavior_states, config.num_scenario_states
    )
    physical = np.concatenate(
        [trajectory.physical_context for trajectory in training], axis=0
    )
    standardized = np.concatenate(
        [trajectory.standardized_context for trajectory in training], axis=0
    )
    acceleration = np.concatenate(
        [trajectory.acceleration for trajectory in training], axis=0
    )
    observation_count = acceleration.size
    if observation_count < max(
        state_space.num_behavior_states, state_space.num_scenario_states
    ):
        raise ValueError("Too few training observations for the requested state counts.")

    scenario_result = _seeded_kmeans(
        standardized,
        state_space.num_scenario_states,
        rng,
        max_iterations=max_kmeans_iterations,
    )
    pooled_initial = np.clip(
        np.exp(priors.idm.log_mean), priors.idm.lower, priors.idm.upper
    )
    pooled_fit = calibrate_idm_acceleration(
        (_calibration_sequence(physical, acceleration, "training_pooled"),),
        initial_parameters=pooled_initial,
        bounds=(priors.idm.lower, priors.idm.upper),
        max_nfev=idm_max_function_evaluations,
    )
    pooled_parameters = np.clip(
        pooled_fit.parameters, priors.idm.lower, priors.idm.upper
    )
    pooled_residual = acceleration - idm_acceleration(
        pooled_parameters, physical
    )
    behavior_result = _canonicalize_1d_clusters(
        _seeded_kmeans(
            pooled_residual[:, np.newaxis],
            state_space.num_behavior_states,
            rng,
            max_iterations=max_kmeans_iterations,
        )
    )

    unperturbed_parameters = np.empty((state_space.num_behavior_states, 5))
    fit_success = []
    fit_evaluations = []
    fit_fallback = []
    for behavior in range(state_space.num_behavior_states):
        selected = behavior_result.labels == behavior
        if int(selected.sum()) < 5:
            unperturbed_parameters[behavior] = pooled_parameters
            fit_success.append(False)
            fit_evaluations.append(0)
            fit_fallback.append(True)
            continue
        fit = calibrate_idm_acceleration(
            (
                _calibration_sequence(
                    physical[selected],
                    acceleration[selected],
                    f"training_behavior_{behavior}",
                ),
            ),
            initial_parameters=pooled_parameters,
            bounds=(priors.idm.lower, priors.idm.upper),
            max_nfev=idm_max_function_evaluations,
        )
        if np.all(np.isfinite(fit.parameters)):
            unperturbed_parameters[behavior] = np.clip(
                fit.parameters, priors.idm.lower, priors.idm.upper
            )
            fallback = False
        else:
            unperturbed_parameters[behavior] = pooled_parameters
            fallback = True
        fit_success.append(bool(fit.success) and not fallback)
        fit_evaluations.append(int(fit.num_function_evaluations))
        fit_fallback.append(fallback)

    realized_perturbation = rng.normal(
        0.0,
        log_parameter_perturbation_sd,
        size=unperturbed_parameters.shape,
    )
    lower_inside = np.nextafter(priors.idm.lower, priors.idm.upper)
    upper_inside = np.nextafter(priors.idm.upper, priors.idm.lower)
    idm_parameters = np.clip(
        unperturbed_parameters * np.exp(realized_perturbation),
        lower_inside,
        upper_inside,
    )
    realized_perturbation = np.log(idm_parameters / unperturbed_parameters)

    residual_variance = np.empty(state_space.num_behavior_states)
    for behavior in range(state_space.num_behavior_states):
        selected = behavior_result.labels == behavior
        residual = acceleration[selected] - idm_acceleration(
            idm_parameters[behavior], physical[selected]
        )
        residual_variance[behavior] = max(
            float(np.mean(residual * residual)), residual_variance_floor
        )

    scenario_mean = np.empty((state_space.num_scenario_states, 3))
    scenario_covariance = np.empty((state_space.num_scenario_states, 3, 3))
    for scenario in range(state_space.num_scenario_states):
        selected_context = standardized[scenario_result.labels == scenario]
        scenario_mean[scenario] = selected_context.mean(axis=0)
        centered = selected_context - scenario_mean[scenario]
        scenario_covariance[scenario] = (
            centered.T @ centered / selected_context.shape[0]
            + scenario_covariance_floor * np.eye(3)
        )

    paths = []
    offset = 0
    for trajectory in training:
        stop = offset + trajectory.acceleration.size
        paths.append(
            state_space.encode_array(
                behavior_result.labels[offset:stop],
                scenario_result.labels[offset:stop],
            )
        )
        offset = stop
    path_tuple = tuple(paths)
    initial_probability, initial_counts = _initial_probability_from_paths(
        path_tuple, state_space, priors
    )
    transition = _transition_from_paths(
        path_tuple, config, priors, state_space
    )
    state = SamplerState(
        paths=path_tuple,
        initial_probability=initial_probability,
        transition=transition,
        idm_parameters=idm_parameters,
        residual_variance=residual_variance,
        scenario_mean=scenario_mean,
        scenario_covariance=scenario_covariance,
    )
    initial_likelihood = marginal_log_likelihood_at_state(
        training, state, state_space
    )
    if not np.isfinite(initial_likelihood):
        raise FloatingPointError(
            "Data-informed initialization produced a non-finite marginal likelihood."
        )
    within_transition_count = int(
        sum(trajectory.acceleration.size - 1 for trajectory in training)
    )
    diagnostics = InitializationDiagnostics(
        strategy="data_informed_training_only",
        seed=int(seed),
        num_trajectories=len(training),
        num_observations=int(observation_count),
        scenario_counts=np.bincount(
            scenario_result.labels, minlength=state_space.num_scenario_states
        ).astype(np.int64),
        behavior_counts=np.bincount(
            behavior_result.labels, minlength=state_space.num_behavior_states
        ).astype(np.int64),
        initial_joint_counts=initial_counts,
        within_trajectory_transition_count=within_transition_count,
        excluded_boundary_count=max(0, len(training) - 1),
        scenario_kmeans=scenario_result.diagnostics,
        behavior_kmeans=behavior_result.diagnostics,
        pooled_idm_parameters=pooled_parameters,
        pooled_idm_rmse=float(pooled_fit.rmse),
        pooled_idm_success=bool(pooled_fit.success),
        pooled_idm_function_evaluations=int(pooled_fit.num_function_evaluations),
        behavior_fit_success=tuple(fit_success),
        behavior_fit_function_evaluations=tuple(fit_evaluations),
        behavior_fit_fallback=tuple(fit_fallback),
        behavior_idm_parameters_before_perturbation=(
            unperturbed_parameters.copy()
        ),
        initial_idm_parameters=idm_parameters.copy(),
        initial_residual_variance=residual_variance.copy(),
        residual_variance_floor=float(residual_variance_floor),
        scenario_covariance_floor=float(scenario_covariance_floor),
        log_parameter_perturbation_sd=float(log_parameter_perturbation_sd),
        realized_log_parameter_perturbation=realized_perturbation,
        initial_marginal_log_likelihood=initial_likelihood,
    )
    return DataInformedInitialization(state, diagnostics)
