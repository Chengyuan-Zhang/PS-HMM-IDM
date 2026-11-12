"""Posterior causal scoring for held-out acceleration trajectories.

Each posterior draw runs its own causal latent-state filter.  At time ``t``
the filter first conditions on the currently observed kinematic context,
scores acceleration without conditioning on ``a_t``, and only then uses
``a_t`` to filter and propagate the state distribution.  Training-posterior
draws retain equal weight throughout evaluation: held-out observations update
trajectory-specific latent states, not the global parameter posterior.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Sequence, Tuple

import numpy as np
from numpy.typing import NDArray
from scipy.special import logsumexp

from ..inference.emissions import (
    idm_acceleration,
    multivariate_normal_log_density,
    normal_log_density,
    stable_cholesky,
)
from ..inference.structures import ContextStandardizer, Trajectory
from ..model.state_space import JointStateSpace
from .causal import (
    CausalFilterResult,
    causal_acceleration_filter,
    normal_mixture_interval,
)


FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class PosteriorDraw:
    """One retained draw from the training posterior.

    Joint-state arrays use the repository's behavior-major convention, where
    the scenario coordinate varies fastest.
    """

    idm_parameters: FloatArray
    residual_variance: FloatArray
    scenario_mean: FloatArray
    scenario_covariance: FloatArray
    initial_probability: FloatArray
    joint_transition: FloatArray

    def __post_init__(self) -> None:
        idm = np.asarray(self.idm_parameters, dtype=float)
        residual = np.asarray(self.residual_variance, dtype=float)
        scenario_mean = np.asarray(self.scenario_mean, dtype=float)
        scenario_covariance = np.asarray(self.scenario_covariance, dtype=float)
        initial = np.asarray(self.initial_probability, dtype=float)
        transition = np.asarray(self.joint_transition, dtype=float)

        if idm.ndim != 2 or idm.shape[0] == 0 or idm.shape[1] != 5:
            raise ValueError("idm_parameters must have shape (behavior_state, 5)")
        if not np.all(np.isfinite(idm)) or np.any(idm <= 0.0):
            raise ValueError("IDM parameters must be finite and positive")
        behavior_count = idm.shape[0]
        if residual.shape != (behavior_count,):
            raise ValueError("residual_variance must have one value per behavior state")
        if not np.all(np.isfinite(residual)) or np.any(residual <= 0.0):
            raise ValueError("residual variances must be finite and positive")

        if (
            scenario_mean.ndim != 2
            or scenario_mean.shape[0] == 0
            or scenario_mean.shape[1] != 3
        ):
            raise ValueError("scenario_mean must have shape (scenario_state, 3)")
        scenario_count = scenario_mean.shape[0]
        if scenario_covariance.shape != (scenario_count, 3, 3):
            raise ValueError(
                "scenario_covariance must have shape (scenario_state, 3, 3)"
            )
        if not np.all(np.isfinite(scenario_mean)):
            raise ValueError("scenario means must be finite")
        for covariance in scenario_covariance:
            stable_cholesky(covariance)

        joint_count = behavior_count * scenario_count
        if initial.shape != (joint_count,):
            raise ValueError("initial_probability has the wrong joint-state dimension")
        if not np.all(np.isfinite(initial)) or np.any(initial < 0.0):
            raise ValueError("initial probabilities must be finite and non-negative")
        if not np.isclose(initial.sum(), 1.0, atol=1.0e-10):
            raise ValueError("initial probabilities must sum to one")
        if transition.shape != (joint_count, joint_count):
            raise ValueError("joint_transition has the wrong joint-state dimensions")
        if not np.all(np.isfinite(transition)) or np.any(transition < 0.0):
            raise ValueError("joint transition must be finite and non-negative")
        if not np.allclose(transition.sum(axis=1), 1.0, atol=1.0e-10):
            raise ValueError("each joint-transition row must sum to one")

        object.__setattr__(self, "idm_parameters", idm.copy())
        object.__setattr__(self, "residual_variance", residual.copy())
        object.__setattr__(self, "scenario_mean", scenario_mean.copy())
        object.__setattr__(self, "scenario_covariance", scenario_covariance.copy())
        object.__setattr__(self, "initial_probability", initial.copy())
        object.__setattr__(self, "joint_transition", transition.copy())

    @property
    def state_space(self) -> JointStateSpace:
        return JointStateSpace(
            num_behavior_states=self.idm_parameters.shape[0],
            num_scenario_states=self.scenario_mean.shape[0],
        )


@dataclass(frozen=True)
class PosteriorPredictionRecord:
    """One held-out posterior-predictive record."""

    trajectory_id: str
    time_index: int
    observed_acceleration: float
    log_predictive_density: float
    predictive_mean: float
    predictive_variance: float
    interval_lower: float
    interval_upper: float
    interval_covered: bool


@dataclass(frozen=True)
class PosteriorPredictiveMetrics:
    """Observation-weighted metrics over all held-out records."""

    mean_log_predictive_density: float
    rmse: float
    mae: float
    interval_level: float
    interval_coverage: float
    mean_interval_width: float
    num_trajectories: int
    num_observations: int
    num_posterior_draws: int


@dataclass(frozen=True)
class PosteriorEvaluationResult:
    """Per-time records and their aggregate held-out metrics."""

    records: Tuple[PosteriorPredictionRecord, ...]
    metrics: PosteriorPredictiveMetrics


@dataclass(frozen=True)
class _DrawTrajectoryPrediction:
    filter_result: CausalFilterResult
    component_means: FloatArray
    component_standard_deviations: FloatArray


def posterior_draws_from_chain_samples(
    samples: object,
    draw_indices: Optional[Iterable[int]] = None,
) -> Tuple[PosteriorDraw, ...]:
    """Convert a ``ChainSamples``-compatible object into individual draws.

    The structural protocol intentionally also accepts a loaded ``.npz``
    namespace or another object exposing the six canonical sampler arrays.
    """

    names = (
        "idm_parameters",
        "residual_variance",
        "scenario_mean",
        "scenario_covariance",
        "initial_probability",
        "joint_transition",
    )
    arrays = {}
    for name in names:
        try:
            value = getattr(samples, name)
        except AttributeError as error:
            raise ValueError(f"samples lacks required array: {name}") from error
        arrays[name] = np.asarray(value, dtype=float)

    draw_count = arrays["idm_parameters"].shape[0]
    if draw_count == 0:
        raise ValueError("samples must contain at least one posterior draw")
    if any(array.ndim == 0 or array.shape[0] != draw_count for array in arrays.values()):
        raise ValueError("all sampler arrays must share a non-empty leading draw axis")

    if draw_indices is None:
        indices = tuple(range(draw_count))
    else:
        indices = tuple(int(index) for index in draw_indices)
        if not indices:
            raise ValueError("draw_indices must select at least one draw")
        if any(index < 0 or index >= draw_count for index in indices):
            raise ValueError("draw_indices contains an out-of-range index")

    return tuple(
        PosteriorDraw(
            idm_parameters=arrays["idm_parameters"][index],
            residual_variance=arrays["residual_variance"][index],
            scenario_mean=arrays["scenario_mean"][index],
            scenario_covariance=arrays["scenario_covariance"][index],
            initial_probability=arrays["initial_probability"][index],
            joint_transition=arrays["joint_transition"][index],
        )
        for index in indices
    )


def _draw_trajectory_prediction(
    draw: PosteriorDraw,
    trajectory: Trajectory,
    standardizer: ContextStandardizer,
) -> _DrawTrajectoryPrediction:
    state_space = draw.state_space
    standardized_context = standardizer.transform(trajectory.physical_context)
    time_count = trajectory.acceleration.size

    behavior_means = np.empty((time_count, state_space.num_behavior_states))
    behavior_log_density = np.empty_like(behavior_means)
    for behavior in range(state_space.num_behavior_states):
        mean = idm_acceleration(
            draw.idm_parameters[behavior], trajectory.physical_context
        )
        behavior_means[:, behavior] = mean
        behavior_log_density[:, behavior] = normal_log_density(
            trajectory.acceleration,
            mean,
            float(draw.residual_variance[behavior]),
        )

    scenario_log_density = np.empty((time_count, state_space.num_scenario_states))
    for scenario in range(state_space.num_scenario_states):
        scenario_log_density[:, scenario] = multivariate_normal_log_density(
            standardized_context,
            draw.scenario_mean[scenario],
            draw.scenario_covariance[scenario],
        )

    scenario_count = state_space.num_scenario_states
    component_means = np.repeat(behavior_means, scenario_count, axis=1)
    component_variances = np.repeat(
        draw.residual_variance, scenario_count
    )[None, :]
    component_standard_deviations = np.broadcast_to(
        np.sqrt(component_variances), component_means.shape
    ).copy()
    response_log_density = np.repeat(
        behavior_log_density, scenario_count, axis=1
    )
    context_log_density = np.tile(
        scenario_log_density, (1, state_space.num_behavior_states)
    )

    filter_result = causal_acceleration_filter(
        log_response_emission=response_log_density,
        component_means=component_means,
        component_standard_deviations=component_standard_deviations,
        initial_probabilities=draw.initial_probability,
        transition=draw.joint_transition,
        log_context_emission=context_log_density,
    )
    return _DrawTrajectoryPrediction(
        filter_result=filter_result,
        component_means=component_means,
        component_standard_deviations=component_standard_deviations,
    )


def evaluate_posterior_predictions(
    *,
    draws: Sequence[PosteriorDraw],
    standardizer: ContextStandardizer,
    trajectories: Sequence[Trajectory],
    interval_level: float = 0.9,
) -> PosteriorEvaluationResult:
    """Causally score held-out trajectories under a posterior mixture.

    Predictive densities are averaged in log space.  Predictive variances use
    the law of total variance across both latent states and posterior draws.
    Equal-tailed intervals are computed from the full finite Normal mixture,
    rather than from a moment-matched Normal approximation.
    """

    posterior_draws = tuple(draws)
    held_out = tuple(trajectories)
    if not posterior_draws:
        raise ValueError("at least one posterior draw is required")
    if not held_out:
        raise ValueError("at least one held-out trajectory is required")
    if not 0.0 < interval_level < 1.0:
        raise ValueError("interval_level must lie strictly between zero and one")
    if len({trajectory.trajectory_id for trajectory in held_out}) != len(held_out):
        raise ValueError("held-out trajectory_id values must be unique")

    state_space = posterior_draws[0].state_space
    if any(draw.state_space != state_space for draw in posterior_draws[1:]):
        raise ValueError("all posterior draws must use the same state space")

    records = []
    draw_count = len(posterior_draws)
    log_draw_count = np.log(float(draw_count))
    for trajectory in held_out:
        predictions = tuple(
            _draw_trajectory_prediction(draw, trajectory, standardizer)
            for draw in posterior_draws
        )
        draw_log_density = np.stack(
            [prediction.filter_result.log_predictive_density for prediction in predictions]
        )
        draw_means = np.stack(
            [prediction.filter_result.predictive_mean for prediction in predictions]
        )
        draw_variances = np.stack(
            [prediction.filter_result.predictive_variance for prediction in predictions]
        )

        posterior_log_density = logsumexp(draw_log_density, axis=0) - log_draw_count
        posterior_mean = draw_means.mean(axis=0)
        posterior_second_moment = np.mean(draw_variances + draw_means**2, axis=0)
        posterior_variance = np.maximum(
            posterior_second_moment - posterior_mean**2, 0.0
        )

        for time_index, observed in enumerate(trajectory.acceleration):
            mixture_weights = np.concatenate(
                [
                    prediction.filter_result.state_weights_before_acceleration[
                        time_index
                    ]
                    / draw_count
                    for prediction in predictions
                ]
            )
            mixture_means = np.concatenate(
                [
                    prediction.component_means[time_index]
                    for prediction in predictions
                ]
            )
            mixture_scales = np.concatenate(
                [
                    prediction.component_standard_deviations[time_index]
                    for prediction in predictions
                ]
            )
            lower, upper = normal_mixture_interval(
                mixture_weights,
                mixture_means,
                mixture_scales,
                level=interval_level,
            )
            records.append(
                PosteriorPredictionRecord(
                    trajectory_id=trajectory.trajectory_id,
                    time_index=time_index,
                    observed_acceleration=float(observed),
                    log_predictive_density=float(
                        posterior_log_density[time_index]
                    ),
                    predictive_mean=float(posterior_mean[time_index]),
                    predictive_variance=float(posterior_variance[time_index]),
                    interval_lower=lower,
                    interval_upper=upper,
                    interval_covered=bool(lower <= observed <= upper),
                )
            )

    observed = np.asarray([record.observed_acceleration for record in records])
    predicted = np.asarray([record.predictive_mean for record in records])
    errors = predicted - observed
    widths = np.asarray(
        [record.interval_upper - record.interval_lower for record in records]
    )
    metrics = PosteriorPredictiveMetrics(
        mean_log_predictive_density=float(
            np.mean([record.log_predictive_density for record in records])
        ),
        rmse=float(np.sqrt(np.mean(errors**2))),
        mae=float(np.mean(np.abs(errors))),
        interval_level=float(interval_level),
        interval_coverage=float(
            np.mean([record.interval_covered for record in records])
        ),
        mean_interval_width=float(np.mean(widths)),
        num_trajectories=len(held_out),
        num_observations=len(records),
        num_posterior_draws=draw_count,
    )
    return PosteriorEvaluationResult(records=tuple(records), metrics=metrics)
