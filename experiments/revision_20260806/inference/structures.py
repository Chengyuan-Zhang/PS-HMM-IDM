"""Validated data, priors, and state containers for revision inference."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Optional, Tuple, Union

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ..model import (
    ContextConditionedTransitions,
    FullJointTransitions,
    IndependentTransitions,
)


FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]
TransitionFamily = Literal["independent", "context_conditioned", "full_joint"]
IDMPriorFamily = Literal[
    "fixed_independent", "hierarchical_log_normal_wishart"
]
TransitionParameters = Union[
    IndependentTransitions, ContextConditionedTransitions, FullJointTransitions
]


def _float_array(value: ArrayLike, name: str) -> FloatArray:
    array = np.asarray(value, dtype=float)
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values.")
    return array


def _positive_definite_matrix(value: ArrayLike, name: str) -> FloatArray:
    matrix = _float_array(value, name)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"{name} must be a square matrix.")
    if not np.allclose(matrix, matrix.T, atol=1.0e-12):
        raise ValueError(f"{name} must be symmetric.")
    if np.min(np.linalg.eigvalsh(matrix)) <= 0.0:
        raise ValueError(f"{name} must be positive definite.")
    return matrix


def precision_wishart_scale_to_inverse_wishart_scale(
    precision_scale: ArrayLike,
) -> FloatArray:
    """Convert a precision-Wishart scale to its covariance-IW scale.

    If ``Lambda ~ Wishart(nu, W)`` under SciPy's scale-matrix
    parameterization and ``Sigma = Lambda**-1``, then
    ``Sigma ~ InvWishart(nu, W**-1)``.  The degrees of freedom are unchanged.
    """

    scale = _positive_definite_matrix(
        precision_scale, "Wishart precision scale"
    )
    inverse = np.linalg.solve(scale, np.eye(scale.shape[0]))
    inverse = 0.5 * (inverse + inverse.T)
    _positive_definite_matrix(inverse, "inverse-Wishart covariance scale")
    return np.asarray(inverse, dtype=float)


def inverse_wishart_scale_to_precision_wishart_scale(
    covariance_scale: ArrayLike,
) -> FloatArray:
    """Return the inverse conversion used by the precision formulation."""

    scale = _positive_definite_matrix(
        covariance_scale, "inverse-Wishart covariance scale"
    )
    inverse = np.linalg.solve(scale, np.eye(scale.shape[0]))
    inverse = 0.5 * (inverse + inverse.T)
    _positive_definite_matrix(inverse, "Wishart precision scale")
    return np.asarray(inverse, dtype=float)


@dataclass(frozen=True)
class ContextStandardizer:
    """Training-set context transform in the fixed ``[speed, gap, dv]`` order."""

    mean: FloatArray
    scale: FloatArray

    def __post_init__(self) -> None:
        mean = _float_array(self.mean, "mean")
        scale = _float_array(self.scale, "scale")
        if mean.shape != (3,) or scale.shape != (3,):
            raise ValueError("mean and scale must both have shape (3,).")
        if np.any(scale <= 0.0):
            raise ValueError("Every context scale must be strictly positive.")
        object.__setattr__(self, "mean", mean.copy())
        object.__setattr__(self, "scale", scale.copy())

    @classmethod
    def fit(cls, training_contexts: Tuple[ArrayLike, ...]) -> "ContextStandardizer":
        """Fit only to contexts explicitly supplied by the training split."""

        if not training_contexts:
            raise ValueError("At least one training context sequence is required.")
        arrays = tuple(_float_array(x, "training context") for x in training_contexts)
        if any(x.ndim != 2 or x.shape[1] != 3 or x.shape[0] == 0 for x in arrays):
            raise ValueError("Each training context must have shape (positive_time, 3).")
        stacked = np.concatenate(arrays, axis=0)
        scale = stacked.std(axis=0)
        if np.any(scale <= 1.0e-12):
            raise ValueError("A training context covariate has effectively zero variance.")
        return cls(stacked.mean(axis=0), scale)

    def transform(self, physical_context: ArrayLike) -> FloatArray:
        context = _float_array(physical_context, "physical_context")
        if context.ndim != 2 or context.shape[1] != 3:
            raise ValueError("physical_context must have shape (time, 3).")
        return (context - self.mean) / self.scale


@dataclass(frozen=True)
class Trajectory:
    """One independent car-following sequence used by the HMM.

    ``physical_context`` is in SI units and ``standardized_context`` must be
    produced using training-only statistics. Columns are speed, net gap, and
    follower-minus-leader relative speed.
    """

    acceleration: FloatArray
    physical_context: FloatArray
    standardized_context: FloatArray
    trajectory_id: str

    def __post_init__(self) -> None:
        acceleration = _float_array(self.acceleration, "acceleration")
        physical = _float_array(self.physical_context, "physical_context")
        standardized = _float_array(
            self.standardized_context, "standardized_context"
        )
        if acceleration.ndim != 1 or acceleration.size == 0:
            raise ValueError("acceleration must be a non-empty one-dimensional array.")
        expected = (acceleration.size, 3)
        if physical.shape != expected or standardized.shape != expected:
            raise ValueError(f"Context arrays must both have shape {expected}.")
        if np.any(physical[:, 0] < 0.0):
            raise ValueError("Follower speed must be non-negative.")
        if np.any(physical[:, 1] <= 0.0):
            raise ValueError("Observed net gaps must be strictly positive.")
        if not self.trajectory_id:
            raise ValueError("trajectory_id must be non-empty.")
        object.__setattr__(self, "acceleration", acceleration.copy())
        object.__setattr__(self, "physical_context", physical.copy())
        object.__setattr__(self, "standardized_context", standardized.copy())

    @classmethod
    def from_physical(
        cls,
        acceleration: ArrayLike,
        physical_context: ArrayLike,
        standardizer: ContextStandardizer,
        trajectory_id: str,
    ) -> "Trajectory":
        physical = np.asarray(physical_context, dtype=float)
        return cls(
            np.asarray(acceleration, dtype=float),
            physical,
            standardizer.transform(physical),
            trajectory_id,
        )


@dataclass(frozen=True)
class NormalInverseWishartPrior:
    mean: FloatArray
    mean_strength: float
    degrees_of_freedom: float
    scale: FloatArray

    def __post_init__(self) -> None:
        mean = _float_array(self.mean, "NIW mean")
        scale = _positive_definite_matrix(self.scale, "NIW scale")
        dimension = mean.size
        if mean.ndim != 1 or scale.shape != (dimension, dimension):
            raise ValueError("NIW mean/scale dimensions are incompatible.")
        if self.mean_strength <= 0.0 or not np.isfinite(self.mean_strength):
            raise ValueError("NIW mean_strength must be finite and positive.")
        if self.degrees_of_freedom <= dimension - 1:
            raise ValueError("NIW degrees_of_freedom must exceed dimension - 1.")
        object.__setattr__(self, "mean", mean.copy())
        object.__setattr__(self, "scale", scale.copy())

    @classmethod
    def from_precision_wishart(
        cls,
        *,
        mean: ArrayLike,
        mean_strength: float,
        degrees_of_freedom: float,
        precision_scale: ArrayLike,
    ) -> "NormalInverseWishartPrior":
        """Represent a manuscript Normal-Wishart prior in covariance form."""

        return cls(
            mean=np.asarray(mean, dtype=float),
            mean_strength=mean_strength,
            degrees_of_freedom=degrees_of_freedom,
            scale=precision_wishart_scale_to_inverse_wishart_scale(
                precision_scale
            ),
        )

    @property
    def precision_wishart_scale(self) -> FloatArray:
        """Return ``W`` when this NIW prior is written on precision."""

        return inverse_wishart_scale_to_precision_wishart_scale(self.scale)


@dataclass(frozen=True)
class LogIDMPrior:
    """Independent Gaussian prior on log IDM parameters with hard bounds."""

    log_mean: FloatArray = field(
        default_factory=lambda: np.log(np.array([33.3, 2.0, 1.5, 1.4, 2.0]))
    )
    log_standard_deviation: FloatArray = field(
        default_factory=lambda: np.array([0.35, 0.70, 0.40, 0.50, 0.50])
    )
    lower: FloatArray = field(
        default_factory=lambda: np.array([5.0, 0.1, 0.1, 0.1, 0.1])
    )
    upper: FloatArray = field(
        default_factory=lambda: np.array([60.0, 15.0, 5.0, 5.0, 8.0])
    )
    proposal_standard_deviation: FloatArray = field(
        default_factory=lambda: np.full(5, 0.04)
    )

    def __post_init__(self) -> None:
        values = {
            name: _float_array(getattr(self, name), name)
            for name in (
                "log_mean",
                "log_standard_deviation",
                "lower",
                "upper",
                "proposal_standard_deviation",
            )
        }
        if any(value.shape != (5,) for value in values.values()):
            raise ValueError("Every IDM prior vector must have shape (5,).")
        if np.any(values["log_standard_deviation"] <= 0.0):
            raise ValueError("IDM prior standard deviations must be positive.")
        if np.any(values["proposal_standard_deviation"] <= 0.0):
            raise ValueError("IDM proposal standard deviations must be positive.")
        if np.any(values["lower"] <= 0.0) or np.any(
            values["upper"] <= values["lower"]
        ):
            raise ValueError("IDM bounds must be positive and ordered.")
        for name, value in values.items():
            object.__setattr__(self, name, value.copy())


@dataclass(frozen=True)
class LogNormalWishartPrior:
    """Normal-Wishart hierarchy for the five log-IDM coefficients.

    The parameterization is

    ``eta_k | m, Lambda ~ N(m, Lambda^-1)``,
    ``m | Lambda ~ N(m0, (kappa0 Lambda)^-1)``, and
    ``Lambda ~ Wishart(nu0, W0)``,

    where ``W0`` is the SciPy Wishart *scale* matrix for precision.
    """

    log_mean_location: FloatArray = field(
        default_factory=lambda: np.log(
            np.array([33.0, 2.0, 1.6, 1.5, 1.67])
        )
    )
    mean_strength: float = 0.01
    degrees_of_freedom: float = 7.0
    precision_scale: FloatArray = field(
        default_factory=lambda: np.diag(np.full(5, 0.1))
    )

    def __post_init__(self) -> None:
        location = _float_array(
            self.log_mean_location, "IDM hierarchy log-mean location"
        )
        scale = _positive_definite_matrix(
            self.precision_scale, "IDM hierarchy Wishart precision scale"
        )
        if location.shape != (5,) or scale.shape != (5, 5):
            raise ValueError(
                "IDM hierarchy location/scale must have shapes (5,) and (5, 5)."
            )
        if not np.isfinite(self.mean_strength) or self.mean_strength <= 0.0:
            raise ValueError(
                "IDM hierarchy mean_strength must be finite and positive."
            )
        if (
            not np.isfinite(self.degrees_of_freedom)
            or self.degrees_of_freedom <= 4.0
        ):
            raise ValueError(
                "IDM hierarchy degrees_of_freedom must exceed dimension - 1."
            )
        object.__setattr__(self, "log_mean_location", location.copy())
        object.__setattr__(self, "precision_scale", scale.copy())


@dataclass(frozen=True)
class InverseGammaPrior:
    shape: float = 2.5
    scale: float = 1.0

    def __post_init__(self) -> None:
        if (
            not np.isfinite(self.shape)
            or not np.isfinite(self.scale)
            or self.shape <= 0.0
            or self.scale <= 0.0
        ):
            raise ValueError("Inverse-gamma shape and scale must be finite and positive.")


@dataclass(frozen=True)
class TransitionPriors:
    initial_total_concentration: float = 1.0
    behavior_total_concentration: float = 1.0
    scenario_total_concentration: float = 1.0
    full_joint_total_concentration: float = 1.0

    def __post_init__(self) -> None:
        values = (
            self.initial_total_concentration,
            self.behavior_total_concentration,
            self.scenario_total_concentration,
            self.full_joint_total_concentration,
        )
        if any(not np.isfinite(value) or value <= 0.0 for value in values):
            raise ValueError("All transition concentrations must be finite and positive.")


@dataclass(frozen=True)
class SamplerPriors:
    idm: LogIDMPrior = field(default_factory=LogIDMPrior)
    residual_variance: InverseGammaPrior = field(default_factory=InverseGammaPrior)
    scenario: NormalInverseWishartPrior = field(
        default_factory=lambda: NormalInverseWishartPrior(
            mean=np.zeros(3),
            mean_strength=0.1,
            degrees_of_freedom=5.0,
            scale=np.eye(3),
        )
    )
    transition: TransitionPriors = field(default_factory=TransitionPriors)
    idm_prior_family: IDMPriorFamily = "fixed_independent"
    idm_hierarchy: LogNormalWishartPrior = field(
        default_factory=LogNormalWishartPrior
    )

    def __post_init__(self) -> None:
        if self.idm_prior_family not in {
            "fixed_independent",
            "hierarchical_log_normal_wishart",
        }:
            raise ValueError("Unknown IDM prior family.")


@dataclass(frozen=True)
class SamplerConfig:
    num_behavior_states: int
    num_scenario_states: int
    transition_family: TransitionFamily
    num_iterations: int
    burn_in: int = 0
    thin: int = 1
    store_latent_paths: bool = False
    adapt_idm_proposals: bool = False
    idm_adaptation_target_acceptance: float = 0.30
    idm_adaptation_gain_scale: float = 0.50
    idm_adaptation_gain_offset: float = 10.0
    idm_adaptation_gain_exponent: float = 0.60
    idm_proposal_standard_deviation_minimum: float = 1.0e-4
    idm_proposal_standard_deviation_maximum: float = 1.0
    use_idm_block_proposal: bool = False
    adapt_idm_block_scale: bool = False
    idm_block_initial_scale: float = 1.0
    idm_block_adaptation_target_acceptance: float = 0.234
    idm_block_adaptation_gain_scale: float = 0.50
    idm_block_adaptation_gain_offset: float = 10.0
    idm_block_adaptation_gain_exponent: float = 0.60
    idm_block_scale_minimum: float = 1.0e-3
    idm_block_scale_maximum: float = 10.0

    def __post_init__(self) -> None:
        if self.num_behavior_states <= 0 or self.num_scenario_states <= 0:
            raise ValueError("State counts must be positive.")
        if self.transition_family not in {
            "independent",
            "context_conditioned",
            "full_joint",
        }:
            raise ValueError("Unknown transition_family.")
        if self.num_iterations <= 0 or self.burn_in < 0 or self.thin <= 0:
            raise ValueError("Invalid iteration, burn-in, or thinning setting.")
        if self.burn_in >= self.num_iterations:
            raise ValueError("burn_in must be smaller than num_iterations.")
        if self.adapt_idm_proposals and self.burn_in <= 0:
            raise ValueError("Proposal adaptation requires a positive burn_in.")
        if not isinstance(self.use_idm_block_proposal, bool) or not isinstance(
            self.adapt_idm_block_scale, bool
        ):
            raise ValueError("IDM block proposal switches must be boolean.")
        if self.adapt_idm_block_scale and not self.use_idm_block_proposal:
            raise ValueError(
                "IDM block-scale adaptation requires the block proposal."
            )
        if self.adapt_idm_block_scale and self.burn_in <= 0:
            raise ValueError(
                "IDM block-scale adaptation requires a positive burn_in."
            )
        if not 0.0 < self.idm_adaptation_target_acceptance < 1.0:
            raise ValueError("IDM adaptation target acceptance must lie in (0, 1).")
        adaptation_values = (
            self.idm_adaptation_gain_scale,
            self.idm_adaptation_gain_offset,
            self.idm_adaptation_gain_exponent,
            self.idm_proposal_standard_deviation_minimum,
            self.idm_proposal_standard_deviation_maximum,
        )
        if any(not np.isfinite(value) or value <= 0.0 for value in adaptation_values):
            raise ValueError("IDM adaptation gains and proposal bounds must be positive.")
        if (
            self.idm_proposal_standard_deviation_maximum
            <= self.idm_proposal_standard_deviation_minimum
        ):
            raise ValueError("IDM proposal adaptation bounds must be strictly ordered.")
        if not 0.0 < self.idm_block_adaptation_target_acceptance < 1.0:
            raise ValueError(
                "IDM block adaptation target acceptance must lie in (0, 1)."
            )
        block_adaptation_values = (
            self.idm_block_initial_scale,
            self.idm_block_adaptation_gain_scale,
            self.idm_block_adaptation_gain_offset,
            self.idm_block_adaptation_gain_exponent,
            self.idm_block_scale_minimum,
            self.idm_block_scale_maximum,
        )
        if any(
            not np.isfinite(value) or value <= 0.0
            for value in block_adaptation_values
        ):
            raise ValueError(
                "IDM block scales and adaptation gains must be finite and positive."
            )
        if self.idm_block_scale_maximum <= self.idm_block_scale_minimum:
            raise ValueError("IDM block-scale bounds must be strictly ordered.")
        if not (
            self.idm_block_scale_minimum
            <= self.idm_block_initial_scale
            <= self.idm_block_scale_maximum
        ):
            raise ValueError("Initial IDM block scale must lie within its bounds.")


@dataclass(frozen=True)
class SamplerState:
    paths: Tuple[IntArray, ...]
    initial_probability: FloatArray
    transition: TransitionParameters
    idm_parameters: FloatArray
    residual_variance: FloatArray
    scenario_mean: FloatArray
    scenario_covariance: FloatArray
    idm_owned_log_parameters: Optional[FloatArray] = None
    idm_hierarchy_mean: Optional[FloatArray] = None
    idm_hierarchy_precision: Optional[FloatArray] = None

    def __post_init__(self) -> None:
        """Copy an optional authoritative slice-kernel state defensively.

        Expanded physical IDM parameters remain the emission-facing state for
        every kernel.  The additional vector is populated only by an owned
        log-coordinate kernel, whose own validation checks its layout, support,
        and exact agreement with the expanded physical view.
        """

        if self.idm_owned_log_parameters is not None:
            owned = np.asarray(self.idm_owned_log_parameters, dtype=float)
            if (
                owned.ndim != 1
                or owned.size == 0
                or not np.all(np.isfinite(owned))
            ):
                raise ValueError(
                    "idm_owned_log_parameters must be a non-empty finite vector."
                )
            object.__setattr__(
                self, "idm_owned_log_parameters", owned.copy()
            )

        hierarchy_mean = self.idm_hierarchy_mean
        hierarchy_precision = self.idm_hierarchy_precision
        if (hierarchy_mean is None) != (hierarchy_precision is None):
            raise ValueError(
                "IDM hierarchy mean and precision must be present together."
            )
        if hierarchy_mean is None:
            return
        mean = _float_array(hierarchy_mean, "IDM hierarchy mean")
        precision = _positive_definite_matrix(
            hierarchy_precision, "IDM hierarchy precision"
        )
        if mean.shape != (5,) or precision.shape != (5, 5):
            raise ValueError(
                "IDM hierarchy mean/precision must have shapes (5,) and (5, 5)."
            )
        object.__setattr__(self, "idm_hierarchy_mean", mean.copy())
        object.__setattr__(
            self, "idm_hierarchy_precision", precision.copy()
        )
