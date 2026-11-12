"""Run a tiny synthetic demonstration; it is not a paper experiment."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np

from experiments.manuscript_storyrev_20260808.sampler import (
    ReconstructionConfig,
    run_chain,
    standardize_trajectories,
)
from experiments.manuscript_storyrev_20260808.target import original_idm_acceleration


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    config = ReconstructionConfig.from_json(
        root / "experiments/manuscript_storyrev_20260808/configs/b2s2.json"
    )
    config = replace(
        config,
        expected_num_trajectories=2,
        allow_observed_negative_speed=False,
        minimum_allowed_speed_mps=0.0,
        expected_negative_speed_count=0,
        expected_negative_speed_fraction=0.0,
        expected_negative_speed_affected_trajectories=0,
        negative_speed_unaffected_case_indices=(),
        num_iterations=4,
        burn_in=2,
        seed_spawn_index=101,
    )
    time = np.arange(20, dtype=float)
    contexts = tuple(
        np.column_stack((10 + 5 * index + 0.03 * time,
                         18 + 7 * index + np.sin(time / 4),
                         0.5 * np.cos(time / 5)))
        for index in range(2)
    )
    parameters = np.array([33.0, 2.0, 1.6, 1.5, 1.67])
    acceleration = tuple(original_idm_acceleration(parameters, item) for item in contexts)
    trajectories, _ = standardize_trajectories(acceleration, contexts, ("example-1", "example-2"))
    result = run_chain(trajectories, config)
    assert result.paths is not None
    print(json.dumps({
        "purpose": "synthetic software demonstration; not a paper result",
        "response_states": config.num_behavior_states,
        "scenario_states": config.num_scenario_states,
        "trajectories": len(trajectories),
        "retained_draws": int(result.idm_parameters.shape[0]),
        "joint_state_path_shape": list(result.paths.shape),
        "meaning": "retrospective state samples conditional on complete synthetic episodes",
    }, indent=2))


if __name__ == "__main__":
    main()
