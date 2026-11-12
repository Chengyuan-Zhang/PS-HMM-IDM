"""Check custom-data boundaries and the complete lightweight fitting path."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from experiments.manuscript_storyrev_20260808.target import original_idm_acceleration
from scripts.fit_custom import DEFAULT_CONFIG, fit, make_config, read_episodes


def _input_frame() -> pd.DataFrame:
    rows = []
    for name, base in (("first", 10.0), ("second", 16.0)):
        for index in range(6):
            rows.append({
                "episode_id": name,
                "time_s": index / 5,
                "speed_mps": base + index * 0.2,
                "gap_m": 15 + base / 2 + np.sin(index),
                "relative_speed_mps": np.cos(index) * 0.5,
                "acceleration_mps2": 0.1 * np.sin(index + base),
            })
    return pd.DataFrame(rows).iloc[::-1].reset_index(drop=True)


def test_custom_csv_is_sorted_and_custom_transition_prior_is_scaled(tmp_path: Path) -> None:
    path = tmp_path / "episodes.csv"
    _input_frame().to_csv(path, index=False)
    frame = read_episodes(path, 5.0)
    assert frame.iloc[0]["episode_id"] == "first"
    np.testing.assert_allclose(frame.iloc[:6]["time_s"], np.arange(6) / 5)
    raw = json.loads(DEFAULT_CONFIG.read_text(encoding="utf-8"))
    raw["states"] = {"behavior": 3, "scenario": 2}
    raw["transition_prior"]["row_total_concentration"] = 12.0
    template = tmp_path / "custom.json"
    template.write_text(json.dumps(raw), encoding="utf-8")
    config = make_config(template, 2, 5.0)
    assert config["transition_prior"]["per_cell_concentration"] == 2.0
    assert config["transition_prior"]["legacy_alpha"] == 6.0
    assert config["data"]["expected_num_trajectories"] == 2


@pytest.mark.parametrize("error", ["duplicate", "gap_in_time", "negative_speed", "zero_gap", "nan"])
def test_custom_csv_rejects_ambiguous_or_invalid_observations(tmp_path: Path, error: str) -> None:
    frame = _input_frame()
    if error == "duplicate":
        frame = pd.concat([frame, frame.iloc[[0]]], ignore_index=True)
    elif error == "gap_in_time":
        frame.loc[0, "time_s"] = 2.0
    elif error == "negative_speed":
        frame.loc[0, "speed_mps"] = -1.0
    elif error == "zero_gap":
        frame.loc[0, "gap_m"] = 0.0
    else:
        frame.loc[0, "relative_speed_mps"] = np.nan
    path = tmp_path / "invalid.csv"
    frame.to_csv(path, index=False)
    with pytest.raises(ValueError):
        read_episodes(path, 5.0)


def test_custom_fit_exports_aligned_probabilities_and_conditional_reconstruction(tmp_path: Path) -> None:
    path = tmp_path / "episodes.csv"
    _input_frame().to_csv(path, index=False)
    raw = json.loads(DEFAULT_CONFIG.read_text(encoding="utf-8"))
    raw["chain"].update(num_iterations=4, burn_in=2)
    config = tmp_path / "custom.json"
    config.write_text(json.dumps(raw), encoding="utf-8")
    output = fit(path, config, 5.0, tmp_path / "fit", checkpoint_interval=2)

    summary = json.loads((output / "posterior_summary.json").read_text(encoding="utf-8"))
    probabilities = pd.read_csv(output / "state_probabilities.csv")
    np.testing.assert_allclose(probabilities[["p_response_1", "p_response_2"]].sum(axis=1), 1)
    np.testing.assert_allclose(probabilities[["p_scenario_1", "p_scenario_2"]].sum(axis=1), 1)
    assert summary["num_episodes"] == 2
    assert summary["num_retained_draws"] == 2
    assert len(list((output / "runs").glob("*/checkpoints/iter_*/receipt.json"))) == 2
    context = probabilities[["speed_mps", "gap_m", "relative_speed_mps"]].to_numpy()
    with np.load(output / "aligned_posterior.npz", allow_pickle=False) as posterior:
        paths = posterior["joint_paths"] // 2
        params = posterior["idm_parameters"]
        expected = np.zeros(len(context))
        for draw in range(len(paths)):
            for state in range(2):
                mask = paths[draw] == state
                if mask.any():
                    expected[mask] += original_idm_acceleration(params[draw, state], context[mask])
        expected /= len(paths)
    np.testing.assert_allclose(probabilities["reconstructed_acceleration_mps2"], expected)
    rmse = np.sqrt(np.mean((probabilities["acceleration_mps2"] - expected) ** 2))
    assert summary["conditional_reconstruction_rmse_mps2"] == pytest.approx(rmse)
    with pytest.raises(FileExistsError):
        fit(path, config, 5.0, output, checkpoint_interval=2)
    assert fit(path, config, 5.0, output, checkpoint_interval=2, resume=True) == output
    assert len(list((output / "runs").glob("*/COMPLETED.json"))) == 1
    raw["chain"]["master_seed"] += 1
    config.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(ValueError, match="config differs"):
        fit(path, config, 5.0, output, checkpoint_interval=2, resume=True)


def test_custom_fit_resumes_checkpoint_to_same_chain(tmp_path: Path, monkeypatch) -> None:
    from experiments.manuscript_storyrev_20260808 import run_exact_reconstruction as engine

    path = tmp_path / "episodes.csv"
    _input_frame().to_csv(path, index=False)
    raw = json.loads(DEFAULT_CONFIG.read_text(encoding="utf-8"))
    raw["chain"].update(num_iterations=4, burn_in=1)
    config = tmp_path / "custom.json"
    config.write_text(json.dumps(raw), encoding="utf-8")
    original_write = engine.write_checkpoint

    def stop_after_checkpoint(*args, **kwargs):
        original_write(*args, **kwargs)
        raise RuntimeError("simulated interruption after durable checkpoint")

    monkeypatch.setattr(engine, "write_checkpoint", stop_after_checkpoint)
    with pytest.raises(RuntimeError, match="simulated interruption"):
        fit(path, config, 5.0, tmp_path / "resumed", checkpoint_interval=2)
    monkeypatch.setattr(engine, "write_checkpoint", original_write)
    fit(path, config, 5.0, tmp_path / "resumed", checkpoint_interval=2, resume=True)
    fit(path, config, 5.0, tmp_path / "uninterrupted", checkpoint_interval=2)
    resumed = next((tmp_path / "resumed" / "runs").glob("*/chain.npz"))
    uninterrupted = next((tmp_path / "uninterrupted" / "runs").glob("*/chain.npz"))
    with np.load(resumed, allow_pickle=False) as actual, np.load(uninterrupted, allow_pickle=False) as expected:
        assert set(actual.files) == set(expected.files)
        for name in actual.files:
            np.testing.assert_array_equal(actual[name], expected[name])


@pytest.mark.parametrize("behavior,scenario", [(1, 1), (3, 1), (1, 3)])
def test_custom_fit_supports_alternative_state_counts(tmp_path: Path, behavior: int, scenario: int) -> None:
    path = tmp_path / "episodes.csv"
    _input_frame().to_csv(path, index=False)
    raw = json.loads(DEFAULT_CONFIG.read_text(encoding="utf-8"))
    raw["states"] = {"behavior": behavior, "scenario": scenario}
    raw["chain"].update(num_iterations=3, burn_in=1)
    config = tmp_path / "custom.json"
    config.write_text(json.dumps(raw), encoding="utf-8")
    output = fit(path, config, 5.0, tmp_path / "fit", checkpoint_interval=2)
    summary = json.loads((output / "posterior_summary.json").read_text(encoding="utf-8"))
    assert summary["response_states"] == behavior
    assert summary["scenario_states"] == scenario
    matrix = np.asarray(summary["transition_matrix_posterior_mean"])
    assert matrix.shape == (behavior * scenario, behavior * scenario)
    np.testing.assert_allclose(matrix.sum(axis=1), 1.0)
