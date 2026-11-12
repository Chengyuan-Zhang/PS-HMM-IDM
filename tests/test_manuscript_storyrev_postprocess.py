"""Synthetic tests for Scheme B label alignment, schema guards, and plots."""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

import experiments.manuscript_storyrev_20260808.postprocess as postprocess_module
from experiments.manuscript_storyrev_20260808.plot_manuscript_results import (
    FIGURE_FILENAMES,
    _categorical_state_handles,
    _fixed_case_ordinal,
    _save_pdf_new,
    _time_space_metadata_sha256,
    plot_context_covariance_correlation,
    plot_state_cooccurrence,
    plot_transition_matrix,
)
from experiments.manuscript_storyrev_20260808.postprocess import (
    AlignedPosterior,
    PosteriorRun,
    PostprocessConfig,
    _effective_sample_size,
    _modal_run_length_segments,
    _physical_scenario_parameters,
    _single_trace_diagnostics,
    _write_text_new,
    align_posterior,
    behavior_response_at_anchors,
    load_posterior_run,
    pointwise_modal_categories,
)


def test_write_text_new_is_lf_normalized_and_append_only(tmp_path: Path) -> None:
    target = tmp_path / "tables.tex"

    _write_text_new(target, "first\nsecond\n")

    assert target.read_bytes() == b"first\nsecond\n"
    with pytest.raises(FileExistsError):
        _write_text_new(target, "replacement\n")


def test_modal_run_length_segments_are_contiguous_and_time_explicit() -> None:
    segments = _modal_run_length_segments(
        np.array([0, 0, 2, 2, 2, 1], dtype=np.int64),
        5.0,
        label_prefix="B",
    )

    assert [segment["state_index_zero_based"] for segment in segments] == [0, 2, 1]
    assert [segment["display_label"] for segment in segments] == ["B #1", "B #3", "B #2"]
    assert [segment["start_sample_zero_based"] for segment in segments] == [0, 2, 5]
    assert [segment["end_sample_exclusive"] for segment in segments] == [2, 5, 6]
    assert [segment["num_samples"] for segment in segments] == [2, 3, 1]
    np.testing.assert_allclose(
        [segment["start_seconds"] for segment in segments], [0.0, 0.4, 1.0]
    )
    np.testing.assert_allclose(
        [segment["end_seconds_exclusive"] for segment in segments], [0.4, 1.0, 1.2]
    )
    np.testing.assert_allclose(
        [segment["last_observation_seconds"] for segment in segments], [0.2, 0.8, 1.0]
    )
    np.testing.assert_allclose(
        [segment["case_fraction"] for segment in segments], [2 / 6, 3 / 6, 1 / 6]
    )
    assert sum(segment["num_samples"] for segment in segments) == 6


def _config() -> PostprocessConfig:
    path = (
        Path(__file__).resolve().parents[1]
        / "experiments"
        / "manuscript_storyrev_20260808"
        / "configs"
        / "postprocess.json"
    )
    return PostprocessConfig.from_json(path)


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("rare_occupancy_threshold", 0.02),
        ("empty_draw_fraction_threshold", 0.40),
        ("case_supported_occupancy_threshold", 0.10),
        ("minimum_passing_fixed_cases", 2),
    ),
)
def test_scheme_b_story_thresholds_are_frozen(field: str, value: float) -> None:
    mutated = replace(_config(), **{field: value})
    with pytest.raises(ValueError, match="Scheme B"):
        mutated.validate()


def _synthetic_permuted_run() -> tuple[PosteriorRun, dict[str, np.ndarray]]:
    draws, behavior, scenario, observations = 4, 3, 2, 8
    canonical_idm = np.array(
        [
            [24.0, 1.0, 2.4, 0.8, 2.4],
            [32.0, 2.0, 1.6, 1.4, 1.8],
            [42.0, 3.0, 0.9, 2.3, 1.1],
        ]
    )
    canonical_variance = np.array([0.2, 0.4, 0.7])
    canonical_mean = np.array([[-1.0, -0.5, 0.8], [1.2, 0.7, -0.6]])
    canonical_covariance = np.array(
        [
            [[0.5, 0.1, 0.0], [0.1, 0.8, 0.05], [0.0, 0.05, 0.4]],
            [[1.2, -0.15, 0.1], [-0.15, 0.6, -0.05], [0.1, -0.05, 0.9]],
        ]
    )
    joint = behavior * scenario
    canonical_initial = np.arange(1, joint + 1, dtype=float)
    canonical_initial /= canonical_initial.sum()
    canonical_transition = np.arange(1, joint * joint + 1, dtype=float).reshape(
        joint, joint
    )
    canonical_transition /= canonical_transition.sum(axis=1, keepdims=True)
    canonical_path = np.array([0, 1, 2, 3, 4, 5, 2, 4], dtype=np.int64)
    behavior_old_for_new = np.array(
        [[0, 1, 2], [2, 0, 1], [1, 2, 0], [0, 2, 1]], dtype=np.int64
    )
    scenario_old_for_new = np.array(
        [[0, 1], [1, 0], [0, 1], [1, 0]], dtype=np.int64
    )
    idm = np.empty((draws, behavior, 5))
    variance = np.empty((draws, behavior))
    mean = np.empty((draws, scenario, 3))
    covariance = np.empty((draws, scenario, 3, 3))
    initial = np.empty((draws, joint))
    transition = np.empty((draws, joint, joint))
    paths = np.empty((draws, observations), dtype=np.int64)
    for draw in range(draws):
        bp = behavior_old_for_new[draw]
        sp = scenario_old_for_new[draw]
        idm[draw, bp] = canonical_idm
        variance[draw, bp] = canonical_variance
        mean[draw, sp] = canonical_mean
        covariance[draw, sp] = canonical_covariance
        old_joint_for_new = (bp[:, None] * scenario + sp[None, :]).reshape(-1)
        initial[draw, old_joint_for_new] = canonical_initial
        transition[draw][np.ix_(old_joint_for_new, old_joint_for_new)] = (
            canonical_transition
        )
        canonical_b = canonical_path // scenario
        canonical_s = canonical_path % scenario
        paths[draw] = bp[canonical_b] * scenario + sp[canonical_s]
    run = PosteriorRun(
        run_directory=Path("synthetic"),
        arm_id="synthetic",
        num_behavior_states=behavior,
        num_scenario_states=scenario,
        idm_parameters=idm,
        residual_variance=variance,
        scenario_mean=mean,
        scenario_covariance=covariance,
        initial_probability=initial,
        transition_probability=transition,
        log_likelihood=np.array([10.0, 3.0, 2.0, 1.0]),
        paths=paths,
        trajectory_offsets=np.array([0, observations], dtype=np.int64),
        standardizer_mean=np.array([10.0, 20.0, 0.0]),
        standardizer_scale=np.array([2.0, 5.0, 1.0]),
        config={
            "data": {
                "observed_negative_speed_policy": {
                    "minimum_allowed_speed_mps": -0.2
                }
            }
        },
        run_metadata={},
    )
    expected = {
        "idm": canonical_idm,
        "variance": canonical_variance,
        "mean": canonical_mean,
        "covariance": canonical_covariance,
        "initial": canonical_initial,
        "transition": canonical_transition,
        "path": canonical_path,
        "behavior_permutation": behavior_old_for_new,
        "scenario_permutation": scenario_old_for_new,
    }
    return run, expected


def test_hungarian_alignment_synchronously_restores_every_state_object() -> None:
    run, expected = _synthetic_permuted_run()
    aligned = align_posterior(run, _config())
    assert isinstance(aligned, AlignedPosterior)
    assert aligned.reference_draw_index == 0
    for draw in range(run.num_draws):
        np.testing.assert_allclose(aligned.idm_parameters[draw], expected["idm"])
        np.testing.assert_allclose(
            aligned.residual_variance[draw], expected["variance"]
        )
        np.testing.assert_allclose(aligned.scenario_mean[draw], expected["mean"])
        np.testing.assert_allclose(
            aligned.scenario_covariance[draw], expected["covariance"]
        )
        np.testing.assert_allclose(
            aligned.initial_probability[draw], expected["initial"]
        )
        np.testing.assert_allclose(
            aligned.transition_probability[draw], expected["transition"]
        )
        np.testing.assert_array_equal(aligned.paths[draw], expected["path"])
    np.testing.assert_array_equal(
        aligned.behavior_old_for_new, expected["behavior_permutation"]
    )
    np.testing.assert_array_equal(
        aligned.scenario_old_for_new, expected["scenario_permutation"]
    )


def test_aligned_response_anchor_summary_is_finite_and_state_synchronous() -> None:
    run, _ = _synthetic_permuted_run()
    config = _config()
    aligned = align_posterior(run, config)
    responses = behavior_response_at_anchors(
        aligned.idm_parameters,
        config.response_anchors,
        minimum_allowed_speed_mps=-0.2,
    )
    assert responses.shape == (
        run.num_draws,
        run.num_behavior_states,
        config.response_anchors.shape[0],
    )
    assert np.all(np.isfinite(responses))
    for draw in range(1, run.num_draws):
        np.testing.assert_allclose(responses[draw], responses[0])


def test_pointwise_state_summary_is_categorical_mode_not_rounded_mean() -> None:
    paths = np.array([[0, 2], [2, 2], [2, 0]], dtype=np.int64)
    modal, probability = pointwise_modal_categories(paths, num_states=3)
    np.testing.assert_array_equal(modal, np.array([2, 2]))
    assert int(np.rint(np.mean(paths[:, 0]))) == 1
    np.testing.assert_allclose(probability.sum(axis=1), 1.0)


def test_single_chain_diagnostics_are_finite_and_explicitly_window_based() -> None:
    trace = np.sin(np.linspace(0.0, 20.0, 200)) + np.linspace(0.0, 0.1, 200)
    diagnostics = _single_trace_diagnostics(trace)
    ess = _effective_sample_size(trace)
    assert 1.0 <= ess <= trace.size
    assert diagnostics["num_retained_draws"] == trace.size
    assert set(diagnostics["autocorrelation"]) == {"1", "5", "10", "20", "50"}
    assert np.isfinite(diagnostics["absolute_window_shift_in_posterior_sd"])


def test_geyer_ess_pairs_lags_zero_one_then_two_three(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    autocorrelation = np.array([1.0, 0.4, 0.3, 0.2, -0.1])
    monkeypatch.setattr(
        postprocess_module,
        "_autocorrelation_fft",
        lambda values: autocorrelation,
    )
    expected_integrated_time = -1.0 + 2.0 * ((1.0 + 0.4) + (0.3 + 0.2))
    assert _effective_sample_size(np.arange(5.0)) == pytest.approx(
        5.0 / expected_integrated_time
    )


def test_scenario_transform_uses_v_gap_dv_and_scales_both_covariance_axes() -> None:
    run, _ = _synthetic_permuted_run()
    aligned = align_posterior(run, _config())
    physical_mean, physical_covariance = _physical_scenario_parameters(aligned, run)
    expected_mean = (
        aligned.scenario_mean * run.standardizer_scale[None, None, :]
        + run.standardizer_mean[None, None, :]
    )
    scale_matrix = np.diag(run.standardizer_scale)
    expected_covariance = np.empty_like(physical_covariance)
    for draw in range(run.num_draws):
        for state in range(run.num_scenario_states):
            expected_covariance[draw, state] = (
                scale_matrix
                @ aligned.scenario_covariance[draw, state]
                @ scale_matrix
            )
    np.testing.assert_allclose(physical_mean, expected_mean)
    np.testing.assert_allclose(physical_covariance, expected_covariance)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_synthetic_run(path: Path) -> None:
    path.mkdir()
    config = {
        "arm_id": "SMOKE_B1_S1_full_joint",
        "states": {"behavior": 1, "scenario": 1},
        "chain": {
            "num_iterations": 2,
            "burn_in": 1,
            "thin": 1,
            "store_latent_paths": True,
        },
    }
    metadata = {
        "arm_id": config["arm_id"],
        "scientific_target_fingerprint": "f" * 64,
        "num_retained_draws": 1,
    }
    objects = {
        "config.json": config,
        "standardizer.json": {
            "context_order": ["speed", "gap", "relative_speed"],
            "mean": [0.0, 0.0, 0.0],
            "scale": [1.0, 1.0, 1.0],
        },
        "target.json": {"scientific_target_fingerprint": "f" * 64},
        "seed_lineage.json": {"master_seed": 1},
        "run_metadata.json": metadata,
    }
    for name, payload in objects.items():
        (path / name).write_text(json.dumps(payload), encoding="utf-8")
    np.savez_compressed(
        path / "chain.npz",
        idm_parameters=np.ones((1, 1, 5)),
        residual_variance=np.ones((1, 1)),
        scenario_mean=np.zeros((1, 1, 3)),
        scenario_covariance=np.eye(3)[None, None, :, :],
        initial_probability=np.ones((1, 1)),
        transition_probability=np.ones((1, 1, 1)),
        log_likelihood=np.zeros(1),
        slice_step_out_evaluations=np.zeros((1, 1, 5), dtype=np.int64),
        slice_shrink_evaluations=np.zeros((1, 1, 5), dtype=np.int64),
        trajectory_offsets=np.array([0, 3], dtype=np.int64),
        paths=np.zeros((1, 3), dtype=np.int16),
    )
    names = ["chain.npz", *objects]
    manifest = {
        "artifacts": {
            name: {"sha256": _sha256(path / name), "bytes": (path / name).stat().st_size}
            for name in names
        }
    }
    (path / "artifact_manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    (path / "COMPLETED.json").write_text("{}", encoding="utf-8")


def test_output_schema_loads_complete_smoke_only_when_explicitly_allowed(
    tmp_path: Path,
) -> None:
    run_path = tmp_path / "run"
    _write_synthetic_run(run_path)
    with pytest.raises(ValueError, match="Non-formal run rejected"):
        load_posterior_run(run_path, require_formal=True)
    loaded = load_posterior_run(run_path, require_formal=False)
    assert loaded.arm_id == "SMOKE_B1_S1_full_joint"
    assert loaded.paths.shape == (1, 3)


def test_schema_rejects_missing_latent_paths(tmp_path: Path) -> None:
    run_path = tmp_path / "run"
    _write_synthetic_run(run_path)
    with np.load(run_path / "chain.npz", allow_pickle=False) as loaded:
        payload = {name: loaded[name] for name in loaded.files if name != "paths"}
    np.savez_compressed(run_path / "chain.npz", **payload)
    manifest = json.loads((run_path / "artifact_manifest.json").read_text(encoding="utf-8"))
    manifest["artifacts"]["chain.npz"] = {
        "sha256": _sha256(run_path / "chain.npz"),
        "bytes": (run_path / "chain.npz").stat().st_size,
    }
    (run_path / "artifact_manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="lacks required arrays"):
        load_posterior_run(run_path, require_formal=False)


def test_plot_helpers_write_vector_pdfs_without_changing_transition(tmp_path: Path) -> None:
    transition = np.array([[0.8, 0.2], [0.1, 0.9]], dtype=float)
    transition_before = transition.copy()
    arrays = {
        "transition_mean": transition,
        "scenario_covariance_physical_median": np.array(
            [[[4.0, 1.0, 0.2], [1.0, 9.0, -0.3], [0.2, -0.3, 1.0]]]
        ),
        "scenario_covariance_physical_mean": np.array(
            [[[4.0, 1.0, 0.2], [1.0, 9.0, -0.3], [0.2, -0.3, 1.0]]]
        ),
        "scenario_mean_physical_median": np.array([[10.0, 20.0, 0.5]]),
        "cooccurrence_mean": np.array([[0.55], [0.45]]),
    }
    figures = (
        plot_transition_matrix(arrays, behavior=2, scenario=1),
        plot_context_covariance_correlation(arrays, scenario=1),
        plot_state_cooccurrence(arrays, behavior=2, scenario=1),
    )
    for index, figure in enumerate(figures):
        output = tmp_path / f"figure_{index}.pdf"
        _save_pdf_new(figure, output)
        assert output.read_bytes().startswith(b"%PDF")
    np.testing.assert_array_equal(transition, transition_before)
    assert FIGURE_FILENAMES == (
        "transition_mat.pdf",
        "cov_corr_mat.pdf",
        "states_stats.pdf",
        "time_space_3by2_state_speed.pdf",
        "alpha1000_KB5_KS5_case2.pdf",
        "alpha1000_KB5_KS5_case3.pdf",
        "alpha1000_KB5_KS5_case6.pdf",
        "alpha1000_KB5_KS5_case9.pdf",
    )


def test_context_covariance_panels_share_one_physical_color_scale() -> None:
    arrays = {
        "scenario_covariance_physical_mean": np.array(
            [
                [[4.0, 1.0, 0.2], [1.0, 9.0, -0.3], [0.2, -0.3, 1.0]],
                [[100.0, 2.0, 0.4], [2.0, 25.0, -0.6], [0.4, -0.6, 4.0]],
            ]
        ),
        "scenario_mean_physical_median": np.array(
            [[10.0, 20.0, 0.5], [15.0, 40.0, -0.2]]
        ),
    }
    figure = plot_context_covariance_correlation(arrays, scenario=2)
    first_clim = figure.axes[0].images[0].get_clim()
    second_clim = figure.axes[1].images[0].get_clim()
    assert first_clim == second_clim == (-100.0, 100.0)


def test_context_covariance_colorbars_do_not_overlap_state_panels() -> None:
    arrays = {
        "scenario_covariance_physical_mean": np.repeat(
            np.array(
                [[[4.0, 1.0, 0.2], [1.0, 9.0, -0.3], [0.2, -0.3, 1.0]]]
            ),
            5,
            axis=0,
        ),
        "scenario_mean_physical_median": np.zeros((5, 3)),
    }
    figure = plot_context_covariance_correlation(arrays, scenario=5)
    figure.canvas.draw()

    panel_axes = figure.axes[:10]
    covariance_colorbar = figure.axes[-2].get_position()
    correlation_colorbar = figure.axes[-1].get_position()
    assert not any(
        axis.get_position().overlaps(covariance_colorbar) for axis in panel_axes[:5]
    )
    assert not any(
        axis.get_position().overlaps(correlation_colorbar) for axis in panel_axes[5:]
    )


def test_fixed_case_titles_and_categorical_legend_labels_use_manuscript_semantics() -> None:
    assert [
        _fixed_case_ordinal(index, (2, 3, 6, 9))
        for index in (2, 3, 6, 9)
    ] == ["I", "II", "III", "IV"]
    with pytest.raises(ValueError, match="not part"):
        _fixed_case_ordinal(4, (2, 3, 6, 9))
    colors = [(1.0, 0.0, 0.0, 1.0), (0.0, 1.0, 0.0, 1.0)]
    handles = _categorical_state_handles(
        np.array([1, 1, 0, 1], dtype=np.int64),
        colors,
        label_prefix="Regime",
    )
    assert [handle.get_label() for handle in handles] == ["Regime #1", "Regime #2"]


def test_time_space_metadata_hash_covers_identity_frames_positions_and_lanes() -> None:
    trajectories = [
        {
            "pair_No": 2,
            "id_Foll": 203,
            "frame_Foll": np.array([100, 105, 110], dtype=np.int64),
            "xFollReal": np.array([1.0, 2.5, 4.0], dtype=np.float64),
            "lane_id": np.array([3, 3, 3], dtype=np.int64),
        }
    ]
    copied = [
        {
            key: value.copy() if isinstance(value, np.ndarray) else value
            for key, value in trajectories[0].items()
        }
    ]
    assert _time_space_metadata_sha256(trajectories) == _time_space_metadata_sha256(
        copied
    )
    copied[0]["xFollReal"][1] += 0.01
    assert _time_space_metadata_sha256(trajectories) != _time_space_metadata_sha256(
        copied
    )
