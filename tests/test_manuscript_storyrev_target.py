import json
from pathlib import Path

import numpy as np
import pytest

from experiments.manuscript_storyrev_20260808.sampler import ReconstructionConfig
from experiments.manuscript_storyrev_20260808.target import (
    ContextStandardizer,
    original_idm_acceleration,
    scientific_target_fingerprint,
)


CONFIG_ROOT = (
    Path(__file__).resolve().parents[1]
    / "experiments"
    / "manuscript_storyrev_20260808"
    / "configs"
)
SCHEME_B_RECEIPT_ROOT = (
    Path(__file__).resolve().parents[1]
    / "experiments"
    / "manuscript_storyrev_20260808"
    / "forensic_outputs"
    / "20260808T054643.732760Z_scheme_b_scope_receipt"
)
FORMAL_STREAMS = {
    "b1s1.json": 0,
    "b2s2.json": 1,
    "b5s1.json": 2,
    "b1s5.json": 3,
    "b5s5.json": 4,
}
DIAGNOSTIC_STREAMS = {
    "smoke_b2s2.json": 100,
    "smoke_b5s5.json": 101,
    "runtime_pilot_b5s5.json": 102,
}
CANONICAL_MASTER_SEED = 20260807


def test_original_idm_matches_historical_untruncated_formula() -> None:
    parameters = np.array([31.0, 1.8, 1.3, 1.6, 2.1])
    context = np.array(
        [[12.0, 18.0, 1.5], [8.0, 7.0, -5.0], [0.0, 3.0, -2.0]]
    )
    speed, gap, relative_speed = context.T
    desired_gap = (
        parameters[1]
        + speed * parameters[2]
        + speed
        * relative_speed
        / (2.0 * np.sqrt(parameters[3] * parameters[4]))
    )
    expected = parameters[3] * (
        1.0 - (speed / parameters[0]) ** 4.0 - (desired_gap / gap) ** 2
    )

    actual = original_idm_acceleration(parameters, context)

    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1.0e-14)


def test_original_idm_does_not_clip_negative_dynamic_gap() -> None:
    parameters = np.array([30.0, 2.0, 0.1, 1.5, 2.0])
    context = np.array([[10.0, 5.0, -10.0]])
    historical = original_idm_acceleration(parameters, context)[0]
    corrected_desired_gap = parameters[1] + max(
        0.0,
        context[0, 0] * parameters[2]
        + context[0, 0]
        * context[0, 2]
        / (2.0 * np.sqrt(parameters[3] * parameters[4])),
    )
    corrected = parameters[3] * (
        1.0
        - (context[0, 0] / parameters[0]) ** 4.0
        - (corrected_desired_gap / context[0, 1]) ** 2
    )

    assert historical != pytest.approx(corrected)


def test_context_standardization_round_trip_preserves_canonical_columns() -> None:
    first = np.array([[10.0, 15.0, -1.0], [11.0, 17.0, 0.5]])
    second = np.array([[20.0, 30.0, 2.0], [25.0, 40.0, -2.0]])
    standardizer = ContextStandardizer.fit((first, second))

    transformed = standardizer.transform(first)
    restored = standardizer.inverse_transform(transformed)

    np.testing.assert_allclose(restored, first, rtol=0.0, atol=1.0e-14)


@pytest.mark.parametrize(
    ("filename", "behavior", "scenario", "per_cell", "row_total"),
    [
        ("b1s1.json", 1, 1, 1000.0, 1000.0),
        ("b2s2.json", 2, 2, 500.0, 2000.0),
        ("b5s1.json", 5, 1, 200.0, 1000.0),
        ("b1s5.json", 1, 5, 1000.0, 5000.0),
        ("b5s5.json", 5, 5, 200.0, 5000.0),
    ],
)
def test_frozen_configs_encode_legacy_effective_transition_prior(
    filename: str,
    behavior: int,
    scenario: int,
    per_cell: float,
    row_total: float,
) -> None:
    config = ReconstructionConfig.from_json(CONFIG_ROOT / filename)

    assert config.num_behavior_states == behavior
    assert config.num_scenario_states == scenario
    assert config.num_joint_states == behavior * scenario
    assert config.transition_per_cell_concentration == per_cell
    assert config.transition_row_total_concentration == row_total
    assert config.num_retained_draws == 2000


def test_formal_random_stream_lineage_matches_canonical_evidence() -> None:
    derived_seed_receipts = []
    for filename, spawn_key in FORMAL_STREAMS.items():
        config = ReconstructionConfig.from_json(CONFIG_ROOT / filename)
        assert config.master_seed == CANONICAL_MASTER_SEED
        assert config.seed_spawn_index == spawn_key
        derived_seed_receipts.append(config.seed)

    assert len(set(derived_seed_receipts)) == len(derived_seed_receipts)


@pytest.mark.parametrize(("filename", "spawn_key"), DIAGNOSTIC_STREAMS.items())
def test_diagnostic_configs_share_canonical_master_seed(
    filename: str, spawn_key: int
) -> None:
    config = ReconstructionConfig.from_json(CONFIG_ROOT / filename)

    assert config.master_seed == CANONICAL_MASTER_SEED
    assert config.seed_spawn_index == spawn_key


def test_config_rejects_a_mislabeled_transition_concentration(tmp_path: Path) -> None:
    raw = json.loads((CONFIG_ROOT / "b5s5.json").read_text(encoding="utf-8"))
    raw["transition_prior"]["row_total_concentration"] = 1.0
    path = tmp_path / "invalid.json"
    path.write_text(json.dumps(raw), encoding="utf-8")

    with pytest.raises(ValueError, match="row_total_concentration"):
        ReconstructionConfig.from_json(path)


def test_scientific_target_fingerprint_is_canonical_and_cohort_specific() -> None:
    cohort_a = "a" * 64
    cohort_b = "b" * 64
    first = scientific_target_fingerprint(
        {"states": {"behavior": 2, "scenario": 2}, "prior": [1, 2]},
        cohort_sha256=cohort_a,
    )
    reordered = scientific_target_fingerprint(
        {"prior": [1, 2], "states": {"scenario": 2, "behavior": 2}},
        cohort_sha256=cohort_a,
    )
    changed = scientific_target_fingerprint(
        {"prior": [1, 2], "states": {"scenario": 2, "behavior": 2}},
        cohort_sha256=cohort_b,
    )

    assert first == reordered
    assert first != changed


def test_smoke_and_formal_b2s2_share_the_same_scientific_target() -> None:
    formal = ReconstructionConfig.from_json(CONFIG_ROOT / "b2s2.json")
    smoke = ReconstructionConfig.from_json(CONFIG_ROOT / "smoke_b2s2.json")

    assert formal.scientific_target_payload() == smoke.scientific_target_payload()
    assert formal.num_iterations == 8000
    assert smoke.num_iterations == 2


def test_smoke_and_formal_b5s5_share_the_same_scientific_target() -> None:
    formal = ReconstructionConfig.from_json(CONFIG_ROOT / "b5s5.json")
    smoke = ReconstructionConfig.from_json(CONFIG_ROOT / "smoke_b5s5.json")

    assert formal.scientific_target_payload() == smoke.scientific_target_payload()
    assert formal.num_iterations == 8000
    assert smoke.num_iterations == 2


def test_runtime_pilot_and_formal_b5s5_share_the_same_scientific_target() -> None:
    formal = ReconstructionConfig.from_json(CONFIG_ROOT / "b5s5.json")
    pilot = ReconstructionConfig.from_json(CONFIG_ROOT / "runtime_pilot_b5s5.json")
    raw_pilot = json.loads(
        (CONFIG_ROOT / "runtime_pilot_b5s5.json").read_text(encoding="utf-8")
    )

    assert formal.scientific_target_payload() == pilot.scientific_target_payload()
    assert formal.num_iterations == 8000
    assert pilot.num_iterations == 20
    assert pilot.burn_in == 10
    assert pilot.seed_spawn_index != formal.seed_spawn_index
    assert raw_pilot["purpose"] == "RUNTIME_PILOT_NOT_MANUSCRIPT_EVIDENCE"


def test_all_current_configs_use_accurate_scheme_b_scope_wording() -> None:
    for config_path in CONFIG_ROOT.glob("*.json"):
        raw = json.loads(config_path.read_text(encoding="utf-8"))
        if "data" not in raw:
            continue
        policy = raw["data"]["observed_negative_speed_policy"]
        assert "Author-approved" not in policy["justification"]
        assert policy["justification"].startswith(
            "Scheme B preserves the frozen historical cohort byte-for-byte"
        )


@pytest.mark.skipif(
    not (SCHEME_B_RECEIPT_ROOT / "cohort_data.npz").is_file(),
    reason="The third-party highD cohort is not redistributed with the code.",
)
def test_scheme_b_receipt_changes_attribution_but_not_cohort_bytes() -> None:
    manifest = json.loads(
        (SCHEME_B_RECEIPT_ROOT / "manifest.json").read_text(encoding="utf-8")
    )
    decision = manifest["cohort"]["domain_audit"]["legacy_continuity_decision"]
    correction = manifest["receipt_correction"]

    assert "author_decision" not in decision
    assert decision["decision_source"] == (
        "scheme_b_scope_and_implementation_disclosure"
    )
    assert correction["cohort_data_copied_byte_for_byte"] is True
    assert correction["target_input_fingerprint_unchanged"] is True
    import hashlib

    actual_npz_sha256 = hashlib.sha256(
        (SCHEME_B_RECEIPT_ROOT / "cohort_data.npz").read_bytes()
    ).hexdigest()
    assert actual_npz_sha256 == correction["cohort_data_sha256"]
