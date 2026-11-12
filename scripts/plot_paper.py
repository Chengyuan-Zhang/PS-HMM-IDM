"""Regenerate the paper's principal figures from verified postprocessed draws."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import pickle
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from experiments.manuscript_storyrev_20260808 import plot_e3_polished_originals as figures
from experiments.manuscript_storyrev_20260808 import plot_state_resolution_sankey as sankey
from experiments.manuscript_storyrev_20260808.export_all_case_b_light_accessibility import (
    build_b_light_case_figure,
    configure_export,
)
from experiments.manuscript_storyrev_20260808.plot_e3_interpretation import (
    configure_matplotlib,
    plot_time_space_states,
)
from experiments.manuscript_storyrev_20260808.plot_manuscript_results import (
    _load_json,
    _load_npz,
    _verify_postprocessed,
)
from experiments.manuscript_storyrev_20260808.postprocess import PostprocessConfig, load_cohort
from scripts.prepare_cohort import PAPER_EPISODES, load_pairs_from_highd


def save_figure(figure: plt.Figure, output: Path) -> None:
    figure.savefig(output, format="pdf", bbox_inches="tight")
    plt.close(figure)


def verify_cohort(pairs: Sequence[Mapping[str, Any]], cohort: Any) -> None:
    if len(pairs) < PAPER_EPISODES:
        raise ValueError("Time-space plot needs the 100 selected follower pairs")
    for index, pair in enumerate(pairs[:PAPER_EPISODES]):
        if "lane_id" not in pair or "xFollReal" not in pair:
            raise ValueError(
                "Time-space metadata lack lane_id or xFollReal; use the raw "
                "highD files or a trusted cache containing both fields"
            )
        start, stop = (int(x) for x in cohort.offsets[index:index + 2])
        context = np.column_stack((pair["vFollReal"], pair["sReal"], pair["dvReal"]))
        if (int(pair["pair_No"]) != int(cohort.pair_nos[index])
                or int(pair["id_Foll"]) != int(cohort.follower_ids[index])
                or not np.array_equal(pair["frame_Foll"], cohort.frames[start:stop])
                or not np.array_equal(context, cohort.physical_context[start:stop])
                or not np.array_equal(pair["aFollReal"], cohort.acceleration[start:stop])):
            raise ValueError(f"Time-space metadata do not match cohort episode {index}")


def write_table_summaries(summary: Mapping[str, Any], output: Path) -> None:
    order = ("B1_S1_full_joint", "B2_S2_full_joint", "B5_S1_full_joint",
             "B1_S5_full_joint", "B5_S5_full_joint")
    with (output / "table_07_reconstruction.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("arm", "response_states", "scenario_states", "conditional_reconstruction_rmse_mps2"))
        for arm in order:
            row = summary["arms"][arm]
            writer.writerow((arm, row["states"]["behavior"], row["states"]["scenario"],
                             row["conditional_reconstruction"]["posterior_mean_rmse_mps2"]))
    main = summary["arms"]["B5_S5_full_joint"]
    for label, key in (("table_05_response_states", "behavior_states"),
                       ("table_06_scenarios", "scenario_states")):
        (output / f"{label}.json").write_text(
            json.dumps(main[key], indent=2) + "\n", encoding="utf-8"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--postprocess-directory", type=Path, required=True)
    parser.add_argument("--cohort-npz", type=Path, required=True)
    parser.add_argument("--cohort-manifest", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--highd-root", type=Path, help="Optional: also reproduce time-space Figure 9")
    source.add_argument("--trusted-pairs-cache", type=Path, help="Optional trusted legacy cache for Figure 9")
    args = parser.parse_args()
    postprocess = args.postprocess_directory.resolve()
    artifact_manifest = _load_json(postprocess / "artifact_manifest.json")
    for name in artifact_manifest.get("artifacts", {}):
        if Path(name).name != name:
            raise ValueError(f"Unsafe postprocess artifact filename: {name}")
    _verify_postprocessed(postprocess)
    summary = _load_json(postprocess / "posterior_summary.json")
    metadata = _load_json(postprocess / "postprocess_metadata.json")
    gate = _load_json(postprocess / "story_gate.json")
    completed = _load_json(postprocess / "COMPLETED.json")
    summary_hash = hashlib.sha256((postprocess / "posterior_summary.json").read_bytes()).hexdigest()
    if completed.get("posterior_summary_sha256") != summary_hash:
        raise ValueError("Completed receipt does not match the posterior summary")
    if (metadata.get("require_formal") is not True or gate.get("status") != "PASS"
            or completed.get("story_gate_status") != "PASS"):
        raise ValueError("Paper figures require completed formal postprocessing and its PASS gate")
    cohort = load_cohort(args.cohort_npz.resolve(), args.cohort_manifest.resolve())
    for key in ("content_sha256", "npz_sha256", "manifest_sha256"):
        if str(summary["cohort"][key]).lower() != str(getattr(cohort, key)).lower():
            raise ValueError(f"Cohort {key} differs from the postprocess receipt")
    config = PostprocessConfig.from_json(postprocess / "config.json")
    for arm in (config.manuscript_arm_id, config.pooled_arm_id):
        name = summary["arm_files"][arm]
        if Path(name).name != name or name not in artifact_manifest["artifacts"]:
            raise ValueError(f"Unsafe or unverified posterior file: {name}")
    main_arrays = _load_npz(postprocess / summary["arm_files"][config.manuscript_arm_id])
    pooled_arrays = _load_npz(postprocess / summary["arm_files"][config.pooled_arm_id])
    if not np.array_equal(main_arrays["trajectory_offsets"], cohort.offsets):
        raise ValueError("Main-state trajectories do not match cohort offsets")
    if not np.array_equal(pooled_arrays["trajectory_offsets"], cohort.offsets):
        raise ValueError("Pooled-model trajectories do not match cohort offsets")

    pairs = None
    if args.highd_root:
        pairs = load_pairs_from_highd(args.highd_root)
    elif args.trusted_pairs_cache:
        # Explicitly trusted pickle only; it is never included in the repository.
        with args.trusted_pairs_cache.open("rb") as handle:
            pairs = pickle.load(handle)
    if pairs is not None:
        verify_cohort(pairs, cohort)

    output = args.output_directory.resolve()
    output.mkdir(parents=True, exist_ok=False)
    sankey.configure_matplotlib()
    response, scenario, _, _ = sankey.load_canonical_panels(postprocess)
    save_figure(sankey.plot_state_resolution_sankey(response, scenario), output / "figure_05_state_resolution.pdf")
    configure_matplotlib()
    save_figure(figures.plot_transition_matrix_original_style(main_arrays, 5, 5), output / "figure_06_transitions.pdf")
    save_figure(figures.plot_context_covariance_correlation_original_style(main_arrays, 5), output / "figure_07_scenario_covariances.pdf")
    save_figure(figures.plot_response_scenario_cooccurrence(main_arrays, 5, 5), output / "figure_08a_cooccurrence.pdf")
    composition = figures.plot_response_composition_original_style(main_arrays, 5, 5)
    composition.axes[0].set_ylabel("Within-scenario response share (%)")
    save_figure(composition, output / "figure_08b_conditional_composition.pdf")
    if pairs is not None:
        time_space, segments = plot_time_space_states(main_arrays, cohort, pairs, config, 5, 5)
        figures.polish_time_space_legends(time_space, 5, 5)
        save_figure(time_space, output / "figure_09_time_space.pdf")
        with (output / "figure_09_time_space_segments.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(segments[0]))
            writer.writeheader()
            writer.writerows(segments)
    configure_export()
    for ordinal, index in enumerate((2, 3, 6, 9), 10):
        figure, _ = build_b_light_case_figure(
            index, cohort=cohort, sampling_frequency_hz=config.sampling_frequency_hz,
            main_arrays=main_arrays, pooled_arrays=pooled_arrays,
        )
        save_figure(figure, output / f"figure_{ordinal:02d}_case_{index}.pdf")
    write_table_summaries(summary, output)
    receipts = {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                for path in sorted(output.iterdir()) if path.is_file()}
    (output / "reproduction_manifest.json").write_text(json.dumps({
        "postprocess_manifest_sha256": hashlib.sha256((postprocess / "artifact_manifest.json").read_bytes()).hexdigest(),
        "cohort_content_sha256": cohort.content_sha256,
        "figures_include_time_space": pairs is not None,
        "files_sha256": receipts,
        "interpretation": "in-sample conditional reconstruction and smoothed state summaries",
    }, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {len(receipts)} verified figures and table summaries to {output}")


if __name__ == "__main__":
    main()
