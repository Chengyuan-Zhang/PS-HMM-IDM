"""Export the author-selected B-light accessibility style for all frozen cases."""

from __future__ import annotations

import argparse
import os
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Mapping, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

from . import export_case6_accessibility_candidates as case6_exporter
from . import plot_case_study_original_style as original
from .case_accessibility_candidates import (
    CONTEXT_STATE_COLORS,
    LINE_COLORS,
    RESPONSE_STATE_COLORS,
    apply_accessible_case_style,
    palette_audit,
    simulate_color_vision,
)
from .case_gap_axis_left import move_gap_axis_to_outer_left
from .case_legend_front_unicode import raise_case_legends_to_front
from .export_case_study_original_style_mixing_mean import (
    replace_response_with_mixing_mean,
)
from .postprocess import _sha256_file, _write_json_new


plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans", "Liberation Sans"]
plt.rcParams["svg.fonttype"] = "none"


Array = NDArray[np.generic]

SELECTED_VARIANT = "B_light"
CASE_INDICES: Tuple[int, ...] = tuple(original.CASE_FILENAMES)
CASE_OUTPUT_STEMS: Mapping[int, str] = {
    case_index: f"alpha1000_KB5_KS5_case{case_index}_B_light_accessible"
    for case_index in CASE_INDICES
}
DIAGNOSTIC_MODES: Tuple[str, ...] = (
    "normal",
    "protanopia",
    "deuteranopia",
    "tritanopia",
    "grayscale",
)


def configure_export() -> None:
    case6_exporter.configure_export()
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
        }
    )
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans", "Liberation Sans"]
    plt.rcParams["svg.fonttype"] = "none"
    plt.rcParams["pdf.fonttype"] = 42


def _case_slice(cohort: Any, case_index: int) -> Tuple[int, int]:
    if case_index not in CASE_INDICES:
        raise ValueError(f"Case {case_index} is not one of the frozen manuscript cases.")
    start, stop = (int(value) for value in cohort.offsets[case_index : case_index + 2])
    if stop - start < 2:
        raise ValueError(f"Case {case_index} is too short to render.")
    return start, stop


def _case_source_data(
    cohort: Any,
    main_arrays: Mapping[str, Array],
    pooled_arrays: Mapping[str, Array],
    case_index: int,
) -> Dict[str, Array]:
    start, stop = _case_slice(cohort, case_index)
    return {
        "response_probability": np.asarray(
            main_arrays["behavior_probability"][start:stop], dtype=float
        ).copy(),
        "context_probability": np.asarray(
            main_arrays["scenario_probability"][start:stop], dtype=float
        ).copy(),
        "observed_acceleration": np.asarray(
            cohort.acceleration[start:stop], dtype=float
        ).copy(),
        "physical_context_v_s_delta_v": np.asarray(
            cohort.physical_context[start:stop], dtype=float
        ).copy(),
        "main_expected_acceleration": np.asarray(
            main_arrays["expected_acceleration"][start:stop], dtype=float
        ).copy(),
        "pooled_expected_acceleration": np.asarray(
            pooled_arrays["expected_acceleration"][start:stop], dtype=float
        ).copy(),
    }


def build_b_light_case_figure(
    case_index: int,
    *,
    cohort: Any,
    sampling_frequency_hz: float,
    main_arrays: Mapping[str, Array],
    pooled_arrays: Mapping[str, Array],
) -> Tuple[plt.Figure, Dict[str, Any]]:
    """Build one selected B-light case while preserving every plotted value."""

    source = _case_source_data(cohort, main_arrays, pooled_arrays, case_index)
    response_before = source["response_probability"].copy()
    context_before = source["context_probability"].copy()
    figure = original.plot_case_original_style(
        case_index,
        cohort=cohort,
        sampling_frequency_hz=sampling_frequency_hz,
        main_arrays=main_arrays,
        pooled_arrays=pooled_arrays,
    )
    response_audit = replace_response_with_mixing_mean(
        figure,
        case_index=case_index,
        cohort=cohort,
        main_arrays=main_arrays,
        pooled_arrays=pooled_arrays,
    )
    front_legend_audit = raise_case_legends_to_front(figure)
    gap_layout_audit = move_gap_axis_to_outer_left(figure)
    style_audit = apply_accessible_case_style(
        figure,
        response_probability=np.asarray(source["response_probability"], dtype=float),
        context_probability=np.asarray(source["context_probability"], dtype=float),
        variant=SELECTED_VARIANT,
    )
    np.testing.assert_array_equal(source["response_probability"], response_before)
    np.testing.assert_array_equal(source["context_probability"], context_before)
    figure.canvas.draw()
    return figure, {
        "case_index_zero_based": case_index,
        "case_ordinal": original.CASE_ORDINALS[case_index],
        "case_length": int(response_before.shape[0]),
        "response_curve": dict(response_audit),
        "pre_accessibility_front_legend": dict(front_legend_audit),
        "gap_axis_layout": dict(gap_layout_audit),
        "accessibility_style": dict(style_audit),
        "source_data": {
            name: {
                "shape": list(np.asarray(value).shape),
                "dtype": str(np.asarray(value).dtype),
                "sha256": case6_exporter._array_sha256(np.asarray(value)),
            }
            for name, value in source.items()
        },
        "panel_audit": {
            "upper": {
                "unique_claim": "Observed and posterior-expected acceleration response over time.",
                "summary": "Observed acceleration, main mixed posterior mean, and pooled mixed posterior mean.",
                "interval": "None in the approved old-style mean display.",
                "labels_and_legends": "PASS",
                "collision_check": "PENDING RENDERED QA",
            },
            "lower": {
                "unique_claim": "Physical kinematics and posterior context composition over the same time support.",
                "summary": "Raw v, s, and delta-v series with marginal posterior context probabilities.",
                "interval": "Not applicable.",
                "labels_and_legends": "PASS",
                "collision_check": "PENDING RENDERED QA",
            },
        },
    }


def _save_figure_bundle_new(figure: plt.Figure, pdf_path: Path) -> Tuple[Path, ...]:
    if pdf_path.suffix.lower() != ".pdf":
        raise ValueError("The figure bundle prefix must be a PDF path.")
    svg_path = pdf_path.with_suffix(".svg")
    png_path = pdf_path.with_suffix(".png")
    paths = (svg_path, pdf_path, png_path)
    conflicts = [path for path in paths if path.exists()]
    if conflicts:
        raise FileExistsError(f"Refusing to overwrite existing figures {conflicts}.")
    temporaries = tuple(path.with_name(f".{path.name}.tmp") for path in paths)
    try:
        figure.savefig(temporaries[0], format="svg")
        figure.savefig(
            temporaries[1],
            format="pdf",
            metadata={
                "Creator": "export_all_case_b_light_accessibility.py",
                "Title": pdf_path.stem.replace("_", " "),
            },
        )
        figure.savefig(temporaries[2], format="png", dpi=300)
        for temporary, path in zip(temporaries, paths):
            os.replace(temporary, path)
    finally:
        plt.close(figure)
        for temporary in temporaries:
            if temporary.exists():
                temporary.unlink()
    return paths


def _save_diagnostic_sheet_new(
    source_png: Path,
    output_png: Path,
    *,
    case_index: int,
) -> None:
    if output_png.exists():
        raise FileExistsError(f"Refusing to overwrite diagnostic sheet {output_png}.")
    image = np.asarray(plt.imread(source_png), dtype=float)
    diagnostics = {
        mode: image if mode == "normal" else simulate_color_vision(image, mode)
        for mode in DIAGNOSTIC_MODES
    }
    labels = {
        "normal": "Normal rendering",
        "protanopia": "Protanopia approximation",
        "deuteranopia": "Deuteranopia approximation",
        "tritanopia": "Tritanopia approximation",
        "grayscale": "Grayscale diagnostic",
    }
    sheet, axes = plt.subplots(5, 1, figsize=(7.2, 12.5))
    for axis, mode in zip(axes.flat, DIAGNOSTIC_MODES):
        axis.imshow(diagnostics[mode])
        axis.set_title(labels[mode], fontsize=10, pad=5)
        axis.set_axis_off()
    sheet.suptitle(
        f"Retained case {original.CASE_ORDINALS[case_index]} "
        f"(index {case_index}) B-light diagnostics (simulation only)",
        fontsize=12,
        y=0.992,
    )
    sheet.subplots_adjust(
        left=0.015,
        right=0.985,
        bottom=0.01,
        top=0.965,
        hspace=0.13,
    )
    temporary = output_png.with_name(f".{output_png.name}.tmp")
    try:
        sheet.savefig(temporary, format="png", dpi=300, facecolor="white")
        os.replace(temporary, output_png)
    finally:
        plt.close(sheet)
        if temporary.exists():
            temporary.unlink()


def _write_revision_record_new(output: Path, metadata: Mapping[str, Any]) -> None:
    record = output / "REVISION_RECORD.md"
    if record.exists():
        raise FileExistsError(f"Refusing to overwrite {record}.")
    record.write_text(
        "\n".join(
            (
                "# Author-selected B-light four-case candidate",
                "",
                "## Figure contract",
                "",
                "Core conclusion: the four retained trajectories show how the same model representation combines ",
                "posterior response composition, posterior context composition, physical kinematics, and ",
                "posterior-expected acceleration reconstruction across distinct examples.",
                "",
                "Archetype: repeated quantitative two-panel time-series cases.",
                "",
                "Evidence boundary: in-sample observed-covariate conditional reconstruction; not open-loop simulation, ",
                "held-out prediction, or evidence that the latent states are literal psychological states.",
                "",
                "## Scope",
                "",
                "- The author selected B-light after reviewing the Case 6 B-light/B-strong comparison.",
                "- This bundle applies B-light to all four frozen retained cases.",
                "- No manuscript LaTeX or existing manuscript figure asset is modified by this export.",
                "- No experiment, sampler, model fit, or postprocessing stage is rerun.",
                "- All state fills are fixed precomposed colors at alpha=1.0.",
                "- Every formal case is delivered as editable SVG, vector PDF, and 300-dpi PNG.",
                "- Each diagnostic sheet is a deterministic design approximation, not clinical validation.",
                "",
                "## Frozen evidence",
                "",
                f"- Postprocess: `{metadata['postprocess_directory']}`",
                f"- Cohort: `{metadata['cohort_npz']}`",
                f"- Case indices: `{metadata['fixed_case_indices_zero_based']}`",
                "",
                "## Pending gate",
                "",
                "The four-case bundle is not integrated into the manuscript until the author reviews this output.",
                "",
            )
        ),
        encoding="utf-8",
    )


def execute_export(
    postprocess_directory: Path,
    cohort_npz: Path,
    cohort_manifest: Path,
    output_root: Path,
) -> Path:
    """Export B-light for all retained cases from the same pinned evidence."""

    started = time.perf_counter()
    bundle, cohort, main_arrays, pooled_arrays = case6_exporter._verify_frozen_inputs(
        postprocess_directory,
        cohort_npz,
        cohort_manifest,
    )
    output_root = output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    output = output_root / f"{timestamp}_all_cases_B_light_accessibility_candidate"
    output.mkdir(parents=False, exist_ok=False)
    configure_export()

    candidate_audits: Dict[str, Any] = {}
    artifact_paths = []
    for case_index in CASE_INDICES:
        figure, case_audit = build_b_light_case_figure(
            case_index,
            cohort=cohort,
            sampling_frequency_hz=bundle.config.sampling_frequency_hz,
            main_arrays=main_arrays,
            pooled_arrays=pooled_arrays,
        )
        prefix = output / CASE_OUTPUT_STEMS[case_index]
        formal_paths = _save_figure_bundle_new(figure, prefix.with_suffix(".pdf"))
        diagnostic_path = output / f"{CASE_OUTPUT_STEMS[case_index]}_diagnostics.png"
        _save_diagnostic_sheet_new(
            prefix.with_suffix(".png"),
            diagnostic_path,
            case_index=case_index,
        )
        artifact_paths.extend((*formal_paths, diagnostic_path))
        case_audit["formal_artifacts"] = [path.name for path in formal_paths]
        case_audit["diagnostic_artifact"] = diagnostic_path.name
        candidate_audits[str(case_index)] = case_audit

    artifacts = {
        path.name: {
            "bytes": path.stat().st_size,
            "sha256": _sha256_file(path),
        }
        for path in artifact_paths
    }
    _write_json_new(output / "figure_artifact_manifest.json", {"artifacts": artifacts})
    completed_at = datetime.now(timezone.utc).isoformat()
    metadata: Dict[str, Any] = {
        "completed_at_utc": completed_at,
        "runtime_seconds": time.perf_counter() - started,
        "purpose": "Author-selected B-light visual encoding for all four retained cases",
        "candidate_status": "AUTHOR SELECTED STYLE; NOT YET INTEGRATED INTO MANUSCRIPT",
        "adaptation_level": "exact structural reuse plus approved B-light style-only inheritance",
        "figure_contract": {
            "core_conclusion": (
                "Four retained trajectories show posterior response/context composition, physical kinematics, "
                "and posterior-expected acceleration reconstruction under one consistent visual grammar."
            ),
            "evidence_chain": (
                "Each retained case contributes one distinct trajectory-level example; upper and lower panels "
                "share the same time support."
            ),
            "archetype": "repeated quantitative two-panel time-series cases",
            "backend": "Python only",
            "export": "7.2 x 4.25 in formal cases; editable SVG/PDF; 300-dpi PNG; >=5 pt glyph floor",
        },
        "source_script": str(Path(__file__).resolve()),
        "source_script_sha256": _sha256_file(Path(__file__).resolve()),
        "accessibility_source": str(
            Path(apply_accessible_case_style.__code__.co_filename).resolve()
        ),
        "accessibility_source_sha256": _sha256_file(
            Path(apply_accessible_case_style.__code__.co_filename).resolve()
        ),
        "postprocess_directory": str(bundle.directory),
        "postprocess_manifest_sha256": bundle.manifest_sha256,
        "posterior_summary_sha256": bundle.posterior_summary_sha256,
        "cohort_npz": str(cohort_npz.resolve()),
        "cohort_npz_sha256": cohort.npz_sha256,
        "cohort_manifest": str(cohort_manifest.resolve()),
        "cohort_manifest_sha256": cohort.manifest_sha256,
        "cohort_content_sha256": cohort.content_sha256,
        "fixed_case_indices_zero_based": list(CASE_INDICES),
        "case_ordinals": {
            str(case_index): original.CASE_ORDINALS[case_index]
            for case_index in CASE_INDICES
        },
        "sampling_frequency_hz": bundle.config.sampling_frequency_hz,
        "selected_variant": SELECTED_VARIANT,
        "data_change": "NONE",
        "excluded_observations": 0,
        "fixed_state_fill_alpha": 1.0,
        "fixed_response_state_colors": list(RESPONSE_STATE_COLORS),
        "fixed_context_state_colors": list(CONTEXT_STATE_COLORS),
        "line_colors": dict(LINE_COLORS),
        "palette_audit": {
            "response": palette_audit(RESPONSE_STATE_COLORS),
            "context": palette_audit(CONTEXT_STATE_COLORS),
            "lower_panel_lines": palette_audit(
                (
                    LINE_COLORS[r"$v$"],
                    LINE_COLORS[r"$s$"],
                    LINE_COLORS[r"$\Delta v$"],
                )
            ),
        },
        "diagnostic_modes": list(DIAGNOSTIC_MODES),
        "interpretation_boundary": (
            "In-sample observed-covariate conditional reconstruction, not open-loop simulation or held-out validation."
        ),
        "diagnostic_boundary": (
            "Color-vision transformations are deterministic approximations for design QA, not clinical validation."
        ),
        "candidate_audits": candidate_audits,
        "python": sys.version,
        "numpy": np.__version__,
        "matplotlib": matplotlib.__version__,
        "platform": platform.platform(),
        "figure_artifacts": artifacts,
    }
    _write_json_new(output / "figure_metadata.json", metadata)
    _write_revision_record_new(output, metadata)
    _write_json_new(
        output / "COMPLETED.json",
        {
            "status": "COMPLETED",
            "completed_at_utc": completed_at,
            "candidate_status": metadata["candidate_status"],
            "figure_artifact_manifest_sha256": _sha256_file(
                output / "figure_artifact_manifest.json"
            ),
            "figure_metadata_sha256": _sha256_file(output / "figure_metadata.json"),
            "revision_record_sha256": _sha256_file(output / "REVISION_RECORD.md"),
        },
    )
    return output


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--postprocess-directory", type=Path, required=True)
    parser.add_argument("--cohort-npz", type=Path, required=True)
    parser.add_argument("--cohort-manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    output = execute_export(
        args.postprocess_directory,
        args.cohort_npz,
        args.cohort_manifest,
        args.output_root,
    )
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

