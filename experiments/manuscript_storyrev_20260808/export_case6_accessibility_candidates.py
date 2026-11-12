"""Export two add-only color-accessibility candidates for frozen manuscript Case 6."""

from __future__ import annotations

import argparse
import hashlib
import json
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

CASE_INDEX = 6
VARIANTS: Tuple[str, ...] = ("B_light", "B_strong")
DIAGNOSTIC_MODES: Tuple[str, ...] = (
    "normal",
    "protanopia",
    "deuteranopia",
    "tritanopia",
    "grayscale",
)


def configure_export() -> None:
    """Match the approved case geometry while preserving editable vector text."""

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "font.size": 7.0,
            "axes.labelsize": 7.5,
            "legend.fontsize": 6.3,
            "xtick.labelsize": 6.5,
            "ytick.labelsize": 6.5,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.03,
        }
    )


def _array_sha256(value: Array) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("utf-8"))
    digest.update(json.dumps(list(array.shape)).encode("utf-8"))
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _verify_frozen_inputs(
    postprocess_directory: Path,
    cohort_npz: Path,
    cohort_manifest: Path,
) -> Tuple[Any, Any, Dict[str, Array], Dict[str, Array]]:
    bundle = original.load_verified_postprocess(postprocess_directory)
    cohort = original.load_cohort(cohort_npz.resolve(), cohort_manifest.resolve())
    receipt = bundle.summary.get("cohort")
    if not isinstance(receipt, Mapping):
        raise ValueError("Pinned posterior summary lacks its cohort receipt.")
    expected_hashes = {
        "content_sha256": cohort.content_sha256,
        "npz_sha256": cohort.npz_sha256,
        "manifest_sha256": cohort.manifest_sha256,
    }
    for key, actual in expected_hashes.items():
        if str(receipt.get(key, "")).lower() != actual.lower():
            raise ValueError(f"Frozen cohort {key} differs from the postprocess receipt.")
    if tuple(bundle.config.fixed_case_indices) != tuple(original.CASE_FILENAMES):
        raise ValueError("The pinned case indices differ from the manuscript cases.")

    arm_files = bundle.summary.get("arm_files")
    if not isinstance(arm_files, Mapping):
        raise ValueError("Pinned posterior summary lacks arm filenames.")
    main_path = bundle.directory / str(arm_files[bundle.config.manuscript_arm_id])
    pooled_path = bundle.directory / str(arm_files[bundle.config.pooled_arm_id])
    main_arrays = original._load_npz_arrays(main_path, original.MAIN_REQUIRED_ARRAYS)
    pooled_arrays = original._load_npz_arrays(pooled_path, original.POOLED_REQUIRED_ARRAYS)
    expected_offsets = np.asarray(cohort.offsets, dtype=np.int64)
    for label, arrays in (("main", main_arrays), ("pooled", pooled_arrays)):
        offsets = np.asarray(arrays["trajectory_offsets"], dtype=np.int64)
        if not np.array_equal(offsets, expected_offsets):
            raise ValueError(f"The {label} trajectory offsets differ from the cohort.")
    return bundle, cohort, main_arrays, pooled_arrays


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
                "Creator": "export_case6_accessibility_candidates.py",
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


def _save_diagnostic_sheet_new(source_png: Path, output_png: Path, variant: str) -> None:
    """Create one Python-only sheet of normal and simulated rendered previews."""

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
        f"Case 6 {variant.replace('_', '-')} accessibility diagnostics "
        "(simulation only)",
        fontsize=12,
        y=0.992,
    )
    sheet.subplots_adjust(left=0.015, right=0.985, bottom=0.01, top=0.965, hspace=0.13)
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
                "# Case 6 accessibility candidates",
                "",
                "## Scope",
                "",
                "This add-only bundle compares B-light and B-strong visual encodings for frozen Case 6. ",
                "It does not replace a manuscript figure and does not modify LaTeX.",
                "",
                "## One-sentence takeaway",
                "",
                "Fixed opaque state fills, white cumulative boundaries, and redundant line patterns improve ",
                "distinguishability across common color-vision diagnostics without changing trajectories or probabilities.",
                "",
                "## Evidence and integrity",
                "",
                f"- Frozen postprocess: `{metadata['postprocess_directory']}`",
                f"- Frozen cohort: `{metadata['cohort_npz']}`",
                f"- Case index: `{CASE_INDEX}` (zero-based)",
                "- No experiment was rerun.",
                "- No data value, time coordinate, posterior probability, or model-response curve was changed.",
                "- The color-vision panels are deterministic diagnostic approximations, not clinical validation.",
                "",
                "## Candidate distinction",
                "",
                "- B-light keeps the PS-HMM-IDM response solid and uses redundant styles for the pooled and lower-panel curves.",
                "- B-strong assigns distinct patterns to both response-model curves as well as the lower-panel variables.",
                "- Both candidates use the same fixed state colors with `alpha=1.0` in SVG, PDF, and PNG.",
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
    """Export two verified Case 6 candidates and diagnostic comparison sheets."""

    started = time.perf_counter()
    bundle, cohort, main_arrays, pooled_arrays = _verify_frozen_inputs(
        postprocess_directory, cohort_npz, cohort_manifest
    )
    start, stop = (int(value) for value in cohort.offsets[CASE_INDEX : CASE_INDEX + 2])
    response_probability = np.asarray(
        main_arrays["behavior_probability"][start:stop], dtype=float
    ).copy()
    context_probability = np.asarray(
        main_arrays["scenario_probability"][start:stop], dtype=float
    ).copy()
    source_data = {
        "response_probability": response_probability,
        "context_probability": context_probability,
        "observed_acceleration": np.asarray(cohort.acceleration[start:stop], dtype=float),
        "physical_context_v_s_delta_v": np.asarray(
            cohort.physical_context[start:stop], dtype=float
        ),
        "main_expected_acceleration": np.asarray(
            main_arrays["expected_acceleration"][start:stop], dtype=float
        ),
        "pooled_expected_acceleration": np.asarray(
            pooled_arrays["expected_acceleration"][start:stop], dtype=float
        ),
    }
    source_data_audit = {
        name: {
            "shape": list(np.asarray(value).shape),
            "dtype": str(np.asarray(value).dtype),
            "sha256": _array_sha256(np.asarray(value)),
        }
        for name, value in source_data.items()
    }

    output_root = output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    output = output_root / f"{timestamp}_case6_accessibility_candidates"
    output.mkdir(parents=False, exist_ok=False)
    configure_export()

    candidate_audits: Dict[str, Any] = {}
    artifact_paths = []
    for variant in VARIANTS:
        figure = original.plot_case_original_style(
            CASE_INDEX,
            cohort=cohort,
            sampling_frequency_hz=bundle.config.sampling_frequency_hz,
            main_arrays=main_arrays,
            pooled_arrays=pooled_arrays,
        )
        response_audit = replace_response_with_mixing_mean(
            figure,
            case_index=CASE_INDEX,
            cohort=cohort,
            main_arrays=main_arrays,
            pooled_arrays=pooled_arrays,
        )
        front_legend_audit = raise_case_legends_to_front(figure)
        gap_layout_audit = move_gap_axis_to_outer_left(figure)
        style_audit = apply_accessible_case_style(
            figure,
            response_probability=response_probability,
            context_probability=context_probability,
            variant=variant,
        )
        figure.canvas.draw()
        prefix = output / f"case6_accessible_{variant}"
        formal_paths = _save_figure_bundle_new(figure, prefix.with_suffix(".pdf"))
        diagnostic_path = output / f"case6_accessible_{variant}_diagnostics.png"
        source_png = prefix.with_suffix(".png")
        _save_diagnostic_sheet_new(source_png, diagnostic_path, variant)
        artifact_paths.extend((*formal_paths, diagnostic_path))
        candidate_audits[variant] = {
            "response_curve": dict(response_audit),
            "pre_accessibility_front_legend": dict(front_legend_audit),
            "gap_axis_layout": dict(gap_layout_audit),
            "accessibility_style": dict(style_audit),
            "formal_artifacts": [path.name for path in formal_paths],
            "diagnostic_artifact": diagnostic_path.name,
        }

    artifacts = {
        path.name: {"bytes": path.stat().st_size, "sha256": _sha256_file(path)}
        for path in artifact_paths
    }
    _write_json_new(output / "figure_artifact_manifest.json", {"artifacts": artifacts})
    completed_at = datetime.now(timezone.utc).isoformat()
    metadata: Dict[str, Any] = {
        "completed_at_utc": completed_at,
        "runtime_seconds": time.perf_counter() - started,
        "purpose": "Case 6 color-vision-accessibility candidate comparison only",
        "candidate_status": "NOT YET SELECTED FOR THE MANUSCRIPT",
        "adaptation_level": "style-only small revision of the approved original-style geometry",
        "source_script": str(Path(__file__).resolve()),
        "source_script_sha256": _sha256_file(Path(__file__).resolve()),
        "accessibility_source": str(Path(apply_accessible_case_style.__code__.co_filename).resolve()),
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
        "case_index_zero_based": CASE_INDEX,
        "sampling_frequency_hz": bundle.config.sampling_frequency_hz,
        "source_data_audit": source_data_audit,
        "data_change": "NONE",
        "geometry_preserved": (
            "Two stacked panels; s outer-left; v and delta-v inner-left; posterior probability right; "
            "all legends on frontmost twin axes."
        ),
        "fixed_state_fill_alpha": 1.0,
        "fixed_response_state_colors": list(RESPONSE_STATE_COLORS),
        "fixed_context_state_colors": list(CONTEXT_STATE_COLORS),
        "line_colors": dict(LINE_COLORS),
        "palette_audit": {
            "response": palette_audit(RESPONSE_STATE_COLORS),
            "context": palette_audit(CONTEXT_STATE_COLORS),
            "lower_panel_lines": palette_audit(
                (LINE_COLORS[r"$v$"], LINE_COLORS[r"$s$"], LINE_COLORS[r"$\Delta v$"])
            ),
        },
        "diagnostic_modes": list(DIAGNOSTIC_MODES),
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

