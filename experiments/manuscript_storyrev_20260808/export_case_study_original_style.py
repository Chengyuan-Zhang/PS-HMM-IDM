"""Export the verified original-style case figures as a complete figure bundle."""

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

from . import plot_case_study_original_style as original
from .postprocess import _sha256_file, _write_json_new


def configure_export() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "font.size": 7.0,
            "axes.labelsize": 7.5,
            "legend.fontsize": 6.3,
            "xtick.labelsize": 6.5,
            "ytick.labelsize": 6.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.03,
        }
    )


def save_figure_bundle_new(figure: plt.Figure, pdf_path: Path) -> Tuple[Path, ...]:
    """Write editable vectors and a 300-dpi preview without overwriting."""

    if pdf_path.suffix.lower() != ".pdf":
        raise ValueError("The figure bundle prefix must be provided as a PDF path.")
    paths = (pdf_path, pdf_path.with_suffix(".svg"), pdf_path.with_suffix(".png"))
    conflicts = [path for path in paths if path.exists()]
    if conflicts:
        raise FileExistsError(f"Refusing to overwrite existing figures {conflicts}.")
    temporaries = tuple(path.with_name(f".{path.name}.tmp") for path in paths)
    try:
        figure.savefig(
            temporaries[0],
            format="pdf",
            metadata={
                "Creator": "export_case_study_original_style.py",
                "Title": pdf_path.stem.replace("_", " "),
            },
        )
        figure.savefig(temporaries[1], format="svg")
        figure.savefig(temporaries[2], format="png", dpi=300)
        for temporary, path in zip(temporaries, paths):
            os.replace(temporary, path)
    finally:
        plt.close(figure)
        for temporary in temporaries:
            if temporary.exists():
                temporary.unlink()
    return paths


def _set_unicode_acceleration_unit(figure: plt.Figure) -> None:
    matches = [axis for axis in figure.axes if axis.get_ylabel().startswith("Acceleration")]
    if len(matches) != 1:
        raise ValueError("Expected exactly one acceleration axis in the original layout.")
    matches[0].set_ylabel("Acceleration (m/s²)")


def execute_export(
    postprocess_directory: Path,
    cohort_npz: Path,
    cohort_manifest: Path,
    output_root: Path,
) -> Path:
    """Verify pinned evidence and export the four frozen cases append-only."""

    started = time.perf_counter()
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

    output_root = output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    output = output_root / f"{timestamp}_original_style_case_figures_final"
    output.mkdir(parents=False, exist_ok=False)
    original.configure_matplotlib()
    configure_export()
    artifacts: Dict[str, Dict[str, Any]] = {}
    for case_index, filename in original.CASE_FILENAMES.items():
        figure = original.plot_case_original_style(
            case_index,
            cohort=cohort,
            sampling_frequency_hz=bundle.config.sampling_frequency_hz,
            main_arrays=main_arrays,
            pooled_arrays=pooled_arrays,
        )
        _set_unicode_acceleration_unit(figure)
        for path in save_figure_bundle_new(figure, output / filename):
            artifacts[path.name] = {
                "bytes": path.stat().st_size,
                "sha256": _sha256_file(path),
            }

    _write_json_new(output / "figure_artifact_manifest.json", {"artifacts": artifacts})
    metadata: Dict[str, Any] = {
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "runtime_seconds": time.perf_counter() - started,
        "source_script": str(Path(__file__).resolve()),
        "source_script_sha256": _sha256_file(Path(__file__).resolve()),
        "structural_plot_script": str(Path(original.__file__).resolve()),
        "structural_plot_script_sha256": _sha256_file(Path(original.__file__).resolve()),
        "adaptation_level": "structural adaptation",
        "preserved_visual_grammar": (
            "Two stacked time-series panels with cumulative posterior-probability "
            "ribbons and overlaid response/kinematic curves."
        ),
        "postprocess_directory": str(bundle.directory),
        "postprocess_manifest_sha256": bundle.manifest_sha256,
        "posterior_summary_sha256": bundle.posterior_summary_sha256,
        "cohort_npz": str(cohort_npz.resolve()),
        "cohort_npz_sha256": cohort.npz_sha256,
        "cohort_manifest": str(cohort_manifest.resolve()),
        "cohort_manifest_sha256": cohort.manifest_sha256,
        "cohort_content_sha256": cohort.content_sha256,
        "fixed_case_indices_zero_based": list(original.CASE_FILENAMES),
        "sampling_frequency_hz": bundle.config.sampling_frequency_hz,
        "response_labels": list(original.RESPONSE_NAMES),
        "context_labels": [f"C{index}" for index in range(1, 6)],
        "state_display": (
            "Cumulative stacks are timewise marginal posterior probabilities, not modal "
            "background blocks or one sampled/Viterbi path."
        ),
        "curve_display": (
            "Solid terracotta is the main posterior median and pointwise 95% interval; "
            "dashed slate is the pooled posterior median and interval; black is observed acceleration."
        ),
        "interpretation_boundary": (
            "In-sample observed-covariate conditional reconstruction, not open-loop "
            "simulation or out-of-sample validation."
        ),
        "python": sys.version,
        "numpy": np.__version__,
        "matplotlib": matplotlib.__version__,
        "platform": platform.platform(),
        "figure_artifacts": artifacts,
    }
    _write_json_new(output / "figure_metadata.json", metadata)
    _write_json_new(
        output / "COMPLETED.json",
        {
            "status": "COMPLETED",
            "completed_at_utc": metadata["completed_at_utc"],
            "figure_artifact_manifest_sha256": _sha256_file(
                output / "figure_artifact_manifest.json"
            ),
            "figure_metadata_sha256": _sha256_file(output / "figure_metadata.json"),
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
