"""Export original-style case figures with posterior-mean mixed IDM responses."""

from __future__ import annotations

import argparse
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Mapping

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

from . import plot_case_study_original_style as original
from .export_case_study_original_style import (
    _set_unicode_acceleration_unit,
    configure_export,
    save_figure_bundle_new,
)
from .postprocess import _sha256_file, _write_json_new


Array = NDArray[np.generic]
FloatArray = NDArray[np.float64]


def _finite_case_series(value: Array, start: int, stop: int, label: str) -> FloatArray:
    series = np.asarray(value, dtype=float)[start:stop]
    if series.shape != (stop - start,) or not np.all(np.isfinite(series)):
        raise ValueError(f"{label} is not a finite case-aligned series.")
    return series


def replace_response_with_mixing_mean(
    figure: plt.Figure,
    *,
    case_index: int,
    cohort: original.CohortArrays,
    main_arrays: Mapping[str, Array],
    pooled_arrays: Mapping[str, Array],
) -> Mapping[str, float]:
    """Replace pointwise medians/bands with old-style mixed posterior means."""

    start, stop = (int(value) for value in cohort.offsets[case_index : case_index + 2])
    observed = _finite_case_series(cohort.acceleration, start, stop, "Observed acceleration")
    main_mean = _finite_case_series(
        main_arrays["expected_acceleration"], start, stop, "Main posterior mean"
    )
    pooled_mean = _finite_case_series(
        pooled_arrays["expected_acceleration"], start, stop, "Pooled posterior mean"
    )
    current_median = np.asarray(
        main_arrays[f"case_{case_index}_prediction_median"], dtype=float
    )
    if current_median.shape != main_mean.shape or not np.all(np.isfinite(current_median)):
        raise ValueError("Current case median is unavailable for the roughness audit.")

    axes = [axis for axis in figure.axes if axis.get_ylabel().startswith("Acceleration")]
    if len(axes) != 1:
        raise ValueError("Expected exactly one acceleration axis in the original layout.")
    axis = axes[0]
    labeled_lines = {line.get_label(): line for line in axis.lines}
    required = {"Pooled IDM", "PS-HMM-IDM", "Observed acceleration"}
    missing = required - set(labeled_lines)
    if missing:
        raise ValueError(f"Original response axis lacks line labels {sorted(missing)}.")
    labeled_lines["Pooled IDM"].set_ydata(pooled_mean)
    labeled_lines["PS-HMM-IDM"].set_ydata(main_mean)
    labeled_lines["Observed acceleration"].set_ydata(observed)

    for collection in list(axis.collections):
        collection.remove()

    lower = float(min(observed.min(), main_mean.min(), pooled_mean.min()))
    upper = float(max(observed.max(), main_mean.max(), pooled_mean.max()))
    span = max(upper - lower, 1.0e-6)
    axis.set_ylim(lower - 0.08 * span, upper + 0.12 * span)
    _set_unicode_acceleration_unit(figure)

    median_roughness = float(np.mean(np.abs(np.diff(current_median, n=2))))
    mean_roughness = float(np.mean(np.abs(np.diff(main_mean, n=2))))
    return {
        "case_index_zero_based": float(case_index),
        "pointwise_median_second_difference_mean_abs": median_roughness,
        "mixed_posterior_mean_second_difference_mean_abs": mean_roughness,
        "roughness_reduction_fraction": (
            1.0 - mean_roughness / median_roughness if median_roughness > 0.0 else 0.0
        ),
    }


def execute_export(
    postprocess_directory: Path,
    cohort_npz: Path,
    cohort_manifest: Path,
    output_root: Path,
) -> Path:
    """Verify pinned evidence and export the revised four-case bundle."""

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
    output = output_root / f"{timestamp}_original_style_mixing_mean_case_figures"
    output.mkdir(parents=False, exist_ok=False)
    original.configure_matplotlib()
    configure_export()
    artifacts: Dict[str, Dict[str, Any]] = {}
    roughness_audit = []
    for case_index, filename in original.CASE_FILENAMES.items():
        figure = original.plot_case_original_style(
            case_index,
            cohort=cohort,
            sampling_frequency_hz=bundle.config.sampling_frequency_hz,
            main_arrays=main_arrays,
            pooled_arrays=pooled_arrays,
        )
        roughness_audit.append(
            replace_response_with_mixing_mean(
                figure,
                case_index=case_index,
                cohort=cohort,
                main_arrays=main_arrays,
                pooled_arrays=pooled_arrays,
            )
        )
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
        "upper_panel_revision": (
            "Main and pooled curves use posterior-mean observed-covariate conditional "
            "responses, matching the submitted mixing-probability logic without random "
            "emission-noise realizations. Pointwise median interval bands are omitted."
        ),
        "roughness_audit": roughness_audit,
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
        "state_display": (
            "Cumulative stacks are timewise marginal posterior probabilities, not modal "
            "background blocks or one sampled/Viterbi path."
        ),
        "curve_display": (
            "Solid terracotta is the main posterior-mean mixed conditional response; "
            "dashed slate is the pooled posterior-mean response; black is observed acceleration."
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
