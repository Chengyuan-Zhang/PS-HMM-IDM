"""Collect exactly one completed run per paper arm, then postprocess them."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from experiments.manuscript_storyrev_20260808.postprocess import (
    PostprocessConfig,
    execute_postprocess,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "experiments/manuscript_storyrev_20260808/configs/postprocess.json"


def find_completed_runs(root: Path, expected_arms: tuple[str, ...]) -> list[Path]:
    found: dict[str, Path] = {}
    for receipt in sorted(root.glob("*/COMPLETED.json")):
        payload = json.loads(receipt.read_text(encoding="utf-8"))
        arm = payload.get("arm_id")
        if arm not in expected_arms:
            continue
        if arm in found:
            raise ValueError(f"More than one completed run for {arm}; select a unique runs directory")
        found[arm] = receipt.parent
    missing = [arm for arm in expected_arms if arm not in found]
    if missing:
        raise ValueError(f"Missing completed formal arms: {missing}")
    return [found[arm] for arm in expected_arms]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root", type=Path, required=True)
    parser.add_argument("--cohort-npz", type=Path, required=True)
    parser.add_argument("--cohort-manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=ROOT / "outputs" / "postprocessed")
    args = parser.parse_args()
    config = PostprocessConfig.from_json(CONFIG)
    runs = find_completed_runs(args.runs_root, config.required_arm_ids)
    output = execute_postprocess(runs, args.cohort_npz, args.cohort_manifest,
                                 CONFIG, args.output_root, require_formal=True)
    print(f"Postprocessed five formal arms: {output}")


if __name__ == "__main__":
    main()
