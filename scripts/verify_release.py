"""Check the scientific source files against the paper-run receipts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    record = json.loads((ROOT / "configs" / "paper_provenance.json").read_text(encoding="utf-8"))
    errors = []
    for relative, expected in record["scientific_source_sha256"].items():
        source = ROOT / relative
        if not source.is_file() or hashlib.sha256(source.read_bytes()).hexdigest() != expected:
            errors.append(relative)
    if errors:
        raise SystemExit(f"Scientific source missing or changed: {errors}")
    print(f"Verified {len(record['scientific_source_sha256'])} source files against the paper-run receipt")


if __name__ == "__main__":
    main()
