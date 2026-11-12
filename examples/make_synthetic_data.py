"""Write an artificial car-following-shaped CSV for the adaptation tutorial."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.manuscript_storyrev_20260808.target import original_idm_acceleration


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("outputs/synthetic_trajectories.csv"))
    args = parser.parse_args()
    rng = np.random.default_rng(2026)
    frames = []
    for episode in range(3):
        time = np.arange(40) / 5.0
        speed = 10.0 + 4.0 * episode + 0.4 * time + np.sin(time / 2)
        gap = 18.0 + 5.0 * episode + 6.0 * np.cos(time / 2 + episode)
        relative_speed = 0.8 * np.sin(time / 2 + episode)
        context = np.column_stack((speed, gap, relative_speed))
        response_state = (np.arange(time.size) // 10 + episode) % 2
        acceleration = np.empty_like(time)
        for state, headway in enumerate((1.1, 2.0)):
            selected = response_state == state
            parameters = np.array([33.0, 2.0, headway, 1.5, 1.67])
            acceleration[selected] = original_idm_acceleration(parameters, context[selected])
        acceleration += rng.normal(0, 0.08, len(time))
        frames.append(pd.DataFrame({
            "episode_id": f"synthetic-{episode + 1}",
            "time_s": time,
            "speed_mps": speed,
            "gap_m": gap,
            "relative_speed_mps": relative_speed,
            "acceleration_mps2": acceleration,
        }))
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    pd.concat(frames, ignore_index=True).to_csv(output, index=False)
    print(f"Wrote artificial example data to {output}")


if __name__ == "__main__":
    main()
