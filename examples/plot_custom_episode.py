"""Plot observed acceleration, reconstruction, and smoothed probabilities."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fit-dir", type=Path, required=True)
    parser.add_argument("--episode-id", help="Episode ID; defaults to the first one")
    parser.add_argument("--output", type=Path, help="New PNG path; defaults to fit-dir/example_episode.png")
    args = parser.parse_args()
    data = pd.read_csv(args.fit_dir / "state_probabilities.csv", dtype={"episode_id": "string"})
    episode_id = args.episode_id if args.episode_id is not None else str(data["episode_id"].iloc[0])
    episode = data[data["episode_id"] == episode_id]
    if episode.empty:
        raise ValueError(f"Episode not found: {episode_id}")
    output = args.output or (args.fit_dir / "example_episode.png")
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(3, 1, figsize=(9, 7), sharex=True, constrained_layout=True)
    axes[0].plot(episode["time_s"], episode["acceleration_mps2"], label="Observed", lw=1)
    axes[0].plot(episode["time_s"], episode["reconstructed_acceleration_mps2"], label="Reconstructed", lw=1.3)
    axes[0].set_ylabel("Acceleration (m/s²)")
    axes[0].legend()
    for prefix, axis, label in (("p_response_", axes[1], "Response probability"),
                                ("p_scenario_", axes[2], "Scenario probability")):
        columns = [column for column in episode.columns if column.startswith(prefix)]
        for column in columns:
            short_label = ("R" if prefix == "p_response_" else "S") + column.removeprefix(prefix)
            axis.plot(episode["time_s"], episode[column], label=short_label)
        axis.set_ylabel(label)
        axis.set_ylim(-0.02, 1.02)
        axis.legend(ncol=min(len(columns), 4), fontsize="small")
    axes[-1].set_xlabel("Time (s)")
    axes[0].set_title(f"Episode {episode_id}: retrospective smoothing")
    figure.savefig(output, dpi=160)
    plt.close(figure)
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
