# Reproducing the paper with highD

## Inputs and preparation

Access the [highD dataset](https://levelxdata.com/highd-dataset/) under your own agreement. Recording 25's `25_tracks.csv` and `25_tracksMeta.csv` must be in the directory supplied to `--highd-root`. For example:

```bash
python -m scripts.reproduce_paper --highd-root "/path/to/highD/data" --output-root outputs/paper --prepare-only
```

The command reconstructs the paper's 100 episodes and 31,557 observations and compares every array, its order, and all five scientific-target fingerprints against the recorded paper run. A mismatch stops the workflow. The historical cached-preprocessing route has reproduced the cohort exactly; the raw-CSV route has been tested end-to-end on synthetic highD-format tracks in both travel directions and against the legacy pair loader, but has not yet been tested here on the actual recording 25 CSV. A successful real-data fingerprint check is therefore essential. See [data and provenance](DATA_AND_PROVENANCE.md) for the selection and preprocessing conventions.

## Run and resume the complete analysis

```bash
python -m scripts.reproduce_paper --highd-root "/path/to/highD/data" --output-root outputs/paper
```

The workflow performs five models in sequence: `(response, scenario) = (1,1), (2,2), (5,1), (1,5), (5,5)`. Each uses its recorded seed and 8,000 iterations, including 6,000 burn-in. The original runs took about 5.6–6.4 hours **per model**, depending on hardware. The runner sets numerical BLAS thread counts to one before importing the model.

The same command can be run again after interruption. Completed arms are reused only after their configuration, target, source, and artifact checks; otherwise the most advanced compatible checkpoint is resumed. Checkpoints are written every 500 iterations by default. `--checkpoint-interval` changes this interval. A checkpoint records the software environment and will not resume if required identity checks disagree. Use the same environment and input files.

The main output directories are:

- `cohort/`: fingerprint-verified, highD-derived input arrays and manifest.
- `paper_runs/`: the five chains, configurations, checkpoints, and run receipts.
- `postprocessed/`: aligned posterior summaries, `tables.tex`, diagnostics, and per-arm outputs.
- `figures/`: Figures 5–13, the Figure 9 segment CSV, Table 5/6 JSON summaries, Table 7 CSV, and output hashes.

The generation code preserves the original postprocessing checks, including its fixed-case structural gate. If that gate does not pass, inspect its recorded output rather than treating a different Monte Carlo realization as identical to the paper. Reuse also checks that postprocessing refers to the same five chains and cohort. Generated plots reproduce quantitative inputs and figure design; fonts and small editorial layout details can differ.

## Individual stages

Each step can be run separately when adapting the workflow to a scheduler. Set one thread for BLAS libraries in your job environment. For example, after preparing `outputs/paper/cohort`:

```bash
python -m experiments.manuscript_storyrev_20260808.run_exact_reconstruction --config experiments/manuscript_storyrev_20260808/configs/b5s5.json --cohort-npz outputs/paper/cohort/cohort_data.npz --manifest outputs/paper/cohort/manifest.json --results-root outputs/paper/paper_runs
```

Use the corresponding `b1s1`, `b2s2`, `b5s1`, and `b1s5` configuration for the other runs. The runner offers `--preflight-only`, `--checkpoint-interval`, and `--resume-checkpoint`; use `--help` for details. A completed checkpoint is a directory such as `checkpoints/iter_00000500`. To collect exactly one completed run for each arm:

```bash
python -m scripts.postprocess_runs --runs-root outputs/paper/paper_runs --cohort-npz outputs/paper/cohort/cohort_data.npz --cohort-manifest outputs/paper/cohort/manifest.json --output-root outputs/paper/postprocessed
python -m scripts.plot_paper --postprocess-directory "/path/to/completed/postprocessing" --cohort-npz outputs/paper/cohort/cohort_data.npz --cohort-manifest outputs/paper/cohort/manifest.json --highd-root "/path/to/highD/data" --output-directory outputs/paper/figures
```

The author's trusted historical pair cache can replace the raw files for preparation via `--trusted-pairs-cache /path/to/cache.pkl`; never unpickle an untrusted file. Figure 9 additionally needs matched position and lane metadata. If those are unavailable, add `--skip-time-space` to the one-command workflow, or omit the source argument when calling `scripts.plot_paper` directly.

The released repository does not contain highD observations, posterior chains, or trajectory-derived figure data. Different BLAS libraries, operating systems, or numerical package builds may produce different Monte Carlo draws even when the scientific target and random seed match.
