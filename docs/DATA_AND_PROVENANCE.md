# Data, provenance, and output map

## Inputs and restrictions

The empirical analysis uses the third-party [highD dataset](https://levelxdata.com/highd-dataset/). Access for non-commercial use is requested through the provider; users must follow their own access agreement. The repository and its MIT licence contain no original highD files, processed highD trajectories, cached follower pairs, latent paths, or other trajectory-derived posterior arrays. Keep those files outside version control; `.gitignore` excludes the usual input and output locations and `.npz`/`.pkl` files.

Dataset citation: R. Krajewski, J. Bock, L. Kloeker, and L. Eckstein, “The highD Dataset: A Drone Dataset of Naturalistic Vehicle Trajectories on German Highways for Validation of Highly Automated Driving Systems,” *2018 IEEE Intelligent Transportation Systems Conference*, 2118–2125. [doi:10.1109/ITSC.2018.8569552](https://doi.org/10.1109/ITSC.2018.8569552).

The frozen paper cohort contains the first 100 eligible pairs in the historical selection order. Recordings 1–24 contributed no eligible pairs and all 100 selected pairs are from recording 25. Input is at 25 Hz; the historical current-observation slice is `[:-6:5]`, producing 5 Hz data. Context order and units are `[follower speed (m/s), net gap (m), follower minus leader speed (m/s)]`; the response is follower acceleration (m/s²).

`scripts.prepare_cohort` preserves the original loader's transformations, selection, ordering, and metadata-indexing conventions. It checks every array's dtype, shape, and SHA-256 as well as the trajectory ordering. The fingerprint record includes hashes and aggregate metadata, not the trajectory arrays. The historic cohort and its 405 observed negative speed values are preserved by the formal configurations; speed is not clipped, rows are not deleted, and relative speed is not recomputed as a consequence of clipping.

The raw-CSV route has end-to-end synthetic highD-format regression checks in both travel directions against the original pair loader. It has **not** yet been exercised end-to-end against the original 25_tracks.csv in this environment. A successful full-input hash check is required before using the output as the paper cohort.

An author who already has the historical preprocessing cache can instead run:

```bash
python -m scripts.prepare_cohort \
  --trusted-pairs-cache /path/to/track_pair_list_temp25_MinLength50_freq_5.pkl \
  --output-directory local_data/cohort
```

Only load a pickle from a trusted source: unpickling can execute code. The historical cache route was checked locally against all paper cohort array hashes. Older caches without lane metadata can reconstruct the cohort but cannot generate the Figure 9 time–space plot; use the original highD CSVs or a trusted cache containing `lane_id` and `xFollReal` for that figure.

## Model and execution provenance

`configs/paper_provenance.json` records the scientific source-file SHA-256 values and the five posterior-target fingerprints from the completed formal runs. `python -m scripts.verify_release` checks the copied source files. Five configurations specify `(K_B, K_S) = (1,1), (2,2), (5,1), (1,5), (5,5)` and master seed 20260807. Each uses 8,000 MCMC iterations, with 6,000 burn-in and 2,000 retained draws. The full model's symmetric Dirichlet transition prior has concentration 200 per cell and 5,000 per row; the `(5,1)` model has a row total of 1,000. See the original [model specification](../experiments/manuscript_storyrev_20260808/MODEL_SPEC.md) and JSON configurations for the remaining priors.

The original run was made in a dirty worktree, so a historical Git commit alone does not identify the implementation. The 24 recorded scientific source hashes in this repository match that run. Different operating systems, BLAS libraries, and package builds can change floating-point sampling trajectories even with identical seeds; use summary agreement rather than assuming byte-identical chains across platforms. Original full-chain payloads and the processed trajectories are not distributed here.

## Paper outputs

| Paper object | Reproduction output |
| --- | --- |
| Tables 5 and 6 | `table_05_response_states.json`, `table_06_scenarios.json`, and `tables.tex` from postprocessing |
| Table 7 | `table_07_reconstruction.csv` and `tables.tex` |
| Figure 5 | `figure_05_state_resolution.pdf` |
| Figure 6 | `assets/product_state_indexing.tex` plus `figure_06_transitions.pdf` |
| Figure 7 | `figure_07_scenario_covariances.pdf` |
| Figure 8 | `figure_08a_cooccurrence.pdf` and `figure_08b_conditional_composition.pdf` |
| Figure 9 | `figure_09_time_space.pdf` and its segment CSV; requires position/lane metadata |
| Figures 10–13 | `figure_10_case_2.pdf` through `figure_13_case_9.pdf` |

For the original retained posterior, Table 7 reports conditional reconstruction RMSE values of 0.406, 0.341, 0.156, 0.406, and 0.267 m/s² in the order shown in the README. The cautious response's conditional shares are approximately 69.4% in C2 and 39.4% in C4. These values describe the retained in-sample, retrospective analysis; a new run is a new Monte Carlo realization. Figure 5's ribbons are posterior-soft overlaps between separately fitted models, not temporal transitions.
