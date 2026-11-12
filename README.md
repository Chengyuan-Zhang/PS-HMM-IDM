# PS-HMM-IDM: response–scenario dynamics in car-following

This repository implements the models and analysis in **“Modeling Within-Driver Heterogeneity in Car-Following through Coupled Response–Scenario Dynamics”**, by Chengyuan Zhang, Cathy Wu, and Lijun Sun, accepted by *Artificial Intelligence for Transportation* (2026). An earlier version is available as an [arXiv preprint](https://arxiv.org/abs/2506.14762).

The product-state hidden Markov model pairs a response state, which selects an IDM acceleration relationship, with a scenario state, which describes observed speed, net gap, and relative speed. The model is fitted to complete episodes using MCMC and forward-filtering backward-sampling. Its state estimates are **retrospective smoothing**; its acceleration reconstruction is in-sample and conditional on observed covariates.

## Install

Create an environment with Python 3.9, then run from the repository root. The optional `data.data_loader.KF_tracks` helper additionally needs `pykalman`; the paper and custom-data workflows do not.

```bash
git clone https://github.com/Chengyuan-Zhang/PS-HMM-IDM.git
cd PS-HMM-IDM
python -m pip install -r requirements.txt
python -m scripts.verify_release
python -m pytest -q
```

The pinned dependencies are the validated environment. One test requiring a separately licensed reference cohort may be skipped. The commands use portable Python entry points; the current release checks were run on Windows.

## Route 1: reproduce the paper with highD

Obtain the third-party [highD dataset](https://levelxdata.com/highd-dataset/) under the provider's access terms. Place `25_tracks.csv` and `25_tracksMeta.csv` in a directory accessible to you. The data are not included or licensed by this repository.

First prepare the paper cohort and verify all five model targets without sampling:

```bash
python -m scripts.reproduce_paper --highd-root "/path/to/highD/data" --output-root outputs/paper --prepare-only
```

Then run the full workflow with the **same** output directory:

```bash
python -m scripts.reproduce_paper --highd-root "/path/to/highD/data" --output-root outputs/paper
```

This runs the five frozen configurations, produces posterior summaries and tables, and generates the principal figures. It reuses completed, verified stages and resumes from the latest compatible checkpoint after an interruption. Each original model run took approximately **5.6–6.4 hours** on the original hardware; the complete workflow is a substantial computation. See the [reproduction guide](docs/REPRODUCE_PAPER.md) for outputs, checkpoints, manual steps, and validation limits.

## Route 2: adapt the model to your own data

No highD data are needed for this route. Generate an artificial example and fit it:

```bash
python -m examples.make_synthetic_data --output outputs/synthetic_trajectories.csv
python -m scripts.fit_custom --csv outputs/synthetic_trajectories.csv --sampling-hz 5 --output-dir outputs/custom_demo
python -m examples.plot_custom_episode --fit-dir outputs/custom_demo --episode-id synthetic-1
```

The example uses 20 MCMC iterations to demonstrate the software. Its summaries are **not** results for interpretation. The generated inputs are artificial, not highD data or trajectories from a kinematic simulator.

To use your own data, supply a CSV with `episode_id`, `time_s`, `speed_mps`, `gap_m`, `relative_speed_mps`, and `acceleration_mps2`. Relative speed is **follower speed minus leader speed**. Copy and edit [configs/custom_example.json](configs/custom_example.json) to choose state counts, priors, and sampling settings, then pass the copy with `--config`. The output includes a posterior summary, aligned posterior draws, observation-level state probabilities, and conditional reconstruction. The [adaptation guide](docs/ADAPTATION.md) explains the data contract, model assumptions, output arrays, and where to change the model.

## Repository map

- `scripts/reproduce_paper.py`: paper-cohort verification, five model runs, postprocessing, and figures.
- `scripts/fit_custom.py`: CSV validation and a general episode-data fitting interface.
- `experiments/manuscript_storyrev_20260808/`: frozen paper target, MCMC sampler, checkpoints, and postprocessing. Dated paths preserve the original source identity.
- `experiments/revision_20260806/model/` and `inference/`: numerical kernels used by the paper.
- `configs/`: paper fingerprints and an editable custom configuration.
- `examples/`, `tests/`, and `docs/`: tutorials, checks, and detailed guidance.

See [data and provenance](docs/DATA_AND_PROVENANCE.md), [CITATION.cff](CITATION.cff), and [LICENSE](LICENSE). The MIT licence applies to the code, not highD data. Generated inputs and results should remain private where required by the user's data agreement.
