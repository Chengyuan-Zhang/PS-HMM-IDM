# PS-HMM-IDM: Bayesian product-state hidden Markov model for within-driver heterogeneity in car-following

[![arXiv](https://img.shields.io/badge/arXiv-2506.14762-b31b1b.svg)](https://arxiv.org/abs/2506.14762)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.9](https://img.shields.io/badge/python-3.9-blue.svg)](requirements.txt)

**PS-HMM-IDM** (Product-State Hidden Markov Model with Intelligent Driver Model) is a Bayesian time-series model of car-following behavior. It separates two sources of variation in a driver's acceleration: *which response relationship* the driver is using (a **response regime**, each an IDM response surface with its own residual dispersion) and *what traffic situation* the driver is in (a **traffic scenario**, each a distribution of speed, net gap, and relative speed). A joint Markov process links the two hidden states, so the same response regime can recur across scenarios and one scenario can accompany several responses.

This repository is the official implementation of **“Modeling Within-Driver Heterogeneity in Car-Following through Coupled Response–Scenario Dynamics”** by Chengyuan Zhang, Cathy Wu, and Lijun Sun, accepted by *Artificial Intelligence for Transportation* (2026). An earlier version is available as an [arXiv preprint](https://arxiv.org/abs/2506.14762) under the title *Markov Regime-Switching Intelligent Driver Model for Interpretable Car-Following Behavior*.

**Keywords:** car-following behavior · within-driver heterogeneity · product-state hidden Markov model · Intelligent Driver Model (IDM) · Bayesian inference · MCMC · forward-filtering backward-sampling (FFBS) · regime switching · highD naturalistic trajectories

## Overview

The model is fitted to complete follower–leader episodes using MCMC (Gibbs sampling with slice-sampled IDM parameters) and exact forward-filtering backward-sampling of the joint hidden-state paths. Its state estimates are **retrospective smoothing**; its acceleration reconstruction is in-sample and conditional on observed covariates. It is a model for describing and comparing behavior, not a trajectory forecaster.

What the repository provides:

- **Paper reproduction** from the raw highD recording: cohort preparation, five frozen model configurations, posterior summaries, tables, and principal figures, with hash-verified provenance and checkpoint/resume.
- **A general fitting interface** for your own car-following episodes from a simple CSV, with configurable numbers of response regimes and traffic scenarios, priors, and sampler settings.
- **Tests** for indexing, parameter bounds, likelihood finiteness, sampler steps, checkpointing, postprocessing, and provenance guards.

## Key findings in the paper

Applied to 100 follower–leader episodes from one highD recording, the fitted model shows a **many-to-many** organization between response regimes and traffic scenarios: multiple responses occur within each of the five scenarios, while the cautious response recurs across all five. The cautious response accounts for 69.4% of fitted assignments within a low-speed closing scenario and 39.4% within a relatively higher-speed, near-steady scenario. Within individual trajectories, a response regime can remain dominant as the scenario changes, and the response composition can shift within a scenario.

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

## FAQ

**What problem does PS-HMM-IDM address?**
A single fixed car-following law cannot represent a driver who responds differently to similar conditions at different moments, and classifying behavior only by the surrounding traffic scenario conflates the driver's response with the situation. PS-HMM-IDM models both as paired hidden states that evolve together over time.

**How is it different from a regime-switching IDM?**
A regime-switching IDM has one hidden state that selects the IDM parameters. PS-HMM-IDM adds a second, scenario state with its own emission model on observed kinematics, and a joint transition matrix over the paired states. This lets the analysis ask which responses occur in which scenarios.

**Do I need the highD dataset?**
Only to reproduce the paper (Route 1). Route 2 runs on any episodes you provide in the CSV format above, and includes a synthetic demo.

**Can it predict future trajectories?**
It is not designed as a forecaster. State estimates use the whole episode (retrospective smoothing), and the acceleration reconstruction is in-sample and conditional on the observed speed, gap, and relative speed.

**How long does a run take?**
The paper's configurations run 8,000 MCMC iterations on 31,557 observations and take several hours each on a desktop CPU. The 20-iteration synthetic demo finishes in seconds.

**What does the inference use?**
Gibbs sampling: bounded slice sampling for IDM parameters, conjugate updates for residual variances, scenario Gaussians (normal–inverse-Wishart prior), and Dirichlet transition rows, plus exact log-space FFBS for the joint latent path.

## Repository map

- `scripts/reproduce_paper.py`: paper-cohort verification, five model runs, postprocessing, and figures.
- `scripts/fit_custom.py`: CSV validation and a general episode-data fitting interface.
- `experiments/manuscript_storyrev_20260808/`: frozen paper target, MCMC sampler, checkpoints, and postprocessing. Dated paths preserve the original source identity.
- `experiments/revision_20260806/model/` and `inference/`: numerical kernels used by the paper.
- `configs/`: paper fingerprints and an editable custom configuration.
- `examples/`, `tests/`, and `docs/`: tutorials, checks, and detailed guidance.

## Citation

If you use this code, please cite the article. GitHub's “Cite this repository” button reads [CITATION.cff](CITATION.cff). Publication identifiers for the journal version are not yet available; until then:

```bibtex
@article{zhang2026pshmmidm,
  title   = {Modeling Within-Driver Heterogeneity in Car-Following through Coupled Response--Scenario Dynamics},
  author  = {Zhang, Chengyuan and Wu, Cathy and Sun, Lijun},
  journal = {Artificial Intelligence for Transportation},
  year    = {2026},
  note    = {Accepted}
}

@misc{zhang2025markov,
  title         = {Markov Regime-Switching Intelligent Driver Model for Interpretable Car-Following Behavior},
  author        = {Zhang, Chengyuan and Wu, Cathy and Sun, Lijun},
  year          = {2025},
  eprint        = {2506.14762},
  archivePrefix = {arXiv},
  primaryClass  = {stat.AP}
}
```

## License and data

See [data and provenance](docs/DATA_AND_PROVENANCE.md) and [LICENSE](LICENSE). The MIT licence applies to the code, not highD data. Generated inputs and results should remain private where required by the user's data agreement.
