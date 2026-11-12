# Adapting the model to your own episodes

The custom interface accepts your own car-following episodes, fits the same product-state HMM–IDM implementation, and exports quantities that you can analyze with Python, R, or a spreadsheet. It does not require highD. Its input observations are never uploaded by this code.

## Start with the synthetic tutorial

From the repository root, run:

```bash
python -m examples.make_synthetic_data --output outputs/synthetic_trajectories.csv
python -m scripts.fit_custom --csv outputs/synthetic_trajectories.csv --sampling-hz 5 --output-dir outputs/custom_demo
python -m examples.plot_custom_episode --fit-dir outputs/custom_demo --episode-id synthetic-1
```

Inspect `outputs/synthetic_trajectories.csv`, `outputs/custom_demo/state_probabilities.csv`, `posterior_summary.json`, `aligned_posterior.npz`, and `example_episode.png`. The generated observations are artificial and their covariates are constructed for a code example; they are not a physically integrated traffic simulation. The 20-iteration configuration is a software demonstration, not evidence of convergence or a recommended research analysis.

## Prepare an episode CSV

Each row is one observation. Supply these columns, with the exact names and units below; additional columns are ignored.

| Column | Meaning |
| --- | --- |
| `episode_id` | Nonempty identifier for a contiguous follower–leader episode. |
| `time_s` | Numeric time in seconds, unique and evenly spaced within each episode. |
| `speed_mps` | Follower longitudinal speed in m/s, oriented to be nonnegative. |
| `gap_m` | Strictly positive bumper-to-bumper net gap in metres. |
| `relative_speed_mps` | Follower speed minus leader speed in m/s; positive means closing. |
| `acceleration_mps2` | Follower longitudinal acceleration in m/s². |

Set `--sampling-hz` to the observations per second after any preprocessing. The same interval must hold within every episode. The reader sorts rows by episode and time and rejects duplicate timestamps, missing or nonfinite values, negative speed, nonpositive gap, and time gaps inconsistent with the specified rate. Split discontinuous intervals or leader changes into separate episodes, or resample beforehand using a method appropriate for your data. No observations are silently dropped, clipped, or interpolated. Each of the three context variables must vary in the pooled data because the scenario model standardizes it.

An `episode_id` identifies one independent sequence for the model. Several episodes may come from the same driver, but driver identity is not a hierarchical effect in this implementation: episodes have separate initial states and share the response/scenario parameter library. Internally, synthetic numeric follower IDs are assigned solely to satisfy the paper runner's input schema; they are not real driver IDs.

For your own file and a configuration copied to `local_data/my_config.json`, for example:

```bash
python -m scripts.fit_custom --csv local_data/my_episodes.csv --sampling-hz 10 --config local_data/my_config.json --output-dir outputs/my_fit
```

The output directory must be new. To resume the same input and configuration after interruption, repeat the command with `--resume`. The code verifies the CSV, prepared arrays, configuration, target, source files, and execution environment before reusing a compatible checkpoint. If no checkpoint has been written, the chain starts again. The default checkpoint interval is 100 iterations; change it with `--checkpoint-interval`. A completed fit is reused rather than sampled again. If summary writing was interrupted midway, move the partial summary files before resuming.

## Choose a model and sampling settings

Copy [the example configuration](../configs/custom_example.json) and review it for your dataset:

- `states.behavior` and `states.scenario` are the numbers of response and scenario components. The sampler uses their Cartesian product as its joint state space.
- `transition_prior.row_total_concentration` sets the total symmetric Dirichlet concentration **per outgoing joint-state row**. The interface derives the per-cell concentration as the row total divided by the number of joint states. A different sampling interval changes the meaning of a one-step transition.
- `idm.log_prior_mean` and `idm.log_prior_standard_deviation` are on the **natural-log parameter scale**, in the order printed in the JSON file. The example uses relatively concentrated paper-derived priors; examine their physical implications and revise the priors, bounds, residual variance, and scenario prior for your application. The custom example's transition prior is a demonstration setting and differs from the frozen paper configuration.
- `chain.num_iterations`, `burn_in`, `thin`, and the seed control the single chain. Set an appropriate schedule for the data and examine mixing and sensitivity. Multiple runs with different seeds require separate output directories. State labels can permute between runs; align or use label-invariant quantities before comparing them.

The model estimates a time-homogeneous, first-order transition matrix at the supplied sampling interval. Scenario Gaussian parameters are estimated after pooling and standardizing the three context variables. The implementation uses the paper's **untruncated** IDM desired-gap formula; it does not insert a `max(0, ...)` term. These choices, and the independence of episodes conditional on shared parameters, should fit the research question before interpreting results.

## Read the outputs

- `posterior_summary.json`: sample counts, in-sample conditional reconstruction RMSE, posterior means/medians/95% equal-tail intervals for IDM parameters, physical scenario means and covariance, state occupancies, and the mean joint transition matrix.
- `state_probabilities.csv`: sorted observations, posterior mean reconstructed acceleration, and marginal response/scenario probabilities at each observation. CSV state names start at 1.
- `aligned_posterior.npz`: aligned parameter draws, transition matrices, joint-state paths, trajectory offsets, standardizer, and label permutations. Array indices start at 0. The joint index is `response_index * number_of_scenario_states + scenario_index`.
- `input/` and `runs/`: prepared arrays, their fingerprints, the effective configuration, original chain, receipts, and checkpoints. These are derived from your data and should be handled under your data agreement.

Within one fit, labels are aligned to a retained draw using IDM parameter profiles and up to eight observed contexts at evenly spaced speed ranks, plus scenario means/covariances. Alignment reduces label switching but is not evidence that statistically similar components are distinct behavioral types. Labels are arbitrary and are not automatically matched across separate fits. Edit the anchor selection in `write_summaries` in [`scripts/fit_custom.py`](../scripts/fit_custom.py) when another observed-context selection is more appropriate, and assess the sensitivity of state-specific interpretations.

The reconstruction is an average of the IDM mean evaluated at each observed context and its sampled response state. The reported probabilities use the complete episode, including its observed acceleration. They are smoothed in-sample quantities, not future predictions, online estimates, or an open-loop simulator.

## Changing the scientific model

The CSV adapter changes the dataset, state counts, priors, and sampling schedule while keeping the paper's three covariates and IDM response. To investigate a different response law, inspect `original_idm_acceleration` and the emission construction in [`target.py`](../experiments/manuscript_storyrev_20260808/target.py), the parameter updates in [`sampler.py`](../experiments/manuscript_storyrev_20260808/sampler.py), and the forward/backward and transition kernels in [`model/`](../experiments/revision_20260806/model/). Replacing a response function alone is insufficient if its parameter constraints, priors, or conditional updates also change. Changing the number or meaning of scenario covariates additionally requires updating standardization and the normal–inverse-Wishart specification.

Make such changes in a separate branch or copy. Modified numerical files will no longer pass `python -m scripts.verify_release`, which deliberately checks identity with the paper run. Keep an unchanged copy for paper reproduction, and validate the modified likelihood and sampler against examples appropriate to the new model.
