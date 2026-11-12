# Frozen Exact Reconstruction Target

This directory reconstructs the manuscript's original five experiment arms
without changing their scientific question. It is independent of the legacy
`MCMC/` implementation and does not overwrite historical caches or figures.

## Posterior target

- Joint state: unrestricted product-state HMM with
  `Q = K_B * K_S`; joint index is `b * K_S + s`.
- Behavior emission: historical IDM desired gap
  `s0 + v*T + v*dv/(2*sqrt(a*b))`, without a non-negative truncation.
- Context emission: Gaussian on standardized `[v, s, dv]`, where
  `dv = follower speed - leader speed`.
- IDM prior: fixed independent Gaussian prior on log parameters, centered at
  `[33, 2, 1.6, 1.5, 1.67]` with log standard deviation `0.1`.
- Scenario prior: `NIW(0, 0.01, 5, 0.01 I)` in standardized units.
- Residual variance: inverse-Gamma with shape `100` and scale `1`.
- Transition prior: the legacy effective concentration is retained exactly.
  Each joint-transition cell has concentration `1000 / K_B`, so the row total
  is `1000 * K_S`. This is strong symmetric regularization, not a sparse prior.
- Initial distribution: sampled from a symmetric Dirichlet posterior with
  total concentration `1`; only the first state of each trajectory is counted.

Complete latent paths are drawn with exact log-space FFBS. IDM parameters are
updated by bounded coordinate-wise slice sampling targeting the fixed
independent log-Gaussian conditional. Trajectory boundaries are respected in
all initial-state and transition counts.

## Frozen arms

Formal configurations are in `configs/`: `b1s1.json`, `b2s2.json`,
`b5s1.json`, `b1s5.json`, and `b5s5.json`. Each requests one 8,000-iteration
chain with 6,000 burn-in draws and 2,000 retained draws. Arm-specific random
streams derive from master seed `20260807` through NumPy `SeedSequence` spawn
keys 0 through 4. This lineage matches the completed formal run directories
and the downstream evidence currently used by the manuscript.

The master seed is reproducibility provenance rather than a scientific model
setting. Future runs may use another documented seed, but they must write to
new timestamped directories and must not be conflated with the existing
realization or its promoted artifacts. No seed change alone is an
evidence-release gate, and the completed `20260807` runs do not require
replacement.

## Data and execution gates

The runner accepts only the frozen flattened `cohort_data.npz` plus its
cryptographic `manifest.json`. It validates every required array, the manifest
input fingerprint, 100 distinct follower IDs, trajectory boundaries, finite
observations, and positive gap before creating a run directory. Negative speed
is rejected by default. The five formal configs explicitly record the Scheme B
implementation scope that preserves 405 observed values down to
`-0.20 m/s`, affecting 23 trajectories but none of cases 2, 3, 6, or 9. The
runner accepts those observations only if their exact count, fraction, lower
bound, affected trajectories, and preservation decision agree with the frozen
manifest. No value is clipped, deleted, or used to recompute relative speed.

After the cohort gate is approved, invoke one arm from the repository root:

```powershell
conda run -n gpytorch --no-capture-output python -m experiments.manuscript_storyrev_20260808.run_exact_reconstruction `
  --config experiments/manuscript_storyrev_20260808/configs/b5s5.json `
  --cohort-npz <forensic-output>/cohort_data.npz `
  --manifest <forensic-output>/manifest.json
```

Append `--preflight-only` to run all hash, cohort, domain, and target checks
without creating a results directory or taking a single MCMC draw.

`configs/runtime_pilot_b5s5.json` is a 20-iteration full-cohort timing pilot.
It has the same scientific-target payload as formal B5S5, but a separate random
stream and chain schedule, and is labeled
`RUNTIME_PILOT_NOT_MANUSCRIPT_EVIDENCE`. Its draws must not be used in tables,
figures, or manuscript claims.

Every run creates a new UTC timestamped directory containing the compressed
chain, standardizer, scientific-target fingerprint, seed lineage, environment,
per-iteration runtime diagnostics, a best-effort process-memory receipt, Git
commit, and artifact hashes. Timing diagnostics are stored separately from the
scientific chain payload. Existing artifacts are never reused unless their
target fingerprint matches exactly.
