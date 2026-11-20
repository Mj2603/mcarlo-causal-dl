# mcarlo-causal-dl: backdoor adjustment on a synthetic SCM

A small, tested demonstration of confounding bias and backdoor adjustment. A structural causal model (SCM) with one observed confounder Z generates SYNTHETIC observational data; the true average treatment effect (ATE) of the treatment X on the outcome Y is known in closed form. The repository compares a naive regression slope, which is biased by confounding, against a backdoor-adjusted (g-formula) estimator that averages a fitted outcome model over the empirical distribution of Z. Despite the repository name, there is no deep learning and no financial market data in this project.

## Model

```
Z ~ N(0, 1)
X = a * 1{Z > 0} + b * Z + U,          U ~ N(0, sigma_u^2)
Y = c * X + d * Z + e * X * Z + eps,   eps ~ N(0, sigma_y^2)
```

Defaults: a = 1, b = 0.1, sigma_u = 0.5, c = 2, d = 0.5, e = 0.3, sigma_y = 0.5. Because E[Z] = 0, the true ATE of do(X = 1) versus do(X = 0) is c = 2.

## What is implemented

- `ConfoundedSCM` (`src/mcarlo_causal/scm.py`): seeded sampling from the observational distribution; the closed-form true ATE; the closed-form population OLS slope of Y on X (the naive estimand); and an oracle Monte Carlo simulation of do(X) that uses the true structural equations, as a check on the simulator.
- Estimators (`src/mcarlo_causal/estimators.py`) that use only observed (z, x, y):
  - `naive_slope`: OLS slope of Y on X, without adjustment.
  - `backdoor_regression_ate`: OLS outcome model for E[Y | X, Z], with or without the X*Z term, averaged over observed z values (g-formula).
  - `bootstrap_ci`: nonparametric percentile bootstrap interval.
- `experiments/run_experiment.py`: repeated-sampling comparison that writes `results/ate_comparison.md` and `results/ate_comparison.csv`.
- `tests/`: checks against closed-form values (true ATE, naive slope bias with and without interaction), large-sample consistency, an exact noiseless case, a no-confounding invariant, and seeded reproducibility.

## How to run

```bash
git clone https://github.com/Mj2603/mcarlo-causal-dl.git
cd mcarlo-causal-dl
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
pytest
python experiments/run_experiment.py
```

`requirements.txt` installs numpy, pytest and this package in editable mode. The experiment takes about 15 seconds.

## Results

Data source: SYNTHETIC samples from `ConfoundedSCM()` with default parameters, seed 20240607, 200 replications of n = 5000. Copied from `results/ate_comparison.md`.

True ATE (closed form): 2.0000. Population naive OLS slope (closed form): 2.5499.

| Estimator | Mean | Bias | Std | RMSE |
|---|---|---|---|---|
| naive OLS slope (no adjustment) | 2.5498 | +0.5498 | 0.0147 | 0.5499 |
| backdoor, linear model without X*Z | 2.0012 | +0.0012 | 0.0135 | 0.0135 |
| backdoor, linear model with X*Z | 2.0008 | +0.0008 | 0.0130 | 0.0130 |
| oracle Monte Carlo do(X) simulation | 1.9993 | -0.0007 | 0.0111 | 0.0111 |

95% percentile bootstrap coverage for the backdoor estimator with X*Z (200 resamples): 0.950 (190/200).

## Limitations

- The confounder is observed and the outcome model with X*Z is correctly specified, so backdoor adjustment is expected to work. The project does not address unobserved confounding, model selection, or high-dimensional covariates.
- The model without X*Z is misspecified; its small bias here is specific to this SCM (E[Z] = 0) and should not be generalised.
- The oracle Monte Carlo row uses the true structural equations and is not available with real data.
- All data are synthetic. Nothing here has been applied to market data, and no trading, risk or regulatory claims are made.
- Counterfactual (level 3) queries are not implemented.
