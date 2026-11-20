# ATE estimator comparison (SYNTHETIC data)

Data source: SYNTHETIC samples from `ConfoundedSCM()` default parameters, seed 20240607. 200 replications, n = 5000 per replication.

True ATE (closed form, do(X=1) vs do(X=0)): 2.0000
Population naive OLS slope (closed form): 2.5499

| Estimator | Mean | Bias | Std | RMSE |
|---|---|---|---|---|
| naive OLS slope (no adjustment) | 2.5498 | +0.5498 | 0.0147 | 0.5499 |
| backdoor, linear model without X*Z | 2.0012 | +0.0012 | 0.0135 | 0.0135 |
| backdoor, linear model with X*Z | 2.0008 | +0.0008 | 0.0130 | 0.0130 |
| oracle Monte Carlo do(X) simulation | 1.9993 | -0.0007 | 0.0111 | 0.0111 |

95% percentile bootstrap coverage for the backdoor estimator with X*Z (200 resamples): 0.950 (190/200).
