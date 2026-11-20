"""Repeated-sampling comparison of ATE estimators on the SYNTHETIC SCM.

All data are SYNTHETIC, drawn from mcarlo_causal.ConfoundedSCM with fixed
seeds. Writes results/ate_comparison.csv and results/ate_comparison.md.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from mcarlo_causal import ConfoundedSCM, backdoor_regression_ate, bootstrap_ci, naive_slope


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--reps", type=int, default=200)
    p.add_argument("--n", type=int, default=5000)
    p.add_argument("--n-boot", type=int, default=200)
    p.add_argument("--seed", type=int, default=20240607)
    p.add_argument("--out", type=Path, default=Path("results"))
    args = p.parse_args()

    scm = ConfoundedSCM()
    truth = scm.true_ate()
    rng = np.random.default_rng(args.seed)

    names = [
        "naive OLS slope (no adjustment)",
        "backdoor, linear model without X*Z",
        "backdoor, linear model with X*Z",
        "oracle Monte Carlo do(X) simulation",
    ]
    est = {k: np.empty(args.reps) for k in names}
    covered = 0
    for r in range(args.reps):
        z, x, y = scm.sample_observational(args.n, rng)
        est[names[0]][r] = naive_slope(x, y)
        est[names[1]][r] = backdoor_regression_ate(z, x, y, interaction=False)
        est[names[2]][r] = backdoor_regression_ate(z, x, y, interaction=True)
        est[names[3]][r] = scm.oracle_mc_ate(args.n, rng)
        lo, hi = bootstrap_ci(backdoor_regression_ate, (z, x, y), rng, n_boot=args.n_boot)
        covered += lo <= truth <= hi

    rows = []
    for k in names:
        v = est[k]
        rows.append(
            {
                "estimator": k,
                "mean": v.mean(),
                "bias": v.mean() - truth,
                "std": v.std(ddof=1),
                "rmse": float(np.sqrt(np.mean((v - truth) ** 2))),
            }
        )
    coverage = covered / args.reps

    args.out.mkdir(parents=True, exist_ok=True)
    with open(args.out / "ate_comparison.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        for row in rows:
            w.writerow({k: (f"{v:.4f}" if isinstance(v, float) else v) for k, v in row.items()})

    lines = [
        "# ATE estimator comparison (SYNTHETIC data)",
        "",
        "Data source: SYNTHETIC samples from `ConfoundedSCM()` default parameters, "
        f"seed {args.seed}. {args.reps} replications, n = {args.n} per replication.",
        "",
        f"True ATE (closed form, do(X=1) vs do(X=0)): {truth:.4f}",
        f"Population naive OLS slope (closed form): {scm.population_naive_slope():.4f}",
        "",
        "| Estimator | Mean | Bias | Std | RMSE |",
        "|---|---|---|---|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['estimator']} | {row['mean']:.4f} | {row['bias']:+.4f} | "
            f"{row['std']:.4f} | {row['rmse']:.4f} |"
        )
    lines += [
        "",
        f"95% percentile bootstrap coverage for the backdoor estimator with X*Z "
        f"({args.n_boot} resamples): {coverage:.3f} ({covered}/{args.reps}).",
        "",
    ]
    (args.out / "ate_comparison.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
