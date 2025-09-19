import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from src.causal_mc import make_toy_scm


def main() -> None:
    scm = make_toy_scm()
    z, x, y = scm.sample_observational(5000)

    # Naive observational ATE via linear fit without adjustment
    beta = np.polyfit(x, y, 1)[0]

    # Backdoor-adjusted ATE via Monte Carlo intervention
    ate = scm.monte_carlo_ate(n=100_000, x0=0.0, x1=1.0)

    os.makedirs("outputs", exist_ok=True)

    df = pd.DataFrame({"z": z, "x": x, "y": y})
    df.sample(10, random_state=0).to_csv("outputs/sample.csv", index=False)

    plt.figure(figsize=(6,4))
    plt.scatter(x, y, s=5, alpha=0.2)
    xline = np.linspace(np.min(x), np.max(x), 200)
    plt.plot(xline, beta * xline + (y.mean() - beta * x.mean()), color="red", label=f"Naive slope={beta:.2f}")
    plt.title(f"Backdoor ATE (x=0->1): {ate:.3f}")
    plt.legend()
    plt.tight_layout()
    plt.savefig("outputs/observational_vs_ate.png", dpi=150)

    print(f"Naive observational slope ~ ATE: {beta:.3f}")
    print(f"Backdoor-adjusted Monte Carlo ATE (x=0->1): {ate:.3f}")


if __name__ == "__main__":
    main()
