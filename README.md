# A Causal Framework for Deep Learning via Monte Carlo Methods

This repo demonstrates a lightweight causal inference framework that integrates Monte Carlo estimation with trainable models. It simulates structural causal models (SCMs), performs intervention queries via Monte Carlo integration, and compares backdoor-adjusted estimates with observational baselines.

Key features:
- Structural causal simulation with confounding
- Backdoor adjustment via Monte Carlo
- Trainable predictor with uncertainty-aware evaluation
- Reproducible experiments in a single script

References for inspiration:
- GitHub: `binDebug3/MLArchitecture` ([link](https://github.com/binDebug3/MLArchitecture))

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python examples/demo.py
```

Outputs:
- Average treatment effect (ATE) via backdoor adjustment
- Observational naive estimate for comparison
- Simple plots saved to `outputs/`

## Project Structure

- `src/causal_mc.py` — SCM simulation and Monte Carlo estimators
- `examples/demo.py` — end-to-end experiment
- `requirements.txt` — dependencies

## Notes
This is a compact research-style scaffold intended for MS Mathematical Finance applications. Extend by:
- Replacing toy SCM with market microstructure factors
- Using intervention queries for strategy stress-tests

