# Economic Capital Aggregation Engine

**EVT, copulas, and capital allocation for banking and reinsurance risk.**

Computes the one-year economic capital requirement of a hypothetical
diversified financial institution holding credit risk (Vasicek one-factor),
market risk (GARCH(1,1) filtered historical simulation), and catastrophe
risk (Poisson frequency + Generalized Pareto severity). The marginals are
aggregated with Gaussian, Student-t and Gumbel copulas to show how tail
dependence destroys diversification benefit precisely when it matters.

**Headline result:** moving from a Gaussian to a Gumbel copula — holding the
overall dependence level (average Kendall tau) fixed — cuts the measured
diversification benefit at the 99.5th percentile from **29.9% to 18.3%**,
raising required capital by **~16%**. The stress tornado shows the two
dominant capital sensitivities are the GPD tail shape and the copula
dependence assumptions, i.e. exactly the parameters least identified by
data: that is the model-risk message of the project.

| | |
|---|---|
| ![Diversification benefit](docs/figures/copula_diversification_benefit.png) | ![Stress tornado](docs/figures/stress_tornado.png) |

![Euler allocation](docs/figures/euler_waterfall.png)

Full numbers (regenerated on every run): [`docs/results.md`](docs/results.md).
Capital metrics on the base (t-copula) model: **VaR 99.9%** (Basel),
**VaR 99.5%** (Solvency II SCR), **ES 97.5%** (FRTB), with Monte Carlo
standard errors and a convergence study; capital is allocated back to risk
types by **Euler allocation** under ES (full-allocation property verified).

## Quickstart

```bash
pip install -e ".[dev]"
pytest                  # 28 unit tests
python run_all.py       # full pipeline: data -> marginals -> copulas -> capital
```

`run_all.py` reproduces every figure in `docs/figures/` and rewrites
`docs/results.md`. All randomness is seeded. Market data is downloaded from
Yahoo Finance on first run and cached. Catastrophe data defaults to a
clearly-flagged synthetic catalog; to use real data, register at
[emdat.be](https://www.emdat.be), export storms/floods as `data/raw/emdat.xlsx`,
delete `data/processed/cat_catalog.csv` and re-run.

## What's inside

| Module | Model | Validation |
|---|---|---|
| `src/credit.py` | Vasicek one-factor portfolio credit model (Basel IRB foundation) | Analytical Vasicek LHP CDF overlay |
| `src/market.py` | GARCH(1,1)-t filtered historical simulation, full 250-day paths (no sqrt-time scaling) | Rolling out-of-sample 99% VaR backtest + Kupiec POF test |
| `src/catastrophe.py` | Compound Poisson / neg-binomial frequency + GPD severity (POT) | Mean excess plot, GPD QQ plot, xi confidence interval |
| `src/aggregation.py` | Gaussian, Student-t and Gumbel copulas, all implemented from scratch | Uniform marginals (KS), independence and t→Gaussian limits |
| `src/capital.py` | VaR/ES + MC standard errors, Euler allocation, stress layer | ES≥VaR, exact full allocation, seeded reproducibility |

Model documentation in the style of an internal model report:
[`docs/methodology.md`](docs/methodology.md). Honest weaknesses:
[`LIMITATIONS.md`](LIMITATIONS.md).
