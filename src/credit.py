"""Credit risk: Vasicek one-factor Gaussian copula portfolio loss model.

The model underlying the Basel IRB formula. A synthetic portfolio of obligors
defaults jointly through a single systematic factor:

    A_i = sqrt(rho) * Z + sqrt(1 - rho) * eps_i,   default iff A_i < Phi^{-1}(PD_i)

Portfolio loss per scenario is the sum of EAD * LGD over defaulted obligors.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

# PDs loosely calibrated to long-run rating-agency annual corporate default
# rates (S&P/Moody's global tables, order of magnitude only).
RATING_BUCKETS = {
    # rating: (PD, portfolio weight by obligor count)
    "AA": (0.0002, 0.10),
    "A": (0.0006, 0.20),
    "BBB": (0.0018, 0.30),
    "BB": (0.0080, 0.25),
    "B": (0.0500, 0.15),
}


@dataclass
class CreditPortfolio:
    table: pd.DataFrame  # columns: rating, pd, lgd, ead
    rho: float

    @property
    def total_exposure(self) -> float:
        return float(self.table["ead"].sum())

    @property
    def max_loss(self) -> float:
        return float((self.table["ead"] * self.table["lgd"]).sum())


def build_portfolio(n_obligors: int = 750, rho: float = 0.18, seed: int = 42) -> CreditPortfolio:
    """Build a synthetic obligor portfolio with skewed exposure sizes.

    EADs are lognormal (a few large exposures, many small), scaled so total
    EAD = 15,000 (currency units of millions). LGD uniform in [0.40, 0.60].
    """
    rng = np.random.default_rng(seed)
    ratings, pds = [], []
    for rating, (pd_, weight) in RATING_BUCKETS.items():
        count = int(round(weight * n_obligors))
        ratings += [rating] * count
        pds += [pd_] * count

    n = len(ratings)
    ead = rng.lognormal(mean=0.0, sigma=1.2, size=n)
    ead *= 15_000.0 / ead.sum()
    lgd = rng.uniform(0.40, 0.60, size=n)

    table = pd.DataFrame({"rating": ratings, "pd": pds, "lgd": lgd, "ead": ead})
    return CreditPortfolio(table=table, rho=rho)


def simulate_losses(
    portfolio: CreditPortfolio,
    n_scenarios: int = 200_000,
    seed: int = 1,
    batch_size: int = 20_000,
) -> np.ndarray:
    """Simulate annual portfolio credit losses under the one-factor model.

    Batched so that the (scenarios x obligors) default indicator matrix never
    exceeds batch_size * n_obligors in memory.
    """
    rng = np.random.default_rng(seed)
    pd_ = portfolio.table["pd"].to_numpy()
    severity = (portfolio.table["ead"] * portfolio.table["lgd"]).to_numpy()
    threshold = stats.norm.ppf(pd_)
    rho = portfolio.rho

    losses = np.empty(n_scenarios)
    done = 0
    while done < n_scenarios:
        m = min(batch_size, n_scenarios - done)
        z = rng.standard_normal((m, 1))
        eps = rng.standard_normal((m, len(pd_)))
        asset = np.sqrt(rho) * z + np.sqrt(1.0 - rho) * eps
        defaults = asset < threshold  # broadcast over obligors
        losses[done : done + m] = defaults @ severity
        done += m
    return losses


def vasicek_lhp_cdf(loss_fraction: np.ndarray, pd_: float, rho: float) -> np.ndarray:
    """Analytical Vasicek large-homogeneous-portfolio loss CDF.

    P(L <= x) = Phi( (sqrt(1-rho) * Phi^{-1}(x) - Phi^{-1}(pd)) / sqrt(rho) )
    where x is the loss fraction of the (unit-LGD) portfolio.
    """
    x = np.clip(np.asarray(loss_fraction, dtype=float), 1e-12, 1 - 1e-12)
    return stats.norm.cdf(
        (np.sqrt(1 - rho) * stats.norm.ppf(x) - stats.norm.ppf(pd_)) / np.sqrt(rho)
    )
