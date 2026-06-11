"""Copula-based aggregation of marginal annual loss distributions.

Three copulas, all implemented from scratch:
- Gaussian (Cholesky):       no tail dependence
- Student-t (Gaussian/chi2): symmetric tail dependence, heavier as nu falls
- Gumbel (Marshall-Olkin):   upper tail dependence -- joint extreme losses

Marginal simulations are mapped to uniforms by their empirical CDFs and back
through inverse empirical CDFs after applying copula dependence.
"""

import numpy as np
from scipy import stats

from src.utils import empirical_quantile, to_uniform

# Baseline dependence assumption (see docs/methodology.md, section 4):
# credit-market dependence is strong (recessions drive both), cat is mostly
# independent of the financial cycle with a mild market link via post-event
# asset sales / insured equity exposure.
RISK_ORDER = ["credit", "market", "cat"]
BASE_CORRELATION = np.array(
    [
        [1.00, 0.50, 0.10],
        [0.50, 1.00, 0.20],
        [0.10, 0.20, 1.00],
    ]
)


def gaussian_copula_sample(corr: np.ndarray, n: int, seed: int = 10) -> np.ndarray:
    """Sample uniforms from a Gaussian copula via Cholesky decomposition."""
    rng = np.random.default_rng(seed)
    chol = np.linalg.cholesky(corr)
    z = rng.standard_normal((n, corr.shape[0])) @ chol.T
    return stats.norm.cdf(z)


def t_copula_sample(corr: np.ndarray, nu: float, n: int, seed: int = 11) -> np.ndarray:
    """Sample uniforms from a Student-t copula.

    Multivariate t = correlated Gaussian / sqrt(chi2_nu / nu); marginals are
    mapped through the univariate t CDF.
    """
    rng = np.random.default_rng(seed)
    chol = np.linalg.cholesky(corr)
    z = rng.standard_normal((n, corr.shape[0])) @ chol.T
    w = rng.chisquare(nu, size=(n, 1)) / nu
    t_samples = z / np.sqrt(w)
    return stats.t.cdf(t_samples, df=nu)


def gumbel_copula_sample(theta: float, dim: int, n: int, seed: int = 12) -> np.ndarray:
    """Sample uniforms from an exchangeable Gumbel copula (theta >= 1).

    Marshall-Olkin construction: with V positive stable with index 1/theta
    and E_i iid Exp(1), U_i = exp(-(E_i / V)^(1/theta)) has the Gumbel copula.
    V is drawn by the Chambers-Mallows-Stuck algorithm. Upper tail dependence
    lambda_U = 2 - 2^(1/theta).
    """
    if theta < 1.0:
        raise ValueError("Gumbel copula requires theta >= 1")
    rng = np.random.default_rng(seed)
    if theta == 1.0:
        return rng.uniform(size=(n, dim))

    alpha = 1.0 / theta
    # Chambers-Mallows-Stuck for positive stable(alpha)
    u = rng.uniform(0, np.pi, size=n)
    e = rng.exponential(size=n)
    v = (
        np.sin(alpha * u)
        / np.sin(u) ** (1.0 / alpha)
        * (np.sin((1.0 - alpha) * u) / e) ** ((1.0 - alpha) / alpha)
    )
    exp_draws = rng.exponential(size=(n, dim))
    return np.exp(-((exp_draws / v[:, None]) ** alpha))


def gumbel_theta_from_corr(corr: np.ndarray) -> float:
    """Calibrate Gumbel theta by matching the average pairwise Kendall tau.

    For the Gaussian copula tau = (2/pi) arcsin(rho); for Gumbel
    tau = 1 - 1/theta. Equating average taus gives a like-for-like overall
    dependence level so that copula comparisons isolate the tail behaviour.
    """
    rho = corr[np.triu_indices_from(corr, k=1)]
    tau = (2.0 / np.pi) * np.arcsin(rho)
    return 1.0 / (1.0 - tau.mean())


def aggregate(
    marginals: dict[str, np.ndarray],
    copula: str,
    n_scenarios: int = 500_000,
    corr: np.ndarray = BASE_CORRELATION,
    nu: float = 4.0,
    theta: float | None = None,
    seed: int = 20,
) -> dict:
    """Aggregate marginal loss simulations under a chosen copula.

    Returns component losses (n_scenarios x n_risks, ordered as RISK_ORDER)
    and their sum.
    """
    names = [k for k in RISK_ORDER if k in marginals]
    dim = len(names)
    if copula == "gaussian":
        u = gaussian_copula_sample(corr[:dim, :dim], n_scenarios, seed=seed)
    elif copula == "t":
        u = t_copula_sample(corr[:dim, :dim], nu, n_scenarios, seed=seed)
    elif copula == "gumbel":
        theta = gumbel_theta_from_corr(corr[:dim, :dim]) if theta is None else theta
        u = gumbel_copula_sample(theta, dim, n_scenarios, seed=seed)
    else:
        raise ValueError(f"Unknown copula: {copula}")

    components = np.column_stack(
        [empirical_quantile(u[:, j], marginals[name]) for j, name in enumerate(names)]
    )
    return {
        "names": names,
        "components": components,
        "total": components.sum(axis=1),
        "uniforms": u,
        "copula": copula,
    }


def diversification_benefit(result: dict, alpha: float = 0.995) -> float:
    """DB = 1 - VaR(total) / sum of standalone VaRs at the same level."""
    standalone = sum(np.quantile(result["components"][:, j], alpha) for j in range(len(result["names"])))
    return 1.0 - np.quantile(result["total"], alpha) / standalone


__all__ = [
    "BASE_CORRELATION",
    "RISK_ORDER",
    "aggregate",
    "diversification_benefit",
    "gaussian_copula_sample",
    "gumbel_copula_sample",
    "gumbel_theta_from_corr",
    "t_copula_sample",
    "to_uniform",
]
