"""Capital metrics, Euler allocation and the stress/sensitivity layer."""

import numpy as np

from src.utils import batch_standard_error, expected_shortfall, var

REGULATORY_METRICS = {
    "VaR 99.9% (Basel)": ("var", 0.999),
    "VaR 99.5% (Solvency II)": ("var", 0.995),
    "ES 97.5% (FRTB)": ("es", 0.975),
}


def capital_metrics(losses: np.ndarray) -> dict[str, dict[str, float]]:
    """Regulatory capital metrics with Monte Carlo standard errors."""
    out = {}
    for label, (kind, alpha) in REGULATORY_METRICS.items():
        fn = var if kind == "var" else expected_shortfall
        out[label] = {
            "value": fn(losses, alpha),
            "mc_se": batch_standard_error(losses, lambda b, a=alpha, f=fn: f(b, a)),
        }
    return out


def euler_allocation_es(
    components: np.ndarray, names: list[str], alpha: float = 0.975
) -> dict[str, float]:
    """Euler capital contributions under Expected Shortfall.

    EC_k = E[L_k | L_total > VaR_alpha(L_total)]. By construction the
    contributions sum to ES of the total (full allocation property).
    """
    total = components.sum(axis=1)
    threshold = np.quantile(total, alpha)
    tail = total > threshold
    return {name: float(components[tail, j].mean()) for j, name in enumerate(names)}


def var_convergence(losses: np.ndarray, alpha: float = 0.995, n_points: int = 30) -> tuple:
    """VaR estimate as a function of the number of scenarios used."""
    sizes = np.unique(np.geomspace(5_000, len(losses), n_points).astype(int))
    estimates = np.array([np.quantile(losses[:m], alpha) for m in sizes])
    return sizes, estimates


def run_stress_tests(rerun, base_var: float) -> dict[str, float]:
    """One-at-a-time sensitivity runs for the tornado chart.

    `rerun` is a callable mapping a dict of overrides to total VaR 99.5%
    (the pipeline wires this to a full re-simulation under a fixed seed).
    Returned values are percentage changes versus base capital.
    """
    shocks = {
        "Copula correlations +0.15": {"corr_shift": 0.15},
        "t-copula nu: 4 -> 3": {"nu": 3.0},
        "GPD xi at upper 95% CI": {"xi_upper": True},
        "Cat frequency +20%": {"freq_scale": 1.2},
        "Credit rho: 0.18 -> 0.24": {"rho": 0.24},
    }
    return {
        label: (rerun(overrides) / base_var - 1.0) * 100.0
        for label, overrides in shocks.items()
    }


def shift_correlation(corr: np.ndarray, shift: float) -> np.ndarray:
    """Shift off-diagonal correlations, capped below 1, keeping PSD by clipping
    eigenvalues if necessary."""
    shifted = np.minimum(corr + shift * (1 - np.eye(len(corr))), 0.99)
    np.fill_diagonal(shifted, 1.0)
    eigval, eigvec = np.linalg.eigh(shifted)
    if eigval.min() <= 0:
        eigval = np.clip(eigval, 1e-8, None)
        shifted = eigvec @ np.diag(eigval) @ eigvec.T
        d = np.sqrt(np.diag(shifted))
        shifted = shifted / np.outer(d, d)
    return shifted
