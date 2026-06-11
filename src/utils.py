"""Shared helpers: probability integral transforms, quantile mapping, plotting."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

FIG_DIR = Path(__file__).resolve().parents[1] / "docs" / "figures"


def to_uniform(x: np.ndarray) -> np.ndarray:
    """Probability integral transform via the empirical CDF.

    Uses rank / (n + 1) so that the result lies strictly inside (0, 1),
    which keeps inverse transforms of other marginals well defined.
    """
    x = np.asarray(x)
    ranks = np.argsort(np.argsort(x)) + 1.0
    return ranks / (len(x) + 1.0)


def empirical_quantile(u: np.ndarray, sample: np.ndarray) -> np.ndarray:
    """Map uniforms back through the inverse empirical CDF of `sample`.

    Linear interpolation between order statistics (type-7 quantile).
    """
    sorted_sample = np.sort(np.asarray(sample))
    n = len(sorted_sample)
    pos = np.clip(np.asarray(u) * (n - 1), 0.0, n - 1.0)
    lo = np.floor(pos).astype(int)
    hi = np.minimum(lo + 1, n - 1)
    frac = pos - lo
    return sorted_sample[lo] * (1 - frac) + sorted_sample[hi] * frac


def var(losses: np.ndarray, alpha: float) -> float:
    """Value-at-Risk at confidence level alpha (losses positive)."""
    return float(np.quantile(losses, alpha))


def expected_shortfall(losses: np.ndarray, alpha: float) -> float:
    """Expected Shortfall: mean loss beyond VaR_alpha."""
    losses = np.asarray(losses)
    threshold = np.quantile(losses, alpha)
    tail = losses[losses > threshold]
    if tail.size == 0:  # degenerate distribution
        return float(threshold)
    return float(tail.mean())


def batch_standard_error(losses: np.ndarray, statistic, n_batches: int = 50) -> float:
    """Monte Carlo standard error of a tail statistic via batch means."""
    losses = np.asarray(losses)
    batches = np.array_split(losses, n_batches)
    estimates = np.array([statistic(b) for b in batches])
    return float(estimates.std(ddof=1) / np.sqrt(n_batches))


def save_figure(fig, name: str) -> Path:
    """Save a figure into docs/figures and close it."""
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    path = FIG_DIR / name
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path
