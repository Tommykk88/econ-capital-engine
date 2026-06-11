"""Catastrophe risk: Poisson/negative-binomial frequency + GPD severity (EVT).

Compound annual loss model. Severity is semiparametric: empirical below a
Peaks-Over-Threshold threshold, Generalized Pareto above it. Frequency is
Poisson unless the annual counts show material overdispersion, in which case
negative binomial is used.
"""

from dataclasses import dataclass

import numpy as np
from scipy import stats


@dataclass
class FrequencyModel:
    distribution: str  # "poisson" or "negbinom"
    mean: float
    dispersion_ratio: float  # variance / mean of annual counts
    nb_n: float | None = None  # negative binomial shape (if used)
    nb_p: float | None = None

    def sample(self, n_years: int, rng: np.random.Generator) -> np.ndarray:
        if self.distribution == "poisson":
            return rng.poisson(self.mean, size=n_years)
        return rng.negative_binomial(self.nb_n, self.nb_p, size=n_years)


@dataclass
class SeverityModel:
    body: np.ndarray  # losses below threshold (empirical resampling)
    threshold: float
    xi: float  # GPD shape
    beta: float  # GPD scale
    p_exceed: float  # P(loss > threshold)
    xi_ci: tuple[float, float]  # 95% CI for xi

    def sample(self, n_events: int, rng: np.random.Generator) -> np.ndarray:
        is_tail = rng.uniform(size=n_events) < self.p_exceed
        out = np.empty(n_events)
        n_tail = int(is_tail.sum())
        out[is_tail] = self.threshold + stats.genpareto.rvs(
            self.xi, scale=self.beta, size=n_tail, random_state=rng
        )
        out[~is_tail] = rng.choice(self.body, size=n_events - n_tail, replace=True)
        return out

    @property
    def tail_mean_is_finite(self) -> bool:
        return self.xi < 1.0


def fit_frequency(annual_counts: np.ndarray, overdispersion_tolerance: float = 1.5) -> FrequencyModel:
    """Fit Poisson; switch to negative binomial if variance/mean exceeds tolerance."""
    counts = np.asarray(annual_counts, dtype=float)
    mean, var = counts.mean(), counts.var(ddof=1)
    ratio = var / mean
    if ratio <= overdispersion_tolerance:
        return FrequencyModel("poisson", mean=mean, dispersion_ratio=ratio)
    # method-of-moments negative binomial: var = mean + mean^2 / n
    nb_n = mean**2 / (var - mean)
    nb_p = nb_n / (nb_n + mean)
    return FrequencyModel("negbinom", mean=mean, dispersion_ratio=ratio, nb_n=nb_n, nb_p=nb_p)


def mean_excess(losses: np.ndarray, n_points: int = 60) -> tuple[np.ndarray, np.ndarray]:
    """Mean excess function e(u) = E[X - u | X > u] over a grid of thresholds.

    Approximately linear-in-u with positive slope indicates a GPD-type heavy
    tail; the threshold is chosen where the plot becomes stable/linear.
    """
    x = np.sort(np.asarray(losses))
    thresholds = np.quantile(x, np.linspace(0.50, 0.98, n_points))
    me = np.array([(x[x > u] - u).mean() for u in thresholds])
    return thresholds, me


def fit_gpd(losses: np.ndarray, threshold: float) -> SeverityModel:
    """Fit a GPD to exceedances above `threshold` by maximum likelihood.

    The 95% CI for xi uses the asymptotic MLE variance (1 + xi)^2 / n.
    """
    losses = np.asarray(losses)
    exceedances = losses[losses > threshold] - threshold
    if len(exceedances) < 30:
        raise ValueError(f"Only {len(exceedances)} exceedances; threshold too high.")
    xi, _, beta = stats.genpareto.fit(exceedances, floc=0.0)
    se_xi = (1 + xi) / np.sqrt(len(exceedances))
    return SeverityModel(
        body=losses[losses <= threshold],
        threshold=float(threshold),
        xi=float(xi),
        beta=float(beta),
        p_exceed=float((losses > threshold).mean()),
        xi_ci=(float(xi - 1.96 * se_xi), float(xi + 1.96 * se_xi)),
    )


def simulate_annual_losses(
    frequency: FrequencyModel,
    severity: SeverityModel,
    n_scenarios: int = 200_000,
    seed: int = 3,
) -> np.ndarray:
    """Compound Monte Carlo: draw event count per year, draw severities, sum."""
    rng = np.random.default_rng(seed)
    counts = frequency.sample(n_scenarios, rng)
    all_events = severity.sample(int(counts.sum()), rng)
    # sum severities per scenario via segment boundaries
    boundaries = np.concatenate([[0], np.cumsum(counts)])
    cumulative = np.concatenate([[0.0], np.cumsum(all_events)])
    return cumulative[boundaries[1:]] - cumulative[boundaries[:-1]]
