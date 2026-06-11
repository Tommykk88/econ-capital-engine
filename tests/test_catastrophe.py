import numpy as np
import pytest
from scipy import stats

from src import catastrophe


@pytest.fixture(scope="module")
def models():
    rng = np.random.default_rng(1)
    losses = np.concatenate([
        rng.lognormal(6.0, 1.0, size=900),
        2_000.0 * (rng.pareto(2.5, size=100) + 1.0),
    ])
    frequency = catastrophe.fit_frequency(rng.poisson(10, size=40))
    severity = catastrophe.fit_gpd(losses, threshold=float(np.quantile(losses, 0.90)))
    return frequency, severity


def test_poisson_chosen_when_no_overdispersion():
    rng = np.random.default_rng(2)
    freq = catastrophe.fit_frequency(rng.poisson(8, size=200))
    assert freq.distribution == "poisson"
    assert freq.mean == pytest.approx(8, rel=0.1)


def test_negbinom_chosen_under_overdispersion():
    rng = np.random.default_rng(3)
    counts = rng.negative_binomial(2, 0.2, size=200)  # var/mean = 1/0.2 = 5
    freq = catastrophe.fit_frequency(counts)
    assert freq.distribution == "negbinom"
    sample = freq.sample(50_000, np.random.default_rng(4))
    assert sample.mean() == pytest.approx(counts.mean(), rel=0.05)


def test_simulated_frequency_matches_parameter(models):
    frequency, _ = models
    sample = frequency.sample(100_000, np.random.default_rng(5))
    assert sample.mean() == pytest.approx(frequency.mean, rel=0.02)


def test_gpd_sampling_matches_scipy(models):
    _, severity = models
    rng = np.random.default_rng(6)
    draws = severity.sample(100_000, rng)
    tail = draws[draws > severity.threshold] - severity.threshold
    # KS test of tail draws against the fitted GPD
    ks = stats.kstest(tail, "genpareto", args=(severity.xi, 0.0, severity.beta))
    assert ks.pvalue > 0.01


def test_aggregate_mean_matches_compound_formula(models):
    frequency, severity = models
    annual = catastrophe.simulate_annual_losses(frequency, severity, n_scenarios=200_000, seed=8)
    severity_mean = severity.sample(500_000, np.random.default_rng(9)).mean()
    assert annual.mean() == pytest.approx(frequency.mean * severity_mean, rel=0.05)


def test_xi_ci_contains_point_estimate(models):
    _, severity = models
    lo, hi = severity.xi_ci
    assert lo < severity.xi < hi
