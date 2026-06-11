import numpy as np
import pytest

from src import capital
from src.utils import expected_shortfall, var


@pytest.fixture(scope="module")
def losses():
    rng = np.random.default_rng(0)
    return rng.lognormal(4, 1, 200_000)


def test_es_geq_var(losses):
    for alpha in [0.95, 0.975, 0.995]:
        assert expected_shortfall(losses, alpha) >= var(losses, alpha)


def test_capital_metrics_structure(losses):
    metrics = capital.capital_metrics(losses)
    assert set(metrics) == set(capital.REGULATORY_METRICS)
    for m in metrics.values():
        assert m["value"] > 0 and m["mc_se"] > 0
    assert (
        metrics["VaR 99.9% (Basel)"]["value"] > metrics["VaR 99.5% (Solvency II)"]["value"]
    )


def test_euler_allocation_sums_to_total_es():
    rng = np.random.default_rng(1)
    components = rng.lognormal(3, 1, size=(100_000, 3))
    alloc = capital.euler_allocation_es(components, ["a", "b", "c"], alpha=0.975)
    total_es = expected_shortfall(components.sum(axis=1), 0.975)
    assert sum(alloc.values()) == pytest.approx(total_es, rel=1e-6)


def test_shift_correlation_stays_valid():
    corr = capital.shift_correlation(np.array([[1.0, 0.9], [0.9, 1.0]]), 0.15)
    assert np.allclose(np.diag(corr), 1.0)
    assert np.all(np.linalg.eigvalsh(corr) > 0)
    assert corr[0, 1] <= 0.99


def test_var_convergence_monotone_sizes(losses):
    sizes, estimates = capital.var_convergence(losses)
    assert (np.diff(sizes) > 0).all()
    # final estimates should be close to the full-sample VaR
    assert estimates[-1] == pytest.approx(var(losses, 0.995), rel=1e-9)
