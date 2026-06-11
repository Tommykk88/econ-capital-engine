import numpy as np
import pandas as pd
import pytest

from src import market


@pytest.fixture(scope="module")
def garch_result():
    # simulate a GARCH(1,1)-t series so tests do not need network access
    rng = np.random.default_rng(0)
    n = 3_000
    omega, a, b, nu = 0.02, 0.08, 0.90, 6.0
    sigma2 = omega / (1 - a - b)
    r = np.empty(n)
    for t in range(n):
        z = rng.standard_t(nu) * np.sqrt((nu - 2) / nu)
        r[t] = np.sqrt(sigma2) * z
        sigma2 = omega + a * r[t] ** 2 + b * sigma2
    series = pd.Series(r, index=pd.bdate_range("2014-01-01", periods=n))
    return market.fit_garch(series), series


def test_garch_stationarity(garch_result):
    result, _ = garch_result
    p = result.params
    assert p["alpha[1]"] + p["beta[1]"] < 1.0


def test_standardized_residuals_unit_variance(garch_result):
    result, _ = garch_result
    z = market.standardized_residuals(result)
    assert z.std() == pytest.approx(1.0, abs=0.1)


def test_var_monotone_in_confidence(garch_result):
    result, _ = garch_result
    losses = market.simulate_annual_losses(result, n_scenarios=30_000, seed=5)
    q = np.quantile(losses, [0.95, 0.99, 0.995])
    assert q[0] < q[1] < q[2]


def test_simulation_reproducible(garch_result):
    result, _ = garch_result
    a = market.simulate_annual_losses(result, n_scenarios=5_000, seed=7)
    b = market.simulate_annual_losses(result, n_scenarios=5_000, seed=7)
    assert np.array_equal(a, b)


def test_kupiec_pof_calibration():
    # exactly the expected number of exceptions -> LR ~ 0, p ~ 1
    out = market.kupiec_pof(n_exceptions=10, n_obs=1_000, alpha=0.99)
    assert out["kupiec_lr"] == pytest.approx(0.0, abs=1e-9)
    assert out["kupiec_pvalue"] > 0.99
    # way too many exceptions -> reject
    out = market.kupiec_pof(n_exceptions=40, n_obs=1_000, alpha=0.99)
    assert out["kupiec_pvalue"] < 0.01
