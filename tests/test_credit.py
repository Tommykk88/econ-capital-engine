import numpy as np
import pytest

from src import credit


@pytest.fixture(scope="module")
def portfolio():
    return credit.build_portfolio(n_obligors=600, rho=0.18, seed=42)


def test_portfolio_construction(portfolio):
    t = portfolio.table
    assert (t["pd"] > 0).all() and (t["pd"] < 1).all()
    assert t["lgd"].between(0.40, 0.60).all()
    assert np.isclose(t["ead"].sum(), 15_000.0)


def test_expected_default_rate_matches_average_pd(portfolio):
    losses = credit.simulate_losses(portfolio, n_scenarios=50_000, seed=0)
    expected_loss = (portfolio.table["ead"] * portfolio.table["lgd"] * portfolio.table["pd"]).sum()
    assert losses.mean() == pytest.approx(expected_loss, rel=0.05)


def test_losses_bounded(portfolio):
    losses = credit.simulate_losses(portfolio, n_scenarios=20_000, seed=1)
    assert (losses >= 0).all()
    assert (losses <= portfolio.max_loss).all()


def test_zero_correlation_reproduces_independent_defaults():
    pf = credit.build_portfolio(n_obligors=600, rho=0.0, seed=3)
    pf.table["lgd"] = 1.0
    pf.table["ead"] = 1.0  # loss = number of defaults
    n_sim = 40_000
    losses = credit.simulate_losses(pf, n_scenarios=n_sim, seed=4)
    # with rho=0 the default count is a sum of independent Bernoullis
    expected_var = (pf.table["pd"] * (1 - pf.table["pd"])).sum()
    assert losses.var() == pytest.approx(expected_var, rel=0.10)


def test_vasicek_cdf_properties():
    x = np.linspace(0.001, 0.999, 100)
    cdf = credit.vasicek_lhp_cdf(x, pd_=0.02, rho=0.18)
    assert (np.diff(cdf) >= 0).all()  # monotone
    assert credit.vasicek_lhp_cdf(0.02, 0.02, 0.18) > 0.5  # median below mean (right skew)
    # CDF evaluated at PD quantile consistency with norm
    assert 0 <= cdf[0] <= cdf[-1] <= 1
