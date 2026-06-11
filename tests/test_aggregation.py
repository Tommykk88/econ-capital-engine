import numpy as np
import pytest
from scipy import stats

from src import aggregation


CORR = aggregation.BASE_CORRELATION


def test_copula_marginals_are_uniform():
    n = 50_000
    samples = {
        "gaussian": aggregation.gaussian_copula_sample(CORR, n),
        "t": aggregation.t_copula_sample(CORR, nu=4, n=n),
        "gumbel": aggregation.gumbel_copula_sample(1.5, 3, n),
    }
    for name, u in samples.items():
        for j in range(u.shape[1]):
            ks = stats.kstest(u[:, j], "uniform")
            assert ks.pvalue > 0.01, f"{name} margin {j} not uniform (p={ks.pvalue:.4f})"


def test_gaussian_identity_correlation_is_independence():
    u = aggregation.gaussian_copula_sample(np.eye(2), 100_000, seed=1)
    corr = np.corrcoef(u.T)[0, 1]
    assert abs(corr) < 0.01


def test_t_copula_converges_to_gaussian_for_large_nu():
    n, seed = 200_000, 2
    u_t = aggregation.t_copula_sample(CORR, nu=1e6, n=n, seed=seed)
    u_g = aggregation.gaussian_copula_sample(CORR, n, seed=seed)
    # compare rank correlations
    tau_t = stats.kendalltau(u_t[:50_000, 0], u_t[:50_000, 1]).statistic
    tau_g = stats.kendalltau(u_g[:50_000, 0], u_g[:50_000, 1]).statistic
    assert tau_t == pytest.approx(tau_g, abs=0.01)


def test_gumbel_theta_one_is_independence():
    u = aggregation.gumbel_copula_sample(1.0, 2, 100_000, seed=3)
    assert abs(np.corrcoef(u.T)[0, 1]) < 0.01


def test_gumbel_kendall_tau():
    theta = 2.0  # tau = 1 - 1/theta = 0.5
    u = aggregation.gumbel_copula_sample(theta, 2, 20_000, seed=4)
    tau = stats.kendalltau(u[:, 0], u[:, 1]).statistic
    assert tau == pytest.approx(0.5, abs=0.02)


def test_aggregate_preserves_marginals_and_sums():
    rng = np.random.default_rng(5)
    marginals = {
        "credit": rng.lognormal(3, 1, 50_000),
        "market": rng.lognormal(3.5, 0.8, 50_000),
        "cat": rng.lognormal(2.5, 1.2, 50_000),
    }
    res = aggregation.aggregate(marginals, "gaussian", n_scenarios=50_000)
    assert res["total"] == pytest.approx(res["components"].sum(axis=1))
    for j, name in enumerate(res["names"]):
        assert res["components"][:, j].mean() == pytest.approx(
            marginals[name].mean(), rel=0.05
        )


def test_diversification_benefit_ordering():
    """Tail-dependent copulas must show lower diversification benefit."""
    rng = np.random.default_rng(6)
    marginals = {k: rng.lognormal(3, 1, 100_000) for k in ["credit", "market", "cat"]}
    n = 200_000
    db = {}
    for cop, kw in [("gaussian", {}), ("t", {"nu": 4.0}), ("gumbel", {})]:
        res = aggregation.aggregate(marginals, cop, n_scenarios=n, **kw)
        db[cop] = aggregation.diversification_benefit(res)
    assert db["gaussian"] > db["t"] > db["gumbel"]
