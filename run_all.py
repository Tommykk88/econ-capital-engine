"""End-to-end pipeline: marginals -> copula aggregation -> capital -> figures.

Reproduces every figure in docs/figures and writes docs/results.md.
All randomness is seeded. Run from the repo root:  python run_all.py
"""

import subprocess
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src import aggregation, capital, catastrophe, credit, market
from src.utils import expected_shortfall, save_figure, var

ROOT = Path(__file__).resolve().parent
N_MARGINAL = 200_000  # scenarios per marginal model
N_JOINT = 500_000  # joint copula scenarios
BASE_NU = 4.0
# The cat catalog records industry-wide event losses; the institution
# underwrites a fixed participation share of them (reinsurance book).
CAT_SHARE = 0.02

# The reported capital model: Student-t copula (nu=4). Gaussian and Gumbel are
# the comparison points in the copula study.
BASE_COPULA = "t"


# ----------------------------------------------------------------- marginals
def run_credit(rho: float = 0.18, n: int = N_MARGINAL,
               make_figures: bool = True) -> tuple[np.ndarray, dict]:
    portfolio = credit.build_portfolio(rho=rho)
    losses = credit.simulate_losses(portfolio, n_scenarios=n)

    # Vasicek LHP validation: simulated loss-fraction CDF vs analytical CDF
    weights = (portfolio.table["ead"] * portfolio.table["lgd"]).to_numpy()
    loss_frac = losses / weights.sum()
    avg_pd = float(np.average(portfolio.table["pd"], weights=weights))
    grid = np.linspace(1e-4, np.quantile(loss_frac, 0.9995), 400)
    empirical_cdf = np.searchsorted(np.sort(loss_frac), grid) / len(loss_frac)
    analytical_cdf = credit.vasicek_lhp_cdf(grid, avg_pd, portfolio.rho)

    if make_figures:
        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.plot(grid * 100, empirical_cdf, label="Simulated portfolio (750 obligors)")
        ax.plot(grid * 100, analytical_cdf, "--", label="Vasicek LHP analytical")
        ax.set(xlabel="Loss as % of total EAD x LGD", ylabel="CDF",
               title=f"Credit loss CDF vs Vasicek LHP (avg PD={avg_pd:.2%}, rho={portfolio.rho})")
        ax.legend()
        save_figure(fig, "credit_vasicek_validation.png")

    ks_distance = float(np.max(np.abs(empirical_cdf - analytical_cdf)))
    return losses, {"avg_pd": avg_pd, "rho": portfolio.rho, "lhp_max_cdf_gap": ks_distance,
                    "total_ead": portfolio.total_exposure}


def run_market(n: int = N_MARGINAL) -> tuple[np.ndarray, dict]:
    prices = market.load_prices()
    returns = market.portfolio_returns(prices)
    result = market.fit_garch(returns)
    losses = market.simulate_annual_losses(result, n_scenarios=n)

    backtest = market.rolling_var_backtest(returns)
    fig, ax = plt.subplots(figsize=(9, 4.5))
    test_returns = -returns.loc[backtest["var_series"].index]
    ax.plot(test_returns.index, test_returns, lw=0.4, alpha=0.6, label="Daily loss (%)")
    ax.plot(backtest["var_series"], color="crimson", lw=1.0, label="99% 1-day VaR (FHS)")
    exc = backtest["exceptions_series"]
    ax.scatter(exc.index[exc], test_returns[exc], color="black", s=12, zorder=5,
               label=f"Exceptions ({backtest['n_exceptions']})")
    ax.set(title=f"Out-of-sample 99% VaR backtest -- Kupiec p={backtest['kupiec_pvalue']:.3f}",
           ylabel="Loss (%)")
    ax.legend()
    save_figure(fig, "market_var_backtest.png")

    p = result.params
    return losses, {
        "alpha_plus_beta": float(p["alpha[1]"] + p["beta[1]"]),
        "t_dof": float(p["nu"]),
        "n_obs_backtest": backtest["n_obs"],
        "n_exceptions": backtest["n_exceptions"],
        "exception_rate": backtest["exception_rate"],
        "kupiec_pvalue": backtest["kupiec_pvalue"],
        "n_returns": len(returns),
    }


def run_catastrophe(xi_override: float | None = None, freq_scale: float = 1.0,
                    n: int = N_MARGINAL, make_figures: bool = True) -> tuple[np.ndarray, dict]:
    catalog_path = ROOT / "data" / "processed" / "cat_catalog.csv"
    if not catalog_path.exists():
        subprocess.run([sys.executable, str(ROOT / "scripts" / "build_cat_catalog.py")], check=True)
    catalog = pd.read_csv(catalog_path)
    events = catalog["loss"].to_numpy()
    years = catalog["year"]
    annual_counts = catalog.groupby("year").size().reindex(
        range(years.min(), years.max() + 1), fill_value=0).to_numpy()

    frequency = catastrophe.fit_frequency(annual_counts)
    frequency.mean *= freq_scale

    threshold = float(np.quantile(events, 0.90))  # chosen from mean-excess diagnostics
    severity = catastrophe.fit_gpd(events, threshold)
    if xi_override is not None:
        severity.xi = xi_override

    if make_figures:
        thresholds, me = catastrophe.mean_excess(events)
        exceedances = np.sort(events[events > threshold] - threshold)
        theo_q = stats.genpareto.ppf((np.arange(1, len(exceedances) + 1) - 0.5) / len(exceedances),
                                     severity.xi, scale=severity.beta)
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
        axes[0].plot(thresholds, me, marker=".", lw=1)
        axes[0].axvline(threshold, color="crimson", ls="--",
                        label=f"chosen threshold (90th pct = {threshold:,.0f})")
        axes[0].set(title="Mean excess plot", xlabel="Threshold u", ylabel="Mean excess e(u)")
        axes[0].legend()
        axes[1].scatter(theo_q, exceedances, s=12)
        lim = [0, max(theo_q.max(), exceedances.max())]
        axes[1].plot(lim, lim, "k--", lw=1)
        axes[1].set(title=f"GPD QQ plot (xi={severity.xi:.2f}, "
                          f"95% CI [{severity.xi_ci[0]:.2f}, {severity.xi_ci[1]:.2f}])",
                    xlabel="Theoretical GPD quantile", ylabel="Empirical exceedance")
        save_figure(fig, "cat_evt_diagnostics.png")

    losses = CAT_SHARE * catastrophe.simulate_annual_losses(frequency, severity, n_scenarios=n)
    return losses, {
        "share": CAT_SHARE,
        "source": catalog["source"].iloc[0],
        "n_events": len(events),
        "frequency_dist": frequency.distribution,
        "frequency_mean": frequency.mean,
        "dispersion_ratio": frequency.dispersion_ratio,
        "threshold": threshold,
        "n_exceedances": int((events > threshold).sum()),
        "xi": severity.xi,
        "xi_ci": severity.xi_ci,
        "beta": severity.beta,
    }


# --------------------------------------------------------------- aggregation
def run_copula_study(marginals: dict[str, np.ndarray]) -> tuple[dict, pd.DataFrame]:
    theta = aggregation.gumbel_theta_from_corr(aggregation.BASE_CORRELATION)
    specs = {"gaussian": {}, "t": {"nu": BASE_NU}, "gumbel": {"theta": theta}}
    results, rows = {}, []
    for name, kwargs in specs.items():
        res = aggregation.aggregate(marginals, copula=name, n_scenarios=N_JOINT, **kwargs)
        db = aggregation.diversification_benefit(res, alpha=0.995)
        results[name] = res
        rows.append({"copula": name, "VaR99.5": var(res["total"], 0.995),
                     "ES97.5": expected_shortfall(res["total"], 0.975),
                     "div_benefit_99.5": db})
    table = pd.DataFrame(rows).set_index("copula")

    fig, ax = plt.subplots(figsize=(7, 4.5))
    labels = ["Gaussian", f"Student-t (nu={BASE_NU:.0f})", f"Gumbel (theta={theta:.2f})"]
    ax.bar(labels, table["div_benefit_99.5"] * 100, color=["#4878a8", "#c8843c", "#a83c3c"])
    ax.set(ylabel="Diversification benefit at 99.5% VaR (%)",
           title="Tail dependence destroys diversification benefit")
    for i, v in enumerate(table["div_benefit_99.5"] * 100):
        ax.text(i, v + 0.3, f"{v:.1f}%", ha="center")
    save_figure(fig, "copula_diversification_benefit.png")

    # joint-tail scatter: credit vs market uniforms, top 2% of total loss
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.3), sharey=True)
    for ax, (name, res) in zip(axes, results.items()):
        tail = res["total"] > np.quantile(res["total"], 0.98)
        u = res["uniforms"][tail][:4000]
        ax.scatter(u[:, 0], u[:, 1], s=3, alpha=0.3)
        ax.set(title=name, xlabel="credit U")
    axes[0].set_ylabel("market U")
    fig.suptitle("Copula uniforms conditional on total loss in worst 2%")
    save_figure(fig, "copula_tail_scatter.png")
    return results, table


# ------------------------------------------------------------------- capital
def run_capital(base_result: dict, marginals: dict[str, np.ndarray],
                cat_info: dict) -> tuple[dict, dict, dict]:
    total = base_result["total"]
    metrics = capital.capital_metrics(total)

    sizes, estimates = capital.var_convergence(total)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.semilogx(sizes, estimates, marker=".")
    ax.axhline(estimates[-1], color="grey", ls=":")
    ax.set(xlabel="Number of scenarios", ylabel="VaR 99.5%",
           title="Monte Carlo convergence of VaR 99.5%")
    save_figure(fig, "mc_convergence.png")

    allocation = capital.euler_allocation_es(base_result["components"], base_result["names"])
    fig, ax = plt.subplots(figsize=(7, 4.5))
    names = list(allocation) + ["Total ES 97.5%"]
    values = list(allocation.values())
    bottoms = np.concatenate([[0], np.cumsum(values)[:-1], [0]])
    heights = values + [sum(values)]
    colors = ["#4878a8", "#c8843c", "#a83c3c", "#555555"]
    ax.bar(names, heights, bottom=bottoms, color=colors)
    for x, b, h in zip(range(len(names)), bottoms, heights):
        ax.text(x, b + h / 2, f"{h:,.0f}", ha="center", va="center", color="white")
    ax.set(ylabel="Capital (millions)", title="Euler allocation of ES 97.5% (t-copula)")
    save_figure(fig, "euler_waterfall.png")

    # ---- stress layer: one-at-a-time re-runs under fixed seeds
    base_var = metrics["VaR 99.5% (Solvency II)"]["value"]

    def rerun(overrides: dict) -> float:
        marg = dict(marginals)
        corr = aggregation.BASE_CORRELATION
        nu = BASE_NU
        if "corr_shift" in overrides:
            corr = capital.shift_correlation(corr, overrides["corr_shift"])
        if "nu" in overrides:
            nu = overrides["nu"]
        if "xi_upper" in overrides:
            marg["cat"], _ = run_catastrophe(xi_override=cat_info["xi_ci"][1], make_figures=False)
        if "freq_scale" in overrides:
            marg["cat"], _ = run_catastrophe(freq_scale=overrides["freq_scale"], make_figures=False)
        if "rho" in overrides:
            marg["credit"], _ = run_credit(rho=overrides["rho"], make_figures=False)
        res = aggregation.aggregate(marg, copula=BASE_COPULA, n_scenarios=N_JOINT, corr=corr, nu=nu)
        return var(res["total"], 0.995)

    stresses = capital.run_stress_tests(rerun, base_var)
    ordered = dict(sorted(stresses.items(), key=lambda kv: abs(kv[1])))
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.barh(list(ordered), list(ordered.values()),
            color=["#a83c3c" if v >= 0 else "#4878a8" for v in ordered.values()])
    ax.axvline(0, color="black", lw=0.8)
    ax.set(xlabel="Change in VaR 99.5% capital (%)", title="Stress sensitivities (tornado)")
    for i, v in enumerate(ordered.values()):
        ax.text(v, i, f" {v:+.1f}% ", va="center", ha="left" if v >= 0 else "right")
    save_figure(fig, "stress_tornado.png")
    return metrics, allocation, stresses


def main() -> None:
    t0 = time.time()
    print("1/5 credit marginal ...")
    credit_losses, credit_info = run_credit()
    print("2/5 market marginal ...")
    market_losses, market_info = run_market()
    print("3/5 catastrophe marginal ...")
    cat_losses, cat_info = run_catastrophe()
    marginals = {"credit": credit_losses, "market": market_losses, "cat": cat_losses}

    print("4/5 copula aggregation ...")
    copula_results, copula_table = run_copula_study(marginals)

    print("5/5 capital, allocation, stress ...")
    metrics, allocation, stresses = run_capital(copula_results[BASE_COPULA], marginals, cat_info)

    standalone = {k: var(v, 0.995) for k, v in marginals.items()}
    lines = [
        "# Results\n",
        f"_Generated by `run_all.py` in {time.time() - t0:.0f}s. "
        f"{N_MARGINAL:,} scenarios per marginal, {N_JOINT:,} joint scenarios._\n",
        f"_Cat data source: **{cat_info['source']}**_\n",
        "## Marginals\n",
        f"- Credit: avg PD {credit_info['avg_pd']:.2%}, rho {credit_info['rho']}, "
        f"max CDF gap to Vasicek LHP {credit_info['lhp_max_cdf_gap']:.3f}",
        f"- Market: GARCH alpha+beta {market_info['alpha_plus_beta']:.3f}, "
        f"t dof {market_info['t_dof']:.1f}; backtest {market_info['n_exceptions']} exceptions "
        f"in {market_info['n_obs_backtest']} days "
        f"({market_info['exception_rate']:.2%} vs 1% expected), "
        f"Kupiec p-value {market_info['kupiec_pvalue']:.3f}",
        f"- Catastrophe: {cat_info['frequency_dist']} frequency "
        f"(mean {cat_info['frequency_mean']:.1f}/yr, dispersion {cat_info['dispersion_ratio']:.2f}), "
        f"GPD xi {cat_info['xi']:.2f} (95% CI [{cat_info['xi_ci'][0]:.2f}, {cat_info['xi_ci'][1]:.2f}]) "
        f"above threshold {cat_info['threshold']:,.0f} ({cat_info['n_exceedances']} exceedances)\n",
        "## Standalone VaR 99.5% (millions)\n",
        *[f"- {k}: {v:,.0f}" for k, v in standalone.items()],
        "\n## Copula comparison\n",
        copula_table.to_markdown(floatfmt=",.3f"),
        f"\n## Capital metrics ({BASE_COPULA}-copula base model, millions)\n",
        *[f"- {k}: {v['value']:,.0f} (MC s.e. {v['mc_se']:,.0f})" for k, v in metrics.items()],
        "\n## Euler allocation of ES 97.5%\n",
        *[f"- {k}: {v:,.0f}" for k, v in allocation.items()],
        f"- sum: {sum(allocation.values()):,.0f} "
        f"(= ES 97.5% {metrics['ES 97.5% (FRTB)']['value']:,.0f})",
        "\n## Stress sensitivities (% change in VaR 99.5%)\n",
        *[f"- {k}: {v:+.1f}%" for k, v in stresses.items()],
        "",
    ]
    out = ROOT / "docs" / "results.md"
    out.write_text("\n".join(lines))
    print(f"\nWrote {out}")
    print(copula_table.to_string(float_format=lambda x: f"{x:,.3f}"))
    for k, v in metrics.items():
        print(f"{k}: {v['value']:,.0f} (s.e. {v['mc_se']:,.0f})")


if __name__ == "__main__":
    main()
