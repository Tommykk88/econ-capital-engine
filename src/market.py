"""Market risk: GARCH(1,1) filtered historical simulation (FHS).

A fixed-weight portfolio of liquid ETFs (equity / bonds / gold). A GARCH(1,1)
with Student-t innovations is fitted to daily portfolio log returns; the
standardized residuals are bootstrapped and propagated through the GARCH
recursion to build full one-year (250-day) P&L paths, so no square-root-of-time
scaling is needed.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from arch import arch_model
from scipy import stats

RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"

TICKERS = ["SPY", "TLT", "GLD"]
WEIGHTS = np.array([0.5, 0.4, 0.1])
PORTFOLIO_VALUE = 2_000.0  # millions, same units as the credit book
TRADING_DAYS = 250


def load_prices(start: str = "2006-01-01", cache: bool = True) -> pd.DataFrame:
    """Download (or load cached) adjusted close prices for the ETF portfolio."""
    cache_path = RAW_DIR / "market_prices.csv"
    if cache and cache_path.exists():
        return pd.read_csv(cache_path, index_col=0, parse_dates=True)

    import yfinance as yf

    prices = yf.download(TICKERS, start=start, auto_adjust=True, progress=False)["Close"]
    prices = prices[TICKERS].dropna()
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    prices.to_csv(cache_path)
    return prices


def portfolio_returns(prices: pd.DataFrame) -> pd.Series:
    """Daily log returns (in percent) of the fixed-weight portfolio."""
    asset_returns = prices.pct_change().dropna()
    port = asset_returns @ WEIGHTS
    return np.log1p(port) * 100.0


def fit_garch(returns: pd.Series):
    """Fit GARCH(1,1) with Student-t innovations to percent returns."""
    model = arch_model(returns, vol="GARCH", p=1, q=1, dist="t", mean="Constant")
    return model.fit(disp="off")


def standardized_residuals(result) -> np.ndarray:
    """Devolatilized residuals (approximately unit variance)."""
    return (result.resid / result.conditional_volatility).to_numpy()


def simulate_annual_losses(
    result,
    n_scenarios: int = 200_000,
    horizon: int = TRADING_DAYS,
    seed: int = 2,
    portfolio_value: float = PORTFOLIO_VALUE,
) -> np.ndarray:
    """One-year losses via multi-step filtered historical simulation.

    Bootstraps standardized residuals and propagates them through the fitted
    GARCH recursion starting from the current conditional variance, capturing
    volatility clustering over the full horizon.
    """
    params = result.params
    mu, omega = params["mu"], params["omega"]
    a, b = params["alpha[1]"], params["beta[1]"]
    z_pool = standardized_residuals(result)

    rng = np.random.default_rng(seed)
    sigma2 = np.full(n_scenarios, float(result.conditional_volatility.iloc[-1]) ** 2)
    cum_log_ret = np.zeros(n_scenarios)
    for _ in range(horizon):
        z = rng.choice(z_pool, size=n_scenarios, replace=True)
        eps = np.sqrt(sigma2) * z
        cum_log_ret += mu + eps
        sigma2 = omega + a * eps**2 + b * sigma2

    pnl = portfolio_value * (np.exp(cum_log_ret / 100.0) - 1.0)
    return -pnl  # losses positive


def rolling_var_backtest(
    returns: pd.Series,
    alpha: float = 0.99,
    min_obs: int = 1_500,
    refit_every: int = 21,
) -> dict:
    """Out-of-sample rolling 1-day VaR backtest with Kupiec POF test.

    GARCH is refitted every `refit_every` days on an expanding window; between
    refits the conditional variance is updated daily with the latest realized
    return, and VaR is the FHS quantile of the standardized residuals scaled
    by the next day's conditional volatility.
    """
    r = returns.to_numpy()
    n = len(r)
    var_forecasts = np.full(n, np.nan)

    t = min_obs
    while t < n:
        result = fit_garch(returns.iloc[:t])
        p = result.params
        mu, omega, a, b = p["mu"], p["omega"], p["alpha[1]"], p["beta[1]"]
        z_q = np.quantile(standardized_residuals(result), 1 - alpha)

        # roll the variance recursion forward through the hold-out block
        sigma2 = float(result.conditional_volatility.iloc[-1]) ** 2
        eps_prev = float(result.resid.iloc[-1])
        for s in range(t, min(t + refit_every, n)):
            sigma2 = omega + a * eps_prev**2 + b * sigma2
            var_forecasts[s] = -(mu + np.sqrt(sigma2) * z_q)  # positive loss number
            eps_prev = r[s] - mu
        t += refit_every

    mask = ~np.isnan(var_forecasts)
    exceptions = (-r[mask]) > var_forecasts[mask]
    n_obs, n_exc = int(mask.sum()), int(exceptions.sum())
    return {
        "n_obs": n_obs,
        "n_exceptions": n_exc,
        "exception_rate": n_exc / n_obs,
        **kupiec_pof(n_exc, n_obs, alpha),
        "var_series": pd.Series(var_forecasts[mask], index=returns.index[mask]),
        "exceptions_series": pd.Series(exceptions, index=returns.index[mask]),
    }


def kupiec_pof(n_exceptions: int, n_obs: int, alpha: float) -> dict:
    """Kupiec proportion-of-failures likelihood ratio test."""
    p = 1 - alpha
    x = n_exceptions
    if x in (0, n_obs):
        lr = -2 * n_obs * (np.log(1 - p) if x == 0 else np.log(p))
    else:
        phat = x / n_obs
        lr = -2 * (
            (n_obs - x) * np.log((1 - p) / (1 - phat)) + x * np.log(p / phat)
        )
    return {"kupiec_lr": float(lr), "kupiec_pvalue": float(stats.chi2.sf(lr, df=1))}
