"""
GRS Test (Gibbons, Ross, and Shanken, 1989)
F-stat: [(T-N-K)/N] * [alpha' Sigma^{-1} alpha] / [1 + mu_f' Omega^{-1} mu_f]
H0: all alphas = 0  |  Statistic ~ F(N, T-N-K)
"""

import numpy as np
import pandas as pd
from scipy import stats
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent

FACTORS = ["Mkt_RF", "SMB", "HML", "RMW", "CMA", "WML"]

FACTORS_FILE   = ROOT / "data" / "fama_french_factors" / "ff6_factors_improved.csv"
RETURNS_PATH   = ROOT / "data" / "results" / "portfolio_returns"
REGRESSION_CSV = ROOT / "data" / "results" / "ff_regression" / "fama_french_regression_results.csv"
RF_FILE        = ROOT / "data" / "riksbanken_tb_1m.csv"


# loaders

def load_factors() -> pd.DataFrame:
    df = pd.read_csv(FACTORS_FILE, parse_dates=["Date"], index_col="Date")
    return df[FACTORS].dropna()


def load_rf() -> pd.Series:
    rf = pd.read_csv(RF_FILE, sep=';')
    rf['Date'] = pd.to_datetime(rf['Period'] + ' 01', format='%Y %B %d')
    rf = rf.rename(columns={'Average': 'RF'})
    rf['RF'] = pd.to_numeric(rf['RF'], errors='coerce')
    rf['RF'] = ((1 + rf['RF'] / 100) ** (1/12) - 1) * 100  # convert annualised to monthly percent
    rf = rf.set_index('Date')['RF']
    return rf


def load_portfolio_returns(metric: str) -> pd.DataFrame:
    path = RETURNS_PATH / f"{metric}_portfolio_returns.csv"
    df = pd.read_csv(path, parse_dates=["Date"], index_col="Date")
    return df.drop(columns=["FormationYear"], errors="ignore")


def load_regression_results(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


# grs core

def align(returns: pd.DataFrame, factors: pd.DataFrame, rf: pd.Series) -> tuple[pd.DataFrame, pd.DataFrame]:
    idx = returns.index.intersection(factors.index).intersection(rf.index)
    return returns.loc[idx], factors.loc[idx], rf.loc[idx] # type: ignore


def compute_residuals(returns, factors, rf):
    T, N = returns.shape
    excess_returns = returns.subtract(rf, axis=0)
    X = np.column_stack([np.ones(T), factors.values])
    Y = excess_returns.values
    coef, _, _, _ = np.linalg.lstsq(X, Y, rcond=None)
    alphas = coef[0]
    resids = Y - X @ coef
    return alphas, resids, excess_returns.values 



def grs_stat(alphas: np.ndarray, resids: np.ndarray, factors: pd.DataFrame, excess_returns: np.ndarray) -> dict:
    T, N = resids.shape
    K = factors.shape[1]

    sigma = (resids.T @ resids) / T
    mu_f  = factors.mean().values
    omega = factors.cov().values

    alpha_term  = alphas @ np.linalg.inv(sigma) @ alphas
    factor_term = mu_f   @ np.linalg.inv(omega) @ mu_f

    df1 = N
    df2 = T - N - K
    grs = (df2 / df1) * alpha_term / (1.0 + factor_term)
    p   = 1.0 - stats.f.cdf(grs, df1, df2)

    # Complementary metrics
    avg_abs_alpha = np.mean(np.abs(alphas))
    avg_abs_mu    = np.mean(np.abs(excess_returns.mean(axis=0)))
    ratio         = avg_abs_alpha / avg_abs_mu if avg_abs_mu != 0 else np.nan

    return {
        "grs_stat":       grs,
        "p_value":        p,
        "df1":            df1,
        "df2":            df2,
        "avg_abs_alpha":  avg_abs_alpha,
        "avg_abs_mu":     avg_abs_mu,
        "ratio":          ratio,
    }


# main execution

def run_grs(regression_csv):
    factors  = load_factors()
    rf       = load_rf()
    reg      = load_regression_results(regression_csv)
    results  = []

    for metric, group in reg.groupby("metric"):
        portfolios = load_portfolio_returns(metric)
        ret, fac, rf_aligned = align(portfolios, factors, rf)

        alphas, resids, excess_returns = compute_residuals(ret, fac, rf_aligned)
        result = grs_stat(alphas, resids, fac, excess_returns)  # pass excess_returns
        results.append({"metric": metric, **result})

    return pd.DataFrame(results).sort_values("p_value").reset_index(drop=True)


if __name__ == "__main__":
    GRS_OUTPUT_DIR = ROOT / "data" / "results" / "GRS"
    GRS_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    results_df = run_grs(REGRESSION_CSV)
    results_df.to_csv(GRS_OUTPUT_DIR / "grs_results.csv", index=False)
    print(results_df.to_string(index=False))
    print(f"\nSaved to {GRS_OUTPUT_DIR / 'grs_results.csv'}")