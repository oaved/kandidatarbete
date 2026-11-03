import pandas as pd
import numpy as np
import statsmodels.api as sm
from pathlib import Path
from typing import Dict, List, Optional

ROOT = Path(__file__).resolve().parent.parent
TERTILES_DIR   = ROOT / "data" / "results" / "rss_tertiles"
RETURNS_FILE   = ROOT / "data" / "LSEG_data" / "raw_metrics" / "TotalReturn.csv"
MARKETCAP_FILE = ROOT / "data" / "LSEG_data" / "raw_metrics" / "MarketCap.csv"
FF_FACTORS     = ROOT / "data" / "fama_french_factors" / "ff6_factors_improved.csv"
RF_FILE        = ROOT / "data" / "riksbanken_tb_1m.csv"
OUTPUT_DIR     = ROOT / "data" / "results" / "ff_dropped_companies"
CUTOFF_DATE    = pd.Timestamp(year=2025, month=6, day=30)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# loaders

def load_returns() -> tuple[pd.DataFrame, dict]:
    df = pd.read_csv(RETURNS_FILE)
    date_columns = [col for col in df.columns if '-' in col and ':' in col]
    date_col_map = {}
    for col in date_columns:
        dt = pd.to_datetime(col, errors='coerce')
        if pd.notna(dt):
            dt_normalized = pd.to_datetime(col, errors='coerce').to_period('M').to_timestamp() # type: ignore
            date_col_map[dt_normalized] = col
    return df, date_col_map


def load_marketcap() -> pd.DataFrame:
    mc = pd.read_csv(MARKETCAP_FILE)
    date_columns = [col for col in mc.columns if '-' in col and ':' in col]
    mc_long = mc[['ISIN Code'] + date_columns].melt(id_vars=['ISIN Code'], var_name='raw_date', value_name='MarketCap')
    mc_long['Date'] = pd.to_datetime(mc_long['raw_date'], errors='coerce').dt.to_period('M').dt.to_timestamp()
    mc_long['MarketCap'] = pd.to_numeric(mc_long['MarketCap'], errors='coerce')
    mc_long = mc_long[['ISIN Code', 'Date', 'MarketCap']].sort_values(['ISIN Code', 'Date'])
    mc_long['MarketCap_lag'] = mc_long.groupby('ISIN Code')['MarketCap'].shift(1)
    return mc_long[['ISIN Code', 'Date', 'MarketCap_lag']]


def load_ff_factors() -> pd.DataFrame:
    ff = pd.read_csv(FF_FACTORS)
    ff['Date'] = pd.to_datetime(ff['Date'])
    rf = pd.read_csv(RF_FILE, sep=';')
    rf['Date'] = pd.to_datetime(rf['Period'] + ' 01', format='%Y %B %d')
    rf = rf.rename(columns={'Average': 'RF'})
    rf['RF'] = pd.to_numeric(rf['RF'], errors='coerce')
    rf['RF'] = ((1 + rf['RF'] / 100) ** (1/12) - 1) * 100  # convert annualised to monthly percent
    return pd.merge(ff, rf[['Date', 'RF']], on='Date', how='left')


# portfolio construction per metric

def build_metric_dropped_portfolio(
    tertiles_df: pd.DataFrame,
    returns_df: pd.DataFrame,
    all_investable: set,
    date_col_map: dict,
    mc_long: pd.DataFrame,
    metric_name: str
) -> pd.DataFrame:
    """Build a dropped portfolio for a single metric.
    
    Dropped = all investable companies (present in TotalReturn.csv)
              minus those assigned to portfolios 1, 2, or 3 for that year.
    This captures both companies with no ESG score and those excluded
    due to insufficient data, as long as they are listed and investable.
    """
    year_cols = [c for c in tertiles_df.columns if c.isdigit()]
    years = sorted([int(c) for c in year_cols])

    results = {"Date": [], "FormationYear": [], "Dropped": [], "n_companies": []}

    for year in years:
        year_col = str(year)

        if year_col not in tertiles_df.columns:
            continue

        # All investable ISINs minus those assigned to a portfolio
        assigned = set(tertiles_df[tertiles_df[year_col].isin([1, 2, 3])]['ISIN Code'].dropna())
        dropped = list(all_investable - assigned)

        print(f"  {year}: {len(all_investable)} investable, {len(assigned)} assigned, {len(dropped)} dropped")

        start_date = pd.Timestamp(year=year, month=7, day=1)
        end_date   = pd.Timestamp(year=year + 1, month=6, day=30)
        selected_months = sorted(
            [(dt, col) for dt, col in date_col_map.items()
             if start_date <= dt <= min(end_date, CUTOFF_DATE)],
            key=lambda x: x[0]
        )

        if not selected_months:
            print(f"  {metric_name} {year}: No return data available")
            continue

        for month_dt, month_col in selected_months:
            results["Date"].append(month_dt.strftime("%Y-%m"))
            results["FormationYear"].append(str(year))

            if not dropped:
                results["Dropped"].append(None)
                results["n_companies"].append(0)
                continue

            company_returns = returns_df[returns_df["ISIN Code"].isin(dropped)][["ISIN Code", month_col]].copy()
            company_returns = company_returns.rename(columns={month_col: 'Return'})
            company_returns['Return'] = pd.to_numeric(company_returns['Return'], errors='coerce')

            mc_month = mc_long[mc_long['Date'] == month_dt][['ISIN Code', 'MarketCap_lag']]
            merged = company_returns.merge(mc_month, on='ISIN Code', how='left')
            merged = merged.dropna(subset=['Return', 'MarketCap_lag'])

            n = len(merged)
            results["n_companies"].append(n)

            if n < 10:
                print(f"  WARNING: {metric_name} {year} {month_dt.strftime('%Y-%m')}: only {n} companies")

            if merged.empty or merged['MarketCap_lag'].sum() <= 0:
                results["Dropped"].append(None)
                continue

            weights = merged['MarketCap_lag'] / merged['MarketCap_lag'].sum()
            results["Dropped"].append(float((weights * merged['Return']).sum()))

    return pd.DataFrame(results)


# regression

def run_ff_regression(portfolio_returns: pd.Series, ff_df: pd.DataFrame) -> dict:
    merged = pd.merge(
        portfolio_returns.reset_index(),
        ff_df,
        on='Date',
        how='inner'
    ).dropna()

    if len(merged) < 10:
        return {}

    merged['excess_return'] = merged['Dropped'] - merged['RF']
    X = sm.add_constant(merged[['Mkt_RF', 'SMB', 'HML', 'RMW', 'CMA', 'WML']])
    y = merged['excess_return']

    res = sm.OLS(y, X).fit()

    return {
        'n_obs':         len(merged),
        'alpha':         res.params['const'],
        'alpha_t':       res.tvalues['const'],
        'alpha_p':       res.pvalues['const'],
        'mkt_rf_beta':   res.params['Mkt_RF'],
        'smb_beta':      res.params['SMB'],
        'hml_beta':      res.params['HML'],
        'rmw_beta':      res.params['RMW'],
        'cma_beta':      res.params['CMA'],
        'wml_beta':      res.params['WML'],
        'r_squared':     res.rsquared,
        'adj_r_squared': res.rsquared_adj,
        'f_stat':        res.fvalue,
        'f_p':           res.f_pvalue,
        'mse':           res.mse_resid,
        'rmse':          np.sqrt(res.mse_resid),
    }


# main execution

if __name__ == "__main__":
    returns_df, date_col_map = load_returns()
    mc_long                  = load_marketcap()
    ff_df                    = load_ff_factors()
    ff_df['Date']            = pd.to_datetime(ff_df['Date']).dt.to_period('M').dt.to_timestamp()

    # All investable ISINs — companies present in TotalReturn.csv
    all_investable = set(returns_df['ISIN Code'].dropna().unique())
    print(f"Total investable companies in TotalReturn.csv: {len(all_investable)}")

    all_regression_results = []

    for tertiles_file in sorted(TERTILES_DIR.glob("*_tertiles.csv")):
        metric_name = tertiles_file.stem.replace("_tertiles", "")
        print(f"\nProcessing: {metric_name}")

        tertiles_df = pd.read_csv(tertiles_file)

        # Build dropped portfolio for this metric
        portfolio_df = build_metric_dropped_portfolio(
            tertiles_df, returns_df, all_investable, date_col_map, mc_long, metric_name
        )
        portfolio_df['Date'] = pd.to_datetime(portfolio_df['Date'], format='%Y-%m')

        # Save portfolio returns CSV
        portfolio_df.to_csv(OUTPUT_DIR / f"{metric_name}_dropped_portfolio.csv", index=False)

        # Run FF regression
        reg_input = portfolio_df.set_index('Date')['Dropped'].dropna()
        result = run_ff_regression(reg_input, ff_df)

        if result:
            result['metric'] = metric_name
            all_regression_results.append(result)
            print(f"  α={result['alpha']:.4f} (t={result['alpha_t']:.2f}), R²={result['r_squared']:.3f}, n={result['n_obs']}")
        else:
            print(f"  Insufficient data for regression")

    # Save combined regression results
    results_df = pd.DataFrame(all_regression_results)
    results_df.to_csv(OUTPUT_DIR / "ff_dropped_companies_by_metric.csv", index=False)
    print(f"\nSaved regression results to {OUTPUT_DIR / 'ff_dropped_companies_by_metric.csv'}")