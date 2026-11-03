import pandas as pd
import numpy as np
import statsmodels.api as sm
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import os

# Define paths
ROOT = Path(__file__).resolve().parent.parent
FF_FACTORS_FILE = ROOT / "data" / "fama_french_factors" / "ff6_factors_improved.csv"
RF_FILE = ROOT / "data" / "riksbanken_tb_1m.csv"
PORTFOLIO_RETURNS_DIR = ROOT / "data" / "results" / "portfolio_returns"
OUTPUT_DIR = ROOT / "data" / "results" / "ff_regression"

# Create output directory if it doesn't exist
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_fama_french_data() -> pd.DataFrame:
    """Load and prepare Fama-French 6-factor data with risk-free rate."""
    print("Loading Fama-French 6-factor data...")
    ff_df = pd.read_csv(FF_FACTORS_FILE)
    ff_df['Date'] = pd.to_datetime(ff_df['Date'])
    print(f"FF factors loaded: {len(ff_df)} months from {ff_df['Date'].min()} to {ff_df['Date'].max()}")

    print("Loading risk-free rate data...")
    rf_df = pd.read_csv(RF_FILE, sep=';')
    rf_df['Date'] = pd.to_datetime(rf_df['Period'] + ' 01', format='%Y %B %d')
    rf_df = rf_df.rename(columns={'Average': 'RF'})
    rf_df['RF'] = pd.to_numeric(rf_df['RF'], errors='coerce')
    rf_df['RF'] = ((1 + rf_df['RF'] / 100) ** (1/12) - 1) * 100  # convert annualised to monthly percent
    rf_df = rf_df[['Date', 'RF']]
    print(f"RF data loaded: {len(rf_df)} months from {rf_df['Date'].min()} to {rf_df['Date'].max()}")

    # Merge FF factors with RF
    ff_df = pd.merge(ff_df, rf_df, on='Date', how='left')
    print(f"Combined data: {len(ff_df)} months\n")

    return ff_df


def run_ff_regression(portfolio_returns: pd.Series, ff_factors: pd.DataFrame, portfolio_name: str) -> Dict:
    """
    Run Fama-French 6-factor regression: Portfolio_Return - RF = alpha + betas*Factors + error
    """
    # Merge portfolio returns with FF factors on date
    merged_df = pd.merge(
        portfolio_returns.reset_index(),
        ff_factors,
        on='Date',
        how='inner'
    )

    if len(merged_df) < 10:  # Need minimum observations
        return {
            'portfolio': portfolio_name,
            'n_obs': len(merged_df),
            'error': 'Insufficient data for regression'
        }

    # Calculate excess return 
    # Long-short is already a zero-investment portfolio; RF cancels out (P3-RF)-(P1-RF) = P3-P1
    if portfolio_name == 'P3_minus_P1':
        merged_df['excess_return'] = merged_df[portfolio_name]
    else:
        merged_df['excess_return'] = merged_df[portfolio_name] - merged_df['RF']

    # Prepare independent variables (FF factors in percent)
    X = merged_df[['Mkt_RF', 'SMB', 'HML', 'RMW', 'CMA', 'WML']]
    X = sm.add_constant(X)  # Add intercept

    # Dependent variable
    y = merged_df['excess_return']

    # Run OLS regression
    model = sm.OLS(y, X, missing='drop')
    results = model.fit()

    # Extract statistics
    stats = {
        'portfolio': portfolio_name,
        'n_obs': len(merged_df),
        'alpha': results.params['const'],
        'alpha_t': results.tvalues['const'],
        'alpha_p': results.pvalues['const'],
        'mkt_rf_beta': results.params['Mkt_RF'],
        'mkt_rf_t': results.tvalues['Mkt_RF'],
        'mkt_rf_p': results.pvalues['Mkt_RF'],
        'smb_beta': results.params['SMB'],
        'smb_t': results.tvalues['SMB'],
        'smb_p': results.pvalues['SMB'],
        'hml_beta': results.params['HML'],
        'hml_t': results.tvalues['HML'],
        'hml_p': results.pvalues['HML'],
        'rmw_beta': results.params['RMW'],
        'rmw_t': results.tvalues['RMW'],
        'rmw_p': results.pvalues['RMW'],
        'cma_beta': results.params['CMA'],
        'cma_t': results.tvalues['CMA'],
        'cma_p': results.pvalues['CMA'],
        'wml_beta': results.params['WML'],
        'wml_t': results.tvalues['WML'],
        'wml_p': results.pvalues['WML'],
        'r_squared': results.rsquared,
        'adj_r_squared': results.rsquared_adj,
        'f_stat': results.fvalue,
        'f_p': results.f_pvalue,
        'mse': results.mse_resid,
        'rmse': np.sqrt(results.mse_resid)
    }

    return stats


def process_individual_portfolios(portfolio_files: List[Path], ff_df: pd.DataFrame) -> List[Dict]:
    """Process individual portfolio returns (P1, P2, P3) and run FF regressions."""
    all_results = []

    print("Processing individual portfolios...")
    for portfolio_file in portfolio_files:
        metric_name = portfolio_file.stem.replace("_portfolio_returns", "")
        print(f"\n{metric_name}")

        # Load portfolio returns
        portfolio_df = pd.read_csv(portfolio_file)
        portfolio_df['Date'] = pd.to_datetime(portfolio_df['Date'], format='%Y-%m')

        # Run regression for each portfolio
        for portfolio_col in ['Portfolio_1', 'Portfolio_2', 'Portfolio_3']:
            if portfolio_col not in portfolio_df.columns:
                continue

            # Get portfolio returns series
            portfolio_returns = portfolio_df.set_index('Date')[portfolio_col].dropna()

            if len(portfolio_returns) == 0:
                print(f"  {portfolio_col}: No valid returns")
                continue

            # Run regression
            regression_results = run_ff_regression(portfolio_returns, ff_df, portfolio_col)

            # Add metric name
            regression_results['metric'] = metric_name

            all_results.append(regression_results)

            if 'error' in regression_results:
                print(f"  {portfolio_col}: {regression_results['error']}")
            else:
                alpha = regression_results['alpha']
                alpha_t = regression_results['alpha_t']
                r2 = regression_results['r_squared']
                n_obs = regression_results['n_obs']
                print(f"  {portfolio_col}: α={alpha:.4f} (t={alpha_t:.2f}), R²={r2:.3f}, n={n_obs}")

    return all_results


def process_long_short_portfolios(portfolio_files: List[Path], ff_df: pd.DataFrame) -> List[Dict]:
    """Process long-short portfolio (P3 - P1) returns and run FF regressions."""
    all_results = []

    print("\n\nProcessing long-short portfolios (P3 - P1)...")
    for portfolio_file in portfolio_files:
        metric_name = portfolio_file.stem.replace("_portfolio_returns", "")
        print(f"\n{metric_name}")

        # Load portfolio returns
        portfolio_df = pd.read_csv(portfolio_file)
        portfolio_df['Date'] = pd.to_datetime(portfolio_df['Date'], format='%Y-%m')

        # Calculate long-short returns: P3 - P1
        if 'Portfolio_1' in portfolio_df.columns and 'Portfolio_3' in portfolio_df.columns:
            portfolio_df['P3_minus_P1'] = portfolio_df['Portfolio_3'] - portfolio_df['Portfolio_1']

            # Get long-short returns series
            long_short_returns = portfolio_df.set_index('Date')['P3_minus_P1'].dropna()

            if len(long_short_returns) == 0:
                print(f"  P3-P1: No valid returns")
                continue

            # Run regression
            regression_results = run_ff_regression(long_short_returns, ff_df, 'P3_minus_P1')

            # Add metric name and portfolio identifier
            regression_results['metric'] = metric_name
            regression_results['portfolio'] = 'P3_minus_P1'

            all_results.append(regression_results)

            if 'error' in regression_results:
                print(f"  P3-P1: {regression_results['error']}")
            else:
                alpha = regression_results['alpha']
                alpha_t = regression_results['alpha_t']
                r2 = regression_results['r_squared']
                n_obs = regression_results['n_obs']
                print(f"  P3-P1: α={alpha:.4f} (t={alpha_t:.2f}), R²={r2:.3f}, n={n_obs}")

    return all_results


def save_and_summarize_results(all_results: List[Dict], output_suffix: str = "") -> None:
    """Save regression results to CSV and print summary statistics."""
    if not all_results:
        print("\nNo valid regression results to save.")
        return

    results_df = pd.DataFrame(all_results)

    # Reorder columns for better readability
    column_order = [
        'metric', 'portfolio', 'n_obs',
        'alpha', 'alpha_t', 'alpha_p',
        'mkt_rf_beta', 'mkt_rf_t', 'mkt_rf_p',
        'smb_beta', 'smb_t', 'smb_p',
        'hml_beta', 'hml_t', 'hml_p',
        'rmw_beta', 'rmw_t', 'rmw_p',
        'cma_beta', 'cma_t', 'cma_p',
        'wml_beta', 'wml_t', 'wml_p',
        'r_squared', 'adj_r_squared', 'f_stat', 'f_p', 'mse', 'rmse'
    ]

    # Only include columns that exist
    available_columns = [col for col in column_order if col in results_df.columns]
    results_df = results_df[available_columns]

    # Save to CSV
    output_file = OUTPUT_DIR / f"fama_french_regression_results{output_suffix}.csv"
    results_df.to_csv(output_file, index=False)
    print(f"\nResults saved to: {output_file}")
    print(f"Total regressions: {len(results_df)}")

    # Print summary statistics
    valid_results = results_df.dropna(subset=['alpha'])
    if not valid_results.empty:
        print("\nSummary:")
        print(f"  Average R²: {valid_results['r_squared'].mean():.3f}")
        print(f"  Average alpha: {valid_results['alpha'].mean():.4f}")
        print(f"  Significant alphas (p<0.05): {len(valid_results[valid_results['alpha_p'] < 0.05])}")
        print(f"  Significant market betas (p<0.05): {len(valid_results[valid_results['mkt_rf_p'] < 0.05])}")

def main():
    """Main execution function."""
    # Load Fama-French data
    ff_df = load_fama_french_data()

    # Get list of portfolio return files
    portfolio_files = sorted(PORTFOLIO_RETURNS_DIR.glob("*_portfolio_returns.csv"))
    print(f"Processing {len(portfolio_files)} metrics...\n")

    # # Check average returns across all portfolio files
    # all_returns = []
    # for f in portfolio_files:
    #     df = pd.read_csv(f)
    #     for col in ['Portfolio_1', 'Portfolio_2', 'Portfolio_3']:
    #         if col in df.columns:
    #             all_returns.extend(df[col].dropna().tolist())

    # all_returns = pd.Series(all_returns)
    # print("\nOverall return distribution across all portfolios:")
    # print(all_returns.describe())
    # print(f"Annualized average (x12): {all_returns.mean() * 12:.2f}%")

    # Process individual portfolios
    individual_results = process_individual_portfolios(portfolio_files, ff_df)
    save_and_summarize_results(individual_results, output_suffix="")

    # Process long-short portfolios
    long_short_results = process_long_short_portfolios(portfolio_files, ff_df)
    save_and_summarize_results(long_short_results, output_suffix="_longshort")

    print("\nDone!")


if __name__ == "__main__":
    main()

