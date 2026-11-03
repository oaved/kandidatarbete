import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Union, Optional
import os

# Define paths
TERTILES_DIR = Path("data/results/rss_tertiles")
RETURNS_FILE = Path("data/LSEG_data/raw_metrics/TotalReturn.csv")
MARKETCAP_FILE = Path("data/LSEG_data/raw_metrics/MarketCap.csv")
OUTPUT_DIR = Path("data/results/portfolio_returns")
CUTOFF_DATE = pd.Timestamp(year=2025, month=6, day=30)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Load returns data
print("Loading TotalReturn.csv...")
returns_df = pd.read_csv(RETURNS_FILE)

# Load market cap data
print("Loading MarketCap.csv...")
mc_df = pd.read_csv(MARKETCAP_FILE)

def get_date_from_column(col: str) -> pd.Timestamp:
    return pd.to_datetime(col, errors='coerce') # type: ignore

def normalize_date_cols(df):
    """Normalize date columns to start-of-month timestamps."""
    date_columns = [col for col in df.columns if '-' in col and ':' in col]
    date_col_map: Dict[pd.Timestamp, str] = {}
    for col in date_columns:
        dt = get_date_from_column(col)
        if pd.notna(dt):
            # Normalize to start of month
            dt_normalized = dt.to_period('M').to_timestamp()
            date_col_map[dt_normalized] = col
    return date_col_map

date_col_map = normalize_date_cols(returns_df)
mc_date_col_map = normalize_date_cols(mc_df)

sorted_dates = sorted(date_col_map.keys())
print(f"Found {len(sorted_dates)} monthly return columns")
print(f"Date range: {sorted_dates[0].date()} to {sorted_dates[-1].date()}")

# Melt market cap to long format for easy lagged lookup
print("Preparing market cap data...")
mc_long = mc_df[['ISIN Code'] + list(mc_date_col_map.values())].copy()
mc_long = mc_long.melt(id_vars=['ISIN Code'], var_name='raw_date', value_name='MarketCap')
mc_long['Date'] = mc_long['raw_date'].apply(lambda x: get_date_from_column(x).to_period('M').to_timestamp())
mc_long['MarketCap'] = pd.to_numeric(mc_long['MarketCap'], errors='coerce')
mc_long = mc_long[['ISIN Code', 'Date', 'MarketCap']].sort_values(['ISIN Code', 'Date'])

# Compute lagged market cap (t-1)
mc_long['MarketCap_lag'] = mc_long.groupby('ISIN Code')['MarketCap'].shift(1)
mc_long = mc_long[['ISIN Code', 'Date', 'MarketCap_lag']]

# Process each tertiles file
tertiles_files = sorted(TERTILES_DIR.glob("*_tertiles.csv"))
print(f"\nProcessing {len(tertiles_files)} metrics...")

for tertiles_file in tertiles_files:
    metric_name = tertiles_file.stem.replace("_tertiles", "")
    print(f"\nProcessing: {metric_name}")

    tertiles_df = pd.read_csv(tertiles_file)
    year_columns = [col for col in tertiles_df.columns if col.isdigit()]
    years = sorted([int(col) for col in year_columns])

    results: Dict[str, Union[List[str], List[Optional[float]]]] = {
        "Date": [],
        "FormationYear": [],
        "Portfolio_1": [],
        "Portfolio_2": [],
        "Portfolio_3": []
    }
    company_counts = {"Portfolio_1": [], "Portfolio_2": [], "Portfolio_3": []}

    for year in years:
        year_col = str(year)

        portfolio_companies = {}
        for portfolio in [1, 2, 3]:
            companies_in_portfolio = tertiles_df[tertiles_df[year_col] == portfolio]["ISIN Code"].dropna().tolist()
            portfolio_companies[portfolio] = companies_in_portfolio
            print(f"  {year} Portfolio {portfolio}: {len(companies_in_portfolio)} companies")

        start_date = pd.Timestamp(year=year, month=7, day=1)
        end_date = pd.Timestamp(year=year + 1, month=6, day=30)
        selected_months = [
            (dt, col) for dt, col in date_col_map.items()
            if start_date <= dt <= min(end_date, CUTOFF_DATE)
        ]
        selected_months = sorted(selected_months, key=lambda x: x[0])

        if not selected_months:
            print(f"  {year}: No return data available for July {year} to June {year + 1}")
            continue

        for month_dt, month_col in selected_months:
            results["Date"].append(month_dt.strftime("%Y-%m")) # type: ignore
            results["FormationYear"].append(str(year)) # type: ignore

            for portfolio in [1, 2, 3]:
                companies = portfolio_companies[portfolio]

                if not companies:
                    results[f"Portfolio_{portfolio}"].append(None) # type: ignore
                    continue

                # Get returns for this month
                company_returns = returns_df[returns_df["ISIN Code"].isin(companies)][["ISIN Code", month_col]].copy()
                company_returns = company_returns.rename(columns={month_col: 'Return'})
                company_returns['Return'] = pd.to_numeric(company_returns['Return'], errors='coerce')

                # Get lagged market cap for this month
                mc_month = mc_long[mc_long['Date'] == month_dt][['ISIN Code', 'MarketCap_lag']]

                # Merge returns with lagged market cap
                merged = company_returns.merge(mc_month, on='ISIN Code', how='left')
                merged = merged.dropna(subset=['Return', 'MarketCap_lag'])

                if merged.empty or merged['MarketCap_lag'].sum() <= 0:
                    results[f"Portfolio_{portfolio}"].append(None) # type: ignore
                    continue
                company_counts[f"Portfolio_{portfolio}"].append(len(merged))
                
                # Minimum company check
                if len(merged) < 10:
                    results[f"Portfolio_{portfolio}"].append(None) # type: ignore
                    continue

                # Value-weighted return
                weights = merged['MarketCap_lag'] / merged['MarketCap_lag'].sum()
                vw_return = float((weights * merged['Return']).sum())
                results[f"Portfolio_{portfolio}"].append(vw_return)  # type: ignore

    output_df = pd.DataFrame(results)
    for portfolio in ['Portfolio_1', 'Portfolio_2', 'Portfolio_3']:
        counts = company_counts[portfolio]
        below_10 = sum(1 for c in counts if c < 10)
        print(f"    {portfolio}: {below_10}/{len(counts)} months have <10 companies (min={min(counts)}, avg={sum(counts)/len(counts):.1f})")
    
    output_file = OUTPUT_DIR / f"{metric_name}_portfolio_returns.csv"
    output_df.to_csv(output_file, index=False)
    print(f"  Saved to: {output_file}")

print("\nDone! Monthly value-weighted portfolio returns saved to", OUTPUT_DIR)