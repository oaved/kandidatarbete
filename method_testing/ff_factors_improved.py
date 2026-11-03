import pandas as pd
import numpy as np
from pathlib import Path

# Purpose: improved FF6 factor construction for method testing.
# Outputs saved under method_testing/results/

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "LSEG_data" / "raw_metrics"
RF_FILE = ROOT / "data" / "riksbanken_tb_1m.csv"
OUT_DIR = Path(__file__).resolve().parent / "results"
OUT_DIR.mkdir(parents=True, exist_ok=True)

MIN_COUNT = 5  # minimum firms per sub-portfolio cell
MOM_MIN_PERIODS = 1  # minimum months required for momentum (relaxed to allow partial history)


def load_monthly(raw_dir: Path):
    tr = pd.read_csv(raw_dir / "TotalReturn.csv")
    mc = pd.read_csv(raw_dir / "MarketCap.csv")

    # normalize column names: date columns may have time component
    def normalize_cols(df):
        df = df.loc[:, ~df.columns.str.startswith("Unnamed")]
        date_cols = [c for c in df.columns if '-' in c]
        id_cols = [c for c in df.columns if c not in date_cols]
        df = df[id_cols + date_cols]
        return df

    tr = normalize_cols(tr)
    mc = normalize_cols(mc)

    # melt to long
    def melt(df, value_name):
        id_vars = [c for c in df.columns if not ('-' in c)]
        date_cols = [c for c in df.columns if '-' in c]
        out = df.melt(id_vars=id_vars, value_vars=date_cols, var_name='Date', value_name=value_name)
        out['Date'] = pd.to_datetime(out['Date'], errors='coerce').dt.to_period('M').dt.to_timestamp()
        out[value_name] = pd.to_numeric(out[value_name], errors='coerce')
        return out

    tr_l = melt(tr, 'Total Return')
    mc_l = melt(mc, 'Market Cap')

    df = tr_l.merge(mc_l, on=['ISIN Code', 'Date'], how='outer')

    # compute return factor and momentum (t-12..t-2)
    df = df.sort_values(['ISIN Code', 'Date'])
    df['Return_factor'] = 1 + df['Total Return'] / 100

    # Robust per-ISIN momentum: product of (1+R) from t-12 to t-2 minus 1
    def compute_mom(s):
        # s is Return_factor series for an ISIN sorted by Date
        shifted = s.shift(2)
        # rolling apply: ignore NaNs inside window when computing product
        def prod_nonan(y):
            arr = np.asarray(y)
            non_na = arr[~np.isnan(arr)]
            if len(non_na) < MOM_MIN_PERIODS:
                return np.nan
            return np.prod(non_na)

        roll = shifted.rolling(window=11, min_periods=MOM_MIN_PERIODS).apply(prod_nonan, raw=True)
        return roll - 1

    df['Momentum'] = df.groupby('ISIN Code')['Return_factor'].transform(compute_mom)

    return df


def load_annual(raw_dir: Path):
    be = pd.read_csv(raw_dir / 'BookEquity.csv')
    ta = pd.read_csv(raw_dir / 'TotalAssets.csv')
    ebit = pd.read_csv(raw_dir / 'EBIT.csv')
    ie = pd.read_csv(raw_dir / 'InterestExpense.csv')

    def melt_year(df, value_name):
        id_vars = [c for c in df.columns if not c.isdigit()]
        year_cols = [c for c in df.columns if c.isdigit()]
        out = df.melt(id_vars=id_vars, value_vars=year_cols, var_name='Year', value_name=value_name)
        out['Date'] = pd.to_datetime(out['Year'], format='%Y', errors='coerce')
        out[value_name] = pd.to_numeric(out[value_name], errors='coerce')
        return out[['ISIN Code', 'Date', value_name]]

    be_l = melt_year(be, 'BE')
    ta_l = melt_year(ta, 'Total Assets')
    ebit_l = melt_year(ebit, 'EBIT')
    ie_l = melt_year(ie, 'Int')

    df = be_l.merge(ta_l, on=['ISIN Code', 'Date'], how='outer')
    df = df.merge(ebit_l, on=['ISIN Code', 'Date'], how='outer')
    df = df.merge(ie_l, on=['ISIN Code', 'Date'], how='outer')

    # compute investment (Inv) on annual data: (A_t - A_{t-1}) / A_{t-1}
    df = df.sort_values(['ISIN Code', 'Date'])
    df['Assets_lag'] = df.groupby('ISIN Code')['Total Assets'].shift(1)
    df['Inv'] = (df['Total Assets'] - df['Assets_lag']) / df['Assets_lag']

    # shift annual dates forward by 6 months (align FY to July)
    df['Date'] = df['Date'] + pd.DateOffset(months=6)
    return df


def load_rf(rf_file: Path):
    rf = pd.read_csv(rf_file, sep=';')
    rf['Date'] = pd.to_datetime(rf['Period'] + ' 01', format='%Y %B %d', errors='coerce')
    rf = rf.rename(columns={'Average': 'RF'})
    rf['RF'] = pd.to_numeric(rf['RF'], errors='coerce')
    rf['RF'] = ((1 + rf['RF'] / 100) ** (1/12) - 1) * 100  # convert annualised to monthly
    return rf[['Date', 'RF']]


def build_universe(monthly_df, annual_df, rf_df):
    # merge annual onto monthly using asof (backward) per ISIN
    monthly_df = monthly_df.sort_values(['ISIN Code', 'Date']).reset_index(drop=True)
    annual_df = annual_df.sort_values(['ISIN Code', 'Date']).reset_index(drop=True)

    out = []
    for isin, grp in monthly_df.groupby('ISIN Code'):
        a_grp = annual_df[annual_df['ISIN Code'] == isin][['Date','BE','Total Assets','EBIT','Int','Inv']].sort_values('Date')
        m = grp.sort_values('Date').reset_index(drop=True)
        merged = pd.merge_asof(m, a_grp, on='Date', direction='backward')
        merged['ISIN Code'] = isin
        out.append(merged)

    df = pd.concat(out, ignore_index=True)

    # add market cap lag (t-1) for size groups and weights
    df = df.sort_values(['ISIN Code', 'Date'])
    df['MarketCap_lag'] = df.groupby('ISIN Code')['Market Cap'].shift(1)

    # merge RF month
    rf_df = rf_df.copy()
    rf_df['Month'] = rf_df['Date'].dt.to_period('M')
    df['Month'] = df['Date'].dt.to_period('M')
    df = pd.merge(df, rf_df[['Month','RF']].drop_duplicates(), on='Month', how='left')

    # keep percent units for Total Return and RF
    df['Excess Return'] = df['Total Return'] - df['RF']

    return df


def vw_return(group, ret_col='Excess Return'):
    # use lagged market cap for weights
    group = group.dropna(subset=[ret_col, 'MarketCap_lag'])
    if group.empty or group['MarketCap_lag'].sum() <= 0:
        return np.nan, 0
    w = group['MarketCap_lag'] / group['MarketCap_lag'].sum()
    return (w * group[ret_col]).sum(), len(group)


def compute_ff6(df):
    factors = []
    diagnostics = []

    for month, month_df in df.groupby('Date'):
        # consider only firms with necessary fields
        month = month_df.copy()

        # require at least some firms
        if month.shape[0] < 10:
            continue

        # size grouping by lagged market cap
        month = month.dropna(subset=['MarketCap_lag'])
        if month.empty:
            continue

        size_med = month['MarketCap_lag'].median()
        month['SizeGroup'] = np.where(month['MarketCap_lag'] < size_med, 'S', 'B')

        # compute characteristics — use Inv and Momentum merged from source data
        month['BM'] = month['BE'] / month['MarketCap_lag']
        month['OP'] = (month['EBIT'] - month['Int']) / month['BE']
        # 'Inv' computed on annual data (merged via asof) — use it directly
        # 'Momentum' precomputed in monthly data — keep it

        # For each characteristic compute long-short portfolios requiring MIN_COUNT per cell
        def factor_for_char(char, high_is_good=True):
            sub = month.dropna(subset=[char, 'Excess Return', 'MarketCap_lag'])
            if len(sub) < 4:
                return np.nan, np.nan, {}

            med = sub[char].median()
            sub['CharGroup'] = np.where(sub[char] >= med, 'H', 'L')

            ports = {}
            counts = {}
            for sg in ['S','B']:
                for cg in ['H','L']:
                    grp = sub[(sub['SizeGroup']==sg) & (sub['CharGroup']==cg)]
                    val, cnt = vw_return(grp)
                    if cnt < MIN_COUNT:
                        val = np.nan
                    ports[f"{sg}{cg}"] = val
                    counts[f"{sg}{cg}"] = cnt

            SH, SL = ports.get('SH', np.nan), ports.get('SL', np.nan)
            BH, BL = ports.get('BH', np.nan), ports.get('BL', np.nan)

            if high_is_good:
                if np.isnan(SH) and np.isnan(BH):
                    return np.nan, np.nan, counts
                long = np.nanmean([v for v in [SH, BH] if not np.isnan(v)])
                short = np.nanmean([v for v in [SL, BL] if not np.isnan(v)])
                if np.isnan(long) or np.isnan(short):
                    return np.nan, np.nan, counts
                factor = long - short
            else:
                if np.isnan(SL) and np.isnan(BL):
                    return np.nan, np.nan, counts
                long = np.nanmean([v for v in [SL, BL] if not np.isnan(v)])
                short = np.nanmean([v for v in [SH, BH] if not np.isnan(v)])
                if np.isnan(long) or np.isnan(short):
                    return np.nan, np.nan, counts
                factor = long - short

            smb = np.nanmean([np.nanmean([ports.get('SH',np.nan), ports.get('SL',np.nan)]), np.nanmean([ports.get('BH',np.nan), ports.get('BL',np.nan)])])
            return factor, smb, counts

        # market return (value-weighted) using lagged market cap
        mkt_val, mkt_cnt = vw_return(month, 'Excess Return')

        hml, smb_bm, counts_bm = factor_for_char('BM', high_is_good=True)
        rmw, smb_op, counts_op = factor_for_char('OP', high_is_good=True)
        cma, smb_inv, counts_inv = factor_for_char('Inv', high_is_good=False)

        # momentum (WML)
        wml, smb_mom, counts_mom = factor_for_char('Momentum', high_is_good=True)

        # SMB: average of available smb components; require at least 2
        smb_components = [s for s in [smb_bm, smb_op, smb_inv, smb_mom] if not np.isnan(s)]
        smb = np.nan
        if len(smb_components) >= 2:
            smb = np.mean(smb_components) # type: ignore

        factors.append({'Date': month['Date'].iloc[0], 'Mkt_RF': mkt_val, 'SMB': smb, 'HML': hml, 'RMW': rmw, 'CMA': cma, 'WML': wml})
        diagnostics.append({'Date': month['Date'].iloc[0], 'n_firms': len(month), 'mkt_cnt': mkt_cnt, 'counts_bm': counts_bm, 'counts_op': counts_op, 'counts_inv': counts_inv, 'counts_mom': counts_mom})

    fac_df = pd.DataFrame(factors)
    diag_df = pd.DataFrame(diagnostics)
    
    return fac_df, diag_df


def main():
    monthly = load_monthly(RAW_DIR)
    annual = load_annual(RAW_DIR)
    rf = load_rf(RF_FILE)

    universe = build_universe(monthly, annual, rf)
    ff6, diag = compute_ff6(universe)

    

    # save
    ff6.to_csv(OUT_DIR / 'ff6_factors_improved.csv', index=False)
    diag.to_csv(OUT_DIR / 'ff6_diagnostics.csv', index=False)
    print('Saved improved factors to', OUT_DIR)


if __name__ == '__main__':
    main()
