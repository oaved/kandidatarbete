import pandas as pd
import numpy as np
from pathlib import Path

# ------ PATHS ------
DATA_DIR     = Path(__file__).parent.parent / "data"
LSEG_DIR     = DATA_DIR / "LSEG_data"
METRICS_DIR  = LSEG_DIR / "raw_metrics"
RESULTS_DIR  = DATA_DIR / "fama_french_factors"

# ------ LOAD DATA ------

NON_DATE_COLS = ["ISIN Code", "Identifier", "CIGS 11 sectors code"]

def melt_wide_to_long(file_path, value_name, date_format="full"):
    df = pd.read_csv(file_path)
    df = df.loc[:, ~df.columns.str.startswith("Unnamed")]

    date_cols = [c for c in df.columns if c not in NON_DATE_COLS]
    df = df.melt(
        id_vars=["ISIN Code"],
        value_vars=date_cols,
        var_name="Date",
        value_name=value_name
    )

    if date_format == "year":
        df["Date"] = pd.to_datetime(df["Date"], format="%Y")
    else:
        df["Date"] = pd.to_datetime(df["Date"])
        # Normalize to start of month so all files align regardless of convention
        df["Date"] = df["Date"].dt.to_period("M").dt.to_timestamp()

    df[value_name] = pd.to_numeric(df[value_name], errors="coerce")
    return df

def load_monthly_data():
    total_return = melt_wide_to_long(METRICS_DIR / "TotalReturn.csv", "Total Return")
    market_cap   = melt_wide_to_long(METRICS_DIR / "MarketCap.csv",   "Market Cap")
    df = total_return.merge(market_cap, on=["ISIN Code", "Date"], how="outer")
    df["Total Return"] = pd.to_numeric(df["Total Return"], errors="coerce")
    df["Market Cap"]   = pd.to_numeric(df["Market Cap"],   errors="coerce")
    return df

def load_annual_data():
    book_equity      = melt_wide_to_long(METRICS_DIR / "BookEquity.csv",      "BE",           date_format="year")
    total_assets     = melt_wide_to_long(METRICS_DIR / "TotalAssets.csv",     "Total Assets", date_format="year")
    ebit             = melt_wide_to_long(METRICS_DIR / "EBIT.csv",            "EBIT",         date_format="year")
    interest_expense = melt_wide_to_long(METRICS_DIR / "InterestExpense.csv", "Int",          date_format="year")

    df = book_equity.merge(total_assets,     on=["ISIN Code", "Date"], how="outer")
    df = df.merge(ebit,                      on=["ISIN Code", "Date"], how="outer")
    df = df.merge(interest_expense,          on=["ISIN Code", "Date"], how="outer")

    for col in ["BE", "Total Assets", "EBIT", "Int"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Shift annual dates forward by 6 months so e.g. 2018 fiscal year
    # data becomes available from July 2019 onwards in the monthly merge
    df["Date"] = df["Date"] + pd.DateOffset(months=6)

    return df

def load_risk_free_rate():
    rf = pd.read_csv(DATA_DIR / "riksbanken_tb_1m.csv", sep=";")
    
    # Parse "2010 January" style dates
    rf["Date"] = pd.to_datetime(rf["Period"], format="%Y %B")
    rf = rf.rename(columns={"Average": "RF"})
    
    # Convert from percent to decimal (0.20632 → 0.00206...)
    rf["RF"] = pd.to_numeric(rf["RF"], errors="coerce") / 100
    
    return rf[["Date", "RF"]]

# ------ FACTOR CALCULATIONS ------

def calc_excess_return(monthly_df, rf):
    """R_t - R_f"""
    monthly_df["Month"] = monthly_df["Date"].dt.to_period("M")
    rf["Month"] = rf["Date"].dt.to_period("M")
    monthly_df = monthly_df.merge(rf[["Month", "RF"]], on="Month", how="left")
    monthly_df["Excess Return"] = monthly_df["Total Return"] / 100 - monthly_df["RF"]
    monthly_df = monthly_df.drop(columns=["Month"])
    return monthly_df

def calc_bm(monthly_df, annual_df):
    """BE / MC — forward-fills annual BE into monthly frequency"""
    annual_sub = annual_df[["ISIN Code", "Date", "BE"]].dropna(subset=["BE", "Date"])
    monthly_df = monthly_df.dropna(subset=["Date"])

    results = []
    for isin, m_group in monthly_df.groupby("ISIN Code"):
        a_group = annual_sub[annual_sub["ISIN Code"] == isin][["Date", "BE"]].sort_values("Date")
        m_group = m_group.sort_values("Date").reset_index(drop=True)
        merged  = pd.merge_asof(m_group, a_group, on="Date", direction="backward")
        results.append(merged)

    merged = pd.concat(results, ignore_index=True)
    merged["BM"] = merged["BE"] / merged["Market Cap"]
    return merged

def calc_op(annual_df):
    """(EBIT - Int) / BE"""
    annual_df["Int"] = annual_df["Int"].fillna(0)
    annual_df["OP"]  = (annual_df["EBIT"] - annual_df["Int"]) / annual_df["BE"]
    return annual_df

def calc_inv(annual_df):
    """(A_t - A_{t-1}) / A_{t-1}"""
    annual_df = annual_df.sort_values(["ISIN Code", "Date"])
    annual_df["Assets_lag"] = annual_df.groupby("ISIN Code")["Total Assets"].shift(1)
    annual_df["Inv"] = (annual_df["Total Assets"] - annual_df["Assets_lag"]) / annual_df["Assets_lag"]
    return annual_df

def calc_momentum(monthly_df):
    """Product of (1 + R) from t-12 to t-2, skip t-1"""
    monthly_df = monthly_df.sort_values(["ISIN Code", "Date"])
    monthly_df["Return_factor"] = 1 + monthly_df["Total Return"] / 100
    monthly_df["Momentum"] = (
        monthly_df.groupby("ISIN Code")["Return_factor"]
        .transform(lambda x: x.shift(2).rolling(window=11).apply(np.prod, raw=True) - 1)
    )
    return monthly_df

def calc_size(monthly_df):
    """MC_t — just the market cap"""
    monthly_df["Size"] = monthly_df["Market Cap"]
    return monthly_df

def calc_ff6_factors(monthly_df, annual_df):
    """
    Construct FF6 factors using median split (2x2).
    Returns a monthly time series of: Mkt_RF, SMB, HML, RMW, CMA, WML.
    """
    annual_sub = annual_df[["ISIN Code", "Date", "OP", "Inv"]].dropna(subset=["Date"])

    results = []
    for isin, m_group in monthly_df.groupby("ISIN Code"):
        a_group = (
            annual_sub[annual_sub["ISIN Code"] == isin][["Date", "OP", "Inv"]]
            .sort_values("Date")
        )
        m_group = m_group.sort_values("Date").reset_index(drop=True)
        merged  = pd.merge_asof(m_group, a_group, on="Date", direction="backward")
        results.append(merged)

    df = pd.concat(results, ignore_index=True)
    
    # --- DEBUG ---
    print("df shape:", df.shape)
    print("df columns:", df.columns.tolist())
    print("Excess Return non-null:", df["Excess Return"].notna().sum())
    print("Size non-null:", df["Size"].notna().sum())
    print("Sample month group sizes:")
    for date, month in list(df.groupby("Date"))[:5]:
        after_drop = month.dropna(subset=["Excess Return", "Size"])
        print(f"  {date}: {len(month)} rows total, {len(after_drop)} after dropna")
    # --- END DEBUG ---

    def vw_return(group):
        if group.empty or group["Size"].sum() == 0:
            return np.nan
        w = group["Size"] / group["Size"].sum()
        return (w * group["Excess Return"]).sum()

    def calc_factor_and_smb(month, char, high_is_good=True):
        sub = month.dropna(subset=[char]).copy()
        if len(sub) < 4:
            return np.nan, np.nan

        char_median  = sub[char].median()
        sub["CharGroup"] = np.where(sub[char] >= char_median, "H", "L")

        ports = {}
        for sg in ["S", "B"]:
            for cg in ["H", "L"]:
                grp = sub[(sub["SizeGroup"] == sg) & (sub["CharGroup"] == cg)]
                ports[f"{sg}{cg}"] = vw_return(grp)

        sh, sl = ports.get("SH", np.nan), ports.get("SL", np.nan)
        bh, bl = ports.get("BH", np.nan), ports.get("BL", np.nan)

        factor = (np.nanmean([sh, bh]) - np.nanmean([sl, bl])) if high_is_good \
            else (np.nanmean([sl, bl]) - np.nanmean([sh, bh]))

        smb = np.nanmean([sh, sl]) - np.nanmean([bh, bl])
        return factor, smb

    factors = []
    for date, month in df.groupby("Date"):
        month = month.dropna(subset=["Excess Return", "Size"]).copy()
        if len(month) < 4:
            continue

        size_median      = month["Size"].median()
        month["SizeGroup"] = np.where(month["Size"] < size_median, "S", "B")

        mkt_rf               = vw_return(month)
        hml, smb_bm  = calc_factor_and_smb(month, "BM",       high_is_good=True)
        rmw, smb_op  = calc_factor_and_smb(month, "OP",       high_is_good=True)
        cma, smb_inv = calc_factor_and_smb(month, "Inv",      high_is_good=False)
        wml, smb_mom = calc_factor_and_smb(month, "Momentum", high_is_good=True)

        smb = np.nanmean([smb_bm, smb_op, smb_inv, smb_mom])

        factors.append({
            "Date"  : date,
            "Mkt_RF": mkt_rf,
            "SMB"   : smb,
            "HML"   : hml,
            "RMW"   : rmw,
            "CMA"   : cma,
            "WML"   : wml,
        })

    return pd.DataFrame(factors)

def convert_ff_factors_to_percent(ff_df):
    """Convert factor returns from decimal form to percent form."""
    factor_cols = ["Mkt_RF", "SMB", "HML", "RMW", "CMA", "WML"]
    ff_df[factor_cols] = ff_df[factor_cols] * 100
    return ff_df


def run_debug_checks():
    """Run debug diagnostics for annual and monthly data alignment."""
    try:
        annual = load_annual_data()
        monthly = load_monthly_data()
    except FileNotFoundError as e:
        print(f"Debug failed: missing file {e.filename}")
        return
    except Exception as e:
        print(f"Debug failed: {type(e).__name__}: {e}")
        return

    print("\n=== DEBUG: Annual dates sample ===")
    print(annual[["ISIN Code", "Date"]].drop_duplicates().head(6).to_string(index=False))

    print("\n=== DEBUG: Monthly sample rows ===")
    print(monthly[["ISIN Code", "Date", "Total Return"]].head(6).to_string(index=False))

    sample_isin = monthly["ISIN Code"].dropna().iloc[0]
    print(f"\n=== DEBUG: Sample ISIN {sample_isin} annual rows ===")
    print(annual[annual["ISIN Code"] == sample_isin].head().to_string(index=False))

    print(f"\n=== DEBUG: Sample ISIN {sample_isin} monthly rows ===")
    print(monthly[monthly["ISIN Code"] == sample_isin].head().to_string(index=False))
    print(f"\n=== DEBUG: Merged alignment for {sample_isin} ===")
    sample_monthly = monthly[monthly["ISIN Code"] == sample_isin].sort_values("Date")
    sample_annual = annual[annual["ISIN Code"] == sample_isin].sort_values("Date")
    merged = pd.merge_asof(
        sample_monthly,
        sample_annual[["Date", "BE", "Total Assets", "EBIT", "Int"]],
        on="Date",
        direction="backward"
    )
    merged = merged.loc[:, ["ISIN Code", "Date", "Total Return", "Market Cap", "BE", "Total Assets", "EBIT", "Int"]]
    print(merged.head(18).to_string(index=False))
if __name__ == "__main__":
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    run_debug_checks()

    monthly_df = load_monthly_data()
    annual_df  = load_annual_data()
    rf         = load_risk_free_rate()

    monthly_df = calc_excess_return(monthly_df, rf)
    monthly_df = calc_momentum(monthly_df)
    monthly_df = calc_size(monthly_df)
    monthly_df = monthly_df.dropna(subset=["Date"])
    monthly_df = calc_bm(monthly_df, annual_df)

    print(monthly_df["BM"].describe())

    # Annual variables
    annual_df = calc_op(annual_df)
    annual_df = calc_inv(annual_df)

    # Save
    monthly_df.to_csv(RESULTS_DIR / "monthly_factors.csv", index=False)
    annual_df.to_csv(RESULTS_DIR  / "annual_factors.csv",  index=False)
    print("Saved monthly_factors.csv and annual_factors.csv")


    ff6_factors = calc_ff6_factors(monthly_df, annual_df)
    ff6_factors = convert_ff_factors_to_percent(ff6_factors)
    ff6_factors.to_csv(RESULTS_DIR / "ff6_factors.csv", index=False)
    print("Saved ff6_factors.csv")
    
    print(ff6_factors.shape)
    print(ff6_factors.head(10))
    print(ff6_factors.isnull().sum())
