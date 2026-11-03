import pandas as pd
from pathlib import Path

METRICS_DIR = Path(__file__).parent.parent / "data" / "LSEG_data" / "raw_metrics"
FF_FILES = {"TotalReturn.csv", "MarketCap.csv", "BookEquity.csv", "TotalAssets.csv", "EBIT.csv", "InterestExpense.csv"}
NON_DATE_COLS = ["ISIN Code", "Identifier", "CIGS 11 sectors code"]
START_YEAR = 2020
END_YEAR = 2024
COVERAGE_THRESHOLD = 0.6
YEARLY_STD_THRESHOLD = 0.15
TOP_N = 25

def gather_intelligence():
    count = 0
    results = []

    for csv_file in METRICS_DIR.glob("*.csv"):
        if csv_file.name in FF_FILES:
            continue

        df = pd.read_csv(csv_file)
        df = df.loc[:, ~df.columns.str.startswith("Unnamed")]

        year_cols = [c for c in df.columns if c not in NON_DATE_COLS
                    and START_YEAR <= int(c) <= END_YEAR]

        n_companies = len(df)
        total_cells = n_companies * len(year_cols)
        filled_cells = df[year_cols].notna().sum().sum()

        results.append({
            "metric"           : csv_file.stem,
            "overall_coverage" : filled_cells / total_cells if total_cells > 0 else 0,
            "coverage_per_year": {int(col): df[col].notna().sum() / n_companies for col in year_cols},
        })
        
        count += 1
        
    print(f"Total social metric files processed: {count}")
    return results

def review_variables(res, coverage_threshold=0.5, yearly_std_threshold=0.15):
    print(f"Coverage threshold: {coverage_threshold:.0%}, Yearly std threshold: {yearly_std_threshold:.0%}")
    
    # Step 1: Apply overall coverage cutoff
    passing = [r for r in res if r["overall_coverage"] >= coverage_threshold]
    failing = [r for r in res if r["overall_coverage"] < coverage_threshold]
    
    print(f"=== STEP 1: Coverage cutoff ({coverage_threshold:.0%}) ===")
    print(f"Passing: {len(passing)}, Failing: {len(failing)}")
    for r in failing:
        print(f"  DROPPED {r['metric']}: {r['overall_coverage']:.2%}")

    # Step 2: Flag inconsistent yearly coverage
    print(f"\n=== STEP 2: Yearly consistency (std threshold {yearly_std_threshold:.0%}) ===")
    stable = []
    for r in passing:
        std = pd.Series(r["coverage_per_year"].values()).std()
        if std > yearly_std_threshold:
            print(f"  FLAGGED {r['metric']}: std={std:.2%}")
        else:
            stable.append(r)

    # Step 3: Correlation between surviving variables
    print(f"\n=== STEP 3: Correlation check ===")
    dfs = []
    for r in stable:
        df = pd.read_csv(METRICS_DIR / f"{r['metric']}.csv")
        df = df.loc[:, ~df.columns.str.startswith("Unnamed")]
        year_cols = [c for c in df.columns if c not in ["ISIN Code", "Identifier", "CIGS 11 sectors code"]
                     and START_YEAR <= int(c) <= END_YEAR]
        melted = df.melt(id_vars=["ISIN Code"], value_vars=year_cols, value_name=r["metric"])
        melted[r["metric"]] = pd.to_numeric(melted[r["metric"]], errors="coerce")
        melted = melted[["ISIN Code", r["metric"]]]
        dfs.append(melted.reset_index(drop=True))

    combined = pd.concat(dfs, axis=1)
    combined = combined.loc[:, ~combined.columns.duplicated()]
    metric_cols = [r["metric"] for r in stable]
    corr = combined[metric_cols].corr()

    high_corr_pairs = []
    for i in range(len(metric_cols)):
        for j in range(i + 1, len(metric_cols)):
            val = corr.iloc[i, j]
            if abs(val) > 0.80:
                high_corr_pairs.append((metric_cols[i], metric_cols[j], val))

    if high_corr_pairs:
        for a, b, val in high_corr_pairs:
            print(f"  HIGH CORR {a} & {b}: {val:.2f}")
    else:
        print("  No highly correlated pairs found.")

    return stable

def forward_fill_gaps(df, start_year=START_YEAR, end_year=END_YEAR):
    year_cols = [c for c in df.columns if c not in ["ISIN Code", "Identifier", "CIGS 11 sectors code"] and start_year <= int(c) <= end_year]
    for i in range(1, len(year_cols)):
        prev_col = year_cols[i - 1]
        curr_col = year_cols[i]
        gap_mask = df[curr_col].isna() & df[prev_col].notna()
        df.loc[gap_mask, curr_col] = df.loc[gap_mask, prev_col]
    return df

def remove_invalid_data(df, start_year=START_YEAR):
    df = df.loc[:, ~df.columns.str.startswith("Unnamed")]
    year_cols = [c for c in df.columns if c not in NON_DATE_COLS]

    # Remove years before start_year
    cols_to_drop = [c for c in year_cols if int(c) < start_year]
    df = df.drop(columns=cols_to_drop)

    # Remove companies with all NULL across remaining year columns
    remaining_year_cols = [c for c in df.columns if c not in NON_DATE_COLS]
    df = df.dropna(subset=remaining_year_cols, how="all")

    return df

if __name__ == "__main__":
    res = gather_intelligence()
    sorted_res = sorted(res, key=lambda x: x["overall_coverage"], reverse=True)
    chosen_metrics = sorted_res[:TOP_N]

    output_dir = Path(__file__).parent.parent / "data" / "LSEG_data" / "finalized_metrics"
    output_dir.mkdir(parents=True, exist_ok=True)

    for r in chosen_metrics:
        print(f"  SELECTED {r['metric']}: {r['overall_coverage']:.2%}")
        df = pd.read_csv(METRICS_DIR / f"{r['metric']}.csv")
        df = forward_fill_gaps(df)
        df = remove_invalid_data(df)
        df.to_csv(output_dir / f"{r['metric']}.csv", index=False)