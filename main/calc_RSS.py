import pandas as pd
from pathlib import Path

FINALIZED_DATA_DIR = Path(__file__).parent.parent / "data" / "LSEG_data" / "finalized_metrics"
NON_DATE_COLS = ["ISIN Code", "Identifier", "CIGS 11 sectors code"]

def get_RSS_variables():
    all_dfs = []

    for csv_file in FINALIZED_DATA_DIR.glob("*.csv"):
        df = pd.read_csv(csv_file)
        df = df.loc[:, ~df.columns.str.startswith("Unnamed")]

        year_cols = [c for c in df.columns if c not in NON_DATE_COLS]

        # Melt to long format
        melted = df.melt(
            id_vars=["ISIN Code", "CIGS 11 sectors code"],
            value_vars=year_cols,
            var_name="Year",
            value_name=csv_file.stem
        )
        melted["Year"] = melted["Year"].astype(int)
        melted[csv_file.stem] = pd.to_numeric(melted[csv_file.stem], errors="coerce")
        all_dfs.append(melted)

    # Merge all metrics on ISIN Code, sector, and year
    combined = all_dfs[0]
    for df in all_dfs[1:]:
        combined = combined.merge(df, on=["ISIN Code", "CIGS 11 sectors code", "Year"], how="outer")

    metric_cols = [c for c in combined.columns if c not in ["ISIN Code", "CIGS 11 sectors code", "Year"]]

    # Compute mean and std per sector per year, excluding NULLs automatically via pandas
    records = []
    for (sector, year), group in combined.groupby(["CIGS 11 sectors code", "Year"]):
        row = {"CIGS 11 sectors code": sector, "Year": year}
        for metric in metric_cols:
            valid = group[metric].dropna()
            row[f"{metric}_mean"] = valid.mean()
            row[f"{metric}_std"]  = valid.std()
        records.append(row)

    df = pd.DataFrame(records).set_index(["CIGS 11 sectors code", "Year"])
    return df

def calc_RSS(rss_vars):
    all_rss = {}

    for csv_file in FINALIZED_DATA_DIR.glob("*.csv"):
        metric = csv_file.stem
        df = pd.read_csv(csv_file)
        df = df.loc[:, ~df.columns.str.startswith("Unnamed")]

        year_cols = [c for c in df.columns if c not in NON_DATE_COLS]

        result_rows = []
        for _, row in df.iterrows():
            isin    = row["ISIN Code"]
            sector  = row["CIGS 11 sectors code"]
            new_row = {"ISIN Code": isin, "CIGS 11 sectors code": sector}

            for year_col in year_cols:
                year  = int(year_col)
                value = pd.to_numeric(row[year_col], errors="coerce")

                try:
                    mean = rss_vars.loc[(sector, year), f"{metric}_mean"]
                    std  = rss_vars.loc[(sector, year), f"{metric}_std"]
                    new_row[year_col] = (value - mean) / std if pd.notna(value) and std > 0 else None
                except KeyError:
                    new_row[year_col] = None

            result_rows.append(new_row)

        all_rss[metric] = pd.DataFrame(result_rows)

    return all_rss

if __name__ == "__main__":
    results_dir = Path(__file__).parent.parent / "data" / "results"
    rss_vars = pd.read_csv(results_dir / "rss_variables.csv", index_col=["CIGS 11 sectors code", "Year"])
    
    all_rss = calc_RSS(rss_vars)
    output_dir = results_dir / "rss_scores"
    output_dir.mkdir(parents=True, exist_ok=True)

    for metric, df in all_rss.items():
        df.to_csv(output_dir / f"{metric}_rss.csv", index=False)
        print(f"Saved RSS scores for {metric} to {output_dir / f'{metric}_rss.csv'}")