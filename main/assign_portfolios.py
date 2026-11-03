import pandas as pd
from pathlib import Path

RSS_SCORES_DIR = Path(__file__).parent.parent / "data" / "results" / "rss_scores"
OUTPUT_DIR     = Path(__file__).parent.parent / "data" / "results" / "rss_tertiles"
NON_DATE_COLS  = ["ISIN Code", "CIGS 11 sectors code"]

def assign_tertiles(df, year_cols):
    result = df[NON_DATE_COLS].copy()
    for year_col in year_cols:
        valid_mask = df[year_col].notna()
        result[year_col] = None
        if valid_mask.sum() >= 3:
            values = df.loc[valid_mask, year_col]
            t33 = values.quantile(1/3)
            t66 = values.quantile(2/3)
            if t33 == t66:
                print(f"  Skipping {year_col}: all companies have same RSS score")
                continue
            
            tertiles = pd.cut(
                values,
                bins=[-float("inf"), t33, t66, float("inf")],
                labels=[1, 2, 3],
                right=False
            ).astype(float)
            result.loc[valid_mask, year_col] = tertiles
            
            print(f"{year_col}: t33={t33:.4f}, t66={t66:.4f}, "
                  f"n_below_t33={(values < t33).sum()}, "
                  f"n_between={((values >= t33) & (values <= t66)).sum()}, "
                  f"n_above={(values > t66).sum()}, "
                  f"unique_values={values.nunique()}")
    return result

if __name__ == "__main__":
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for csv_file in sorted(RSS_SCORES_DIR.glob("*_rss.csv")):
        metric = csv_file.stem.replace("_rss", "")
        df     = pd.read_csv(csv_file)
        df     = df.loc[:, ~df.columns.str.startswith("Unnamed")]

        year_cols = [c for c in df.columns if c not in NON_DATE_COLS]

        tertile_df = assign_tertiles(df, year_cols)