import importlib.util
from pathlib import Path
import pandas as pd

p = Path(__file__).resolve().parent / 'ff_factors_improved.py'
spec = importlib.util.spec_from_file_location('ffm', str(p))
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

monthly = mod.load_monthly(mod.RAW_DIR)
annual = mod.load_annual(mod.RAW_DIR)
rf = mod.load_rf(mod.RF_FILE)

universe = mod.build_universe(monthly, annual, rf)

print('Total rows in universe:', len(universe))
print('Momentum column present:', 'Momentum' in universe.columns)
print('Momentum non-null overall:', int(universe['Momentum'].notna().sum()))
print('\nReturn_factor non-null overall:', int(universe['Return_factor'].notna().sum()))
print('Sample ISINs with most Return_factor entries:')
print(universe.groupby('ISIN Code')['Return_factor'].apply(lambda s: s.notna().sum()).sort_values(ascending=False).head(10).to_string())

# show most-populated ISIN sample rows
top_isin = universe.groupby('ISIN Code')['Return_factor'].apply(lambda s: s.notna().sum()).idxmax()
print(f"\nTop ISIN: {top_isin}")
print(universe[universe['ISIN Code']==top_isin].sort_values('Date').tail(20).to_string())
print('\nSample per-month non-null counts (first 20 months):')
counts = universe.groupby(universe['Date'].dt.to_period('M'))['Momentum'].apply(lambda s: s.notna().sum())
print(counts.head(20).to_string())
print('\nSample rows with Momentum non-null:')
print(universe[universe['Momentum'].notna()].head(10).to_string())
