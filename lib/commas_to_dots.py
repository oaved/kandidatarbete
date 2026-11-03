import pandas as pd

# Read the CSV file with semicolon as the separator
df = pd.read_csv("data/riksbanken__tb_1m.csv", sep=';')
# Replace commas with dots in the 'Average' column and convert to float
df['Average'] = df['Average'].str.replace(',', '.').astype(float)
# Save the modified DataFrame back to a CSV file
df.to_csv("data/riksbanken__tb_1m.csv", sep=';', index=False)