import pandas as pd
import os

# get data from excel
df = pd.read_excel("NASDAQ_STHLM_companies.xlsx")
df.to_csv("NASDAQ_STHLM_companies.csv", index=False)

# save the csv file to the data folder
if not os.path.exists("data"):
    os.makedirs("data")
os.rename("NASDAQ_STHLM_companies.csv", "data/NASDAQ_STHLM_companies.csv")