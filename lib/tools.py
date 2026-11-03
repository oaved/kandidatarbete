import pandas as pd

def read_company_isins():
    companies_df = pd.read_csv("data/NASDAQ_STHLM_companies.csv")
    return companies_df["ISIN Code"].tolist()

def read_risk_free_rate(start_date):
    tb_1m_df = pd.read_csv("data/riksbanken__tb_1m.csv", sep=';')
    tb_1m_df["Period"] = pd.to_datetime(tb_1m_df["Period"], format="%Y %B")
    tb_1m_df = tb_1m_df[tb_1m_df["Period"] >= start_date]
    tb_1m_df = tb_1m_df.rename(columns={"Period": "Date", "Average": "RF"})
    tb_1m_df["RF"] = tb_1m_df["RF"] / 100
    return tb_1m_df[["Date", "RF"]]