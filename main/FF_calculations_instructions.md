# Datapoints needed:

**Per Stock, Per Month**

| Variable | Symbol | LSEG Field | Source | Used For |
|---|---|---|---|---|
| Total Return (price + dividends) | $R_t$ | `TR.TotalReturn` | LSEG | Mkt-RF, WML |
| Market Cap | $MC$ | `TR.CompanyMarketCap` | LSEG | SMB, HML, Value-weighting |
| Risk-free Rate | $R_f$ | SE TB 1 Month | Riksbank | Mkt-RF |

---

**Per Stock, Annually**

| Variable | Symbol | LSEG Field | Used For |
|---|---|---|---|
| Common Equity (Book Equity) | $BE$ | `TR.F.ComEqTot` | HML, RMW |
| Total Assets (t and t-1) | $A_t,\ A_{t-1}$ | `TR.F.TotAssets` | CMA |
| EBIT | $\text{EBIT}$ | `TR.F.EBIT` | RMW |
| Interest Expense | $\text{Int}$ | `TR.F.IntrExpn` | RMW |

---

# What we then calculate:

| Derived | Formula | FF Factor |
|---|---|---|
| Excess Return | $$R_t - R_f$$ | Mkt-RF |
| B/M Ratio | $$\frac{BE}{MC}$$ | HML |
| Operating Profitability | $$OP = \frac{\text{EBIT} - \text{Int}}{BE}$$ | RMW |
| Asset Growth | $$Inv = \frac{A_t - A_{t-1}}{A_{t-1}}$$ | CMA |
| Prior Return (Momentum) | $$\text{Mom} = \prod_{s=t-12}^{t-2}(1 + R_s) - 1$$ | WML |
| Size (Market Cap rank) | $$MC_t$$ | SMB |