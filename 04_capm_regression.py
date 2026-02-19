import numpy as np
import pandas as pd
import yfinance as yf
import statsmodels.api as sm
import matplotlib.pyplot as plt

start_date = "2018-01-01"

# 1) Download adjusted prices
stock_df = yf.download("AAPL", start=start_date, progress=False, auto_adjust=True)
market_df = yf.download("SPY", start=start_date, progress=False, auto_adjust=True)

if stock_df.empty or market_df.empty:
    raise ValueError("Download returned empty data. Check internet or ticker symbols.")

stock = stock_df["Close"]
market = market_df["Close"]

# 2) Align dates + compute returns
data = pd.concat([stock, market], axis=1)
data.columns = ["Stock", "Market"]
data = data.dropna()

returns = data.pct_change().dropna()

# Risk-free (simple assumption)
rf_annual = 0.02
rf_daily = (1 + rf_annual)**(1/252) - 1

returns["Stock_excess"] = returns["Stock"] - rf_daily
returns["Market_excess"] = returns["Market"] - rf_daily

# 3) CAPM regression
X = sm.add_constant(returns["Market_excess"])
y = returns["Stock_excess"]

model = sm.OLS(y, X).fit()

alpha = model.params["const"]
beta = model.params["Market_excess"]

print("CAPM Regression Results (AAPL vs SPY)")
print("-------------------------------------")
print(f"Alpha (daily): {alpha:.6f}")
print(f"Alpha (annual approx): {alpha*252:.4f}")
print(f"Beta: {beta:.4f}")
print(f"R-squared: {model.rsquared:.4f}")
print("\nT-stats:")
print(model.tvalues)
print("\nP-values:")
print(model.pvalues)

# 4) Plot
plt.figure()
plt.scatter(returns["Market_excess"], returns["Stock_excess"], alpha=0.3)
plt.plot(returns["Market_excess"], model.predict(X), linewidth=2)
plt.title("CAPM Regression: AAPL vs SPY (Excess Returns)")
plt.xlabel("Market Excess Return")
plt.ylabel("Stock Excess Return")
plt.tight_layout()
plt.savefig("capm_regression_plot.png", dpi=200)
plt.show()
