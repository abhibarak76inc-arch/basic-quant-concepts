import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

# ============================================================
# 10 - Rolling Correlation (Time-Varying Relationships)
# ============================================================

# 1. Download data
ticker_a = "AAPL"
ticker_b = "SPY"
start_date = "2018-01-01"

data = yf.download([ticker_a, ticker_b], start=start_date, auto_adjust=True, progress=False)["Close"]
data = data.dropna()

# 2. Compute returns
returns = data.pct_change().dropna()

# 3. Rolling correlation
window = 60  # 60-day rolling window

rolling_corr = returns[ticker_a].rolling(window).corr(returns[ticker_b])

# 4. Plot
plt.figure()

plt.plot(rolling_corr, label="Rolling Correlation (60D)")
plt.axhline(0, linestyle="--")

plt.title(f"Rolling Correlation: {ticker_a} vs {ticker_b}")
plt.xlabel("Date")
plt.ylabel("Correlation")
plt.legend()

plt.tight_layout()
plt.savefig("rolling_correlation.png", dpi=200)
plt.show()

# 5. Print summary
print("Rolling Correlation Summary:")
print(f"Mean correlation: {rolling_corr.mean():.4f}")
print(f"Max correlation: {rolling_corr.max():.4f}")
print(f"Min correlation: {rolling_corr.min():.4f}")
