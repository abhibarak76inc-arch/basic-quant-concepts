import numpy as np
import pandas as pd
import yfinance as yf
import statsmodels.api as sm
import matplotlib.pyplot as plt

# ============================================================
# 11 - Portfolio Beta (Factor Exposure)
# ============================================================

tickers = ["AAPL", "MSFT", "GOOGL"]
market = "SPY"
start_date = "2018-01-01"

# 1. Download data
data = yf.download(tickers + [market], start=start_date, auto_adjust=True, progress=False)["Close"]
data = data.dropna()

# 2. Compute returns
returns = data.pct_change().dropna()
market_returns = returns[market]

# 3. Compute beta for each stock
betas = {}

for ticker in tickers:
    X = sm.add_constant(market_returns)
    y = returns[ticker]

    model = sm.OLS(y, X).fit()
    betas[ticker] = model.params[market]

# Print individual betas
print("Individual Betas:")
for k, v in betas.items():
    print(f"{k}: {v:.4f}")

# 4. Define portfolio weights
weights = {
    "AAPL": 0.4,
    "MSFT": 0.3,
    "GOOGL": 0.3
}

# 5. Compute portfolio beta
portfolio_beta = 0
for ticker in tickers:
    portfolio_beta += weights[ticker] * betas[ticker]

print("\nPortfolio Beta:", round(portfolio_beta, 4))

# ============================================================
# 6. Visualization
# ============================================================

labels = list(betas.keys()) + ["Portfolio"]
values = list(betas.values()) + [portfolio_beta]

plt.figure()

plt.bar(labels, values)

plt.axhline(1, linestyle="--")  # market benchmark

plt.title("Individual Betas vs Portfolio Beta")
plt.ylabel("Beta")

plt.tight_layout()
plt.savefig("portfolio_beta.png", dpi=200)
plt.show()
