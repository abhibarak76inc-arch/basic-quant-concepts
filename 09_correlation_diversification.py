import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

# ============================================================
# 09 - Correlation, Covariance & Diversification
# ============================================================

# 1. Download data
tickers = ["AAPL", "MSFT", "SPY", "GLD"]
start_date = "2018-01-01"

data = yf.download(tickers, start=start_date, auto_adjust=True, progress=False)["Close"]
data = data.dropna()

# 2. Compute returns
returns = data.pct_change().dropna()

# 3. Covariance matrix
cov_matrix = returns.cov()

# 4. Correlation matrix
corr_matrix = returns.corr()

print("Covariance Matrix:")
print(cov_matrix)

print("\nCorrelation Matrix:")
print(corr_matrix)

# ------------------------------------------------------------
# 5. Diversification Example (2-asset portfolio)
# ------------------------------------------------------------

# Choose two assets
r1 = returns["AAPL"]
r2 = returns["GLD"]

# Equal weights
w1 = 0.5
w2 = 0.5

# Individual vol
vol_1 = r1.std()
vol_2 = r2.std()

# Portfolio return
portfolio_returns = w1 * r1 + w2 * r2

# Portfolio vol
portfolio_vol = portfolio_returns.std()

print("\nVolatility Comparison:")
print(f"AAPL Volatility: {vol_1:.4f}")
print(f"GLD Volatility: {vol_2:.4f}")
print(f"Portfolio Volatility (50/50): {portfolio_vol:.4f}")

# ------------------------------------------------------------
# 6. Plot Correlation Heatmap
# ------------------------------------------------------------

plt.figure()

# simple heatmap using imshow
plt.imshow(corr_matrix, cmap="coolwarm", aspect="auto")

plt.colorbar(label="Correlation")

plt.xticks(range(len(tickers)), tickers)
plt.yticks(range(len(tickers)), tickers)

plt.title("Correlation Matrix Heatmap")

plt.tight_layout()
plt.savefig("correlation_heatmap.png", dpi=200)
plt.show()
