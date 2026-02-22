import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

# -----------------------------
# 1. Download Data
# -----------------------------
tickers = ["AAPL", "MSFT", "SPY", "GLD", "TLT"]
start_date = "2018-01-01"

data = yf.download(tickers, start=start_date, auto_adjust=True, progress=False)["Close"]
data = data.dropna()

# -----------------------------
# 2. Compute Returns
# -----------------------------
returns = data.pct_change().dropna()

mean_returns = returns.mean() * 252        # annualized return
cov_matrix = returns.cov() * 252           # annualized covariance

rf = 0.02  # risk-free rate

# -----------------------------
# 3. Monte Carlo Simulation
# -----------------------------
n_portfolios = 10000

results = np.zeros((3, n_portfolios))
weights_record = []

for i in range(n_portfolios):
    weights = np.random.random(len(tickers))
    weights /= np.sum(weights)

    weights_record.append(weights)

    portfolio_return = np.dot(weights, mean_returns)
    portfolio_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    sharpe = (portfolio_return - rf) / portfolio_vol

    results[0, i] = portfolio_return
    results[1, i] = portfolio_vol
    results[2, i] = sharpe

# -----------------------------
# 4. Identify Optimal Portfolios
# -----------------------------
max_sharpe_idx = np.argmax(results[2])
min_vol_idx = np.argmin(results[1])

max_sharpe_weights = weights_record[max_sharpe_idx]
min_vol_weights = weights_record[min_vol_idx]

# -----------------------------
# 5. Print Results
# -----------------------------
print("Maximum Sharpe Portfolio")
print("-------------------------")
print("Return:", round(results[0, max_sharpe_idx], 4))
print("Volatility:", round(results[1, max_sharpe_idx], 4))
print("Sharpe:", round(results[2, max_sharpe_idx], 4))
print("Weights:")
for ticker, weight in zip(tickers, max_sharpe_weights):
    print(f"{ticker}: {round(weight,4)}")

print("\nMinimum Volatility Portfolio")
print("-----------------------------")
print("Return:", round(results[0, min_vol_idx], 4))
print("Volatility:", round(results[1, min_vol_idx], 4))
print("Sharpe:", round(results[2, min_vol_idx], 4))
print("Weights:")
for ticker, weight in zip(tickers, min_vol_weights):
    print(f"{ticker}: {round(weight,4)}")

# -----------------------------
# 6. Plot Efficient Frontier
# -----------------------------
plt.figure()
plt.scatter(results[1], results[0], c=results[2])
plt.scatter(results[1, max_sharpe_idx], results[0, max_sharpe_idx], marker="*", s=300)
plt.scatter(results[1, min_vol_idx], results[0, min_vol_idx], marker="*", s=300)

plt.xlabel("Volatility")
plt.ylabel("Expected Return")
plt.title("Efficient Frontier (Monte Carlo Simulation)")
plt.tight_layout()

plt.savefig("efficient_frontier.png", dpi=200)
plt.show()
