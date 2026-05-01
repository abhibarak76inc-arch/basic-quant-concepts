import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

# ============================================================
# 13 - Backtesting with Transaction Costs
# ============================================================

TICKER = "AAPL"
START = "2015-01-01"
LOOKBACK = 20
TRADING_DAYS = 252

# Transaction cost (e.g., 0.1% per trade)
COST = 0.001

# 1. Download data
data = yf.download(TICKER, start=START, auto_adjust=True, progress=False)
prices = data["Close"].squeeze().dropna()

# 2. Returns
returns = prices.pct_change().dropna()

# 3. Momentum signal
momentum = prices.pct_change(LOOKBACK)
signal = (momentum > 0).astype(int)

# 4. Avoid look-ahead bias
signal = signal.shift(1).fillna(0)

# 5. Strategy returns (no cost)
strategy_returns = signal * returns

# ------------------------------------------------------------
# 6. Transaction cost logic
# ------------------------------------------------------------

# Detect trades
trades = signal.diff().abs()

# Apply cost when trade happens
costs = trades * COST

# Strategy returns after cost
strategy_returns_cost = strategy_returns - costs

# ------------------------------------------------------------
# 7. Equity curves
# ------------------------------------------------------------

equity_no_cost = (1 + strategy_returns).cumprod()
equity_with_cost = (1 + strategy_returns_cost).cumprod()

# ------------------------------------------------------------
# 8. Metrics
# ------------------------------------------------------------

def compute_metrics(equity, returns):
    ann_return = equity.iloc[-1] ** (TRADING_DAYS / len(equity)) - 1
    ann_vol = returns.std(ddof=1) * np.sqrt(TRADING_DAYS)

    if ann_vol == 0:
        sharpe = np.nan
    else:
        sharpe = ann_return / ann_vol

    max_dd = (equity / equity.cummax() - 1).min()

    return ann_return, ann_vol, sharpe, max_dd


r1 = compute_metrics(equity_no_cost, strategy_returns)
r2 = compute_metrics(equity_with_cost, strategy_returns_cost)

print("WITHOUT COSTS")
print("-------------")
print(f"Return: {r1[0]:.2%}, Sharpe: {r1[2]:.2f}, Max DD: {r1[3]:.2%}")

print("\nWITH COSTS")
print("-----------")
print(f"Return: {r2[0]:.2%}, Sharpe: {r2[2]:.2f}, Max DD: {r2[3]:.2%}")

# ------------------------------------------------------------
# 9. Plot
# ------------------------------------------------------------

plt.figure()

plt.plot(equity_no_cost, label="No Cost")
plt.plot(equity_with_cost, label="With Cost")

plt.title("Impact of Transaction Costs")
plt.xlabel("Date")
plt.ylabel("Growth of $1")
plt.legend()

plt.tight_layout()
plt.savefig("transaction_costs.png", dpi=200)
plt.show()
