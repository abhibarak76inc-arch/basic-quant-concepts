import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

# ============================================================
# 12 - Backtesting Framework (Momentum Strategy)
# ============================================================

TICKER = "AAPL"
START = "2015-01-01"
LOOKBACK = 20
TRADING_DAYS = 252

# 1. Download data
data = yf.download(TICKER, start=START, auto_adjust=True, progress=False)

# FIX: ensure Series (not DataFrame)
prices = data["Close"].squeeze().dropna()

# 2. Compute returns
returns = prices.pct_change().dropna()

# 3. Momentum signal
momentum = prices.pct_change(LOOKBACK)
signal = (momentum > 0).astype(int)

# 4. Avoid look-ahead bias
signal = signal.shift(1).fillna(0)

# 5. Strategy returns
strategy_returns = signal * returns

# 6. Equity curves
equity_strategy = (1 + strategy_returns).cumprod()
equity_buy_hold = (1 + returns).cumprod()

# 7. Performance metrics

# Annual return
ann_return = equity_strategy.iloc[-1] ** (TRADING_DAYS / len(equity_strategy)) - 1

# Volatility
ann_vol = strategy_returns.std(ddof=1) * np.sqrt(TRADING_DAYS)

# FIX: avoid ambiguous boolean
if ann_vol == 0:
    sharpe = np.nan
else:
    sharpe = ann_return / ann_vol

# Max drawdown
max_dd = (equity_strategy / equity_strategy.cummax() - 1).min()

print("Momentum Strategy Results")
print("--------------------------")
print(f"Annual Return: {ann_return:.2%}")
print(f"Annual Volatility: {ann_vol:.2%}")
print(f"Sharpe Ratio: {sharpe:.2f}")
print(f"Max Drawdown: {max_dd:.2%}")

# 8. Plot equity curves
plt.figure()

plt.plot(equity_strategy, label="Momentum Strategy")
plt.plot(equity_buy_hold, label="Buy & Hold")

plt.title("Backtest: Momentum Strategy vs Buy & Hold")
plt.xlabel("Date")
plt.ylabel("Growth of $1")
plt.legend()

plt.tight_layout()
plt.savefig("momentum_backtest.png", dpi=200)
plt.show()
