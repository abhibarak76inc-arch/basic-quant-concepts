import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

# -----------------------------
# 06 - Rolling Risk Dashboard
# -----------------------------

TICKER = "AAPL"
START = "2018-01-01"
RF_ANNUAL = 0.02
TRADING_DAYS = 252

# 1) Download adjusted prices
df = yf.download(TICKER, start=START, auto_adjust=True, progress=False)
if df.empty:
    raise ValueError("No data downloaded. Check internet/ticker.")

prices = df["Close"].dropna()

# 2) Returns
rets = prices.pct_change().dropna()

# Convert annual risk-free rate to daily
rf_daily = (1 + RF_ANNUAL) ** (1 / TRADING_DAYS) - 1
excess = rets - rf_daily

# 3) Helper functions
def rolling_volatility(r, window):
    # annualized rolling vol
    return r.rolling(window).std(ddof=1) * np.sqrt(TRADING_DAYS)

def rolling_sharpe(excess_r, window):
    # annualized rolling Sharpe
    mean = excess_r.rolling(window).mean() * TRADING_DAYS
    vol = excess_r.rolling(window).std(ddof=1) * np.sqrt(TRADING_DAYS)
    return mean / vol

def rolling_max_drawdown(r, window):
    """
    Rolling max drawdown over a window:
    - create equity curve within each window
    - compute drawdown within that window
    - take minimum drawdown (most negative)
    """
    eq = (1 + r).cumprod()
    # Rolling peak and drawdown
    roll_peak = eq.rolling(window, min_periods=window).max()
    dd = eq / roll_peak - 1.0
    # Rolling min drawdown
    return dd.rolling(window, min_periods=window).min()

# 4) Compute rolling metrics
vol_30 = rolling_volatility(rets, 30)
sharpe_60 = rolling_sharpe(excess, 60)
mdd_252 = rolling_max_drawdown(rets, 252)

# 5) Plot dashboard (3 charts)
plt.figure(figsize=(10, 8))

# Equity curve
plt.subplot(3, 1, 1)
equity = (1 + rets).cumprod()
plt.plot(equity)
plt.title(f"{TICKER} - Equity Curve (Growth of $1)")
plt.ylabel("Value")

# Rolling volatility
plt.subplot(3, 1, 2)
plt.plot(vol_30)
plt.title("Rolling 30D Volatility (Annualized)")
plt.ylabel("Vol")

# Rolling Sharpe + Rolling Max Drawdown
plt.subplot(3, 1, 3)
plt.plot(sharpe_60, label="Rolling 60D Sharpe")
plt.plot(mdd_252, label="Rolling 252D Max Drawdown")
plt.title("Rolling Risk Metrics")
plt.ylabel("Value")
plt.legend()

plt.tight_layout()
plt.savefig("rolling_risk_dashboard.png", dpi=200)
plt.show()

print("Latest Rolling 30D Volatility:", float(vol_30.dropna().iloc[-1]))
print("Latest Rolling 60D Sharpe:", float(sharpe_60.dropna().iloc[-1]))
print("Latest Rolling 252D Max Drawdown:", float(mdd_252.dropna().iloc[-1]))
