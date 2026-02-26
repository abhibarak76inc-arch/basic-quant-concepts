import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

# -----------------------------
# 07 - Pairs Trading (Z-score Spread)
# -----------------------------
# Pair: KO vs PEP
# Idea:
# - If two stocks move together, their price ratio tends to revert to a mean.
# - When ratio is "too high" (z-score large), short the ratio (short KO, long PEP).
# - When ratio is "too low" (z-score very negative), long the ratio (long KO, short PEP).

TICKER_A = "KO"
TICKER_B = "PEP"
START = "2018-01-01"

LOOKBACK = 60      # rolling window for mean/std of ratio
ENTRY_Z = 2.0      # enter when |z| > 2
EXIT_Z = 0.5       # exit when |z| < 0.5

# 1) Download adjusted prices
df = yf.download([TICKER_A, TICKER_B], start=START, auto_adjust=True, progress=False)["Close"]
df = df.dropna()
df.columns = [TICKER_A, TICKER_B]

# 2) Compute ratio and z-score
ratio = df[TICKER_A] / df[TICKER_B]
rolling_mean = ratio.rolling(LOOKBACK).mean()
rolling_std = ratio.rolling(LOOKBACK).std(ddof=1)
zscore = (ratio - rolling_mean) / rolling_std

# 3) Create trading signals
# Position = +1 means long ratio (long A, short B)
# Position = -1 means short ratio (short A, long B)
position = pd.Series(0, index=df.index)

in_trade = 0  # 0=no position, +1=long ratio, -1=short ratio

for i in range(len(df)):
    date = df.index[i]
    z = zscore.iloc[i]

    if np.isnan(z):
        position.iloc[i] = 0
        continue

    # If not in trade, check entry
    if in_trade == 0:
        if z > ENTRY_Z:
            in_trade = -1  # short ratio
        elif z < -ENTRY_Z:
            in_trade = +1  # long ratio

    # If in trade, check exit
    else:
        if abs(z) < EXIT_Z:
            in_trade = 0

    position.iloc[i] = in_trade

# 4) Compute strategy returns
# Use daily returns of each asset
ret_a = df[TICKER_A].pct_change().fillna(0)
ret_b = df[TICKER_B].pct_change().fillna(0)

# Long ratio: +ret_a - ret_b
# Short ratio: -ret_a + ret_b
strategy_ret = position.shift(1).fillna(0) * (ret_a - ret_b)

equity = (1 + strategy_ret).cumprod()

# 5) Basic performance stats
ann_return = equity.iloc[-1] ** (252 / len(equity)) - 1
ann_vol = strategy_ret.std(ddof=1) * np.sqrt(252)
sharpe = ann_return / ann_vol if ann_vol != 0 else np.nan
max_dd = (equity / equity.cummax() - 1).min()

print("Pairs Trading Strategy (Z-score Spread)")
print("--------------------------------------")
print(f"Pair: {TICKER_A} / {TICKER_B}")
print(f"Lookback: {LOOKBACK}, Entry Z: {ENTRY_Z}, Exit Z: {EXIT_Z}")
print(f"Annualized Return: {ann_return:.2%}")
print(f"Annualized Vol: {ann_vol:.2%}")
print(f"Sharpe (approx): {sharpe:.2f}")
print(f"Max Drawdown: {max_dd:.2%}")
print(f"Number of trade-days (in position): {int((position != 0).sum())}")

# 6) Plot: ratio + z-score + equity
plt.figure(figsize=(10, 8))

plt.subplot(3, 1, 1)
plt.plot(df[TICKER_A], label=TICKER_A)
plt.plot(df[TICKER_B], label=TICKER_B)
plt.title("Prices (Adjusted)")
plt.legend()

plt.subplot(3, 1, 2)
plt.plot(zscore, label="Z-score of Ratio")
plt.axhline(ENTRY_Z, linestyle="--", label="Entry +2")
plt.axhline(-ENTRY_Z, linestyle="--", label="Entry -2")
plt.axhline(EXIT_Z, linestyle=":", label="Exit ±0.5")
plt.axhline(-EXIT_Z, linestyle=":")
plt.title(f"Z-score of Ratio: {TICKER_A}/{TICKER_B}")
plt.legend()

plt.subplot(3, 1, 3)
plt.plot(equity, label="Strategy Equity Curve")
plt.title("Pairs Trading Equity Curve")
plt.legend()

plt.tight_layout()
plt.savefig("pairs_trading_spread.png", dpi=200)
plt.show()
