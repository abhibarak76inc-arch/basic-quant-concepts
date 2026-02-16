import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Create synthetic price series
np.random.seed(42)
days = 252
prices = 100 * np.exp(np.cumsum(np.random.normal(0.0005, 0.02, days)))

prices = pd.Series(prices)

# Compute simple returns
simple_returns = prices.pct_change().dropna()

# Compute log returns
log_returns = np.log(prices / prices.shift(1)).dropna()

# Cumulative returns
cum_simple = (1 + simple_returns).cumprod()
cum_log = np.exp(log_returns.cumsum())

# Plot
plt.figure()
plt.plot(cum_simple, label="Cumulative Simple Returns", linewidth=2)
plt.plot(cum_log, label="Cumulative Log Returns", linewidth=2, alpha=0.6)
plt.legend()
plt.title("Simple vs Log Returns")
plt.show()
