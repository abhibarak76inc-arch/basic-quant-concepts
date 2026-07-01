import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# ============================================================
# 14 - Black-Scholes Option Pricing
# ============================================================

# Parameters
S = 100      # Stock Price
K = 100      # Strike Price
T = 1        # Time to Maturity (years)
r = 0.05     # Risk-Free Rate
sigma = 0.20 # Volatility

# ------------------------------------------------------------
# Calculate d1 and d2
# ------------------------------------------------------------

d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
d2 = d1 - sigma * np.sqrt(T)

# ------------------------------------------------------------
# Black-Scholes formulas
# ------------------------------------------------------------

call_price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

put_price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

print("Black-Scholes Option Pricing")
print("----------------------------")
print(f"Call Option Price : {call_price:.2f}")
print(f"Put Option Price  : {put_price:.2f}")

# ------------------------------------------------------------
# Sensitivity to Stock Price
# ------------------------------------------------------------

stock_prices = np.linspace(50, 150, 100)

call_values = []

for price in stock_prices:

    d1 = (np.log(price / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    value = price * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

    call_values.append(value)

# ------------------------------------------------------------
# Plot
# ------------------------------------------------------------

plt.figure(figsize=(8,5))

plt.plot(stock_prices, call_values)

plt.title("Black-Scholes Call Option Value")
plt.xlabel("Stock Price")
plt.ylabel("Call Option Price")

plt.grid(True)

plt.tight_layout()

plt.savefig("black_scholes_option.png", dpi=200)

plt.show()
