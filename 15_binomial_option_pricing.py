import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# 15 - Binomial Option Pricing Model
# ============================================================

# Parameters
S = 100          # Current stock price
K = 100          # Strike price
T = 1            # Time to maturity
r = 0.05         # Risk-free rate
sigma = 0.20     # Volatility
N = 100          # Number of time steps

# Time step
dt = T / N

# Up and down factors
u = np.exp(sigma * np.sqrt(dt))
d = 1 / u

# Risk-neutral probability
p = (np.exp(r * dt) - d) / (u - d)

# Discount factor
discount = np.exp(-r * dt)

# ------------------------------------------------------------
# Stock prices at maturity
# ------------------------------------------------------------

stock_prices = np.array([
    S * (u ** (N - i)) * (d ** i)
    for i in range(N + 1)
])

# ------------------------------------------------------------
# Call option payoff at maturity
# ------------------------------------------------------------

option_values = np.maximum(stock_prices - K, 0)

# ------------------------------------------------------------
# Backward induction
# ------------------------------------------------------------

for step in range(N - 1, -1, -1):
    option_values = discount * (
        p * option_values[:-1] +
        (1 - p) * option_values[1:]
    )

call_price = option_values[0]

print("Binomial Option Pricing")
print("-----------------------")
print(f"Call Option Price: {call_price:.4f}")

# ------------------------------------------------------------
# Plot option value vs stock price at maturity
# ------------------------------------------------------------

plt.figure(figsize=(8,5))

plt.plot(stock_prices,
         np.maximum(stock_prices-K,0),
         linewidth=2)

plt.title("Call Option Payoff at Expiration")
plt.xlabel("Stock Price at Expiration")
plt.ylabel("Option Payoff")

plt.grid(True)

plt.tight_layout()

plt.savefig("binomial_option_payoff.png", dpi=200)

plt.show()
