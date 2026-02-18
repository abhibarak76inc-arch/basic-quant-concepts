import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# -----------------------------
# 03 - Volatility Scaling (sqrt(T))
# -----------------------------
# Idea:
# If daily returns are i.i.d. with volatility sigma_1d,
# then volatility over T days scales roughly as sigma_T = sigma_1d * sqrt(T)

np.random.seed(42)

# Simulate daily returns (e.g., ~10 years of trading days)
TRADING_DAYS = 252
years = 10
n = TRADING_DAYS * years

mu_daily = 0.0003     # ~0.03% average daily return (toy assumption)
sigma_daily_true = 0.01  # 1% daily volatility (toy assumption)

daily_returns = pd.Series(np.random.normal(mu_daily, sigma_daily_true, n))

# 1) Estimate daily volatility from the simulated sample
sigma_1d = daily_returns.std(ddof=1)

# 2) Define horizons to test
horizons = [1, 5, 21, 63]  # 1d, 1w, 1m, 1q (approx)

realized_vols = []
predicted_vols = []

for T in horizons:
    if T == 1:
        agg = daily_returns
    else:
        # Aggregate by summing returns over T-day blocks (approx for small returns)
        agg = daily_returns.groupby(np.arange(len(daily_returns)) // T).sum()

    realized_sigma = agg.std(ddof=1)
    predicted_sigma = sigma_1d * np.sqrt(T)

    realized_vols.append(realized_sigma)
    predicted_vols.append(predicted_sigma)

# 3) Print results
print("Estimated daily volatility (sigma_1d):", round(float(sigma_1d), 6))
print("\nHorizon (days) | Realized Vol | Predicted Vol (sqrt(T))")
for T, rv, pv in zip(horizons, realized_vols, predicted_vols):
    print(f"{T:>12} | {rv:>12.6f} | {pv:>20.6f}")

# 4) Plot realized vs predicted
plt.figure()
plt.plot(horizons, realized_vols, marker="o", label="Realized volatility")
plt.plot(horizons, predicted_vols, marker="o", label="sqrt(T) prediction")

plt.title("Volatility Scaling: Realized vs sqrt(T) Prediction")
plt.xlabel("Horizon (days)")
plt.ylabel("Volatility (std of aggregated returns)")
plt.xticks(horizons)
plt.legend()
plt.tight_layout()

# Save for GitHub
plt.savefig("vol_scaling.png", dpi=200)
plt.show()
