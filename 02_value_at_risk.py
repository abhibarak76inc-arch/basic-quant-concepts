import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# 1) Create returns (synthetic)
# -----------------------------
np.random.seed(42)
n_days = 1000

# Example: average daily return ~ 0.05% and daily volatility ~ 2%
mu = 0.0005
sigma = 0.02

returns = np.random.normal(mu, sigma, n_days)

# Confidence levels
cl_95 = 0.95
cl_99 = 0.99

alpha_95 = 1 - cl_95
alpha_99 = 1 - cl_99

# -----------------------------
# 2) Historical VaR
# -----------------------------
var_hist_95 = np.quantile(returns, alpha_95)
var_hist_99 = np.quantile(returns, alpha_99)

# -----------------------------
# 3) Parametric (Gaussian) VaR
# VaR = mean + z_alpha * std
# -----------------------------
mean_r = returns.mean()
std_r = returns.std(ddof=1)

z_95 = np.quantile(np.random.normal(0, 1, 200000), alpha_95)  # approx z
z_99 = np.quantile(np.random.normal(0, 1, 200000), alpha_99)

var_para_95 = mean_r + z_95 * std_r
var_para_99 = mean_r + z_99 * std_r

# -----------------------------
# 4) Monte Carlo VaR
# -----------------------------
mc_sims = 200000
mc_returns = np.random.normal(mean_r, std_r, mc_sims)

var_mc_95 = np.quantile(mc_returns, alpha_95)
var_mc_99 = np.quantile(mc_returns, alpha_99)

# -----------------------------
# 5) Print results
# (VaR is a loss threshold, so it's typically reported as positive number)
# -----------------------------
print("VaR results (1-day). Reported as positive loss numbers:")
print(f"Historical VaR 95%: {-var_hist_95:.4f}")
print(f"Historical VaR 99%: {-var_hist_99:.4f}\n")

print(f"Parametric VaR 95%: {-var_para_95:.4f}")
print(f"Parametric VaR 99%: {-var_para_99:.4f}\n")

print(f"Monte Carlo VaR 95%: {-var_mc_95:.4f}")
print(f"Monte Carlo VaR 99%: {-var_mc_99:.4f}")

# -----------------------------
# 6) Plot histogram + VaR lines
# -----------------------------
plt.figure()
plt.hist(returns, bins=50)

# lines for historical VaR
plt.axvline(var_hist_95, linestyle="--", label="Hist VaR 95%")
plt.axvline(var_hist_99, linestyle="--", label="Hist VaR 99%")

plt.title("Return Distribution with Historical VaR")
plt.xlabel("Daily Return")
plt.ylabel("Frequency")
plt.legend()
plt.tight_layout()

# Save image for GitHub
plt.savefig("value_at_risk_plot.png", dpi=200)
plt.show()
