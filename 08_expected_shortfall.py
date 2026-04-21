import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# 08 - Expected Shortfall (CVaR)
# ============================================================

np.random.seed(42)

# Simulate daily returns
n_days = 1000
mu = 0.0005
sigma = 0.02
returns = np.random.normal(mu, sigma, n_days)

# Confidence level
cl = 0.95
alpha = 1 - cl

# ------------------------------------------------------------
# 1. Historical VaR
# ------------------------------------------------------------
var_hist = np.quantile(returns, alpha)

# Historical CVaR = average of returns worse than VaR
cvar_hist = returns[returns <= var_hist].mean()

# ------------------------------------------------------------
# 2. Parametric (Gaussian) VaR and CVaR
# ------------------------------------------------------------
mean_r = returns.mean()
std_r = returns.std(ddof=1)

# approximate z-score using simulation
z_alpha = np.quantile(np.random.normal(0, 1, 200000), alpha)

# Parametric VaR
var_para = mean_r + z_alpha * std_r

# Parametric CVaR via Monte Carlo approximation from same Gaussian model
gaussian_sims = np.random.normal(mean_r, std_r, 200000)
cvar_para = gaussian_sims[gaussian_sims <= var_para].mean()

# ------------------------------------------------------------
# 3. Print results
# ------------------------------------------------------------
print("Expected Shortfall / CVaR Results")
print("---------------------------------")
print(f"Historical VaR 95%: {-var_hist:.4f}")
print(f"Historical CVaR 95%: {-cvar_hist:.4f}")
print()
print(f"Parametric VaR 95%: {-var_para:.4f}")
print(f"Parametric CVaR 95%: {-cvar_para:.4f}")

# ------------------------------------------------------------
# 4. Plot
# ------------------------------------------------------------
plt.figure()
plt.hist(returns, bins=50)

plt.axvline(var_hist, linestyle="--", label="Historical VaR 95%")
plt.axvline(cvar_hist, linestyle=":", label="Historical CVaR 95%")
plt.axvline(var_para, linestyle="--", label="Parametric VaR 95%")
plt.axvline(cvar_para, linestyle=":", label="Parametric CVaR 95%")

plt.title("Expected Shortfall (CVaR) vs VaR")
plt.xlabel("Daily Return")
plt.ylabel("Frequency")
plt.legend()
plt.tight_layout()
plt.savefig("expected_shortfall_plot.png", dpi=200)
plt.show()
