import numpy as np
import math

# Parameters
lam = 0.13          # lambda (mean rate)
k = 2               # number of events
n_sim = 1_000_000   # number of simulations (increase for better accuracy)

# 1. Theoretical probability using Poisson PMF
theoretical_prob = (math.exp(-lam) * (lam ** k)) / math.factorial(k)

# 2. Simulate Poisson samples
np.random.seed(42)  # for reproducibility
samples = np.random.poisson(lam=lam, size=n_sim)

# 3. Empirical probability: proportion of samples equal to k
empirical_prob = np.mean(samples == k)

# 4. Output results
print(f"Theoretical P(K = {k}) = {theoretical_prob:.6f}")
print(f"Empirical   P(K = {k}) = {empirical_prob:.6f}")
print(f"Difference             = {abs(theoretical_prob - empirical_prob):.6f}")

# Optional: Show distribution of rare events
print(f"\nNote: With λ = {lam}, most samples are 0 or 1.")
print(f"Sample counts: 0 → {np.sum(samples == 0):,}, 1 → {np.sum(samples == 1):,}, 2 → {np.sum(samples == 2):,}")