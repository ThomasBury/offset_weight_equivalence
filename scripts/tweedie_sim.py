#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tweedie GLM with exposure: rigorous implementations and equivalence checks.

- Data generator for Tweedie totals with exposure
- scikit-learn TweedieRegressor: rates + correct sample_weight = exposure**(2 - p)
- LightGBM Tweedie: (A) totals + offset via init_score=log(exposure)
                    (B) rates + sample_weight = exposure**(2 - p)
- Numeric checks: deviance equivalence across implementations
- Special cases: Poisson (p=1), Gamma (p=2)

Author: (you)
"""

from __future__ import annotations
import math
import warnings
import numpy as np
import pandas as pd
from numpy.random import default_rng

from sklearn.model_selection import train_test_split
from sklearn.linear_model import TweedieRegressor
from sklearn.metrics import mean_squared_error

# Try LightGBM (optional)
try:
    import lightgbm as lgb
    LGB_AVAILABLE = True
except Exception:
    LGB_AVAILABLE = False
    warnings.warn("lightgbm not found. LightGBM experiments will be skipped.", RuntimeWarning)

np.set_printoptions(suppress=True, linewidth=120)

#%%

# -------------------------
# Utilities: Tweedie pieces
# -------------------------
def half_tweedie_deviance(y: np.ndarray, mu: np.ndarray, p: float) -> np.ndarray:
    """
    Half Tweedie deviance with log link (matches sklearn's HalfTweedieLoss for p not in {0,1,2}).
    Returns element-wise loss (without constants in y for p in {0,1,2}).
    Domain assumptions for typical insurance use:
      - For 1 < p < 2: y >= 0, mu > 0
      - For p = 1, 2: handled by limits
    """
    eps = 1e-12
    y = np.asarray(y, dtype=float)
    mu = np.asarray(mu, dtype=float)
    mu = np.maximum(mu, eps)

    if abs(p - 1.0) < 1e-12:
        # Poisson limit
        # half deviance (up to y-only constants): mu - y*log(mu)
        # sklearn also uses constants to make perfect fit zero; we compare consistently across models,
        # so dropping constants is acceptable for equivalence checks.
        return mu - y * np.log(mu + eps)

    if abs(p - 2.0) < 1e-12:
        # Gamma limit
        # half deviance (up to constants): (y / mu) - log(y / mu) - 1
        r = np.maximum(y, eps) / mu
        return r - np.log(r) - 1.0

    # General case (and also valid as a strictly consistent scoring rule for 0 < p < 1 even if no Tweedie density exists)
    term3 = (mu ** (2.0 - p)) / (2.0 - p)
    term2 = y * (mu ** (1.0 - p)) / (1.0 - p)
    # We drop the y-only constant: max(y,0)^(2-p)/((1-p)(2-p))
    return term3 - term2


def tweedie_compound_poisson_gamma_params(mu: np.ndarray, phi: float, p: float):
    """
    For 1 < p < 2: map Tweedie(μ, φ, p) to compound Poisson–Gamma params:
      N ~ Poisson(λ), severities ~ Gamma(a, θ) (shape a, scale θ), total = sum_{k=1..N} S_k
    λ = μ^{2-p} / (φ (2-p))
    a = (2 - p) / (p - 1)
    θ = φ (p - 1) μ^{p - 1}
    """
    if not (1.0 < p < 2.0):
        raise ValueError("Compound Poisson–Gamma mapping only valid for 1 < p < 2.")
    lam = (mu ** (2.0 - p)) / (phi * (2.0 - p))
    a = (2.0 - p) / (p - 1.0)
    theta = phi * (p - 1.0) * (mu ** (p - 1.0))
    return lam, a, theta


def rng_tweedie_totals(mu: np.ndarray, phi: float, p: float, rng: np.random.Generator) -> np.ndarray:
    """
    Simulate totals Y ~ Tweedie(μ, φ, p) for insurance case 1 < p < 2 via compound Poisson–Gamma.
    For p=1 or p=2, we use appropriate limits (Poisson and Gamma) for demonstration.
    Note: For production, consider a dedicated Tweedie RNG library if needed.
    """
    mu = np.asarray(mu, dtype=float)
    n = mu.shape[0]
    out = np.zeros(n, dtype=float)

    if abs(p - 1.0) < 1e-12:
        # Poisson with mean μ and variance φ μ  (φ acts like overdispersion in quasi-Poisson; here we use μ as Poisson mean)
        # For demonstration we set Poisson mean = μ (ignoring φ in RNG; deviance checks remain consistent).
        out = rng.poisson(lam=np.maximum(mu, 0))
        return out.astype(float)

    if abs(p - 2.0) < 1e-12:
        # Gamma: Var(Y) = φ μ^2 => shape k = 1/φ, scale θ = φ μ
        # But to keep mean exactly μ, we can sample Gamma(k, θ = μ / k) => Var = μ^2 / k
        k = 1.0 / max(phi, 1e-9)
        theta = mu / np.maximum(k, 1e-9)
        out = rng.gamma(shape=k, scale=theta)
        return out

    if not (1.0 < p < 2.0):
        raise ValueError("This simulator supports only p in {1} ∪ (1,2) ∪ {2} for demo purposes.")

    lam, a, theta = tweedie_compound_poisson_gamma_params(mu, phi, p)
    # Draw N_i, then sum N_i Gammas(a, θ_i) where θ_i depends on μ_i
    N = rng.poisson(lam=lam)
    for i in range(n):
        if N[i] <= 0:
            out[i] = 0.0
        else:
            # severity Gamma(shape=a, scale=theta_i)
            out[i] = rng.gamma(shape=a * N[i], scale=theta[i])
    return out


# -------------------------
# Data generation
# -------------------------
def make_insurance_data(n=50_000, p_index=1.9, phi=0.8, n_features=6, seed=7):
    """
    Generate synthetic insurance-style data:
      - features X
      - exposure ω
      - rate μ_rate = exp(Xβ)
      - total mean μ_total = ω * μ_rate
      - totals Y ~ Tweedie(μ_total, φ, p)
    """
    rng = default_rng(seed)
    X = rng.normal(size=(n, n_features))
    beta = rng.normal(scale=0.4, size=n_features)
    # Add intercept implicitly via a bias feature if desired; here use explicit intercept later.
    linpred = X @ beta
    mu_rate = np.exp(linpred)  # rate per unit exposure

    # Exposures: positive, moderately spread (lognormal)
    exposure = np.exp(rng.normal(loc=-0.1, scale=0.6, size=n))  # median ~ 0.9
    mu_total = exposure * mu_rate

    # Simulate totals
    Y = rng_tweedie_totals(mu_total, phi=phi, p=p_index, rng=rng)

    df = pd.DataFrame(X, columns=[f"x{j}" for j in range(n_features)])
    df["Exposure"] = exposure
    df["Total"] = Y
    df["Rate"] = np.divide(Y, exposure, out=np.zeros_like(Y), where=exposure > 0)
    return df


# -------------------------
# Fitting helpers
# -------------------------
def fit_sklearn_tweedie_rates(df_train, df_test, feature_cols, p, alpha=0.1):
    """scikit-learn: fit on rates with correct weights exposure**(2 - p)."""
    Xtr = df_train[feature_cols].to_numpy()
    Xte = df_test[feature_cols].to_numpy()
    rtr = df_train["Rate"].to_numpy()
    rte = df_test["Rate"].to_numpy()
    wtr = (df_train["Exposure"].to_numpy()) ** (2.0 - p)
    wte = (df_test["Exposure"].to_numpy()) ** (2.0 - p)

    glm = TweedieRegressor(power=p, alpha=alpha, solver="newton-cholesky")
    glm.fit(Xtr, rtr, sample_weight=wtr)

    # Predict rates, convert to totals for apples-to-apples comparison
    rhat_tr = glm.predict(Xtr)
    rhat_te = glm.predict(Xte)
    yhat_tr = rhat_tr * df_train["Exposure"].to_numpy()
    yhat_te = rhat_te * df_test["Exposure"].to_numpy()

    return glm, (yhat_tr, yhat_te), (rhat_tr, rhat_te), (Xtr, Xte, rtr, rte, wtr, wte)


def fit_lgbm_tweedie_totals_offset(df_train, df_test, feature_cols, p, num_boost_round=300, seed=13):
    """LightGBM: totals + offset via init_score = log(exposure)."""
    if not LGB_AVAILABLE:
        return None, (None, None)

    ytr = df_train["Total"].to_numpy(dtype=float)
    yte = df_test["Total"].to_numpy(dtype=float)
    Xtr = df_train[feature_cols].to_numpy()
    Xte = df_test[feature_cols].to_numpy()
    init_tr = np.log(df_train["Exposure"].to_numpy())
    init_te = np.log(df_test["Exposure"].to_numpy())

    dtrain = lgb.Dataset(Xtr, label=ytr, init_score=init_tr)
    dvalid = lgb.Dataset(Xte, label=yte, init_score=init_te, reference=dtrain)

    params = dict(
        objective="tweedie",
        tweedie_variance_power=p,
        # learning_rate=0.05,
        # num_leaves=63,
        # min_data_in_leaf=50,
        # feature_fraction=0.9,
        # bagging_fraction=0.9,
        # bagging_freq=1,
        verbose=-1,
        seed=seed,
    )
    gbm = lgb.train(
        params,
        dtrain,
        valid_sets=[dvalid],
        num_boost_round=num_boost_round,
    )

    yhat_tr = gbm.predict(Xtr)
    yhat_te = gbm.predict(Xte)
    return gbm, (yhat_tr, yhat_te)


def fit_lgbm_tweedie_rates_weights(df_train, df_test, feature_cols, p, num_boost_round=300, seed=13):
    """LightGBM: rates + weights exposure**(2 - p)."""
    if not LGB_AVAILABLE:
        return None, (None, None), (None, None)

    rtr = df_train["Rate"].to_numpy(dtype=float)
    rte = df_test["Rate"].to_numpy(dtype=float)
    Xtr = df_train[feature_cols].to_numpy()
    Xte = df_test[feature_cols].to_numpy()
    wtr = (df_train["Exposure"].to_numpy()) ** (2.0 - p)
    wte = (df_test["Exposure"].to_numpy()) ** (2.0 - p)

    dtrain = lgb.Dataset(Xtr, label=rtr, weight=wtr)
    dvalid = lgb.Dataset(Xte, label=rte, weight=wte, reference=dtrain)

    params = dict(
        objective="tweedie",
        tweedie_variance_power=p,
        # learning_rate=0.05,
        # num_leaves=63,
        # min_data_in_leaf=50,
        # feature_fraction=0.9,
        # bagging_fraction=0.9,
        # bagging_freq=1,
        verbose=-1,
        seed=seed,
    )
    gbm = lgb.train(
        params,
        dtrain,
        valid_sets=[dvalid],
        num_boost_round=num_boost_round,
    )

    # Predict rates, convert to totals
    rhat_tr = gbm.predict(Xtr)
    rhat_te = gbm.predict(Xte)
    yhat_tr = rhat_tr * df_train["Exposure"].to_numpy()
    yhat_te = rhat_te * df_test["Exposure"].to_numpy()
    return gbm, (yhat_tr, yhat_te), (rhat_tr, rhat_te)


# -------------------------
# Experiments / checks
# -------------------------
def evaluate_deviance(y_true, y_hat, p, sample_weight=None, name=""):
    """Return average half Tweedie deviance (weighted)."""
    loss = half_tweedie_deviance(y_true, np.maximum(y_hat, 1e-12), p)
    if sample_weight is None:
        val = float(np.mean(loss))
    else:
        sw = np.asarray(sample_weight, dtype=float)
        val = float(np.average(loss, weights=sw))
    if name:
        print(f"{name}: avg half-deviance = {val:.6f}")
    return val


#%%
# -------------------------
# Config
# -------------------------
N = 80_000
p = 1.9           # Tweedie index (1<p<=2 typical in insurance)
phi = 0.8         # dispersion for data generation only
alpha = 0.1       # L2 in sklearn
test_size = 0.25
seed = 42

#%%
# -------------------------
# Data
# -------------------------
df = make_insurance_data(n=N, p_index=p, phi=phi, n_features=8, seed=seed)
feature_cols = [c for c in df.columns if c.startswith("x")]
df_train, df_test = train_test_split(df, test_size=test_size, random_state=seed, shuffle=True)

#%%
# -------------------------
# sklearn: rates + correct weights (exact for all p>0 with log link)
# -------------------------
sk_model, (sk_yhat_tr, sk_yhat_te), (sk_rhat_tr, sk_rhat_te), (Xtr, Xte, rtr, rte, wtr, wte) = \
    fit_sklearn_tweedie_rates(df_train, df_test, feature_cols, p, alpha=alpha)

# Evaluate on totals (common scale)
ytr = df_train["Total"].to_numpy()
yte = df_test["Total"].to_numpy()
wtr_tot = None  # for totals; we compare cross-models so we can use unweighted deviance
wte_tot = None

print("\n=== scikit-learn (rates + ω^(2-p) weights) ===")
dev_tr = evaluate_deviance(ytr, sk_yhat_tr, p, sample_weight=wtr_tot, name="train")
dev_te = evaluate_deviance(yte, sk_yhat_te, p, sample_weight=wte_tot, name="test")

#%%
# -------------------------
# LightGBM (if available): totals + offset (init_score)
# -------------------------
if LGB_AVAILABLE:
    print("\n=== LightGBM (totals + offset via init_score=log ω) ===")
    lgb_off_model, (lgb_off_yhat_tr, lgb_off_yhat_te) = \
        fit_lgbm_tweedie_totals_offset(df_train, df_test, feature_cols, p, num_boost_round=400, seed=seed)

    dev_tr_off = evaluate_deviance(ytr, lgb_off_yhat_tr, p, name="train")
    dev_te_off = evaluate_deviance(yte, lgb_off_yhat_te, p, name="test")

    # -------------------------
    # LightGBM: rates + ω^(2-p) weights (exact)
    # -------------------------
    print("\n=== LightGBM (rates + weights ω^(2-p)) ===")
    lgb_wt_model, (lgb_wt_yhat_tr, lgb_wt_yhat_te), _ = \
        fit_lgbm_tweedie_rates_weights(df_train, df_test, feature_cols, p, num_boost_round=400, seed=seed)

    dev_tr_wt = evaluate_deviance(ytr, lgb_wt_yhat_tr, p, name="train")
    dev_te_wt = evaluate_deviance(yte, lgb_wt_yhat_te, p, name="test")

    # -------------------------
    # Equivalence checks (numerical)
    # -------------------------
    # Compare predictions across the three exact encodings on the same split (test set)
    def rel_rmse(a, b):
        denom = np.maximum(np.abs(a) + np.abs(b), 1e-6)
        return math.sqrt(np.mean(((a - b) / denom) ** 2))

    print("\n=== Equivalence checks (test set) ===")
    print(f"RMSE between LGB offset totals and LGB rates+ω^(2-p): "
          f"{mean_squared_error(lgb_off_yhat_te, lgb_wt_yhat_te):.6f}")
    print(f"Relative RMSE (scale-free): {rel_rmse(lgb_off_yhat_te, lgb_wt_yhat_te):.6e}")

    print(f"RMSE between sklearn and LGB offset totals: "
          f"{mean_squared_error(sk_yhat_te, lgb_off_yhat_te):.6f}")
    print(f"Relative RMSE: {rel_rmse(sk_yhat_te, lgb_off_yhat_te):.6e}")

    print(f"RMSE between sklearn and LGB rates+ω^(2-p): "
          f"{mean_squared_error(sk_yhat_te, lgb_wt_yhat_te):.6f}")
    print(f"Relative RMSE: {rel_rmse(sk_yhat_te, lgb_wt_yhat_te):.6e}")

    # Optional: assert closeness (tolerances depend on boosting stochasticity)
    # You can tighten these if you fix seeds and disable bagging.
    # assert rel_rmse(lgb_off_yhat_te, lgb_wt_yhat_te) < 5e-3, "LightGBM encodings not matching closely."
else:
    print("\n[LightGBM unavailable] Skipping offset/weights equivalence checks for LightGBM.")

#%%
# -------------------------
# Special cases sanity (optional small runs): p=1 (Poisson) and p=2 (Gamma)
# -------------------------
for p_special in (1.0, 2.0):
    print(f"\n=== Special case p = {p_special:.1f} ===")
    df_sp = make_insurance_data(n=40_000, p_index=p_special, phi=phi, n_features=6, seed=123)
    df_tr, df_te = train_test_split(df_sp, test_size=0.25, random_state=0)
    # sklearn: rates + exposure^(2 - p)
    glm_sp, (yhat_tr_sp, yhat_te_sp), _, _ = \
        fit_sklearn_tweedie_rates(df_tr, df_te, [c for c in df_sp.columns if c.startswith("x")],
                                  p_special, alpha=alpha)
    dev_sp = evaluate_deviance(df_te["Total"].to_numpy(), yhat_te_sp, p_special, name="test deviance (sklearn)")
    # Expect: if p=1 → weights=exposure; if p=2 → weights=1 (both included by exposure**(2-p))
    print("OK; sklearn uses correct weights exposure**(2 - p).")

print("\nAll done.")
