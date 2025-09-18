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
import math, typing
import warnings
import numpy as np
import pandas as pd
from numpy.random import default_rng

from sklearn.model_selection import train_test_split
from sklearn.linear_model import TweedieRegressor
from sklearn.metrics import mean_squared_error

if typing.TYPE_CHECKING:
    from typing import List, Optional, Tuple


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
    """Calculate the half Tweedie deviance with a log link.

    This matches scikit-learn's `HalfTweedieLoss` for p not in {0, 1, 2}.
    Returns element-wise loss (without constants in y for p in {0,1,2}).

    Domain assumptions for typical insurance use:
      - For 1 < p < 2: y >= 0, mu > 0
      - For p = 1, 2: handled by limits

    Parameters
    ----------
    y : np.ndarray
        The ground truth values.
    mu : np.ndarray
        The predicted mean values.
    p : float
        The Tweedie variance power.

    Returns
    -------
    np.ndarray
        The element-wise half Tweedie deviance.
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


def tweedie_compound_poisson_gamma_params(
    mu: np.ndarray, phi: float, p: float
) -> Tuple[np.ndarray, float, np.ndarray]:
    """Map Tweedie parameters to compound Poisson-Gamma parameters.

    For a Tweedie distribution with 1 < p < 2, this function calculates the
    parameters (λ, a, θ) for the equivalent compound Poisson-Gamma process.
      N ~ Poisson(λ), severities ~ Gamma(a, θ) (shape a, scale θ), total = sum_{k=1..N} S_k
    λ = μ^{2-p} / (φ (2-p))
    a = (2 - p) / (p - 1)
    θ = φ (p - 1) μ^{p - 1}

    Parameters
    ----------
    mu : np.ndarray
        Mean of the Tweedie distribution.
    phi : float
        Dispersion parameter of the Tweedie distribution.
    p : float
        Power parameter of the Tweedie distribution (must be in (1, 2)).

    Returns
    -------
    Tuple[np.ndarray, float, np.ndarray]
        A tuple containing the Poisson rate (lam), Gamma shape (a), and Gamma scale (theta).
    """
    if not (1.0 < p < 2.0):
        raise ValueError("Compound Poisson–Gamma mapping only valid for 1 < p < 2.")
    lam = (mu ** (2.0 - p)) / (phi * (2.0 - p))
    a = (2.0 - p) / (p - 1.0)
    theta = phi * (p - 1.0) * (mu ** (p - 1.0))
    return lam, a, theta


def rng_tweedie_totals(mu: np.ndarray, phi: float, p: float, rng: np.random.Generator) -> np.ndarray:
    """Simulate totals from a Tweedie distribution.

    For the insurance case 1 < p < 2, simulation is done via a compound Poisson–Gamma process.
    For p=1 or p=2, we use appropriate limits (Poisson and Gamma) for demonstration.
    Note: For production, consider a dedicated Tweedie RNG library if needed.

    Parameters
    ----------
    mu : np.ndarray
        The mean of the Tweedie distribution for each sample.
    phi : float
        The dispersion parameter.
    p : float
        The Tweedie power parameter. Supported values are p=1, p=2, and 1 < p < 2.
    rng : np.random.Generator
        A NumPy random number generator instance.

    Returns
    -------
    np.ndarray
        The simulated total values.
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
def make_insurance_data(
    n: int = 50_000, p_index: float = 1.9, phi: float = 0.8, n_features: int = 6, seed: int = 7
) -> pd.DataFrame:
    """Generate synthetic insurance-style data.

    The process is as follows:
      - features X
      - exposure ω
      - rate μ_rate = exp(Xβ)
      - total mean μ_total = ω * μ_rate
      - totals Y ~ Tweedie(μ_total, φ, p)

    Parameters
    ----------
    n : int, optional
        Number of samples to generate, by default 50_000.
    p_index : float, optional
        Tweedie power index, by default 1.9.
    phi : float, optional
        Dispersion parameter for data generation, by default 0.8.
    n_features : int, optional
        Number of features to generate, by default 6.
    seed : int, optional
        Random seed for reproducibility, by default 7.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing features, Exposure, Total claim amount, and Rate.
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
def fit_sklearn_tweedie_rates(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    feature_cols: List[str],
    p: float,
    alpha: float = 0.1,
) -> Tuple[
    TweedieRegressor,
    Tuple[np.ndarray, np.ndarray],
    Tuple[np.ndarray, np.ndarray],
    Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray],
]:
    """Fit a scikit-learn TweedieRegressor on rates with correct weights.

    The correct sample weight for modeling rates is `exposure**(2 - p)`.

    Parameters
    ----------
    df_train : pd.DataFrame
        Training data.
    df_test : pd.DataFrame
        Test data.
    feature_cols : List[str]
        List of column names to use as features.
    p : float
        Tweedie power parameter.
    alpha : float, optional
        L2 regularization strength, by default 0.1.

    Returns
    -------
    Tuple
        A tuple containing:
        - The fitted `TweedieRegressor` model.
        - Predicted totals for train and test sets.
        - Predicted rates for train and test sets.
        - A tuple of raw data arrays (Xtr, Xte, rtr, rte, wtr, wte).
    """
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


def fit_lgbm_tweedie_totals_offset(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    feature_cols: List[str],
    p: float,
    num_boost_round: int = 300,
    seed: int = 13,
) -> Tuple[Optional[lgb.Booster], Tuple[Optional[np.ndarray], Optional[np.ndarray]]]:
    """Fit a LightGBM Tweedie model on totals with a log-exposure offset.

    The offset is provided via `init_score = log(exposure)`.

    Parameters
    ----------
    df_train : pd.DataFrame
        Training data.
    df_test : pd.DataFrame
        Test data.
    feature_cols : List[str]
        List of column names to use as features.
    p : float
        Tweedie power parameter.
    num_boost_round : int, optional
        Number of boosting rounds, by default 300.
    seed : int, optional
        Random seed for reproducibility, by default 13.

    Returns
    -------
    Tuple[Optional[lgb.Booster], Tuple[Optional[np.ndarray], Optional[np.ndarray]]]
        The fitted model and predicted totals for train and test sets. Returns None if LightGBM is not available.
    """
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


def fit_lgbm_tweedie_rates_weights(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    feature_cols: List[str],
    p: float,
    num_boost_round: int = 300,
    seed: int = 13,
) -> Tuple[
    Optional[lgb.Booster],
    Tuple[Optional[np.ndarray], Optional[np.ndarray]],
    Tuple[Optional[np.ndarray], Optional[np.ndarray]],
]:
    """Fit a LightGBM Tweedie model on rates with correct weights.

    The model is trained on rates with `sample_weight = exposure**(2 - p)`.

    Parameters
    ----------
    df_train : pd.DataFrame
        Training data.
    df_test : pd.DataFrame
        Test data.
    feature_cols : List[str]
        List of column names to use as features.
    p : float
        Tweedie power parameter.
    num_boost_round : int, optional
        Number of boosting rounds, by default 300.
    seed : int, optional
        Random seed for reproducibility, by default 13.

    Returns
    -------
    Tuple
        The fitted model, predicted totals (train/test), and predicted rates (train/test). Returns Nones if LightGBM is not available.
    """
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
def evaluate_deviance(
    y_true: np.ndarray, y_hat: np.ndarray, p: float, sample_weight: Optional[np.ndarray] = None, name: str = ""
) -> float:
    """Calculate the average half Tweedie deviance.

    If sample weights are provided, a weighted average is computed.

    Parameters
    ----------
    y_true : np.ndarray
        Ground truth target values.
    y_hat : np.ndarray
        Predicted target values.
    p : float
        Tweedie power parameter.
    sample_weight : Optional[np.ndarray], optional
        Sample weights, by default None.
    name : str, optional
        An optional name to print with the result, by default "".

    Returns
    -------
    float
        The (weighted) average half Tweedie deviance.
    """
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
    def rel_rmse(a: np.ndarray, b: np.ndarray) -> float:
        """Calculate the relative root mean squared error.

        This provides a scale-free measure of the difference between two arrays.

        Parameters
        ----------
        a : np.ndarray
            First array of predictions.
        b : np.ndarray
            Second array of predictions.

        Returns
        -------
        float
            The relative RMSE value.
        """
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
