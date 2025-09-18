# Empirical test: effect of `boost_from_average` when using an offset via `init_score=log(Exposure)`
# We'll run two experiments:
#  (A) Poisson frequency with counts label + offset log(exposure)
#  (B) Tweedie severity with totals label + offset log(exposure)
# For each, compare models trained with boost_from_average=True vs False.
from typing import Dict, Optional, Tuple
import numpy as np, pandas as pd, lightgbm as lgb
from sklearn.metrics import mean_poisson_deviance, mean_tweedie_deviance

rng = np.random.default_rng(42)

def make_poisson_synth(n: int = 200_000) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generate synthetic data for a Poisson regression model.

    Creates features, a highly variable exposure, a true underlying rate,
    and observed counts based on a Poisson distribution where the mean is
    the product of exposure and rate.

    Parameters
    ----------
    n : int, optional
        Number of samples to generate, by default 200_000.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        A tuple containing:
        - X: Feature matrix (n_samples, n_features).
        - exposure: Exposure array (n_samples,).
        - rate: True rate per unit of exposure (n_samples,).
        - counts: Simulated claim counts (n_samples,).
    """
    # features
    x1 = rng.normal(size=n)
    x2 = rng.uniform(-1, 1, size=n)
    X = np.c_[x1, x2]

    # exposure (highly variable)
    exposure = rng.lognormal(mean=0.0, sigma=1.0, size=n)  # median ~1, heavy tail
    exposure = np.clip(exposure, 1e-6, None)

    # true log-rate
    eta = 0.3 * x1 - 0.7 * x2 + 0.2  # linear predictor for rate
    rate = np.exp(eta)  # per-exposure rate

    # counts with offset (mean = exposure * rate)
    counts = rng.poisson(exposure * rate)

    return X, exposure, rate, counts

def make_tweedie_synth(n: int = 200_000, p: float = 1.5) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generate synthetic data for a Tweedie regression model.

    Creates features, exposure, a true rate, and total claim amounts.
    The totals are simulated from a Gamma distribution to approximate a
    Tweedie distribution with a power parameter `p` between 1 and 2.

    Parameters
    ----------
    n : int, optional
        Number of samples to generate, by default 200_000.
    p : float, optional
        Tweedie variance power, by default 1.5.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        A tuple containing:
        - X: Feature matrix (n_samples, n_features).
        - exposure: Exposure array (n_samples,).
        - rate: True rate (pure premium) per unit of exposure (n_samples,).
        - totals: Simulated total claim amounts (n_samples,).
    """
    # features
    x1 = rng.normal(size=n)
    x2 = rng.uniform(-1, 1, size=n)
    X = np.c_[x1, x2]

    exposure = rng.lognormal(mean=0.0, sigma=1.0, size=n)
    exposure = np.clip(exposure, 1e-6, None)

    # true rate (pure premium per exposure)
    eta = 0.4 * x1 + 0.2 * x2 - 0.1
    rate = np.exp(eta)

    # Tweedie totals ~ Tweedie(mean = exposure*rate, power=p)
    # We'll simulate totals using a Poisson-gamma compound for p in (1,2).
    # For simplicity, approximate with gamma noise around mean (not exact Tweedie but sufficient for testing offsets).
    mean_tot = exposure * rate
    phi = 1.0  # dispersion scale
    # gamma with mean=mean_tot and variance=phi*mean_tot**p => shape=k, scale=theta => k*theta=mean, k*theta^2=var
    var = phi * (mean_tot ** p)
    theta = var / mean_tot
    k = mean_tot / theta
    # Avoid numerical issues
    theta = np.clip(theta, 1e-9, None)
    k = np.clip(k, 1e-6, None)
    totals = rng.gamma(shape=k, scale=theta)

    return X, exposure, rate, totals

def fit_lgb_offset(
    X: np.ndarray,
    exposure: np.ndarray,
    label: np.ndarray,
    objective: str,
    power: Optional[float] = None,
    boost_from_average: bool = False,
    lr: float = 0.05,
    rounds: int = 400,
) -> Tuple[lgb.Booster, np.ndarray]:
    """Fit a LightGBM model using log(exposure) as an offset via init_score.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix.
    exposure : np.ndarray
        Exposure values for each sample. Used to compute the offset.
    label : np.ndarray
        Target variable (e.g., counts or totals).
    objective : str
        LightGBM objective function (e.g., 'poisson' or 'tweedie').
    power : Optional[float], optional
        Tweedie variance power, required if objective is 'tweedie'. By default None.
    boost_from_average : bool, optional
        LightGBM's `boost_from_average` parameter. By default False.
    lr : float, optional
        Learning rate. By default 0.05.
    rounds : int, optional
        Number of boosting rounds. By default 400.

    Returns
    -------
    Tuple[lgb.Booster, np.ndarray]
        A tuple containing the trained LightGBM model and the predicted rates.
    """
    eps = 1e-9
    init = np.log(np.maximum(exposure, eps))
    dtrain = lgb.Dataset(X, label=label, init_score=init, free_raw_data=True)
    params = dict(objective=objective, learning_rate=lr, verbose=-1, num_leaves=31, max_depth=-1, min_data_in_leaf=50)
    if objective == 'tweedie':
        params['tweedie_variance_power'] = power
    params['boost_from_average'] = boost_from_average
    model = lgb.train(params, dtrain, num_boost_round=rounds)
    pred_rate = model.predict(X)  # rate per exposure (given offset usage)
    return model, pred_rate

def summarize_poisson(
    rate_true: np.ndarray, exposure: np.ndarray, counts: np.ndarray, rate_pred: np.ndarray
) -> Dict[str, float]:
    """Calculate and summarize metrics for a Poisson model.

    Computes the exposure-weighted mean Poisson deviance between true and
    predicted rates, and compares the aggregate true and predicted counts.

    Parameters
    ----------
    rate_true : np.ndarray
        The true underlying rate.
    exposure : np.ndarray
        The exposure for each observation.
    counts : np.ndarray
        The observed counts (unused in calculation but kept for context).
    rate_pred : np.ndarray
        The predicted rate from the model.

    Returns
    -------
    Dict[str, float]
        A dictionary of metrics: 'poisson_dev', 'agg_true_counts',
        'agg_pred_counts', and 'ratio'.
    """
    # Evaluate on rate-level with exposure weighting
    sw = exposure
    dev = mean_poisson_deviance(rate_true, rate_pred, sample_weight=sw)
    # Compare aggregate counts
    agg_true = (rate_true * exposure).sum()
    agg_pred = (rate_pred * exposure).sum()
    return dict(poisson_dev=dev, agg_true_counts=agg_true, agg_pred_counts=agg_pred, ratio=agg_pred/agg_true)

def summarize_tweedie(
    rate_true: np.ndarray, exposure: np.ndarray, totals: np.ndarray, rate_pred: np.ndarray, p: float
) -> Dict[str, float]:
    """Calculate and summarize metrics for a Tweedie model.

    Computes the correctly weighted mean Tweedie deviance between true and
    predicted rates, and compares the aggregate true and predicted totals.

    Parameters
    ----------
    rate_true : np.ndarray
        The true underlying rate (pure premium).
    exposure : np.ndarray
        The exposure for each observation.
    totals : np.ndarray
        The observed total amounts (unused in calculation but kept for context).
    rate_pred : np.ndarray
        The predicted rate from the model.
    p : float
        The Tweedie variance power used for the deviance calculation.

    Returns
    -------
    Dict[str, float]
        A dictionary of metrics: 'tweedie_dev', 'agg_true_totals',
        'agg_pred_totals', and 'ratio'.
    """
    sw = exposure ** (2 - p)  # GLM-correct rate weighting
    dev = mean_tweedie_deviance(rate_true, rate_pred, power=p, sample_weight=sw)
    agg_true = (rate_true * exposure).sum()
    agg_pred = (rate_pred * exposure).sum()
    return dict(tweedie_dev=dev, agg_true_totals=agg_true, agg_pred_totals=agg_pred, ratio=agg_pred/agg_true)

# ===== Experiment A: Poisson counts + offset =====
X, expA, rateA, countsA = make_poisson_synth(n=250_000)

m_false, rate_pred_false = fit_lgb_offset(X, expA, countsA, objective='poisson', boost_from_average=False, lr=0.05, rounds=300)
m_true,  rate_pred_true  = fit_lgb_offset(X, expA, countsA, objective='poisson', boost_from_average=True,  lr=0.05, rounds=300)

sumA_false = summarize_poisson(rateA, expA, countsA, rate_pred_false)
sumA_true  = summarize_poisson(rateA, expA, countsA, rate_pred_true)

print("=== Poisson counts + offset ===")
print("boost_from_average = False:", sumA_false)
print("boost_from_average = True :", sumA_true)

# ===== Experiment B: Tweedie totals + offset =====
p = 1.5
X, expB, rateB, totalsB = make_tweedie_synth(n=100_000, p=p)

mt_false, rate_pred_false_B = fit_lgb_offset(X, expB, totalsB, objective='tweedie', power=p, boost_from_average=False, lr=0.05, rounds=400)
mt_true,  rate_pred_true_B  = fit_lgb_offset(X, expB, totalsB, objective='tweedie', power=p, boost_from_average=True,  lr=0.05, rounds=400)

sumB_false = summarize_tweedie(rateB, expB, totalsB, rate_pred_false_B, p=p)
sumB_true  = summarize_tweedie(rateB, expB, totalsB, rate_pred_true_B, p=p)

print("\n=== Tweedie totals + offset (p=1.5) ===")
print("boost_from_average = False:", sumB_false)
print("boost_from_average = True :", sumB_true)
