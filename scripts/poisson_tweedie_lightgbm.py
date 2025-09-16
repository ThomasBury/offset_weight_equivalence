# %% [markdown]
# # Real-data comparison on French MTPL (freMTPL2):
# Tweedie regression ONLY — exposure handling variants compared scientifically.
# 
# We compare, for p=1.9:
# 
# (1) scikit-learn TweedieRegressor (log link, no per-row offsets):
#
# - Exact:   label = PurePremium,   sample_weight = Exposure ** (2 - p)
# - Heuristic (Poisson-style):      label = PurePremium,   sample_weight = Exposure
# 
# (2) LightGBM Tweedie (log link):
#
# - Offset (totals + offset):       label = ClaimAmount,   init_score = log(Exposure)
# - Exact rates + weights:          label = PurePremium,   weight = Exposure ** (2 - p)
# - Heuristic rates + weights:      label = PurePremium,   weight = Exposure
# 
# Evaluation:
# - On TEST split, metrics on PurePremium (rate) with BOTH evaluation weightings:
#     a) Exposure ** (2 - p)  [GLM-correct for Tweedie rates]
#     b) Exposure             [portfolio / business weighting]
# - Lorenz curves (exposure-weighted) and calibration-by-feature plots.
# 
# Note:
# - LightGBM part requires `lightgbm`. If unavailable, those fits are skipped.
# - For speed, you can limit n_samples via N_SAMPLES below.

# %%
from __future__ import annotations

import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Use a non-interactive backend to avoid tkinter issues
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import (
    FunctionTransformer,
    KBinsDiscretizer,
    OneHotEncoder,
    StandardScaler,
)

from sklearn.linear_model import TweedieRegressor, PoissonRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    mean_tweedie_deviance,
    auc,
    mean_poisson_deviance,
)

# Optional LightGBM
try:
    import lightgbm as lgb
    LGB_AVAILABLE = True
except Exception: 
    LGB_AVAILABLE = False
    warnings.warn("lightgbm not found. LightGBM parts will be skipped.", RuntimeWarning)


# Config
# Tweedie variance power. Note: the top-level docstring mentions p=1.9, but the code uses 1.25.
P = 1.75
ALPHA = 0.1
RANDOM_STATE = 42
TEST_SIZE = 0.20
N_SAMPLES = None        # set to e.g. 200_000 for faster runs; None for full dataset
NUM_BOOST_ROUND = 100   # LightGBM
LEARNING_RATE = 0.1
np.random.seed(RANDOM_STATE)

# %% [markdown]
# ## Data loading utilities

# %%
def load_mtpl2(n_samples=None):
    """Fetch French MTPL (freq + sev), aggregate severity to totals, clean columns."""
    # freMTPL2freq
    df_freq = fetch_openml(data_id=41214, as_frame=True).data
    df_freq["IDpol"] = df_freq["IDpol"].astype(int)
    df_freq.set_index("IDpol", inplace=True)
    # freMTPL2sev
    df_sev = fetch_openml(data_id=41215, as_frame=True).data
    df_sev = df_sev.groupby("IDpol").sum()

    df = df_freq.join(df_sev, how="left")
    df["ClaimAmount"] = df["ClaimAmount"].fillna(0)

    # strip quotes in string columns
    for col in df.columns[[t is object for t in df.dtypes.values]]:
        df[col] = df[col].str.strip("'")

    return df.iloc[:n_samples]


def basic_cleaning(df):
    """Same cleaning logic as sklearn example, plus derived targets."""
    df = df.copy()
    df["ClaimNb"] = df["ClaimNb"].clip(upper=4)
    df["Exposure"] = df["Exposure"].clip(upper=1)
    df["ClaimAmount"] = df["ClaimAmount"].clip(upper=200_000)
    df.loc[(df["ClaimAmount"] == 0) & (df["ClaimNb"] >= 1), "ClaimNb"] = 0

    # Derived targets (rates)
    df["PurePremium"] = df["ClaimAmount"] / df["Exposure"]
    df["Frequency"] = df["ClaimNb"] / df["Exposure"]
    df["AvgClaimAmount"] = df["ClaimAmount"] / np.fmax(df["ClaimNb"], 1)
    return df

# %% [markdown]
# ## Feature engineering

# %%
def build_column_transformer():
    # Make OHE robust to unseen categories in test
    ohe = OneHotEncoder(handle_unknown="ignore")
    log_scale_transformer = make_pipeline(
        FunctionTransformer(func=np.log), StandardScaler()
    )
    column_trans = ColumnTransformer(
        transformers=[
            (
                "binned_numeric",
                KBinsDiscretizer(n_bins=10, encode="onehot-dense", strategy="quantile"),
                # Use quantile binning to handle skewed distributions like age.
                # Each bin will contain approximately the same number of samples.
                KBinsDiscretizer(
                    n_bins=10, encode="onehot-dense", strategy="quantile"
                ),
                ["VehAge", "DrivAge"],
            ),
            ("onehot_categorical", ohe, ["VehBrand", "VehPower", "VehGas", "Region", "Area"]),
            ("passthrough_numeric", "passthrough", ["BonusMalus"]),
            ("log_scaled_numeric", log_scale_transformer, ["Density"]),
        ],
        remainder="drop",
    )
    return column_trans

# %% [markdown]
# ## Plot helpers (rate-level calibration, Lorenz)

# %%
def _get_aggregated_rates(df, feature, weight_name, rate_values):
    """Helper to aggregate rates by a feature, exposure-weighted."""
    w = df[weight_name].to_numpy()
    tmp = pd.DataFrame({
        feature: df[feature].to_numpy(),
        "w": w,
        "rate_total": rate_values * w,
    })
    grp = tmp.groupby(feature)[["w", "rate_total"]].sum()
    grp["rate"] = grp["rate_total"] / grp["w"].replace(0, np.nan)
    return grp


def lorenz_curve(y_true_rate, y_pred_rate, exposure):
    y_true_rate = np.asarray(y_true_rate)
    y_pred_rate = np.asarray(y_pred_rate)
    exposure = np.asarray(exposure)

    order = np.argsort(y_pred_rate)
    ex = exposure[order]
    num = np.cumsum(ex * y_true_rate[order])
    num = num / num[-1] if num[-1] > 0 else num
    den = np.cumsum(ex) / np.sum(ex)
    return den, num

# %% [markdown]
# ## Metrics

# %%
def d2_explained(y_true_rate, y_pred_rate, sample_weight):
    """
    D^2 explained (GLM deviance analogue) computed from mean Tweedie deviance
    at the modeling power P; we use the same sample_weight for both terms.
    """
    sw = None if sample_weight is None else np.asarray(sample_weight)
    dev = mean_tweedie_deviance(y_true_rate, y_pred_rate, power=P, sample_weight=sw)
    y_bar = np.average(y_true_rate, weights=sw)
    dev_null = mean_tweedie_deviance(y_true_rate, np.full_like(y_true_rate, y_bar), power=P, sample_weight=sw)
    return 1.0 - (dev / dev_null if dev_null > 0 else np.nan)


def evaluate_models_table(df_test, pred_dict, weights_for_eval=("Exposure", "Exposure_2mp")):
    """
    Build a tidy table of metrics for each model under two evaluation weightings:
    - Exposure
    - Exposure ** (2 - p)
    Metrics on rates: MAE, MSE, mean Tweedie dev (power=P), and D^2 explained.
    """
    rows = []
    y_true_rate = df_test["PurePremium"].to_numpy()
    for wname in weights_for_eval:
        if wname == "Exposure":
            sw = df_test["Exposure"].to_numpy()
        elif wname == "Exposure_2mp":
            sw = (df_test["Exposure"].to_numpy()) ** (2.0 - P)
        else:
            raise ValueError("Unknown weight name")

        for model_name, y_pred_rate in pred_dict.items():
            mae = mean_absolute_error(y_true_rate, y_pred_rate, sample_weight=sw)
            mse = mean_squared_error(y_true_rate, y_pred_rate, sample_weight=sw)
            dev = mean_tweedie_deviance(y_true_rate, y_pred_rate, power=P, sample_weight=sw)
            d2 = d2_explained(y_true_rate, y_pred_rate, sample_weight=sw)
            rows.append({
                "eval_weight": wname,
                "model": model_name,
                "MAE_rate": mae,
                "MSE_rate": mse,
                f"MeanTweedieDev(p={P})": dev,
                "D2_explained": d2,
            })
    res = pd.DataFrame(rows).set_index(["eval_weight", "model"])
    return res

def d2_poisson_explained(y_true_rate, y_pred_rate, sample_weight):
    """
    D^2 explained for Poisson deviance, evaluated on rates with exposure as sample_weight.
    """
    sw = None if sample_weight is None else np.asarray(sample_weight)
    dev = mean_poisson_deviance(y_true_rate, y_pred_rate, sample_weight=sw)
    y_bar = np.average(y_true_rate, weights=sw)
    dev_null = mean_poisson_deviance(y_true_rate, np.full_like(y_true_rate, y_bar), sample_weight=sw)
    return 1.0 - (dev / dev_null if dev_null > 0 else np.nan)


def evaluate_frequency_models_table(df_test, pred_dict):
    """
    Build a tidy table of metrics for frequency models.
    Evaluation is always exposure-weighted.
    Metrics on rates: MAE, MSE, mean Poisson dev, and D^2 explained.
    """
    rows = []
    y_true_rate = df_test["Frequency"].to_numpy()
    sw = df_test["Exposure"].to_numpy()

    for model_name, y_pred_rate in pred_dict.items():
        mae = mean_absolute_error(y_true_rate, y_pred_rate, sample_weight=sw)
        mse = mean_squared_error(y_true_rate, y_pred_rate, sample_weight=sw)
        dev = mean_poisson_deviance(y_true_rate, y_pred_rate, sample_weight=sw)
        d2 = d2_poisson_explained(y_true_rate, y_pred_rate, sample_weight=sw)
        rows.append({
            "model": model_name,
            "MAE_freq": mae,
            "MSE_freq": mse,
            "MeanPoissonDev": dev,
            "D2_explained": d2,
        })
    res = pd.DataFrame(rows).set_index("model")
    return res
# %% [markdown]
# ## Fitting: sklearn TweedieRegressor (rates)

# %%
def fit_sklearn_tweedie_rates(X_tr, X_te, df_train, df_test, weight_scheme):
    """
    weight_scheme in {"exact", "poisson"}:
      - "exact":   s = Exposure ** (2 - P)
      - "poisson": s = Exposure
    Returns (model, y_pred_rate_train, y_pred_rate_test)
    """
    if weight_scheme == "exact":
        wtr = (df_train["Exposure"].to_numpy()) ** (2.0 - P)
        wte = (df_test["Exposure"].to_numpy()) ** (2.0 - P)
        tag = "sklearn_rate_w=exp^(2-p)"
    elif weight_scheme == "poisson":
        wtr = df_train["Exposure"].to_numpy()
        wte = df_test["Exposure"].to_numpy()
        tag = "sklearn_rate_w=exp"
    else:
        raise ValueError("weight_scheme must be 'exact' or 'poisson'.")

    y_tr = df_train["PurePremium"].to_numpy()
    y_te = df_test["PurePremium"].to_numpy()

    glm = TweedieRegressor(power=P, alpha=ALPHA, solver="newton-cholesky")
    glm.fit(X_tr, y_tr, sample_weight=wtr)
    yhat_tr_rate = glm.predict(X_tr)
    yhat_te_rate = glm.predict(X_te)
    return tag, glm, yhat_tr_rate, yhat_te_rate

# %% [markdown]
# ## Fitting: LightGBM Tweedie

# %%
def lgb_offset(X_tr, X_te, df_train, df_test):
    """
    LightGBM: totals + offset
      label = ClaimAmount
      init_score = log(Exposure)
      objective = tweedie, tweedie_variance_power=P
    Returns (model, yhat_tr_rate, yhat_te_rate)
    """
    if not LGB_AVAILABLE:
        return None, None, None

    y_tr = df_train["ClaimAmount"].to_numpy(dtype=float)
    init_tr = np.log(np.fmax(df_train["Exposure"].to_numpy(), 1e-3))

    dtrain = lgb.Dataset(X_tr, label=y_tr, init_score=init_tr)
    # The validation dataset also needs to be configured correctly if used for early stopping
    y_te = df_test["ClaimAmount"].to_numpy(dtype=float)
    init_te = np.log(np.fmax(df_test["Exposure"].to_numpy(), 1e-3))
    dvalid = lgb.Dataset(X_te, label=y_te, init_score=init_te, reference=dtrain)


    params = dict(
        objective="tweedie",
        tweedie_variance_power=P,
        learning_rate=LEARNING_RATE,
        verbose=-1,
        seed=RANDOM_STATE,
    )
    gbm = lgb.train(params,
                    dtrain,
                    valid_sets=[dvalid],
                    num_boost_round=NUM_BOOST_ROUND,
                    )

    yhat_te_rate = gbm.predict(X_te)
    yhat_te_tot = yhat_te_rate *  np.fmax(df_test["Exposure"].to_numpy(), 1e-12)

    # Check: total count of claims predicted on test set
    print("Total predicted claims (offset):", yhat_te_tot.sum())
    print("Total true claims:", df_test["ClaimAmount"].sum())

    # Do the same for the training set if needed for evaluation
    yhat_tr_rate = gbm.predict(X_tr)
    yhat_tr_tot = yhat_tr_rate * np.fmax(df_train["Exposure"].to_numpy(), 1e-12)

    return gbm, yhat_tr_rate, yhat_te_rate

def fit_lgb_rates(X_tr, X_te, df_train, df_test, weight_scheme):
    """
    LightGBM: rates + weights
      label = PurePremium
      weight = Exposure ** (2 - P)  (exact)  OR  Exposure (poisson-style)
    Returns (model, yhat_tr_rate, yhat_te_rate)
    """
    if not LGB_AVAILABLE:
        return None, None, None

    y_tr = df_train["PurePremium"].to_numpy(dtype=float)
    y_te = df_test["PurePremium"].to_numpy(dtype=float)

    if weight_scheme == "exact":
        wtr = (df_train["Exposure"].to_numpy()) ** (2.0 - P)
        wte = (df_test["Exposure"].to_numpy()) ** (2.0 - P)
        tag = "lgb_rate_w=exp^(2-p)"
    elif weight_scheme == "poisson":
        wtr = df_train["Exposure"].to_numpy()
        wte = df_test["Exposure"].to_numpy()
        tag = "lgb_rate_w=exp"
    else:
        raise ValueError("weight_scheme must be 'exact' or 'poisson'.")

    dtrain = lgb.Dataset(X_tr, label=y_tr, weight=wtr)
    dvalid = lgb.Dataset(X_te, label=y_te, weight=wte, reference=dtrain)

    params = dict(
        objective="tweedie",
        tweedie_variance_power=P,
        learning_rate=LEARNING_RATE,
        verbose=-1,
        seed=RANDOM_STATE,
    )
    gbm = lgb.train(params, 
                    dtrain, 
                    valid_sets=[dvalid], 
                    num_boost_round=NUM_BOOST_ROUND, 
                    )

    yhat_tr_rate = gbm.predict(X_tr)
    yhat_te_rate = gbm.predict(X_te)
    return tag, gbm, yhat_tr_rate, yhat_te_rate

# %% [markdown]
# ## Fitting: Poisson Models for Frequency

# %%
def fit_sklearn_poisson_rates(X_tr, X_te, df_train, df_test):
    """
    sklearn PoissonRegressor: rates + weights
      label = Frequency
      sample_weight = Exposure
    Returns (model, y_pred_rate_train, y_pred_rate_test)
    """
    tag = "sklearn_poisson_rate_w=exp"
    wtr = df_train["Exposure"].to_numpy()
    y_tr = df_train["Frequency"].to_numpy()

    glm = PoissonRegressor(alpha=ALPHA, solver="newton-cholesky")
    glm.fit(X_tr, y_tr, sample_weight=wtr)
    yhat_tr_rate = glm.predict(X_tr)
    yhat_te_rate = glm.predict(X_te)
    return tag, glm, yhat_tr_rate, yhat_te_rate


def fit_lgb_poisson_offset_counts(X_tr, X_te, df_train, df_test):
    """
    LightGBM (Poisson) with log-exposure offset via init_score.
      - label = ClaimNb (counts)
      - init_score = log(Exposure)
      - objective = 'poisson'
      - predict() returns RATE per unit exposure (not counts)
    Returns: (tag, model, yhat_tr_rate, yhat_te_rate)
    """
    if not LGB_AVAILABLE:
        return None, None, None, None

    tag = "lgb_poisson_offset"

    # Labels (counts)
    y_tr = df_train["ClaimNb"].to_numpy(dtype=float)
    y_te = df_test["ClaimNb"].to_numpy(dtype=float)

    # Offsets: log(exposure) with tiny floor
    eps = 1e-9
    exp_tr = df_train["Exposure"].to_numpy(dtype=float)
    exp_te = df_test["Exposure"].to_numpy(dtype=float)
    init_tr = np.log(np.fmax(exp_tr, eps))
    init_te = np.log(np.fmax(exp_te, eps))

    dtrain = lgb.Dataset(X_tr, label=y_tr, init_score=init_tr)
    dvalid = lgb.Dataset(X_te, label=y_te, init_score=init_te, reference=dtrain)

    params = dict(
        objective="poisson",
        learning_rate=LEARNING_RATE,
        verbose=-1,
        seed=RANDOM_STATE,
    )

    gbm = lgb.train(
        params=params,
        train_set=dtrain,
        valid_sets=[dvalid],
        num_boost_round=NUM_BOOST_ROUND,
    )

    # Predictions are RATES (per unit exposure)
    yhat_tr_rate = gbm.predict(X_tr)
    yhat_te_rate = gbm.predict(X_te)

    # Check: total count of claims predicted on test set
    yhat_te_tot = yhat_te_rate * exp_te
    print("Total predicted claims (poisson offset):", yhat_te_tot.sum())
    print("Total true claims:", y_te.sum())

    return tag, gbm, yhat_tr_rate, yhat_te_rate


def fit_lgb_poisson_rates_weights(X_tr, X_te, df_train, df_test):
    """
    LightGBM: rates + weights
      label = Frequency
      weight = Exposure
      objective = poisson
    Returns (model, yhat_tr_rate, yhat_te_rate)
    """
    if not LGB_AVAILABLE:
        return None, None, None, None

    tag = "lgb_poisson_rate_w=exp"
    y_tr = df_train["Frequency"].to_numpy(dtype=float)
    y_te = df_test["Frequency"].to_numpy(dtype=float)
    wtr = df_train["Exposure"].to_numpy()
    wte = df_test["Exposure"].to_numpy()

    dtrain = lgb.Dataset(X_tr, label=y_tr, weight=wtr)
    dvalid = lgb.Dataset(X_te, label=y_te, weight=wte, reference=dtrain)

    params = dict(
        objective="poisson",
        learning_rate=LEARNING_RATE,
        boost_from_average=True, # Default is fine here
        verbose=-1,
        seed=RANDOM_STATE,
    )
    gbm = lgb.train(params, dtrain, valid_sets=[dvalid], num_boost_round=NUM_BOOST_ROUND)

    yhat_tr_rate = gbm.predict(X_tr)
    yhat_te_rate = gbm.predict(X_te)
    return tag, gbm, yhat_tr_rate, yhat_te_rate

# %% [markdown]
# ## Data loading and preprocessing

# %%
# Load and clean data
df = load_mtpl2(n_samples=N_SAMPLES)
df = basic_cleaning(df)

# Split BEFORE fitting transformer to avoid leakage (unsupervised transforms, but good hygiene)
df_train, df_test = train_test_split(df, test_size=TEST_SIZE, random_state=RANDOM_STATE, shuffle=True)

# Build transformer on train, transform both
column_trans = build_column_transformer()
X_train = column_trans.fit_transform(df_train)
X_test = column_trans.transform(df_test)

print(f"Train size: {len(df_train):,}, Test size: {len(df_test):,}")
print(f"Tweedie power p = {P}, alpha = {ALPHA}, LightGBM rounds = {NUM_BOOST_ROUND}")

# %% [markdown]
# ## Pure Premium Model Fitting (Tweedie)

# %%
# --- sklearn: rates + weights (exact and poisson-style) ---
tag_sk_exact, sk_exact, sk_exact_tr, sk_exact_te = fit_sklearn_tweedie_rates(X_train, X_test, df_train, df_test, "exact")
tag_sk_pois,  sk_pois,  sk_pois_tr,  sk_pois_te  = fit_sklearn_tweedie_rates(X_train, X_test, df_train, df_test, "poisson")

# --- LightGBM variants (if available) ---
lgb_pred = {}
if LGB_AVAILABLE:
    lgb_off, lgb_off_tr, lgb_off_te = lgb_offset(X_train, X_test, df_train, df_test)
    tag_lgb_exact, lgb_exact, lgb_exact_tr, lgb_exact_te = fit_lgb_rates(X_train, X_test, df_train, df_test, "exact")
    tag_lgb_pois,  lgb_pois,  lgb_pois_tr,  lgb_pois_te  = fit_lgb_rates(X_train, X_test, df_train, df_test, "poisson")
    lgb_pred = {
        "lgb_offset": lgb_off_te,
        tag_lgb_exact: lgb_exact_te,
        tag_lgb_pois: lgb_pois_te,
    }

# Collect predictions on TEST (rates)
pred_rate_test = {
    tag_sk_exact: sk_exact_te,
    tag_sk_pois:  sk_pois_te,
}
pred_rate_test.update(lgb_pred)

# %% [markdown]
# ## Lorenz curves (TEST), exposure-weighted

# %%
y_true_rate = df_test["PurePremium"].to_numpy()
fig, ax = plt.subplots(figsize=(7, 7))
# models
exposure_test = df_test["Exposure"].to_numpy()
for label, y_pred_rate in pred_rate_test.items():
    cx, cy = lorenz_curve(y_true_rate, y_pred_rate, exposure_test)
    gini = 1 - 2 * auc(cx, cy)
    ax.plot(cx, cy, label=f"{label} (Gini={gini:.3f})" )
# Oracle
cx, cy = lorenz_curve(y_true_rate, y_true_rate, exposure_test)
gini = 1 - 2 * auc(cx, cy)
ax.plot(cx, cy, linestyle="-.", label=f"Oracle (Gini={gini:.3f})" )
# Random
ax.plot([0, 1], [0, 1], linestyle="--", label="Random baseline")
ax.set(
    title="Lorenz curves on TEST (exposure-weighted)",
    xlabel="Cumulative exposure (sorted by predicted risk, low→high)",
    ylabel="Cumulative claim amounts",
)
ax.legend(loc="lower right")
plt.tight_layout()
plt.savefig("lorenz_curve_tweedie.png")
plt.close(fig)
print("\nSaved Tweedie Lorenz curve to lorenz_curve_tweedie.png")

# %% [markdown]
# ## Metrics tables (TEST) under two evaluation weightings

# %%
metrics_tbl = evaluate_models_table(df_test, pred_rate_test, weights_for_eval=("Exposure", "Exposure_2mp"))
print("\n=== Metrics on TEST (rates), under two evaluation weightings ===")
with pd.option_context("display.max_rows", 100, "display.width", 160):
    print(metrics_tbl.sort_index())

# %% [markdown]
# ## Aggregate totals comparison (TEST)

# %%
y_true_tot = (df_test["PurePremium"].to_numpy() * exposure_test).sum()
agg_rows = [
    ("Observed totals", y_true_tot),
    (tag_sk_exact, np.sum(exposure_test * sk_exact_te)),
    (tag_sk_pois,  np.sum(exposure_test * sk_pois_te)),
]
if LGB_AVAILABLE:
    agg_rows.extend([
        ("lgb_offset", np.sum(exposure_test * lgb_off_te)),
        (tag_lgb_exact, np.sum(exposure_test * lgb_exact_te)),
        (tag_lgb_pois,  np.sum(exposure_test * lgb_pois_te)),
    ])
agg_df = pd.DataFrame(agg_rows, columns=["model", "sum_predicted_totals"]).set_index("model")
print("\n=== Aggregate predicted totals on TEST ===")
with pd.option_context("display.float_format", "{:.2f}".format):
    print(agg_df)

# %% [markdown]
# ## Calibration by feature (TEST): pick 2 interpretable features

# %%
feature_list = ["DrivAge", "VehPower"]
for feat in feature_list:
    # --- Sklearn models comparison ---
    fig, ax = plt.subplots(figsize=(8, 5))

    # Get observed rates and distribution
    grp_obs = _get_aggregated_rates(df_test, feat, "Exposure", y_true_rate)
    
    # Plot observed rate as gray line without markers
    grp_obs["rate"].plot(style="-", color="gray", ax=ax, label="Observed", linewidth=1.5, alpha=0.8)

    # Plot predicted rates for each sklearn model with consistent colors and thicker lines
    model_styles_sklearn = {
        tag_sk_exact: ("-", "blue", "Predicted (exact: ω^(2-p))"),
        tag_sk_pois: ("-", "red", "Predicted (Poisson-style: ω)"),
    }
    
    for model_tag, (linestyle, color, label) in model_styles_sklearn.items():
        grp_pred = _get_aggregated_rates(df_test, feat, "Exposure", pred_rate_test[model_tag])
        grp_pred["rate"].plot(style=linestyle, color=color, ax=ax, label=label, linewidth=3)

    # Add a shaded area for the feature's distribution
    y_max = ax.get_ylim()[1]
    x_values = (grp_obs.index.astype(float) if np.issubdtype(grp_obs.index.dtype, np.number) 
                else np.arange(len(grp_obs)))
    
    ax.fill_between(
        x_values,
        0, y_max * 0.5 * grp_obs["w"] / np.nanmax(grp_obs["w"]),
        alpha=0.1, color='grey', label=f"{feat} distribution"
    )

    ax.set(
        title=f"TEST: Calibration by {feat} (scikit-learn models)",
        xlabel=feat,
        ylabel="Pure premium (per exposure)"
    )
    ax.legend()
    plt.tight_layout()
    plt.savefig(f"calibration_sklearn_{feat}.png")
    plt.close(fig)
    print(f"Saved sklearn calibration plot for {feat} to calibration_sklearn_{feat}.png")

    # --- LightGBM models comparison ---
    if LGB_AVAILABLE:
        fig, ax = plt.subplots(figsize=(8, 5))

        # Plot observed rate as gray line without markers
        grp_obs["rate"].plot(style="-", color="gray", ax=ax, label="Observed", linewidth=1.5, alpha=0.8)

        # Plot predicted rates for each LightGBM model with consistent colors and thicker lines
        model_styles_lgb = {
            "lgb_offset": ("-", "green", "Predicted (offset)"),
            tag_lgb_exact: ("-", "orange", "Predicted (exact: ω^(2-p))"),
            tag_lgb_pois: ("-", "purple", "Predicted (Poisson-style: ω)"),
        }
        
        for model_tag, (linestyle, color, label) in model_styles_lgb.items():
            grp_pred = _get_aggregated_rates(df_test, feat, "Exposure", pred_rate_test[model_tag])
            grp_pred["rate"].plot(style=linestyle, color=color, ax=ax, label=label, linewidth=3)

        # Add distribution shading
        ax.fill_between(
            x_values,
            0, y_max * 0.5 * grp_obs["w"] / np.nanmax(grp_obs["w"]),
            alpha=0.1, color='grey', label=f"{feat} distribution"
        )

        ax.set(
            title=f"TEST: Calibration by {feat} (LightGBM models)",
            xlabel=feat,
            ylabel="Pure premium (per exposure)"
        )
        ax.legend()
        plt.tight_layout()
        plt.savefig(f"calibration_lgbm_{feat}.png")
        plt.close(fig)
        print(f"Saved LightGBM calibration plot for {feat} to calibration_lgbm_{feat}.png")

# %% [markdown]
# ---
# # FREQUENCY ANALYSIS (Poisson)
# 
# Now we repeat a similar analysis, but for claim frequency (`ClaimNb / Exposure`), using Poisson models.
# 
# We compare:
# 1.  **scikit-learn `PoissonRegressor`**: `label=Frequency`, `sample_weight=Exposure`
# 2.  **LightGBM `objective=poisson`**:
#     - Offset method: `label=ClaimNb`, `init_score=log(Exposure)`
#     - Weighted-rate method: `label=Frequency`, `weight=Exposure`
# 
# Evaluation is on the test set, using exposure-weighted metrics on the predicted frequency.

# %% [markdown]
# ## Frequency Model Fitting (Poisson)

# %%
# --- sklearn: rates + weights ---
tag_sk_poi, sk_poi_model, sk_poi_tr, sk_poi_te = fit_sklearn_poisson_rates(X_train, X_test, df_train, df_test)

# --- LightGBM variants (if available) ---
lgb_poi_pred = {}
if LGB_AVAILABLE:
    tag_lgb_off, lgb_off_model, lgb_off_tr, lgb_off_te = fit_lgb_poisson_offset_counts(X_train, X_test, df_train, df_test)
    tag_lgb_w, lgb_w_model, lgb_w_tr, lgb_w_te = fit_lgb_poisson_rates_weights(X_train, X_test, df_train, df_test)
    lgb_poi_pred = {
        tag_lgb_off: lgb_off_te,
        tag_lgb_w: lgb_w_te,
    }

# Collect all frequency predictions on TEST
pred_freq_test = { tag_sk_poi: sk_poi_te }
pred_freq_test.update(lgb_poi_pred)

# %% [markdown]
# ## Frequency Metrics (TEST)

# %%
metrics_freq_tbl = evaluate_frequency_models_table(df_test, pred_freq_test)
print("\n=== Metrics on TEST (frequency), exposure-weighted ===")
with pd.option_context("display.max_rows", 100, "display.width", 160):
    print(metrics_freq_tbl.sort_index())

# %% [markdown]
# ## Aggregate Claim Count Comparison (TEST)

# %%
exposure_test_freq = df_test["Exposure"].to_numpy()
y_true_counts = df_test["ClaimNb"].to_numpy().sum()

agg_rows_freq = [("Observed counts", y_true_counts)]
for model_name, pred_rate in pred_freq_test.items():
    pred_counts = np.sum(exposure_test * pred_rate)
    agg_rows_freq.append((model_name, pred_counts))

agg_df_freq = pd.DataFrame(agg_rows_freq, columns=["model", "sum_predicted_counts"]).set_index("model")
print("\n=== Aggregate predicted claim counts on TEST ===")
with pd.option_context("display.float_format", "{:,.2f}".format):
    print(agg_df_freq)

# %% [markdown]
# ## Lorenz & Calibration Plots for Frequency (TEST)

# %%
y_true_freq = df_test["Frequency"].to_numpy()

# --- Lorenz Curve ---
fig, ax = plt.subplots(figsize=(7, 7))
for label, y_pred_freq in pred_freq_test.items():
    cx, cy = lorenz_curve(y_true_freq, y_pred_freq, exposure_test_freq)
    gini = 1 - 2 * auc(cx, cy)
    ax.plot(cx, cy, label=f"{label} (Gini={gini:.3f})")

# Oracle
cx, cy = lorenz_curve(y_true_freq, y_true_freq, exposure_test)
gini = 1 - 2 * auc(cx, cy)
ax.plot(cx, cy, linestyle="-.", label=f"Oracle (Gini={gini:.3f})")

ax.plot([0, 1], [0, 1], linestyle="--", label="Random baseline")
ax.set(
    title="Lorenz curves for Frequency on TEST (exposure-weighted)",
    xlabel="Cumulative exposure (sorted by predicted frequency, low→high)",
    ylabel="Cumulative claim counts",
)
ax.legend(loc="lower right")
plt.tight_layout()
plt.savefig("lorenz_curve_frequency.png")
plt.close(fig)
print("\nSaved Frequency Lorenz curve to lorenz_curve_frequency.png")

# --- Calibration Plot ---
feat = "DrivAge" # Pick one feature for demonstration
fig, ax = plt.subplots(figsize=(8, 5))

# Get observed rates and distribution
grp_obs = _get_aggregated_rates(df_test, feat, "Exposure", y_true_freq)

# Plot observed rate as gray line without markers
grp_obs["rate"].plot(style="-", color="gray", ax=ax, label="Observed", linewidth=1.5, alpha=0.8)

# Plot predicted rates for each model with consistent colors and thicker lines
model_styles_freq = {
    "sklearn_poisson_rate_w=exp": ("-", "blue", "Predicted (sklearn)"),
    "lgb_poisson_offset": ("-", "green", "Predicted (LGB offset)"),
    "lgb_poisson_rate_w=exp": ("-", "orange", "Predicted (LGB rates)"),
}

for model_tag, (linestyle, color, label) in model_styles_freq.items():
    grp_pred = _get_aggregated_rates(df_test, feat, "Exposure", pred_freq_test[model_tag])
    grp_pred["rate"].plot(style=linestyle, color=color, ax=ax, label=label, linewidth=3)

# Add a shaded area for the feature's distribution
y_max = ax.get_ylim()[1]
x_values = (grp_obs.index.astype(float) if np.issubdtype(grp_obs.index.dtype, np.number)
            else np.arange(len(grp_obs)))

ax.fill_between(
    x_values,
    0, y_max * 0.5 * grp_obs["w"] / np.nanmax(grp_obs["w"]),
    alpha=0.1, color='grey', label=f"{feat} distribution"
)

ax.set(title=f"TEST: Frequency Calibration by {feat}", xlabel=feat, ylabel="Claim Frequency (per exposure)")
ax.legend()
plt.tight_layout()
plt.savefig("calibration_frequency_DrivAge.png")
plt.close(fig)
print("Saved Frequency calibration plot for DrivAge to calibration_frequency_DrivAge.png")
# %%
pred_freq_test
# %%
