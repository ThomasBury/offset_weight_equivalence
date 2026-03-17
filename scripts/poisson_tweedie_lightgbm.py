# %% [markdown]
# # Real-data comparison on French MTPL (freMTPL2):
# Tweedie regression ONLY — exposure handling variants compared scientifically.
# 
# We compare several ways to encode exposure for a Tweedie model with log link.
# In this script, `P = 1.75`, so the mathematically correct rate weight is
# `Exposure ** (2 - P) = Exposure ** 0.25`.
# 
# The key tutorial point is:
# 
# - Totals + offset: `E[Y_i] = Exposure_i * mu_i`
# - Rates + exact weights: `R_i = Y_i / Exposure_i`, `sample_weight = Exposure_i ** (2 - P)`
# 
# Under a Tweedie GLM with log link, those two formulations have the same score.
# For LightGBM, to make that equivalence visible in practice, we must also disable
# `boost_from_average`; otherwise LightGBM injects a different global starting value
# depending on whether we train on totals or rates.
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
# - For the LightGBM exact comparison, we also print the max prediction gap
#   between `offset` and `rates + Exposure ** (2 - p)`.
# 
# Note:
# - LightGBM part requires `lightgbm`. If unavailable, those fits are skipped.
# - For speed, you can limit n_samples via N_SAMPLES below.

# %%
from __future__ import annotations

import warnings
import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple
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

# Optional Rich terminal rendering
try:
    from rich import box
    from rich.console import Console
    from rich.table import Table

    RICH_AVAILABLE = True
    console = Console()
except Exception:
    RICH_AVAILABLE = False
    console = None
    warnings.warn("rich not found. Falling back to plain text tables.", RuntimeWarning)


# Config
# `P` is the Tweedie variance power used everywhere in this tutorial.
P = 1.75
# Keep a small ridge penalty in sklearn for numerical stability with the dense OHE design.
# The exact offset-vs-weight theorem is unpenalized; here the sklearn section is mainly a
# practical rate-model tutorial, while the LightGBM section demonstrates the exact encoding.
ALPHA = 0.1
RANDOM_STATE = 42
TEST_SIZE = 0.20
N_SAMPLES = None        # set to e.g. 200_000 for faster runs; None for full dataset
NUM_BOOST_ROUND = 100   # LightGBM
LEARNING_RATE = 0.1
EXPOSURE_FLOOR = 1e-9
# Important tutorial switch:
# disable LightGBM's automatic global base score so that "totals + offset" and
# "rates + exact weights" start from the same raw-score convention.
LGB_DISABLE_BOOST_FROM_AVERAGE = True
np.random.seed(RANDOM_STATE)

# %% [markdown]
# ## Exposure helpers
#
# The two helpers below keep the tutorial implementation aligned with the derivation:
#
# - `tweedie_rate_weights(exposure)` returns the exact rate weight `omega ** (2 - p)`.
# - `log_exposure_offset(exposure)` returns the additive log-offset used in totals models.
#
# Using helpers instead of repeating the formulas makes it harder to accidentally mix
# Poisson-style `omega` weights into the Tweedie examples.

# %%
def tweedie_rate_weights(exposure: np.ndarray) -> np.ndarray:
    """Return the exact Tweedie rate weights omega ** (2 - p)."""
    exposure = np.asarray(exposure, dtype=float)
    return exposure ** (2.0 - P)


def log_exposure_offset(exposure: np.ndarray, floor: float = EXPOSURE_FLOOR) -> np.ndarray:
    """Return log(exposure) with a safety floor for tiny exposures."""
    exposure = np.asarray(exposure, dtype=float)
    return np.log(np.fmax(exposure, floor))


# %% [markdown]
# ## Rich / console output helpers
#
# The tutorial prints a fair amount of tabular information. Rich tables make the
# terminal output easier to scan, especially for side-by-side comparisons between:
#
# - exact vs heuristic weights,
# - offset vs weighted-rate formulations,
# - Tweedie vs Poisson metrics.
#
# The helpers below keep the main modeling code uncluttered and fall back to plain
# pandas/text output if `rich` is unavailable in the environment.

# %%
def emit(message: str, style: Optional[str] = None) -> None:
    """Print a message with Rich when available, else fall back to plain print."""
    if RICH_AVAILABLE:
        console.print(message, style=style)
    else:
        print(message)


def _format_scalar(value: object) -> str:
    """Format scalar values for readable terminal tables."""
    if value is None:
        return "-"
    if isinstance(value, str):
        return value
    try:
        if pd.isna(value):
            return "-"
    except TypeError:
        pass

    if isinstance(value, (np.integer, int)):
        return f"{int(value):,}"

    if isinstance(value, (np.floating, float)):
        value = float(value)
        abs_value = abs(value)
        if abs_value >= 10_000:
            return f"{value:,.2f}"
        if abs_value >= 1:
            return f"{value:,.4f}"
        if abs_value == 0:
            return "0.0000"
        return f"{value:.6f}"

    return str(value)


def _row_style_from_model(model_name: str) -> str:
    """Use model naming conventions to assign a meaningful Rich row style."""
    lower_name = model_name.lower()
    if "observed" in lower_name:
        return "bold white"
    if "offset" in lower_name:
        return "bold cyan"
    if "2-p" in lower_name:
        return "bold green"
    if "_w=exp" in lower_name or "poisson-style" in lower_name:
        return "yellow"
    return ""


def print_summary_table(title: str, rows: Tuple[Tuple[str, object], ...]) -> None:
    """Print a compact two-column summary table."""
    if not RICH_AVAILABLE:
        print(f"\n=== {title} ===")
        for key, value in rows:
            print(f"{key}: {_format_scalar(value)}")
        return

    table = Table(title=title, box=box.ROUNDED, header_style="bold magenta", show_edge=True)
    table.add_column("Item", style="bold")
    table.add_column("Value", justify="right", style="cyan")
    for key, value in rows:
        table.add_row(key, _format_scalar(value))
    console.print(table)


def print_dataframe_table(
    df: pd.DataFrame,
    title: str,
    model_col: str = "model",
    caption: Optional[str] = None,
) -> None:
    """Render a DataFrame as a Rich table with sensible numeric formatting."""
    if not RICH_AVAILABLE:
        print(f"\n=== {title} ===")
        if caption:
            print(caption)
        with pd.option_context("display.max_rows", 100, "display.width", 180):
            print(df)
        return

    table = Table(
        title=title,
        caption=caption,
        box=box.SIMPLE_HEAVY,
        header_style="bold magenta",
        row_styles=["none", "dim"],
        show_edge=True,
    )

    numeric_cols = {col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])}
    for col in df.columns:
        justify = "right" if col in numeric_cols else "left"
        style = "bold" if col == model_col else ""
        table.add_column(str(col), justify=justify, style=style, no_wrap=(col == model_col))

    for _, row in df.iterrows():
        row_style = _row_style_from_model(str(row[model_col])) if model_col in df.columns else ""
        table.add_row(*[_format_scalar(row[col]) for col in df.columns], style=row_style)

    console.print(table)


def print_grouped_metrics_tables(metrics_df: pd.DataFrame, title_prefix: str) -> None:
    """Print one metrics table per evaluation weighting."""
    metrics_reset = metrics_df.reset_index().copy()
    weight_labels = {
        "Exposure": "Business weighting: exposure",
        "Exposure_2mp": f"Exact Tweedie weighting: exposure^(2-p) = exposure^{2.0 - P:.2f}",
    }

    for eval_weight, subset in metrics_reset.groupby("eval_weight", sort=False):
        display_df = subset.drop(columns=["eval_weight"]).copy()
        display_df = display_df.sort_values("model").reset_index(drop=True)
        caption = weight_labels.get(eval_weight, eval_weight)
        print_dataframe_table(
            display_df,
            title=f"{title_prefix} [{eval_weight}]",
            caption=caption,
        )


def build_aggregate_comparison_df(
    observed_label: str,
    observed_value: float,
    rows: Tuple[Tuple[str, float], ...],
    value_col: str,
) -> pd.DataFrame:
    """Build an aggregate comparison table with absolute and relative error columns."""
    table_rows = [(observed_label, observed_value, np.nan, np.nan)]
    for model_name, predicted_value in rows:
        abs_error = predicted_value - observed_value
        rel_error_pct = 100.0 * abs_error / observed_value if observed_value != 0 else np.nan
        table_rows.append((model_name, predicted_value, abs_error, rel_error_pct))

    return pd.DataFrame(
        table_rows,
        columns=["model", value_col, "abs_error", "rel_error_pct"],
    )


# %% [markdown]
# ## Data loading utilities

# %%
def load_mtpl2(n_samples: Optional[int] = None) -> pd.DataFrame:
    """Fetch French MTPL (freq + sev), aggregate severity to totals, clean columns.

    Parameters
    ----------
    n_samples : Optional[int], optional
        Number of samples to load, by default None (all samples).

    Returns
    -------
    pd.DataFrame
        The cleaned DataFrame with aggregated totals.
    """
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


def basic_cleaning(df: pd.DataFrame) -> pd.DataFrame:
    """Apply cleaning logic and derive targets.

    Same cleaning logic as sklearn example, plus derived targets.

    Parameters
    ----------
    df : pd.DataFrame
        The input DataFrame to clean.

    Returns
    -------
    pd.DataFrame
        The cleaned DataFrame with derived targets.
    """
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
def build_column_transformer() -> ColumnTransformer:
    """Build a column transformer for preprocessing features.

    Creates a ColumnTransformer with binning for numeric features,
    one-hot encoding for categorical features, and log scaling for density.

    Returns
    -------
    ColumnTransformer
        The configured column transformer.
    """
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
def _get_aggregated_rates(df: pd.DataFrame, feature: str, weight_name: str, rate_values: np.ndarray) -> pd.DataFrame:
    """Aggregate rates by a feature, exposure-weighted.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame containing the data.
    feature : str
        The feature to group by.
    weight_name : str
        The column name for weights.
    rate_values : np.ndarray
        The rate values to aggregate.

    Returns
    -------
    pd.DataFrame
        The aggregated DataFrame with rates.
    """
    w = df[weight_name].to_numpy()
    tmp = pd.DataFrame({
        feature: df[feature].to_numpy(),
        "w": w,
        "rate_total": rate_values * w,
    })
    grp = tmp.groupby(feature)[["w", "rate_total"]].sum()
    grp["rate"] = grp["rate_total"] / grp["w"].replace(0, np.nan)
    return grp


def lorenz_curve(y_true_rate: np.ndarray, y_pred_rate: np.ndarray, exposure: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Compute the Lorenz curve for exposure-weighted rates.

    Parameters
    ----------
    y_true_rate : np.ndarray
        True rate values.
    y_pred_rate : np.ndarray
        Predicted rate values.
    exposure : np.ndarray
        Exposure values.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        Cumulative exposure and cumulative true rates.
    """
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
def d2_explained(y_true_rate: np.ndarray, y_pred_rate: np.ndarray, sample_weight: Optional[np.ndarray]) -> float:
    """Compute D^2 explained using Tweedie deviance.

    D^2 explained (GLM deviance analogue) computed from mean Tweedie deviance
    at the modeling power P; we use the same sample_weight for both terms.

    Parameters
    ----------
    y_true_rate : np.ndarray
        True rate values.
    y_pred_rate : np.ndarray
        Predicted rate values.
    sample_weight : Optional[np.ndarray]
        Sample weights.

    Returns
    -------
    float
        The D^2 explained value.
    """
    sw = None if sample_weight is None else np.asarray(sample_weight)
    dev = mean_tweedie_deviance(y_true_rate, y_pred_rate, power=P, sample_weight=sw)
    y_bar = np.average(y_true_rate, weights=sw)
    dev_null = mean_tweedie_deviance(y_true_rate, np.full_like(y_true_rate, y_bar), power=P, sample_weight=sw)
    return 1.0 - (dev / dev_null if dev_null > 0 else np.nan)


def evaluate_models_table(df_test: pd.DataFrame, pred_dict: Dict[str, np.ndarray], weights_for_eval: Tuple[str, ...] = ("Exposure", "Exposure_2mp")) -> pd.DataFrame:
    """Build a table of metrics for each model under different weightings.

    Build a tidy table of metrics for each model under two evaluation weightings:
    - Exposure
    - Exposure ** (2 - p)
    Metrics on rates: MAE, MSE, mean Tweedie dev (power=P), and D^2 explained.

    Parameters
    ----------
    df_test : pd.DataFrame
        Test data.
    pred_dict : Dict[str, np.ndarray]
        Dictionary of model predictions.
    weights_for_eval : Tuple[str, ...], optional
        Weighting schemes, by default ("Exposure", "Exposure_2mp").

    Returns
    -------
    pd.DataFrame
        Table of metrics.
    """
    rows = []
    y_true_rate = df_test["PurePremium"].to_numpy()
    for wname in weights_for_eval:
        if wname == "Exposure":
            sw = df_test["Exposure"].to_numpy()
        elif wname == "Exposure_2mp":
            sw = tweedie_rate_weights(df_test["Exposure"].to_numpy())
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

def d2_poisson_explained(y_true_rate: np.ndarray, y_pred_rate: np.ndarray, sample_weight: Optional[np.ndarray]) -> float:
    """Compute D^2 explained using Poisson deviance.

    D^2 explained for Poisson deviance, evaluated on rates with exposure as sample_weight.

    Parameters
    ----------
    y_true_rate : np.ndarray
        True rate values.
    y_pred_rate : np.ndarray
        Predicted rate values.
    sample_weight : Optional[np.ndarray]
        Sample weights.

    Returns
    -------
    float
        The D^2 explained value.
    """
    sw = None if sample_weight is None else np.asarray(sample_weight)
    dev = mean_poisson_deviance(y_true_rate, y_pred_rate, sample_weight=sw)
    y_bar = np.average(y_true_rate, weights=sw)
    dev_null = mean_poisson_deviance(y_true_rate, np.full_like(y_true_rate, y_bar), sample_weight=sw)
    return 1.0 - (dev / dev_null if dev_null > 0 else np.nan)


def evaluate_frequency_models_table(df_test: pd.DataFrame, pred_dict: Dict[str, np.ndarray]) -> pd.DataFrame:
    """Build a table of metrics for frequency models.

    Build a tidy table of metrics for frequency models.
    Evaluation is always exposure-weighted.
    Metrics on rates: MAE, MSE, mean Poisson dev, and D^2 explained.

    Parameters
    ----------
    df_test : pd.DataFrame
        Test data.
    pred_dict : Dict[str, np.ndarray]
        Dictionary of model predictions.

    Returns
    -------
    pd.DataFrame
        Table of metrics.
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
def fit_sklearn_tweedie_rates(X_tr: np.ndarray, X_te: np.ndarray, df_train: pd.DataFrame, df_test: pd.DataFrame, weight_scheme: str) -> Tuple[str, TweedieRegressor, np.ndarray, np.ndarray]:
    """Fit scikit-learn TweedieRegressor with different weight schemes.

    weight_scheme in {"exact", "poisson"}:
      - "exact":   s = Exposure ** (2 - P)
      - "poisson": s = Exposure
    Returns (tag, model, y_pred_rate_train, y_pred_rate_test)

    Parameters
    ----------
    X_tr : np.ndarray
        Training features.
    X_te : np.ndarray
        Test features.
    df_train : pd.DataFrame
        Training data.
    df_test : pd.DataFrame
        Test data.
    weight_scheme : str
        Weight scheme ("exact" or "poisson").

    Returns
    -------
    Tuple[str, TweedieRegressor, np.ndarray, np.ndarray]
        Tag, model, train predictions, test predictions.
    """
    if weight_scheme == "exact":
        # Exact Tweedie rate weighting from Var(Y / omega) = phi * mu^p / omega^(2-p).
        wtr = tweedie_rate_weights(df_train["Exposure"].to_numpy())
        wte = tweedie_rate_weights(df_test["Exposure"].to_numpy())
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
def lgb_offset(X_tr: np.ndarray, X_te: np.ndarray, df_train: pd.DataFrame, df_test: pd.DataFrame) -> Tuple[Optional[lgb.Booster], Optional[np.ndarray], Optional[np.ndarray]]:
    """Fit LightGBM with totals and offset.

    LightGBM: totals + offset
      label = ClaimAmount
      init_score = log(Exposure)
      objective = tweedie, tweedie_variance_power=P
      boost_from_average = False to preserve the exact offset/weight equivalence
    Returns (model, yhat_tr_rate, yhat_te_rate)

    Parameters
    ----------
    X_tr : np.ndarray
        Training features.
    X_te : np.ndarray
        Test features.
    df_train : pd.DataFrame
        Training data.
    df_test : pd.DataFrame
        Test data.

    Returns
    -------
    Tuple[Optional[lgb.Booster], Optional[np.ndarray], Optional[np.ndarray]]
        Model and predictions.
    """
    if not LGB_AVAILABLE:
        return None, None, None

    y_tr = df_train["ClaimAmount"].to_numpy(dtype=float)
    exposure_tr = df_train["Exposure"].to_numpy(dtype=float)
    init_tr = log_exposure_offset(exposure_tr)

    dtrain = lgb.Dataset(X_tr, label=y_tr, init_score=init_tr)
    # Validation must receive the same offset convention; otherwise any validation
    # loss / early stopping logic would be comparing a different objective.
    y_te = df_test["ClaimAmount"].to_numpy(dtype=float)
    exposure_te = df_test["Exposure"].to_numpy(dtype=float)
    init_te = log_exposure_offset(exposure_te)
    dvalid = lgb.Dataset(X_te, label=y_te, init_score=init_te, reference=dtrain)


    params = dict(
        objective="tweedie",
        tweedie_variance_power=P,
        learning_rate=LEARNING_RATE,
        boost_from_average=not LGB_DISABLE_BOOST_FROM_AVERAGE,
        verbose=-1,
        seed=RANDOM_STATE,
    )
    gbm = lgb.train(params,
                    dtrain,
                    valid_sets=[dvalid],
                    num_boost_round=NUM_BOOST_ROUND,
                    )

    # LightGBM predict() returns exp(raw_score_from_trees). The per-row init_score
    # used during training is NOT re-supplied at prediction time, so the returned
    # values are interpreted here as rates per unit exposure.
    yhat_te_rate = gbm.predict(X_te)
    yhat_te_tot = yhat_te_rate * np.fmax(exposure_te, 1e-12)

    # Sanity check on reconstructed totals.
    print_summary_table(
        "LightGBM Tweedie Offset Sanity Check",
        (
            ("Predicted total claim amount", yhat_te_tot.sum()),
            ("Observed total claim amount", df_test["ClaimAmount"].sum()),
        ),
    )

    # Do the same for the training set if needed for evaluation.
    yhat_tr_rate = gbm.predict(X_tr)

    return gbm, yhat_tr_rate, yhat_te_rate

def fit_lgb_rates(X_tr: np.ndarray, X_te: np.ndarray, df_train: pd.DataFrame, df_test: pd.DataFrame, weight_scheme: str) -> Tuple[str, Optional[lgb.Booster], np.ndarray, np.ndarray]:
    """Fit LightGBM with rates and weights.

    LightGBM: rates + weights
      label = PurePremium
      weight = Exposure ** (2 - P)  (exact)  OR  Exposure (poisson-style)
    Returns (tag, model, yhat_tr_rate, yhat_te_rate)

    Parameters
    ----------
    X_tr : np.ndarray
        Training features.
    X_te : np.ndarray
        Test features.
    df_train : pd.DataFrame
        Training data.
    df_test : pd.DataFrame
        Test data.
    weight_scheme : str
        Weight scheme.

    Returns
    -------
    Tuple[str, Optional[lgb.Booster], np.ndarray, np.ndarray]
        Tag, model, train predictions, test predictions.
    """
    if not LGB_AVAILABLE:
        return None, None, None

    y_tr = df_train["PurePremium"].to_numpy(dtype=float)
    y_te = df_test["PurePremium"].to_numpy(dtype=float)

    if weight_scheme == "exact":
        # This is the exact Tweedie analogue of Poisson's exposure weights.
        wtr = tweedie_rate_weights(df_train["Exposure"].to_numpy())
        wte = tweedie_rate_weights(df_test["Exposure"].to_numpy())
        tag = "lgb_rate_w=exp^(2-p)"
    elif weight_scheme == "poisson":
        # Included on purpose as a heuristic baseline: correct only in the Poisson case p=1.
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
        boost_from_average=not LGB_DISABLE_BOOST_FROM_AVERAGE,
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
def fit_sklearn_poisson_rates(X_tr: np.ndarray, X_te: np.ndarray, df_train: pd.DataFrame, df_test: pd.DataFrame) -> Tuple[str, PoissonRegressor, np.ndarray, np.ndarray]:
    """Fit scikit-learn PoissonRegressor.

    sklearn PoissonRegressor: rates + weights
      label = Frequency
      sample_weight = Exposure
    Returns (tag, model, y_pred_rate_train, y_pred_rate_test)

    Parameters
    ----------
    X_tr : np.ndarray
        Training features.
    X_te : np.ndarray
        Test features.
    df_train : pd.DataFrame
        Training data.
    df_test : pd.DataFrame
        Test data.

    Returns
    -------
    Tuple[str, PoissonRegressor, np.ndarray, np.ndarray]
        Tag, model, train predictions, test predictions.
    """
    tag = "sklearn_poisson_rate_w=exp"
    wtr = df_train["Exposure"].to_numpy()
    y_tr = df_train["Frequency"].to_numpy()

    glm = PoissonRegressor(alpha=ALPHA, solver="newton-cholesky")
    glm.fit(X_tr, y_tr, sample_weight=wtr)
    yhat_tr_rate = glm.predict(X_tr)
    yhat_te_rate = glm.predict(X_te)
    return tag, glm, yhat_tr_rate, yhat_te_rate


def fit_lgb_poisson_offset_counts(X_tr: np.ndarray, X_te: np.ndarray, df_train: pd.DataFrame, df_test: pd.DataFrame) -> Tuple[Optional[str], Optional[lgb.Booster], Optional[np.ndarray], Optional[np.ndarray]]:
    """Fit LightGBM Poisson with offset.

    LightGBM (Poisson) with log-exposure offset via init_score.
      - label = ClaimNb (counts)
      - init_score = log(Exposure)
      - objective = 'poisson'
      - predict() returns RATE per unit exposure (not counts)
    Returns: (tag, model, yhat_tr_rate, yhat_te_rate)

    Parameters
    ----------
    X_tr : np.ndarray
        Training features.
    X_te : np.ndarray
        Test features.
    df_train : pd.DataFrame
        Training data.
    df_test : pd.DataFrame
        Test data.

    Returns
    -------
    Tuple[Optional[str], Optional[lgb.Booster], Optional[np.ndarray], Optional[np.ndarray]]
        Tag, model, train predictions, test predictions.
    """
    if not LGB_AVAILABLE:
        return None, None, None, None

    tag = "lgb_poisson_offset"

    # Labels (counts)
    y_tr = df_train["ClaimNb"].to_numpy(dtype=float)
    y_te = df_test["ClaimNb"].to_numpy(dtype=float)

    exp_tr = df_train["Exposure"].to_numpy(dtype=float)
    exp_te = df_test["Exposure"].to_numpy(dtype=float)
    init_tr = log_exposure_offset(exp_tr)
    init_te = log_exposure_offset(exp_te)

    dtrain = lgb.Dataset(X_tr, label=y_tr, init_score=init_tr)
    dvalid = lgb.Dataset(X_te, label=y_te, init_score=init_te, reference=dtrain)

    params = dict(
        objective="poisson",
        learning_rate=LEARNING_RATE,
        boost_from_average=not LGB_DISABLE_BOOST_FROM_AVERAGE,
        verbose=-1,
        seed=RANDOM_STATE,
    )

    gbm = lgb.train(
        params=params,
        train_set=dtrain,
        valid_sets=[dvalid],
        num_boost_round=NUM_BOOST_ROUND,
    )

    # As in the Tweedie offset example above, predict() is interpreted here as a
    # per-unit-exposure rate. Reconstruct totals by multiplying back by exposure.
    yhat_tr_rate = gbm.predict(X_tr)
    yhat_te_rate = gbm.predict(X_te)

    # Check: total count of claims predicted on test set
    yhat_te_tot = yhat_te_rate * exp_te
    print_summary_table(
        "LightGBM Poisson Offset Sanity Check",
        (
            ("Predicted total claim count", yhat_te_tot.sum()),
            ("Observed total claim count", y_te.sum()),
        ),
    )

    return tag, gbm, yhat_tr_rate, yhat_te_rate


def fit_lgb_poisson_rates_weights(X_tr: np.ndarray, X_te: np.ndarray, df_train: pd.DataFrame, df_test: pd.DataFrame) -> Tuple[Optional[str], Optional[lgb.Booster], Optional[np.ndarray], Optional[np.ndarray]]:
    """Fit LightGBM Poisson with rates and weights.

    LightGBM: rates + weights
      label = Frequency
      weight = Exposure
      objective = poisson
    Returns (tag, model, yhat_tr_rate, yhat_te_rate)

    Parameters
    ----------
    X_tr : np.ndarray
        Training features.
    X_te : np.ndarray
        Test features.
    df_train : pd.DataFrame
        Training data.
    df_test : pd.DataFrame
        Test data.

    Returns
    -------
    Tuple[Optional[str], Optional[lgb.Booster], Optional[np.ndarray], Optional[np.ndarray]]
        Tag, model, train predictions, test predictions.
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
        boost_from_average=not LGB_DISABLE_BOOST_FROM_AVERAGE,
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

print_summary_table(
    "Run Configuration",
    (
        ("Train size", len(df_train)),
        ("Test size", len(df_test)),
        ("Tweedie power p", P),
        ("Ridge alpha (sklearn)", ALPHA),
        ("LightGBM rounds", NUM_BOOST_ROUND),
        (
            "Exact-equivalence mode",
            "boost_from_average=False"
            if LGB_DISABLE_BOOST_FROM_AVERAGE
            else "boost_from_average=True",
        ),
    ),
)

# %% [markdown]
# ## Pure Premium Model Fitting (Tweedie)
# 
# The two "exact" Tweedie LightGBM models below should now be numerically very close:
# 
# - `lgb_offset`: totals + `init_score = log(exposure)`
# - `lgb_rate_w=exp^(2-p)`: rates + exact Tweedie weights
# 
# If they are not close, the first thing to check is whether some default such as
# `boost_from_average` reintroduced a different starting raw score.

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
    print_summary_table(
        "LightGBM Exact Equivalence Check",
        (
            (
                "Max abs gap: offset vs exact weighted rate",
                np.max(np.abs(lgb_off_te - lgb_exact_te)),
            ),
        ),
    )

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
emit("Saved Tweedie Lorenz curve to lorenz_curve_tweedie.png", style="green")

# %% [markdown]
# ## Metrics tables (TEST) under two evaluation weightings

# %%
metrics_tbl = evaluate_models_table(df_test, pred_rate_test, weights_for_eval=("Exposure", "Exposure_2mp"))
print_grouped_metrics_tables(metrics_tbl.sort_index(), title_prefix="Tweedie Test Metrics")

# %% [markdown]
# ## Aggregate totals comparison (TEST)

# %%
y_true_tot = (df_test["PurePremium"].to_numpy() * exposure_test).sum()
agg_rows = [
    (tag_sk_exact, np.sum(exposure_test * sk_exact_te)),
    (tag_sk_pois,  np.sum(exposure_test * sk_pois_te)),
]
if LGB_AVAILABLE:
    agg_rows.extend([
        ("lgb_offset", np.sum(exposure_test * lgb_off_te)),
        (tag_lgb_exact, np.sum(exposure_test * lgb_exact_te)),
        (tag_lgb_pois,  np.sum(exposure_test * lgb_pois_te)),
    ])
agg_df = build_aggregate_comparison_df(
    observed_label="Observed totals",
    observed_value=y_true_tot,
    rows=tuple(agg_rows),
    value_col="sum_predicted_totals",
)
print_dataframe_table(
    agg_df,
    title="Aggregate Predicted Totals On TEST",
    caption="Absolute and relative error are measured against the observed total claim amount.",
)

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
    emit(f"Saved sklearn calibration plot for {feat} to calibration_sklearn_{feat}.png", style="green")

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
        emit(f"Saved LightGBM calibration plot for {feat} to calibration_lgbm_{feat}.png", style="green")

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
print_dataframe_table(
    metrics_freq_tbl.sort_index().reset_index(),
    title="Poisson Frequency Test Metrics [Exposure]",
    caption="All frequency metrics are exposure-weighted, matching the Poisson rate formulation.",
)

# %% [markdown]
# ## Aggregate Claim Count Comparison (TEST)

# %%
exposure_test_freq = df_test["Exposure"].to_numpy()
y_true_counts = df_test["ClaimNb"].to_numpy().sum()

agg_rows_freq = []
for model_name, pred_rate in pred_freq_test.items():
    pred_counts = np.sum(exposure_test * pred_rate)
    agg_rows_freq.append((model_name, pred_counts))

agg_df_freq = build_aggregate_comparison_df(
    observed_label="Observed counts",
    observed_value=y_true_counts,
    rows=tuple(agg_rows_freq),
    value_col="sum_predicted_counts",
)
print_dataframe_table(
    agg_df_freq,
    title="Aggregate Predicted Claim Counts On TEST",
    caption="Absolute and relative error are measured against the observed total claim count.",
)

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
emit("Saved Frequency Lorenz curve to lorenz_curve_frequency.png", style="green")

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
emit("Saved Frequency calibration plot for DrivAge to calibration_frequency_DrivAge.png", style="green")
# %%
pred_freq_test
# %%
