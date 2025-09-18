# %%
import pandas as pd
import numpy as np
from typing import Tuple
import lightgbm as lgb
from scipy.stats import poisson

# %%
# Set seed for reproducibility
np.random.seed(100)

# %% [markdown]
# # DATA GENERATION
# %%
# DATA GENERATION
data_basic = pd.DataFrame({
    'var1': ['A'] * 5000 + ['B'] * 5000,
    'var2': ['C'] * 2500 + ['D'] * 2500 + ['C'] * 2500 + ['D'] * 2500,
    'expos': 1
})

data_easy = pd.DataFrame({
    'var1': ['A'] * 5000 + ['B'] * 5000,
    'var2': ['C'] * 2500 + ['D'] * 2500 + ['C'] * 2500 + ['D'] * 2500,
    'expos': np.tile(np.arange(0.1, 1.1, 0.1), 1000)
})

data_mod = pd.DataFrame({
    'var1': ['A'] * 5000 + ['B'] * 5000,
    'var2': ['C'] * 2500 + ['D'] * 2500 + ['C'] * 2500 + ['D'] * 2500,
    'expos': np.round(np.random.uniform(size=10000), 4)
})

var_impact = pd.DataFrame({
    'var1': ['A', 'B', 'A', 'B'],
    'var2': ['C', 'C', 'D', 'D'],
    'lambda_base': [0.3, 0.7, 1.3, 1.9]
})

# %%
def generate_claim_counts(dt: pd.DataFrame, var_impact: pd.DataFrame) -> pd.DataFrame:
    """Generate claim counts based on exposure and base lambda.

    Merges the base data with feature impacts to get a base lambda,
    calculates the expected claim count (lambda) as `expos * lambda_base`,
    and then simulates the actual claim counts from a Poisson distribution.

    Parameters
    ----------
    dt : pd.DataFrame
        The input DataFrame with features and exposure.
    var_impact : pd.DataFrame
        A DataFrame mapping feature combinations to a `lambda_base`.

    Returns
    -------
    pd.DataFrame
        The input DataFrame augmented with lambda, claim_count, and claim_count_adjusted.
    """
    dt = dt.merge(var_impact, on=['var1', 'var2'])
    dt['lambda'] = dt['expos'] * dt['lambda_base']
    dt['claim_count'] = [poisson.rvs(mu) for mu in dt['lambda']]
    dt['claim_count_adjusted'] = dt['claim_count'] / dt['expos']
    return dt

# %%
data_basic = generate_claim_counts(data_basic, var_impact)
data_easy = generate_claim_counts(data_easy, var_impact)
data_mod = generate_claim_counts(data_mod, var_impact)

# %% [markdown]
# # CHECKS
# %%
# CHECKS
print("Data Basic counts:")
print(data_basic.groupby(['var1', 'var2', 'expos']).size().reset_index(name='N'))
print("\nData Easy Poisson check:")
print(data_easy.groupby(['var1', 'var2', 'expos', 'lambda'])['claim_count'].mean().reset_index())

# %% [markdown]
# # SOLUTION 1: init_score
# %%
# SOLUTION 1: init_score
def solution_1_predict(data_curr: pd.DataFrame) -> np.ndarray:
    """Train a Poisson LightGBM model using log(exposure) as an offset.

    The offset is passed via the `init_score` parameter. `boost_from_average`
    is set to False, which is crucial for offset models.

    Parameters
    ----------
    data_curr : pd.DataFrame
        The training data, containing features, 'claim_count', and 'expos'.

    Returns
    -------
    np.ndarray
        The predicted rates (per unit of exposure).
    """
    data_curr_recoded = data_curr[['var1', 'var2']].copy()
    for col in ['var1', 'var2']:
        data_curr_recoded[col] = pd.Categorical(data_curr_recoded[col]).codes
    
    dtrain = lgb.Dataset(
        data_curr_recoded.values,
        label=data_curr['claim_count'].values,
        init_score=np.log(np.fmax(data_curr['expos'].values, 1e-9)),
        categorical_feature=[0, 1]
    )
    
    param = {
        'objective': 'poisson',
        'num_iterations': 100,
        'learning_rate': 0.5,
        'verbose': -1,
        'boost_from_average': False  # Crucial for offset models
    }
    
    lgb_model = lgb.train(param, dtrain)
    return lgb_model.predict(data_curr_recoded.values)

# %%
predicted_counts_easy = solution_1_predict(data_easy)
data_easy['sol_1_predict'] = predicted_counts_easy *  np.fmax(data_easy['expos'], 1e-9)
data_easy['sol_1_predict_raw'] = predicted_counts_easy 
predicted_counts_mod = solution_1_predict(data_mod)
data_mod['sol_1_predict'] = predicted_counts_mod * np.fmax(data_mod['expos'], 1e-9)
data_mod['sol_1_predict_raw'] = predicted_counts_mod 

# %% [markdown]
# # SOLUTION 1B: Tweedie
# %%
# SOLUTION 1B: Tweedie
def solution_1b_predict(data_curr: pd.DataFrame) -> np.ndarray:
    """Train a Tweedie(p=1) LightGBM model using log(exposure) as an offset.

    This is equivalent to the Poisson model in Solution 1. The offset is
    passed via `init_score`. `boost_from_average` is set to False.

    Parameters
    ----------
    data_curr : pd.DataFrame
        The training data, containing features, 'claim_count', and 'expos'.

    Returns
    -------
    np.ndarray
        The predicted totals. For p=1, this is equivalent to predicted counts.
    """
    data_curr_recoded = data_curr[['var1', 'var2']].copy()
    for col in ['var1', 'var2']:
        data_curr_recoded[col] = pd.Categorical(data_curr_recoded[col]).codes
    
    dtrain = lgb.Dataset(
        data_curr_recoded.values,
        label=data_curr['claim_count'].values,
        init_score=np.log(np.fmax(data_curr['expos'].values, 1e-9)),
        categorical_feature=[0, 1]
    )
    
    param = {
        'objective': 'tweedie',
        'tweedie_variance_power': 1,
        'num_iterations': 100,
        'learning_rate': 0.5,
        'verbose': -1,
        'boost_from_average': False  # Crucial for offset models
    }
    
    lgb_model = lgb.train(param, dtrain)
    return lgb_model.predict(data_curr_recoded.values)

# %%
predicted_counts_easy_1b = solution_1b_predict(data_easy)
data_easy['sol_1b_predict'] = predicted_counts_easy_1b
data_easy['sol_1b_predict_raw'] = predicted_counts_easy_1b / np.fmax(data_easy['expos'], 1e-9)
predicted_counts_mod_1b = solution_1b_predict(data_mod)
data_mod['sol_1b_predict'] = predicted_counts_mod_1b
data_mod['sol_1b_predict_raw'] = predicted_counts_mod_1b / np.fmax(data_mod['expos'], 1e-9)

# %% [markdown]
# # SOLUTION 2: Adjusted claim counts with weights
# %%
# SOLUTION 2: Adjusted claim counts with weights
def solution_2_predict(data_curr: pd.DataFrame) -> np.ndarray:
    """Train a Poisson LightGBM model on rates with exposure as weights.

    The label is the rate (`claim_count / expos`), and the sample weights
    are set to the exposure. This is the mathematically equivalent "weights"
    approach to using an offset.

    Parameters
    ----------
    data_curr : pd.DataFrame
        The training data, containing features and pre-calculated 'claim_count_adjusted' and 'expos'.

    Returns
    -------
    np.ndarray
        The predicted rates.
    """
    data_curr_recoded = data_curr[['var1', 'var2']].copy()
    for col in ['var1', 'var2']:
        data_curr_recoded[col] = pd.Categorical(data_curr_recoded[col]).codes
    
    dtrain = lgb.Dataset(
        data_curr_recoded.values,
        label=data_curr['claim_count_adjusted'].values,
        weight=data_curr['expos'].values,
        categorical_feature=[0, 1]
    )
    
    param = {
        'objective': 'poisson',
        'num_iterations': 100,
        'learning_rate': 0.5,
        'verbose': -1  # boost_from_average=True (default) is correct here
    }
    
    lgb_model = lgb.train(param, dtrain)
    return lgb_model.predict(data_curr_recoded.values)

# %%
data_easy['sol_2_predict_raw'] = solution_2_predict(data_easy)  # This is a predicted RATE
data_easy['sol_2_predict'] = data_easy['sol_2_predict_raw'] * data_easy['expos']  # Convert to COUNT
data_mod['sol_2_predict_raw'] = solution_2_predict(data_mod)    # This is a predicted RATE
data_mod['sol_2_predict'] = data_mod['sol_2_predict_raw'] * data_mod['expos']    # Convert to COUNT

# %% [markdown]
# # SOLUTION 3: Custom objective function
# %%
def solution_3_predict(data_curr: pd.DataFrame) -> np.ndarray:
    """Train a Poisson LightGBM model with a custom objective function.

    The exposure is handled by adding log(exposure) to the model's raw prediction
    inside the objective function to calculate the full linear predictor.

    Parameters
    ----------
    data_curr : pd.DataFrame
        The training data, containing features, 'claim_count', and 'expos'.

    Returns
    -------
    np.ndarray
        The predicted rates (per unit of exposure).
    """
    data_curr_recoded = data_curr[['var1', 'var2']].copy()
    for col in ['var1', 'var2']:
        data_curr_recoded[col] = pd.Categorical(data_curr_recoded[col]).codes

    # We use a closure to pass the exposure array to the objective function,
    # which is cleaner than using a global variable.
    exposure_values = data_curr['expos'].values

    def custom_poisson_obj(y_pred: np.ndarray, data: lgb.Dataset) -> Tuple[np.ndarray, np.ndarray]:
        """Custom Poisson objective function with exposure offset.

        Parameters
        ----------
        y_pred : np.ndarray
            Raw score from the booster (F(x)).
        data : lgb.Dataset
            The lgb.Dataset object, from which we get the labels.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            Gradient and Hessian of the Poisson loss.
        """
        y_true = data.get_label()
        # Add the log-exposure offset to get the full linear predictor (eta)
        eta = y_pred + np.log(np.fmax(exposure_values, 1e-9))
        # The predicted mean (mu) is the exponent of the linear predictor
        mu = np.exp(eta)

        # Gradient of Poisson loss `mu - y_true*log(mu)` w.r.t. y_pred
        grad = mu - y_true
        # Hessian of Poisson loss w.r.t. y_pred
        hess = mu
        return grad, hess

    dtrain = lgb.Dataset(
        data_curr_recoded.values,
        label=data_curr['claim_count'].values,
        categorical_feature=[0, 1]
    )

    param = {
        'objective': custom_poisson_obj,
        'num_iterations': 100,
        'learning_rate': 0.5,
        'verbose': -1,
    }

    lgb_model = lgb.train(param, dtrain)
    return lgb_model.predict(data_curr_recoded.values)

# %%
data_easy['sol_3_predict_raw'] = solution_3_predict(data_easy)
data_easy['sol_3_predict'] = np.exp(data_easy['sol_3_predict_raw']) * data_easy['expos']
data_mod['sol_3_predict_raw'] = solution_3_predict(data_mod)
data_mod['sol_3_predict'] = np.exp(data_mod['sol_3_predict_raw']) * data_mod['expos']

# %%
# ANALYSIS (example checks)

print("\n\n--- Analysis of Predicted Counts (should match lambda) ---")
for i in [1, 2, 3]:
    print(f"\nSolution {i} (Easy Data) Check:")
    # For a given lambda, the predicted count should be close to it
    print(data_easy.groupby('lambda')[f'sol_{i}_predict'].mean())


print("\n\n--- Aggregate Claim Count Comparison (Mod Data) ---")
print(f"Observed counts: {data_mod['claim_count'].sum():.2f}")
print(f"Theoretical counts: {data_mod['lambda'].sum():.2f}")
for i in [1, 2, 3]:
    pred_sum = data_mod[f'sol_{i}_predict'].sum()
    print(f"Solution {i} predicted counts: {pred_sum:.2f}")



print("\n\n--- Equivalence Checks (comparing final predicted counts) ---")
for sol_pair in [("sol_1_predict", "sol_2_predict"), ("sol_1_predict", "sol_3_predict")]:
    mismatches = data_easy[~np.isclose(data_easy[sol_pair[0]], data_easy[sol_pair[1]])]
    print(f"Mismatches between {sol_pair[0]} and {sol_pair[1]} (Easy Data): {len(mismatches)}")

    mismatches_mod = data_mod[~np.isclose(data_mod[sol_pair[0]], data_mod[sol_pair[1]])]
    print(f"Mismatches between {sol_pair[0]} and {sol_pair[1]} (Mod Data):  {len(mismatches_mod)}")
# %%
data_easy.head()
# %%
