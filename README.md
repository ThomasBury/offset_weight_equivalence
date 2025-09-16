# Offset–Weight Equivalence in Poisson and Tweedie Regression (GLM/GBM)

This repository demonstrates, with code and figures, the equivalence between two common ways to encode exposure in insurance modeling:

- Offset formulation: model totals with a log-exposure offset.
- Rates + weights formulation: model per-exposure rates with appropriate sample weights.

It covers both Poisson (frequency) and Tweedie (aggregate cost) models using scikit-learn and LightGBM, and includes simulation and real-data experiments.

## Key idea (short)

- Poisson with log link: either model counts with an offset log(ω) or model rates with sample_weight = ω. Both give identical score and Hessian, hence the same estimator.
- Tweedie with log link and variance V(μ) = μ^p: either model totals with an offset log(ω), or model rates with sample_weight = ω^(2−p). Again, both produce identical estimating equations under the usual GLM conditions.

For precise statements and derivations, see [PDF](.\docs\offset_and_weight_equivalence.pdf).

## Core terms (beginner glossary)

- Non-life insurance pricing: pricing for property & casualty (e.g., auto, home). We model expected claims cost rather than final premium (which adds expenses, profit, taxes).
- Exposure (ω): the amount of risk observed (e.g., policy-years). Partial-term policies have ω < 1. Exposure scales both expected counts and totals.
- Frequency: expected number of claims per unit exposure. Often modeled with a Poisson GLM.
- Severity: size of an individual claim. Often modeled with a Gamma GLM. Aggregate cost combines frequency and severity.
- Pure premium (rate): expected claim amount per unit exposure; equivalently Total/Exposure. This is the modeling target before adding loadings.
- Totals vs rates: Total = Exposure × Rate. You can model totals with an offset or rates with weights; this repo shows how they match.
- GLM (generalized linear model): models a transformed mean with a link function, e.g., log(mean) = Xβ.
- GBM (gradient boosting machine): ensemble of decision trees trained on gradients; LightGBM is a popular GBM library used here.
- Log link: link function using log(mean). With a log link, multiplying the mean by ω adds a constant log(ω) to the linear predictor, enabling offsets.
- Offset: a known additive term in the linear predictor (here log(ω)). Lets the model learn rates while predictions scale correctly to totals.
- Sample weight: per-row weight used during fitting. For Poisson rates use weight = ω; for Tweedie rates use weight = ω^(2−p).
- Tweedie regression: GLM family with variance Var(Y) = φ·μ^p. Special cases: p=1 Poisson (frequency), p=2 Gamma (severity), 1<p<2 compound Poisson–Gamma (aggregate claims).
- Hessian: matrix of second derivatives of the loss; used by optimizers/GBMs. The offset and weighted-rate encodings yield the same gradients/Hessians.
- Calibration plot: compares observed vs predicted rates across bins of a feature; good models track the observed curve.
- Lorenz curve / Gini: ranks risks by predicted rate and plots cumulative share; higher Gini indicates stronger segmentation.

## Repository layout

- scripts/
  - poisson_sim.py — synthetic Poisson example: offset vs weighted rates equivalence, incl. LightGBM variants.
  - tweedie_sim.py — synthetic Tweedie example (1 < p ≤ 2): equivalence checks across implementations.
  - poisson_tweedie_lightgbm.py — real-data experiment (French MTPL: freMTPL2) comparing scikit-learn and LightGBM encodings; saves Lorenz and calibration plots.
  - tweedie_poisson_offset_from_average.py — empirical check of LightGBM’s boost_from_average interaction with offsets.
- docs/
  - nonlife_regression.tex — slides explaining the mathematics of the equivalence and practical caveats.
  - offset_and_weight_equivalence.pdf — (if present) compiled slides.
- figures/ — generated figures (created when running scripts that save plots).
- pyproject.toml — Python package metadata and runtime dependencies.
- uv.lock — lockfile for reproducible installs with uv.

Note: Some duplicate or misspelled script names may exist at the repository root; use the scripts/ versions as authoritative.

## Quick start (recommended paths)

First, clone the repository to your local machine and navigate into the project directory:

```powershell
git clone https://github.com/ThomasBury/offset_weight_equivalence
cd offset_weight_equivalence
```

Then, pick **ONE** of the two setup methods below. Both create an isolated environment and install the dependencies declared in `pyproject.toml`.
All commands below are intended for PowerShell (pwsh).

### Option A — Setup with Conda

1. **Install Miniforge or Miniconda** (Conda) if you don’t have it.
    - Miniforge (conda-forge first): <https://conda-forge.org/download/>

2. **Create and activate an environment** (Python 3.13):

    ```powershell
    # Create a new environment named 'tweedie-regr'
    conda create -n tweedie-regr -c conda-forge python=3.13
    
    # Activate it
    conda activate tweedie-regr
    ```

3. **Install dependencies** from conda-forge:

    ```powershell
    conda install -c conda-forge numpy pandas scikit-learn matplotlib lightgbm jupyter ipykernel
    ```

4. **Verify installation**:

    ```powershell
    python -c "import numpy, pandas, sklearn, matplotlib, lightgbm; print('OK')"
    ```

5. **Run scripts**:

    ```powershell
    python scripts/poisson_sim.py
    python scripts/tweedie_sim.py
    python scripts/poisson_tweedie_lightgbm.py
    python scripts/tweedie_poisson_offset_from_average.py
    ```

Notes:

- `poisson_tweedie_lightgbm.py` fetches OpenML datasets (internet required on first run).
- The scripts print metrics/tables and may save PNG figures to the repository root (e.g., lorenz_curve_tweedie.png, calibration_*.png).

### Option B — Setup with uv (Fast, Lockfile-based)

`uv` is a fast Python package installer and environment manager. These steps will create a virtual environment and run the scripts without needing to manually activate it.

1. **Install `uv`** (PowerShell):

    ```powershell
    powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
    ```

    Close and reopen your terminal to ensure `uv` is in your `PATH`. Verify with `uv --version`.

    or using `pipx` (to install `pipx`: run `pip install -U pipx`)

    ```powershell
    pipx install uv
    ```

2. **Sync dependencies** from `uv.lock`:

    ```powershell
    uv sync
    ```

3. **Verify and Run Scripts**: Use `uv run` to execute commands within the managed environment.

    ```powershell
    # Verify installation
    uv run python -c "import numpy, pandas, sklearn, matplotlib, lightgbm; print('OK')"

    # Run the main scripts
    uv run python scripts/poisson_sim.py
    uv run python scripts/tweedie_sim.py
    uv run python scripts/poisson_tweedie_lightgbm.py
    uv run python scripts/tweedie_poisson_offset_from_average.py
    ```

Notes:

- If `uv sync` reports issues with `lightgbm` wheels on your platform, you can either:
  - fall back to Conda for `lightgbm`, or
  - install a prebuilt wheel compatible with your Python/CPU.

## What Each Script Demonstrates

- **`scripts/poisson_sim.py`**
  - **Goal**: Show equivalence for Poisson models in LightGBM.
  - **Methods**: Compares (1) `init_score` offset, (2) weighted rates, and (3) a custom objective function.
  - **Result**: Confirms that all three methods produce numerically identical predictions.

- **`scripts/tweedie_sim.py`**
  - **Goal**: Show equivalence for Tweedie models across `scikit-learn` and `LightGBM`.
  - **Methods**: Compares `scikit-learn` (rates + `weight=ω^(2-p)`) with `LightGBM` (offset and weighted rates).
  - **Result**: Confirms that predictions are nearly identical, validating the equivalence for Tweedie models.

- **`scripts/poisson_tweedie_lightgbm.py`**
  - **Goal**: Apply and compare all methods on a real-world insurance dataset.
  - **Methods**: Fits Poisson and Tweedie models using both `scikit-learn` and `LightGBM` with offset, exact weights, and heuristic weights.
  - **Result**: Produces comparative metrics, Lorenz curves, and calibration plots, demonstrating the practical implications of each method.

- **`scripts/tweedie_poisson_offset_from_average.py`**
  - **Goal**: Isolate and test the crucial `boost_from_average` parameter in `LightGBM` when using an offset.
  - **Result**: Empirically proves that `boost_from_average` must be set to `False` when using `init_score` for an offset, as setting it to `True` leads to significant prediction bias.

## Key Findings

1. **Equivalence is Confirmed**: For both Poisson and Tweedie models with a log-link, modeling **totals with a log-exposure offset** is mathematically and empirically equivalent to modeling **rates with the correct sample weights** (`weight=exposure` for Poisson, `weight=exposure^(2-p)` for Tweedie).
2. **`boost_from_average` is Critical for Offsets**: When using the offset method (`init_score`) in LightGBM, it is essential to set `boost_from_average=False`. The default `True` value causes the model to double-count the base rate, leading to incorrect predictions.
3. **Heuristic Weights are Not Equivalent**: Using a "Poisson-style" weight (`weight=exposure`) for a Tweedie model is a heuristic, not an exact equivalent to the offset method. The experiments show it produces different results from the theoretically correct approaches.

## Reproducibility

- Seeds are set in scripts where applicable; minor numerical differences can occur between libraries/platforms.
- For LightGBM runs using early stopping/validation, tree growth is deterministic given the same seed and data ordering.

## Troubleshooting

- lightgbm installation on Windows fails with pip/uv:
  - Use Conda: `conda install -c conda-forge lightgbm`.
- OpenML download fails or is slow:
  - Ensure internet access, try again later, or reduce `N_SAMPLES` in the script to speed up experimentation.
- Very small exposures cause numerical issues:
  - Scripts clamp exposure with small epsilons (e.g., `1e-9`) where needed.
- Offsets and early stopping in LightGBM:
  - When using `init_score` for offsets, ensure validation datasets also carry the same `init_score`; the provided code does this correctly.
