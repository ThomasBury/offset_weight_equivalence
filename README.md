# Offset-Weight Equivalence in Poisson and Tweedie Regression

This repository is a tutorial for a specific modeling question that shows up constantly in non-life insurance work:

When a target is observed over different exposures, should you

- model totals with a log-exposure offset, or
- model rates with sample weights?

For Poisson models, the exact rate weight is `omega`.
For Tweedie models with log link and variance `V(mu) = mu^p`, the exact rate weight is `omega^(2-p)`.

This repo explains the math, then shows the consequence in code with `scikit-learn` and `LightGBM`. For the mathematical review, see the slide deck in [docs/offset_and_weight_equivalence.pdf](docs/offset_and_weight_equivalence.pdf).

## Scripts At A Glance

- [scripts/poisson_sim.py](scripts/poisson_sim.py)
  Minimal Poisson toy example: offset, weighted rates, and a custom objective all agree.
- [scripts/tweedie_sim.py](scripts/tweedie_sim.py)
  Synthetic Tweedie example: shows why the exact rate weight is `exposure^(2-p)`.
- [scripts/tweedie_poisson_offset_from_average.py](scripts/tweedie_poisson_offset_from_average.py)
  LightGBM caveat demo: isolates the effect of `boost_from_average` when using offsets.
- [scripts/poisson_tweedie_lightgbm.py](scripts/poisson_tweedie_lightgbm.py)
  Full real-data tutorial: compares exact and heuristic encodings on French MTPL data.

## Who This Is For

This tutorial is a good fit if you are:

- a pricing or actuarial practitioner working with exposure, claim counts, or pure premium,
- a data scientist moving from generic ML toward insurance GLMs and GBMs,
- a student who understands basic regression but wants the offset vs weight question made concrete,
- a LightGBM user who wants to know when `init_score` is a true offset and when it is not enough by itself.

This repo is probably not the best starting point if you are looking for a general introduction to GLMs from scratch. It is focused on one narrow but important modeling equivalence.

## What You Will Learn

By the end, you should be able to answer all of these clearly:

- Why Poisson counts with `offset = log(exposure)` and Poisson rates with `sample_weight = exposure` are equivalent.
- Why the Tweedie analogue is not `sample_weight = exposure`, but `sample_weight = exposure^(2-p)`.
- Why `sample_weight = exposure` is only a heuristic for Tweedie when `p != 1`.
- Why LightGBM needs special care with `boost_from_average` when using offsets.
- How to reconstruct totals correctly from a rate model.

## Fastest Path Through The Repo

If you only want the best onboarding path, do this in order:

1. Read the slide deck in [docs/offset_and_weight_equivalence.pdf](docs/offset_and_weight_equivalence.pdf).
2. Run [scripts/tweedie_poisson_offset_from_average.py](scripts/tweedie_poisson_offset_from_average.py) to see why `boost_from_average=False` matters in LightGBM.
3. Run [scripts/poisson_sim.py](scripts/poisson_sim.py) for the Poisson case.
4. Run [scripts/tweedie_sim.py](scripts/tweedie_sim.py) for the Tweedie case.
5. Run [scripts/poisson_tweedie_lightgbm.py](scripts/poisson_tweedie_lightgbm.py) for the full real-data tutorial.

That sequence moves from concept to implementation to realistic usage.

## Core Takeaway

With a log link:

- Poisson: `totals + offset(log(omega))` is equivalent to `rates + weight=omega`
- Tweedie: `totals + offset(log(omega))` is equivalent to `rates + weight=omega^(2-p)`

The second line is the one many practitioners get wrong.

For Tweedie, dividing by exposure changes the variance:

`Var(Y / omega) = phi * mu^p / omega^(2-p)`

That is why the exact rate weight is `omega^(2-p)`.

## Repository Map

- [scripts/poisson_sim.py](scripts/poisson_sim.py)
  Synthetic Poisson demonstration of offset vs weighted rates.
- [scripts/tweedie_sim.py](scripts/tweedie_sim.py)
  Synthetic Tweedie demonstration showing the exact `omega^(2-p)` weighting.
- [scripts/poisson_tweedie_lightgbm.py](scripts/poisson_tweedie_lightgbm.py)
  Main tutorial on real French MTPL data with `scikit-learn` and `LightGBM`.
- [scripts/tweedie_poisson_offset_from_average.py](scripts/tweedie_poisson_offset_from_average.py)
  Focused experiment showing the interaction between offsets and `boost_from_average` in `LightGBM`.
- [docs/nonlife_regression.tex](docs/nonlife_regression.tex)
  Source for the mathematical slides.
- [docs/offset_and_weight_equivalence.pdf](docs/offset_and_weight_equivalence.pdf)
  Compiled slides, if available.

## Get The Repository

If you have not cloned the repo yet:

```powershell
git clone https://github.com/ThomasBury/offset_weight_equivalence
cd offset_weight_equivalence
```

The commands in this README are written for PowerShell on Windows, since that is the most likely environment for the intended audience. The Python concepts and scripts are not Windows-specific.

## Setup

The project metadata currently requires Python `>=3.14`, as declared in [pyproject.toml](pyproject.toml).

If you want the smoothest setup, use `uv`.
If you already live in Conda, that is also fine.

### Option A: Recommended Setup With `uv`

Install `uv` if needed:

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Restart your terminal, then from the repository root run:

```powershell
uv sync
uv run python -c "import numpy, pandas, sklearn, matplotlib, lightgbm; print('Environment OK')"
```

Then run the tutorial scripts:

```powershell
uv run python scripts/tweedie_poisson_offset_from_average.py
uv run python scripts/poisson_sim.py
uv run python scripts/tweedie_sim.py
uv run python scripts/poisson_tweedie_lightgbm.py
```

### Option B: Setup With Conda

If you prefer Conda, create an environment with a Python version compatible with [pyproject.toml](pyproject.toml):

```powershell
conda create -n offset-weight-equivalence -c conda-forge python=3.14
conda activate offset-weight-equivalence
conda install -c conda-forge numpy pandas scikit-learn matplotlib lightgbm jupyter ipykernel pyarrow
python -c "import numpy, pandas, sklearn, matplotlib, lightgbm; print('Environment OK')"
```

Then run:

```powershell
python scripts/tweedie_poisson_offset_from_average.py
python scripts/poisson_sim.py
python scripts/tweedie_sim.py
python scripts/poisson_tweedie_lightgbm.py
```

## First Run Expectations

Here is what a first-time user should expect:

- [scripts/poisson_sim.py](scripts/poisson_sim.py) should show that offset and weighted-rate Poisson formulations line up numerically.
- [scripts/tweedie_sim.py](scripts/tweedie_sim.py) should show the same story for Tweedie when the exact weight `omega^(2-p)` is used.
- [scripts/tweedie_poisson_offset_from_average.py](scripts/tweedie_poisson_offset_from_average.py) should show that `boost_from_average=True` breaks the clean offset equivalence in LightGBM.
- [scripts/poisson_tweedie_lightgbm.py](scripts/poisson_tweedie_lightgbm.py) should print comparative metrics and save figures such as Lorenz curves and calibration plots.

The real-data tutorial fetches OpenML datasets on first use, so internet access is required for that script.

## Best Starting Script

If you are unsure where to begin, start with [scripts/poisson_tweedie_lightgbm.py](scripts/poisson_tweedie_lightgbm.py).

It is the most complete tutorial in the repo because it:

- explains the exact Tweedie weight inline,
- compares exact and heuristic weighting side by side,
- includes both `scikit-learn` and `LightGBM`,
- prints a diagnostic gap between the exact LightGBM offset fit and the exact weighted-rate fit,
- produces plots that are easier to interpret than raw optimization output alone.

## Reading The Results

When you run the main tutorial, interpret the output like this:

- `sklearn_rate_w=exp^(2-p)` is the exact Tweedie rate formulation.
- `lgb_offset` is the totals-plus-offset LightGBM formulation.
- `lgb_rate_w=exp^(2-p)` is the exact Tweedie weighted-rate LightGBM formulation.
- `lgb_rate_w=exp` is the Poisson-style heuristic weight, included on purpose as a contrast.

The important comparison is:

- `lgb_offset` vs `lgb_rate_w=exp^(2-p)`

Those should be very close when the tutorial is behaving correctly.

The interesting non-equivalence is:

- `lgb_rate_w=exp^(2-p)` vs `lgb_rate_w=exp`

Those usually differ, because `exp` is not the exact Tweedie rate weight unless `p = 1`.

## Beginner Glossary

- Exposure `omega`
  Amount of observed risk, often policy-years.
- Frequency
  Expected claims per unit exposure.
- Severity
  Expected claim size given a claim.
- Pure premium
  Expected claim amount per unit exposure.
- Offset
  A fixed additive term in the linear predictor, here typically `log(exposure)`.
- Sample weight
  A per-row weight used during fitting.
- Log link
  The link where `log(mu)` is modeled linearly.
- Tweedie power `p`
  The exponent in `Var(Y) = phi * mu^p`.
- Lorenz curve
  A rank-ordering plot used to evaluate segmentation.
- Calibration plot
  A comparison of observed and predicted rates across bins or categories.

## Practical Lessons From This Repo

- If you model Poisson rates, use `sample_weight = exposure`.
- If you model Tweedie rates, use `sample_weight = exposure^(2-p)`.
- If you model totals with a log-exposure offset in LightGBM, do not trust the default base-score behavior blindly.
- If you use LightGBM offsets, make sure train and validation sets receive the same offset convention.
- If you train on rates, remember that totals are reconstructed as `exposure * predicted_rate`.

## Known Pitfalls

- Using `sample_weight = exposure` for Tweedie because it worked for Poisson.
  This is the most common conceptual mistake the repo is meant to correct.
- Forgetting that `boost_from_average` changes the starting raw score in LightGBM.
  This can make two mathematically equivalent encodings look different in practice.
- Comparing totals from one model to rates from another without reconstructing them consistently.
- Ignoring tiny exposures.
  Very small exposures can create numerical instability and are clamped where needed in the scripts.

## Troubleshooting

- `lightgbm` fails to install on Windows
  Try the Conda path first: `conda install -c conda-forge lightgbm`.
- OpenML download is slow or fails
  Retry later, check internet access, or work through the simulation scripts first.
- The real-data tutorial feels too long for a first pass
  Start with [scripts/tweedie_poisson_offset_from_average.py](scripts/tweedie_poisson_offset_from_average.py), then [scripts/tweedie_sim.py](scripts/tweedie_sim.py).
- Your exact LightGBM offset and weighted-rate fits are not close
  Check whether `boost_from_average` has been re-enabled and whether the same exposure treatment is used on validation data.

## Recommended Study Order

If you want a structured learning path, use this order:

1. Slide deck: [docs/offset_and_weight_equivalence.pdf](docs/offset_and_weight_equivalence.pdf)
2. Offset/base-score caveat: [scripts/tweedie_poisson_offset_from_average.py](scripts/tweedie_poisson_offset_from_average.py)
3. Poisson equivalence: [scripts/poisson_sim.py](scripts/poisson_sim.py)
4. Tweedie equivalence: [scripts/tweedie_sim.py](scripts/tweedie_sim.py)
5. Full tutorial: [scripts/poisson_tweedie_lightgbm.py](scripts/poisson_tweedie_lightgbm.py)

## Summary

This repo exists to make one point unambiguous:

- Poisson rates use weight `omega`
- Tweedie rates use weight `omega^(2-p)`

If you remember only one practical rule from this tutorial, remember the second line.
