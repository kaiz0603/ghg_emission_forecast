# GHG Emission Forecast for ASEAN Countries

This repository contains a notebook-based forecasting workflow for **country-level greenhouse gas (GHG) emissions** across selected ASEAN countries. The project builds country-specific feature panels, selects optimal lags for external drivers, compares several forecasting candidates with rolling backtests, and exports forecast and evaluation files.

The main working artifact is:

- [ghg_emission_forecast_kaggle.ipynb](./ghg_emission_forecast_kaggle.ipynb)

You can also find the notebook at [Kaggle](https://www.kaggle.com/code/kaizhi0603/ghg-emission-forecast).

## Scope

The project currently forecasts GHG emissions for:

- Brunei Darussalam
- Timor-Leste
- Cambodia
- Indonesia
- Lao PDR
- Malaysia
- Myanmar
- Philippines
- Singapore
- Thailand

It uses the following exogenous variables:

- temperature
- energy consumption
- sustainable energy consumption
- forest area percentage
- GDP per capita

## Repository Layout

```text
ghg_emission_forecast/
|-- data_effect_climate_change/                 # Input datasets required by the notebook
|-- ghg_emission_forecast_kaggle.ipynb          # Main forecasting notebook
|-- country_model_evaluation.csv                # Example generated evaluation output
|-- country_ghg_forecast_2024_2028.csv          # Example generated forecast output
|-- implementation_testing_validation_answers.md
|-- pyproject.toml
`-- uv.lock
```

## Analytical Overview

The workflow does the following:

1. Loads wide-format country datasets from the local `data_effect_climate_change/` folder.
2. Extracts country-level annual time series.
3. Selects country-specific lags for external variables using Spearman correlation.
4. Engineers autoregressive, rolling, and change-based features.
5. Compares multiple forecasting candidates:
   - naive persistence
   - 3-year moving average
   - 5-year trend
   - Ridge regression on delta
   - Histogram Gradient Boosting on a log target
6. Uses rolling one-step backtesting to select the best model by horizon.
7. Exports:
   - `country_model_evaluation.csv`
   - `country_ghg_forecast_<start>_<end>.csv`

## Requirements

- Python 3.9 or newer
- Jupyter Notebook or JupyterLab
- Local access to the dataset folder:
  - `data_effect_climate_change/`

The Python dependencies declared in [pyproject.toml](./pyproject.toml) are:

- `ipykernel`
- `lightgbm`
- `pandas`
- `scikit-learn`
- `scipy`
- `seaborn`
- `odfpy`
- `jinja2`

Note:

- The current notebook uses `matplotlib`, `numpy`, `pandas`, `scipy`, and `scikit-learn`.
- The repository currently does **not** include `forecast_pipeline.py`, so replication should be done through the notebook.

## Setup Instructions

### Option 1: Setup with `uv`

If you use `uv`, from the project root run:

```powershell
uv sync
```

Then start Jupyter with the environment:

```powershell
uv run jupyter notebook
```

or

```powershell
uv run jupyter lab
```

### Option 2: Setup with `pip`

Create and activate a virtual environment:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

Install the core dependencies:

```powershell
pip install pandas scikit-learn scipy seaborn matplotlib numpy odfpy jinja2 ipykernel notebook
```

If you prefer JupyterLab:

```powershell
pip install jupyterlab
```

## Data Setup

The notebook expects the input files to exist inside:

```text
data_effect_climate_change/
```

Required files:

- `OWID_CB_TOTAL_GHG_WIDEF.ods`
- `WB_CCKP_TAS_WIDEF.csv`
- `OWID_CB_PRIMARY_ENERGY_CONSUMPTION_WIDEF.csv`
- `WB_SE4ALL_EG_FCON_RNEW_WIDEF.csv`
- `WB_WDI_AG_LND_FRST_ZS_WIDEF.csv`
- `WB_WDI_NY_GDP_PCAP_KN_WIDEF.csv`

The notebook reads this folder directly from the repository root using:

```python
DATA_ROOT_CANDIDATES = [
    Path("data_effect_climate_change"),
]
```

That means replication works as long as the notebook is run from the project root and the dataset folder name remains unchanged.

## Replication Instructions

To fully replicate the project results:

### 1. Clone or open the repository

Make sure you are in the project root:

```powershell
cd /path/to/ghg_emission_forecast
```

### 2. Install dependencies

Use either the `uv` or `pip` setup above.

### 3. Verify the input data folder

Confirm that `data_effect_climate_change/` contains all required files.

### 4. Launch Jupyter

```powershell
uv run jupyter notebook
```

or, if using `pip`:

```powershell
jupyter notebook
```

### 5. Open the notebook

Open:

- [ghg_emission_forecast_kaggle.ipynb](./ghg_emission_forecast_kaggle.ipynb)

### 6. Run all cells from top to bottom

The notebook is organized into these stages:

1. global configuration
2. source-data loading
3. feature engineering and lag selection
4. feature diagnostics
5. candidate model definitions
6. rolling backtest evaluation
7. forecast generation
8. export and visualization

### 7. Collect the generated outputs

Running the notebook writes the final outputs to the project root:

- `country_model_evaluation.csv`
- `country_ghg_forecast_<start>_<end>.csv`

With the current configuration (`FORECAST_YEARS = 5`), the existing example export is:

- `country_ghg_forecast_2024_2028.csv`

## Reproducing With Different Settings

You can change the main experiment settings in the configuration cell near the top of the notebook:

```python
AUTOREGRESSIVE_LAGS = 5
MAX_EXOGENOUS_LAG = 5
MIN_TRAIN_SIZE = 18
MAX_BACKTEST_YEARS = 6
FORECAST_YEARS = 5
FLOAT_PRECISION = 3
```

Common changes:

- Increase `FORECAST_YEARS` to extend the forecast horizon.
- Adjust `MAX_EXOGENOUS_LAG` to test longer lag relationships.
- Adjust `MIN_TRAIN_SIZE` if you want stricter or looser backtest requirements.
- Change `COUNTRIES` to run only a subset of countries.

After changing configuration values, rerun the full notebook from the top.

## Expected Outputs

### `country_model_evaluation.csv`

This file summarizes the selected 1-year-ahead model for each country, including:

- selected model
- selected model family
- backtest years
- RMSE
- MAE
- MAPE
- R²
- benchmark comparison against the naive model
- reliability tier

### `country_ghg_forecast_<start>_<end>.csv`

This file contains multi-year direct forecasts with:

- country
- source year
- prediction year
- horizon
- selected model
- predicted GHG value
- year-over-year change
- backtest RMSE
- benchmark RMSE
- reliability tier

## Notes on Reproducibility

- The notebook uses deterministic settings where applicable, including `random_state=42` for Histogram Gradient Boosting.
- Model selection is based on rolling backtest RMSE, then MAE, then a fixed model ranking.
- Forecasting is done independently per country and per horizon.
- Some output values can still vary slightly across environments due to dependency versions and numeric behavior.

## Limitations

- The repository currently centers on the notebook workflow rather than a packaged CLI pipeline.
- `pyproject.toml` references a console script target (`forecast_pipeline:main`) that is not present in the current repository snapshot.
- Replication instructions in this README therefore focus on the notebook, which is the fully available implementation.

## Troubleshooting

### Missing `odf` or Excel read support

If reading the `.ods` file fails, make sure `odfpy` is installed:

```powershell
pip install odfpy
```

### Jupyter kernel not found

Install and register the environment:

```powershell
pip install ipykernel
python -m ipykernel install --user --name ghg-emission-forecast
```

### File not found errors

Check that:

- you launched Jupyter from the project root
- the folder is named exactly `data_effect_climate_change`
- all required input files are present
