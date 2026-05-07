from __future__ import annotations

import os
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import ConstantInputWarning
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "4")
warnings.filterwarnings("ignore", category=ConstantInputWarning)


# Easy-to-tune global parameters.
DATA_ROOT = Path("data_effect_climate_change")
MIN_TRAIN_SIZE = 18
MAX_BACKTEST_YEARS = 6
MAX_EXOGENOUS_LAG = 5
FORECAST_YEARS = 5
FLOAT_PRECISION = 3


COUNTRIES = [
    "Brunei Darussalam",
    "Timor-Leste",
    "Cambodia",
    "Indonesia",
    "Lao PDR",
    "Malaysia",
    "Myanmar",
    "Philippines",
    "Singapore",
    "Thailand",
]

EXOGENOUS_FEATURES = [
    "temperature",
    "energy_consumption",
    "sustainable_energy",
    "forest_area_pct",
    "gdp_per_capita",
]


@dataclass(frozen=True)
class ForecastConfig:
    data_root: Path = DATA_ROOT
    min_train_size: int = MIN_TRAIN_SIZE
    max_backtest_years: int = MAX_BACKTEST_YEARS
    max_exogenous_lag: int = MAX_EXOGENOUS_LAG
    forecast_years: int = FORECAST_YEARS
    float_precision: int = FLOAT_PRECISION

    @property
    def direct_forecast_horizons(self) -> tuple[int, ...]:
        return tuple(range(1, self.forecast_years + 1))


@dataclass
class CountryDataset:
    country: str
    panel: pd.DataFrame
    model_df: pd.DataFrame
    feature_cols: list[str]
    lags: dict[str, int]
    series_map: dict[str, pd.Series]


@dataclass(frozen=True)
class CandidateSpec:
    name: str
    family: str
    fit: Callable[[pd.DataFrame, list[str]], object | None]
    predict: Callable[[object | None, pd.DataFrame, pd.Series, list[str], int], float]


class DataRepository:
    def __init__(self, data_root: Path) -> None:
        self.data_root = data_root
        self.df_ghg = pd.read_excel(data_root / "OWID_CB_TOTAL_GHG_WIDEF.ods")
        self.df_temp = pd.read_csv(data_root / "WB_CCKP_TAS_WIDEF.csv")
        self.df_energy_consumption = pd.read_csv(
            data_root / "OWID_CB_PRIMARY_ENERGY_CONSUMPTION_WIDEF.csv"
        )
        self.df_sustainable_energy = pd.read_csv(
            data_root / "WB_SE4ALL_EG_FCON_RNEW_WIDEF.csv"
        )
        self.df_forest_area = pd.read_csv(
            data_root / "WB_WDI_AG_LND_FRST_ZS_WIDEF.csv"
        )
        self.df_gdp = pd.read_csv(data_root / "WB_WDI_NY_GDP_PCAP_KN_WIDEF.csv")

    def get_country_series(self, country: str) -> dict[str, pd.Series]:
        return {
            "ghg_current": self._extract_series(country, self.df_ghg),
            "temperature": self._extract_series(country, self.df_temp),
            "energy_consumption": self._extract_series(
                country, self.df_energy_consumption
            ),
            "sustainable_energy": self._extract_series(
                country, self.df_sustainable_energy
            ),
            "forest_area_pct": self._extract_series(country, self.df_forest_area),
            "gdp_per_capita": self._extract_series(country, self.df_gdp),
        }

    @staticmethod
    def _extract_series(
        country: str, df: pd.DataFrame, min_year: int = 1950
    ) -> pd.Series:
        frame = df.copy()
        frame.columns = frame.columns.astype(str)
        matches = frame.index[frame["REF_AREA_LABEL"] == country]
        if len(matches) == 0:
            raise KeyError(f"{country} is missing from source data.")

        year_cols = sorted(
            [
                column
                for column in frame.columns
                if column.isdigit() and min_year <= int(column) <= 2100
            ],
            key=int,
        )
        series = pd.to_numeric(frame.loc[matches[0], year_cols], errors="coerce")
        series.index = pd.Index([int(year) for year in year_cols], name="year")
        return series.astype(float).sort_index()


def compute_optimal_lag(target: pd.Series, variable: pd.Series, max_lag: int) -> int:
    best_lag = 0
    best_abs_rho = -1.0

    for lag in range(max_lag + 1):
        trial = pd.DataFrame({"target": target, "variable": variable.shift(lag)}).dropna()
        if len(trial) < 3:
            continue
        rho, _ = stats.spearmanr(trial["variable"], trial["target"])
        if pd.notna(rho) and abs(rho) > best_abs_rho:
            best_lag = lag
            best_abs_rho = abs(rho)

    return best_lag


def safe_pct_change(series: pd.Series, periods: int = 1) -> pd.Series:
    return series.pct_change(periods=periods, fill_method=None).replace(
        [np.inf, -np.inf], np.nan
    )


def build_country_dataset(
    repository: DataRepository, country: str, config: ForecastConfig
) -> CountryDataset:
    series_map = repository.get_country_series(country)
    ghg = series_map["ghg_current"]
    lags = {
        feature_name: compute_optimal_lag(
            ghg, series_map[feature_name], max_lag=config.max_exogenous_lag
        )
        for feature_name in EXOGENOUS_FEATURES
    }

    years = sorted(ghg.index)
    panel = pd.DataFrame({"year": years, "country": country})
    panel["ghg_current"] = ghg.reindex(years).values

    for feature_name, lag in lags.items():
        panel[feature_name] = series_map[feature_name].shift(lag).reindex(years).values

    for lag in range(1, 6):
        panel[f"ghg_lag{lag}"] = panel["ghg_current"].shift(lag)

    panel["ghg_roll3"] = panel["ghg_current"].shift(1).rolling(3).mean()
    panel["ghg_roll5"] = panel["ghg_current"].shift(1).rolling(5).mean()
    panel["ghg_std3"] = panel["ghg_current"].shift(1).rolling(3).std()
    panel["ghg_diff1"] = panel["ghg_current"].diff(1)
    panel["ghg_diff2"] = panel["ghg_current"].diff(2)
    panel["ghg_growth1"] = safe_pct_change(panel["ghg_current"], periods=1)
    panel["year_index"] = panel["year"] - panel["year"].min()
    panel["ghg_next_year"] = panel["ghg_current"].shift(-1)
    panel["ghg_delta_next_year"] = panel["ghg_next_year"] - panel["ghg_current"]

    feature_cols = [
        "year",
        "year_index",
        "ghg_current",
        "ghg_lag1",
        "ghg_lag2",
        "ghg_lag3",
        "ghg_lag4",
        "ghg_lag5",
        "ghg_roll3",
        "ghg_roll5",
        "ghg_std3",
        "ghg_diff1",
        "ghg_diff2",
        "ghg_growth1",
        *EXOGENOUS_FEATURES,
    ]

    model_df = panel.dropna(subset=["ghg_next_year"]).copy().reset_index(drop=True)
    return CountryDataset(
        country=country,
        panel=panel,
        model_df=model_df,
        feature_cols=feature_cols,
        lags=lags,
        series_map=series_map,
    )


def build_horizon_training_df(dataset: CountryDataset, horizon_years: int) -> pd.DataFrame:
    training_df = dataset.panel.copy()
    training_df["target_ghg"] = training_df["ghg_current"].shift(-horizon_years)
    training_df["target_delta"] = training_df["target_ghg"] - training_df["ghg_current"]
    return training_df.dropna(subset=["target_ghg"]).copy().reset_index(drop=True)


def fit_naive(_: pd.DataFrame, __: list[str]) -> None:
    return None


def predict_naive(
    _: object | None,
    feature_row: pd.DataFrame,
    __: pd.Series,
    ___: list[str],
    ____: int,
) -> float:
    return max(0.0, float(feature_row["ghg_current"].iloc[0]))


def fit_mean3(_: pd.DataFrame, __: list[str]) -> None:
    return None


def predict_mean3(
    _: object | None,
    __: pd.DataFrame,
    history: pd.Series,
    ___: list[str],
    ____: int,
) -> float:
    recent = history.dropna().tail(3)
    if len(recent) == 0:
        return np.nan
    return max(0.0, float(recent.mean()))


def fit_trend5(_: pd.DataFrame, __: list[str]) -> None:
    return None


def predict_trend5(
    _: object | None,
    __: pd.DataFrame,
    history: pd.Series,
    ___: list[str],
    horizon_years: int,
) -> float:
    observed = history.dropna().astype(float).sort_index().tail(5)
    if len(observed) == 0:
        return np.nan
    if len(observed) == 1:
        return float(observed.iloc[-1])

    x = observed.index.to_numpy(dtype=float)
    y = observed.to_numpy(dtype=float)
    slope, intercept = np.polyfit(x, y, 1)
    forecast_year = float(x[-1] + horizon_years)
    return max(0.0, float(slope * forecast_year + intercept))


def fit_ridge_delta(train_df: pd.DataFrame, feature_cols: list[str]) -> Pipeline:
    model = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", Ridge(alpha=1.0)),
        ]
    )
    model.fit(train_df[feature_cols], train_df["target_delta"])
    return model


def predict_ridge_delta(
    model: Pipeline,
    feature_row: pd.DataFrame,
    _: pd.Series,
    feature_cols: list[str],
    __: int,
) -> float:
    current = float(feature_row["ghg_current"].iloc[0])
    delta = float(model.predict(feature_row[feature_cols])[0])
    return max(0.0, current + delta)


def fit_histgb_log(train_df: pd.DataFrame, feature_cols: list[str]) -> Pipeline:
    model = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            (
                "model",
                HistGradientBoostingRegressor(
                    learning_rate=0.05,
                    max_depth=3,
                    max_iter=300,
                    l2_regularization=0.1,
                    random_state=42,
                ),
            ),
        ]
    )
    target = np.log1p(np.clip(train_df["target_ghg"].to_numpy(dtype=float), 0.0, None))
    model.fit(train_df[feature_cols], target)
    return model


def predict_histgb_log(
    model: Pipeline,
    feature_row: pd.DataFrame,
    _: pd.Series,
    feature_cols: list[str],
    __: int,
) -> float:
    prediction = float(np.expm1(model.predict(feature_row[feature_cols])[0]))
    return max(0.0, prediction)


def get_candidate_specs() -> list[CandidateSpec]:
    return [
        CandidateSpec(
            name="naive",
            family="baseline_persistence",
            fit=fit_naive,
            predict=predict_naive,
        ),
        CandidateSpec(
            name="mean3",
            family="baseline_recent_average",
            fit=fit_mean3,
            predict=predict_mean3,
        ),
        CandidateSpec(
            name="trend5",
            family="baseline_recent_linear_trend",
            fit=fit_trend5,
            predict=predict_trend5,
        ),
        CandidateSpec(
            name="ridge_delta",
            family="regularized_linear_delta",
            fit=fit_ridge_delta,
            predict=predict_ridge_delta,
        ),
        CandidateSpec(
            name="histgb_log",
            family="boosted_tree_log_target",
            fit=fit_histgb_log,
            predict=predict_histgb_log,
        ),
    ]


def build_backtest_history(train_df: pd.DataFrame, test_row: pd.DataFrame) -> pd.Series:
    history = pd.concat(
        [
            train_df[["year", "ghg_current"]],
            test_row[["year", "ghg_current"]],
        ],
        ignore_index=True,
    )
    history = history.drop_duplicates("year", keep="last").sort_values("year")
    return history.set_index("year")["ghg_current"]


def run_rolling_backtest(
    dataset: CountryDataset,
    candidate_specs: list[CandidateSpec],
    config: ForecastConfig,
    horizon_years: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    model_df = (
        build_horizon_training_df(dataset, horizon_years)
        .sort_values("year")
        .reset_index(drop=True)
    )
    available_splits = len(model_df) - config.min_train_size
    if available_splits < 3:
        return pd.DataFrame(), pd.DataFrame()

    n_splits = min(config.max_backtest_years, available_splits)
    test_start = len(model_df) - n_splits
    prediction_rows: list[dict[str, object]] = []
    candidate_metric_rows: list[dict[str, object]] = []

    for spec in candidate_specs:
        spec_predictions: list[dict[str, object]] = []
        for test_idx in range(test_start, len(model_df)):
            train_df = model_df.iloc[:test_idx].copy()
            test_row = model_df.iloc[[test_idx]].copy()
            history = build_backtest_history(train_df, test_row)
            model = spec.fit(train_df, dataset.feature_cols)
            prediction = spec.predict(
                model, test_row, history, dataset.feature_cols, horizon_years
            )
            prediction_rows.append(
                {
                    "country": dataset.country,
                    "forecast_horizon_years": horizon_years,
                    "candidate_model": spec.name,
                    "candidate_family": spec.family,
                    "source_year": int(test_row["year"].iloc[0]),
                    "predict_for_year": int(test_row["year"].iloc[0] + horizon_years),
                    "actual_ghg_mtco2e": float(test_row["target_ghg"].iloc[0]),
                    "predicted_ghg_mtco2e": float(prediction),
                }
            )
            spec_predictions.append(prediction_rows[-1])

        prediction_frame = pd.DataFrame(spec_predictions)
        actual = prediction_frame["actual_ghg_mtco2e"].to_numpy(dtype=float)
        predicted = prediction_frame["predicted_ghg_mtco2e"].to_numpy(dtype=float)
        candidate_metric_rows.append(
            {
                "country": dataset.country,
                "forecast_horizon_years": horizon_years,
                "candidate_model": spec.name,
                "candidate_family": spec.family,
                "backtest_years": int(len(prediction_frame)),
                "rmse_mtco2e": float(np.sqrt(mean_squared_error(actual, predicted))),
                "mae_mtco2e": float(mean_absolute_error(actual, predicted)),
                "mape_pct": float(mape(actual, predicted)),
                "r2": float(r2_score(actual, predicted)) if len(prediction_frame) > 1 else np.nan,
            }
        )

    return pd.DataFrame(prediction_rows), pd.DataFrame(candidate_metric_rows)


def choose_best_candidate(candidate_metrics_df: pd.DataFrame) -> pd.Series:
    sort_order = {
        "naive": 0,
        "mean3": 1,
        "trend5": 2,
        "ridge_delta": 3,
        "histgb_log": 4,
    }
    ranked = candidate_metrics_df.copy()
    ranked["model_rank"] = ranked["candidate_model"].map(sort_order).fillna(99)
    return (
        ranked.sort_values(["rmse_mtco2e", "mae_mtco2e", "model_rank"])
        .reset_index(drop=True)
        .iloc[0]
    )


def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = y_true != 0
    if mask.sum() == 0:
        return np.nan
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100.0)


def direct_multi_year_forecast(
    dataset: CountryDataset,
    config: ForecastConfig,
    candidate_specs: list[CandidateSpec],
    spec_lookup: dict[str, CandidateSpec],
) -> pd.DataFrame:
    latest_feature_row = dataset.panel.sort_values("year").tail(1).copy()
    history = (
        dataset.panel.dropna(subset=["ghg_current"])
        .sort_values("year")
        .set_index("year")["ghg_current"]
    )
    latest_observed_year = int(latest_feature_row["year"].iloc[0])
    forecast_rows: list[dict[str, object]] = []

    for horizon_years in config.direct_forecast_horizons:
        candidate_predictions_df, candidate_metrics_df = run_rolling_backtest(
            dataset=dataset,
            candidate_specs=candidate_specs,
            config=config,
            horizon_years=horizon_years,
        )
        if candidate_predictions_df.empty or candidate_metrics_df.empty:
            continue

        best_candidate = choose_best_candidate(candidate_metrics_df)
        best_spec = spec_lookup[str(best_candidate["candidate_model"])]
        benchmark_row = (
            candidate_metrics_df.loc[candidate_metrics_df["candidate_model"] == "naive"]
            .iloc[0]
        )
        training_df = build_horizon_training_df(dataset, horizon_years)
        fitted_model = best_spec.fit(training_df, dataset.feature_cols)
        prediction = best_spec.predict(
            fitted_model,
            latest_feature_row,
            history,
            dataset.feature_cols,
            horizon_years,
        )
        previous_value = float(history.iloc[-1])
        yoy_change = prediction - previous_value
        forecast_rows.append(
            {
                "country": dataset.country,
                "latest_observed_year": latest_observed_year,
                "source_year": latest_observed_year,
                "predict_for_year": latest_observed_year + horizon_years,
                "forecast_horizon_years": horizon_years,
                "selected_model": best_spec.name,
                "selected_model_family": best_spec.family,
                "predicted_ghg_mtco2e": prediction,
                "previous_ghg_mtco2e": previous_value,
                "yoy_change_mtco2e": yoy_change,
                "yoy_change_pct": (
                    yoy_change / previous_value * 100.0 if previous_value != 0 else np.nan
                ),
                "backtest_rmse_mtco2e": float(best_candidate["rmse_mtco2e"]),
                "benchmark_rmse_mtco2e": float(benchmark_row["rmse_mtco2e"]),
                "rmse_improvement_vs_benchmark_mtco2e": float(
                    benchmark_row["rmse_mtco2e"] - best_candidate["rmse_mtco2e"]
                ),
                "reliability_tier": reliability_tier(
                    float(best_candidate["rmse_mtco2e"]),
                    float(benchmark_row["rmse_mtco2e"]),
                ),
                "forecast_basis": "direct_horizon_model_selection",
            }
        )

    return pd.DataFrame(forecast_rows)


def reliability_tier(rmse: float, naive_rmse: float) -> str:
    if pd.isna(rmse) or pd.isna(naive_rmse):
        return "low"
    if rmse <= naive_rmse * 0.95:
        return "high"
    if rmse <= naive_rmse * 1.05:
        return "medium"
    return "low"


def round_numeric(df: pd.DataFrame, digits: int) -> pd.DataFrame:
    rounded = df.copy()
    numeric_columns = rounded.select_dtypes(include=["number"]).columns
    rounded[numeric_columns] = rounded[numeric_columns].round(digits)
    return rounded


def build_output_tables(config: ForecastConfig) -> dict[str, pd.DataFrame]:
    repository = DataRepository(config.data_root)
    candidate_specs = get_candidate_specs()
    spec_lookup = {spec.name: spec for spec in candidate_specs}

    evaluation_rows: list[dict[str, object]] = []
    multi_year_frames: list[pd.DataFrame] = []

    for country in COUNTRIES:
        dataset = build_country_dataset(repository, country, config)
        latest_feature_year = int(dataset.panel["year"].max())
        horizon_one_df = build_horizon_training_df(dataset, 1)

        if len(horizon_one_df) < config.min_train_size + 3:
            continue

        _, candidate_metrics_df = run_rolling_backtest(
            dataset=dataset,
            candidate_specs=candidate_specs,
            config=config,
            horizon_years=1,
        )
        if candidate_metrics_df.empty:
            continue

        best_candidate = choose_best_candidate(candidate_metrics_df)
        best_name = str(best_candidate["candidate_model"])

        benchmark_row = (
            candidate_metrics_df.loc[candidate_metrics_df["candidate_model"] == "naive"]
            .iloc[0]
        )

        rmse_value = float(best_candidate["rmse_mtco2e"])
        naive_rmse = float(benchmark_row["rmse_mtco2e"])
        evaluation_rows.append(
            {
                "country": country,
                "selected_model": best_name,
                "selected_model_family": str(best_candidate["candidate_family"]),
                "backtest_years": int(best_candidate["backtest_years"]),
                "train_rows_full": int(len(horizon_one_df)),
                "latest_observed_year": latest_feature_year,
                "forecast_horizon_years": 1,
                "rmse_mtco2e": rmse_value,
                "mae_mtco2e": float(best_candidate["mae_mtco2e"]),
                "mape_pct": float(best_candidate["mape_pct"]),
                "r2": float(best_candidate["r2"]),
                "benchmark_model": "naive",
                "benchmark_rmse_mtco2e": naive_rmse,
                "benchmark_mae_mtco2e": float(benchmark_row["mae_mtco2e"]),
                "rmse_improvement_vs_benchmark_mtco2e": naive_rmse - rmse_value,
                "rmse_improvement_vs_benchmark_pct": (
                    (naive_rmse - rmse_value) / naive_rmse * 100.0
                    if naive_rmse != 0
                    else np.nan
                ),
                "reliability_tier": reliability_tier(rmse_value, naive_rmse),
                "validation_strategy": "rolling_one_step_backtest_with_model_selection",
            }
        )

        direct_forecast_df = direct_multi_year_forecast(
            dataset=dataset,
            config=config,
            candidate_specs=candidate_specs,
            spec_lookup=spec_lookup,
        )
        multi_year_frames.append(direct_forecast_df)
    evaluation_df = (
        pd.DataFrame(evaluation_rows)
        .sort_values(["forecast_horizon_years", "reliability_tier", "rmse_mtco2e", "country"])
        .reset_index(drop=True)
        if evaluation_rows
        else pd.DataFrame()
    )
    multi_year_df = (
        pd.concat(multi_year_frames, ignore_index=True)
        .sort_values(["country", "predict_for_year"])
        .reset_index(drop=True)
        if multi_year_frames
        else pd.DataFrame()
    )
    multi_year_file_name = f"country_ghg_forecast_{config.forecast_years}_years.csv"
    if not multi_year_df.empty:
        start_year = int(multi_year_df["predict_for_year"].min())
        end_year = int(multi_year_df["predict_for_year"].max())
        multi_year_file_name = f"country_ghg_forecast_{start_year}_{end_year}.csv"

    return {
        "country_model_evaluation.csv": evaluation_df,
        multi_year_file_name: multi_year_df,
    }


def save_outputs(output_tables: dict[str, pd.DataFrame], config: ForecastConfig) -> None:
    for file_name, table in output_tables.items():
        table_to_save = round_numeric(table, config.float_precision)
        table_to_save.to_csv(file_name, index=False)


def main() -> None:
    config = ForecastConfig()
    output_tables = build_output_tables(config)
    save_outputs(output_tables, config)

    for file_name, table in output_tables.items():
        print(f"Saved {file_name}: {len(table)} rows")


if __name__ == "__main__":
    main()
