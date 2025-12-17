import os
from pathlib import Path
from typing import Any

import matplotlib
import mlflow
import optuna
import polars as pl
import polars.selectors as cs
from dotenv import load_dotenv
from numpy.typing import NDArray
from sklearn.metrics import r2_score
from sklearn.utils import shuffle  # type: ignore
from xgboost import XGBRegressor

load_dotenv(Path(__file__).parent.parent / ".env.example")

os.environ["MLFLOW_S3_ENDPOINT_URL"] = os.getenv(
	"MLFLOW_S3_ENDPOINT_URL", "http://localhost:9000"
)
os.environ["AWS_ACCESS_KEY_ID"] = os.getenv("AWS_ACCESS_KEY_ID", "mlflow")
os.environ["AWS_SECRET_ACCESS_KEY"] = os.getenv("AWS_SECRET_ACCESS_KEY", "mlflow123")

matplotlib.use("Agg")  # type: ignore

mlflow.set_tracking_uri(os.getenv("MLFLOW_URL", "http://localhost:5001"))
optuna.logging.set_verbosity(optuna.logging.WARNING)


def cleanup(lf: pl.LazyFrame, drop: list[str] | None = None) -> pl.LazyFrame:
	"""
	Cleans up the input LazyFrame by removing duplicates and handling missing values.

	Args:
	    lf (pl.LazyFrame): Input lazy dataframe containing battery health data.
	    drop (list[str] | None): List of columns to drop from the dataframe. Defaults to None.

	Returns:
	    pl.LazyFrame: Cleaned lazy dataframe.
	"""

	if drop is None:
		drop = ["device_id"]

	return (
		lf.drop(drop)
		.drop_nulls(["battery_health_percent"])
		.unique()
		.with_columns(
			cs.string(include_categorical=True)
			.str.strip_chars()
			.str.to_lowercase()
			.str.replace(" ", "_"),
			cs.numeric().fill_null(strategy="mean"),
		)
	)


def features1(lf: pl.LazyFrame) -> tuple[NDArray[Any], NDArray[Any], list[str]]:
	"""
	Returns the following features:
	        - total_usage_hours: Estimated total usage hours of the device
	        - charging_cycles_per_year: Average number of charging cycles per year
	        - avg_decline_per_year: Average battery health decline per year
	        - One-hot encoded categorical features: brand, model_year, os, usage_type, overheating

	Args:
	        lf (pl.LazyFrame): Input lazy dataframe containing battery health data.
	        drop (list[str] | None): List of columns to drop from the dataframe. Defaults to None.

	Returns:
	        tuple[NDArray[Any], NDArray[Any]]: A tuple containing the feature matrix (X) and target vector (y).
	"""

	df = (
		lf.with_columns(
			(pl.col("daily_usage_hours") * (2025 - pl.col("model_year")) * 365).alias(
				"total_usage_hours"
			),
			(pl.col("charging_cycles") / (2025 - pl.col("model_year"))).alias(
				"charging_cycles_per_year"
			),
			(
				(100 - pl.col("battery_health_percent")) / (2025 - pl.col("model_year"))
			).alias("avg_decline_per_year"),
		)
		.collect()
		.to_dummies(
			[
				"brand",
				"model_year",
				"os",
				"usage_type",
				"overheating_issues",
				"performance_rating",
			],
			drop_first=True,
			drop_nulls=True,
		)
	)

	X_df, y_df = (
		df.select(~cs.by_name("battery_health_percent")),
		df.select(cs.by_name("battery_health_percent")),
	)
	return X_df.to_numpy(), y_df.to_numpy(), X_df.collect_schema().names()


def features2(lf: pl.LazyFrame) -> tuple[NDArray[Any], NDArray[Any], list[str]]:
	"""
	Returns the following additional features & creates intervals:
	        - total_usage_hours: Estimated total usage hours of the device
	        - charging_cycles_per_year: Average number of charging cycles per year
	        - avg_decline_per_year: Average battery health decline per year
	        - One-hot encoded categorical features: brand, model_year, os, usage_type, overheating

	Args:
	        lf (pl.LazyFrame): Input lazy dataframe containing battery health data.

	Returns:
	        tuple[NDArray[Any], NDArray[Any]]: A tuple containing the feature matrix (X) and target vector (y).
	"""

	df = (
		lf.with_columns(
			(pl.col("daily_usage_hours") * (2025 - pl.col("model_year")) * 365).alias(
				"total_usage_hours"
			),
			(pl.col("charging_cycles") / (2025 - pl.col("model_year"))).alias(
				"charging_cycles_per_year"
			),
			(
				(100 - pl.col("battery_health_percent")) / (2025 - pl.col("model_year"))
			).alias("avg_decline_per_year"),
		)
		.with_columns(
			pl.col("total_usage_hours").qcut(
				10,
				labels=[f"total_usage_hours_bin_{i}" for i in range(10)],
				allow_duplicates=True,
			),
			pl.col("charging_cycles_per_year").qcut(
				10,
				labels=[f"charging_cycles_per_year_bin_{i}" for i in range(10)],
				allow_duplicates=True,
			),
			pl.col("avg_decline_per_year").qcut(
				10,
				labels=[f"avg_decline_per_year_bin_{i}" for i in range(10)],
				allow_duplicates=True,
			),
			pl.col("battery_age_months").qcut(
				10,
				labels=[f"battery_age_months_bin_{i}" for i in range(10)],
				allow_duplicates=True,
			),
		)
		.collect()
		.to_dummies(
			[
				"brand",
				"model_year",
				"os",
				"usage_type",
				"overheating_issues",
				"performance_rating",
				"total_usage_hours",
				"charging_cycles_per_year",
				"avg_decline_per_year",
				"battery_age_months",
			],
			drop_first=True,
			drop_nulls=True,
		)
	)

	X_df, y_df = (
		df.select(~cs.by_name("battery_health_percent")),
		df.select(cs.by_name("battery_health_percent")),
	)
	return X_df.to_numpy(), y_df.to_numpy(), X_df.collect_schema().names()


def find_best_model(
	X: NDArray[Any], y: NDArray[Any]
) -> tuple[XGBRegressor, float, dict[str, Any]]:
	shuffled = shuffle(X, y, random_state=42)
	if not shuffled or len(shuffled) != 2:
		raise ValueError
	X_shuff, y_shuff = shuffled
	split_idx = int(0.8 * len(X_shuff))  # type: ignore
	X_train, X_test = X_shuff[:split_idx], X_shuff[split_idx:]  # type: ignore
	y_train, y_test = y_shuff[:split_idx], y_shuff[split_idx:]  # type: ignore

	def objective(trial: optuna.Trial) -> float:
		with mlflow.start_run(nested=True):
			search_space = {
				"n_estimators": trial.suggest_int("n_estimators", 50, 200),
				"max_depth": trial.suggest_int("max_depth", 3, 10),
				"learning_rate": trial.suggest_float(
					"learning_rate", 0.01, 0.3, log=True
				),
				"subsample": trial.suggest_float("subsample", 0.7, 1.0),
				"colsample_bytree": trial.suggest_float("colsample_bytree", 0.7, 1.0),
			}

			model = XGBRegressor(**search_space, random_state=42, n_jobs=-1).fit(
				X_train, y_train
			)
			y_pred = model.predict(X_test)
			score = r2_score(y_test, y_pred)  # type: ignore

			mlflow.log_params(search_space)
			mlflow.log_metric("r2_score", score)

			return score

	study = optuna.create_study(direction="maximize")
	study.optimize(objective, n_trials=15, show_progress_bar=True)

	best_model = XGBRegressor(**study.best_params, random_state=42, n_jobs=-1)
	best_model.fit(X, y.ravel())

	return best_model, study.best_value, study.best_params


def find_best_model_feature_combo(
	lf: pl.LazyFrame, drop_any: list[str] | None = None
) -> tuple[XGBRegressor, float, list[str]]:
	mlflow.set_experiment("battery_health")

	lf = cleanup(lf)

	with mlflow.start_run(run_name="feature_comparison"):
		print("\n=== Testing features1 ===")
		with mlflow.start_run(run_name="features1", nested=True):
			X1, y1, features1_names = features1(lf)
			print(f"Features1 shape: {X1.shape}")

			model1, score1, params1 = find_best_model(X1, y1)
			print(f"Features1 best R²: {score1:.4f}")
			mlflow.log_metric("best_r2", score1)
			mlflow.log_params(params1)

		print("\n=== Testing features2 ===")
		with mlflow.start_run(run_name="features2", nested=True):
			X2, y2, features2_names = features2(lf)
			print(f"Features2 shape: {X2.shape}")

			model2, score2, params2 = find_best_model(X2, y2)
			print(f"Features2 best R²: {score2:.4f}")
			mlflow.log_metric("best_r2", score2)
			mlflow.log_params(params2)

		best_feature_set = "features1" if score1 > score2 else "features2"
		mlflow.log_param("best_feature_set", best_feature_set)
		mlflow.log_metric("best_overall_r2", max(score1, score2))

		if score1 > score2:
			best_model = model1
			best_features = features1_names
			best_params = params1
		else:
			best_model = model2
			best_features = features2_names
			best_params = params2

		print("\n=== Saving champion model ===")
		best_model._estimator_type = "regressor"  # type: ignore
		mlflow.xgboost.log_model(  # type: ignore
			best_model,
			artifact_path="battery_health_predictor",
			registered_model_name="battery_health_predictor",
		)

		mlflow.log_dict({"features": best_features}, "feature_names.json")
		mlflow.log_params({f"champion_{k}": v for k, v in best_params.items()})

		print(
			f"\n=== Winner: {best_feature_set} with R² = {max(score1, score2):.4f} ==="
		)

		return best_model, max(score1, score2), best_features


if __name__ == "__main__":
	lf = pl.scan_csv("./data/laptop_battery_health_usage.csv", separator=",")
	print(lf.collect_schema())
	best_model, best_score, best_features = find_best_model_feature_combo(lf)
	print(f"\nFinal Best R² score: {best_score:.4f}")
	print(f"Number of features: {len(best_features)}")
	print("Model registered in MLflow as 'battery_health_predictor'")
