"""Student Academic Performance Prediction

Reproducible workflow for the mathematics subset of the UCI Student
Performance dataset. Predicts final grade G3 using three regressors and
compares early (without G1/G2) and later (with G1/G2) prediction scenarios.

Dataset source: UCI Machine Learning Repository, DOI: 10.24432/C5TG7T
"""

from __future__ import annotations

import sys
import urllib.request
import zipfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeRegressor


# ---------------------------------------------------------------------------
# Paths and experiment settings
# ---------------------------------------------------------------------------
PROJECT_DIR = Path(__file__).resolve().parents[1]
RAW_DATA_DIR = PROJECT_DIR / "data" / "raw"
OUTPUT_DIR = PROJECT_DIR / "outputs"
FIGURE_DIR = OUTPUT_DIR / "figures"
DATA_FILE = RAW_DATA_DIR / "student-mat.csv"
ZIP_FILE = RAW_DATA_DIR / "student-performance.zip"
DATASET_URL = "https://archive.ics.uci.edu/static/public/320/student+performance.zip"

RANDOM_STATE = 42
TEST_SIZE = 0.20
TARGET = "G3"


# ---------------------------------------------------------------------------
# Dataset acquisition and loading
# ---------------------------------------------------------------------------
def download_dataset() -> None:
    """Download the official UCI archive and extract student-mat.csv."""
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    print("Official mathematics dataset not found. Downloading from UCI...")

    request = urllib.request.Request(
        DATASET_URL,
        headers={"User-Agent": "Mozilla/5.0 (compatible; StudentPerformanceResearch/1.0)"},
    )
    try:
        with urllib.request.urlopen(request, timeout=45) as response:
            ZIP_FILE.write_bytes(response.read())

        with zipfile.ZipFile(ZIP_FILE, "r") as archive:
            member = next(
                (name for name in archive.namelist() if name.lower().endswith("student-mat.csv")),
                None,
            )
            if member is None:
                raise FileNotFoundError("student-mat.csv was not present in the downloaded UCI archive.")
            DATA_FILE.write_bytes(archive.read(member))

        print(f"Dataset saved to: {DATA_FILE}")
    except Exception as exc:
        raise RuntimeError(
            "Could not download the UCI dataset automatically. Download the official "
            "Student Performance ZIP from https://archive.ics.uci.edu/dataset/320/student+performance, "
            "extract student-mat.csv, and place it in data/raw/student-mat.csv. "
            f"Original error: {exc}"
        ) from exc


def load_data() -> pd.DataFrame:
    """Load mathematics course records; the file is semicolon-separated."""
    if not DATA_FILE.exists():
        download_dataset()

    data = pd.read_csv(DATA_FILE, sep=";", encoding="utf-8")
    if TARGET not in data.columns:
        raise ValueError(f"Expected target column '{TARGET}' was not found in the dataset.")
    if "G1" not in data.columns or "G2" not in data.columns:
        raise ValueError("Expected columns G1 and G2 were not found. Check that student-mat.csv is used.")
    data[TARGET] = pd.to_numeric(data[TARGET], errors="coerce")
    data = data.dropna(subset=[TARGET]).copy()
    return data


# ---------------------------------------------------------------------------
# Initial inspection and exploratory data analysis
# ---------------------------------------------------------------------------
def inspect_data(data: pd.DataFrame) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    report_path = OUTPUT_DIR / "data_inspection.txt"

    lines = [
        "STUDENT ACADEMIC PERFORMANCE DATA INSPECTION",
        "=" * 55,
        f"Rows: {data.shape[0]}",
        f"Columns: {data.shape[1]}",
        f"Duplicate rows: {int(data.duplicated().sum())}",
        "",
        "Missing values by column:",
        data.isna().sum().to_string(),
        "",
        "Column data types:",
        data.dtypes.astype(str).to_string(),
        "",
        "Numerical summary:",
        data.describe(include=[np.number]).round(3).to_string(),
        "",
        "First five rows:",
        data.head().to_string(index=False),
    ]
    report_path.write_text("\n".join(lines), encoding="utf-8")

    print("\nDATASET OVERVIEW")
    print("-" * 60)
    print(f"Shape: {data.shape[0]} rows x {data.shape[1]} columns")
    print(f"Duplicate rows: {int(data.duplicated().sum())}")
    print(f"Missing cells: {int(data.isna().sum().sum())}")
    print("\nFirst five records:")
    print(data.head())
    print("\nNumerical summary:")
    print(data.describe(include=[np.number]).round(2))
    print(f"\nDetailed inspection saved to: {report_path}")


def save_eda_plots(data: pd.DataFrame) -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Distribution of final mathematics grades
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(data[TARGET], bins=np.arange(-0.5, 21.5, 1), edgecolor="black")
    ax.set(title="Distribution of Final Mathematics Grades (G3)", xlabel="Final grade (G3)", ylabel="Number of students")
    ax.set_xticks(range(0, 21, 2))
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "01_final_grade_distribution.png", dpi=160)
    plt.close(fig)

    # 2. Study time and final grade
    if "studytime" in data.columns:
        fig, ax = plt.subplots(figsize=(8, 5))
        groups = [data.loc[data["studytime"] == value, TARGET].dropna() for value in sorted(data["studytime"].dropna().unique())]
        labels = [str(value) for value in sorted(data["studytime"].dropna().unique())]
        ax.boxplot(groups, labels=labels)
        ax.set(title="Study Time and Final Mathematics Grade", xlabel="Study time category", ylabel="Final grade (G3)")
        fig.tight_layout()
        fig.savefig(FIGURE_DIR / "02_studytime_vs_final_grade.png", dpi=160)
        plt.close(fig)

    # 3. Absences and final grade
    if "absences" in data.columns:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.scatter(data["absences"], data[TARGET], alpha=0.65)
        ax.set(title="School Absences and Final Mathematics Grade", xlabel="Number of absences", ylabel="Final grade (G3)")
        fig.tight_layout()
        fig.savefig(FIGURE_DIR / "03_absences_vs_final_grade.png", dpi=160)
        plt.close(fig)

    # 4. Correlation heatmap for key numerical academic variables
    numeric = data.select_dtypes(include=[np.number])
    selected = [column for column in ["age", "studytime", "failures", "absences", "G1", "G2", "G3"] if column in numeric.columns]
    corr = numeric[selected].corr()
    fig, ax = plt.subplots(figsize=(8, 6))
    image = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(selected)), selected, rotation=45, ha="right")
    ax.set_yticks(range(len(selected)), selected)
    ax.set_title("Correlation Heatmap of Selected Numerical Variables")
    for row in range(len(selected)):
        for col in range(len(selected)):
            ax.text(col, row, f"{corr.iloc[row, col]:.2f}", ha="center", va="center", fontsize=8)
    fig.colorbar(image, ax=ax, label="Correlation")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "04_correlation_heatmap.png", dpi=160)
    plt.close(fig)

    print(f"EDA figures saved in: {FIGURE_DIR}")


# ---------------------------------------------------------------------------
# Preprocessing and model definitions
# ---------------------------------------------------------------------------
def build_pipeline(X: pd.DataFrame, estimator) -> Pipeline:
    """Build preprocessing + model pipeline; fit preprocessing on training data only."""
    numeric_columns = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_columns = X.select_dtypes(exclude=[np.number]).columns.tolist()

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_columns),
            ("categorical", categorical_pipeline, categorical_columns),
        ],
        remainder="drop",
    )

    return Pipeline(steps=[("preprocessor", preprocessor), ("model", estimator)])


def get_models() -> dict:
    return {
        "Linear Regression": LinearRegression(),
        "Decision Tree": DecisionTreeRegressor(
            max_depth=5,
            min_samples_leaf=3,
            random_state=RANDOM_STATE,
        ),
        "Random Forest": RandomForestRegressor(
            n_estimators=300,
            min_samples_leaf=2,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
    }


# ---------------------------------------------------------------------------
# Experiments: early prediction (no G1/G2) vs later prediction (with G1/G2)
# ---------------------------------------------------------------------------
def run_experiments(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    y = data[TARGET].astype(float)
    all_features = data.drop(columns=[TARGET])

    feature_sets = {
        "Early prediction": all_features.drop(columns=["G1", "G2"]),
        "Later prediction": all_features.copy(),
    }

    # Generate one common split so both scenarios use the same student records.
    all_indices = np.arange(len(data))
    train_indices, test_indices = train_test_split(
        all_indices,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )
    y_test = y.iloc[test_indices]

    metric_rows: list[dict] = []
    prediction_frames: list[pd.DataFrame] = []
    fitted_models: dict = {}

    for scenario, X in feature_sets.items():
        X_train = X.iloc[train_indices]
        X_test = X.iloc[test_indices]
        fitted_models[scenario] = {}

        print(f"\n{'=' * 68}\n{scenario.upper()}\n{'=' * 68}")
        print(f"Number of input features before encoding: {X.shape[1]}")

        for model_name, estimator in get_models().items():
            pipeline = build_pipeline(X_train, estimator)
            pipeline.fit(X_train, y.iloc[train_indices])
            predictions = pipeline.predict(X_test)

            mae = mean_absolute_error(y_test, predictions)
            rmse = float(np.sqrt(mean_squared_error(y_test, predictions)))
            r2 = r2_score(y_test, predictions)

            metric_rows.append(
                {
                    "Scenario": scenario,
                    "Model": model_name,
                    "MAE": mae,
                    "RMSE": rmse,
                    "R2": r2,
                }
            )
            prediction_frames.append(
                pd.DataFrame(
                    {
                        "Scenario": scenario,
                        "Model": model_name,
                        "Actual_G3": y_test.to_numpy(),
                        "Predicted_G3": predictions,
                    }
                )
            )
            fitted_models[scenario][model_name] = pipeline

            print(f"{model_name:20s} | MAE: {mae:.4f} | RMSE: {rmse:.4f} | R2: {r2:.4f}")

    metrics = pd.DataFrame(metric_rows).sort_values(["Scenario", "MAE"]).reset_index(drop=True)
    predictions = pd.concat(prediction_frames, ignore_index=True)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(OUTPUT_DIR / "model_comparison_metrics.csv", index=False)
    predictions.to_csv(OUTPUT_DIR / "test_set_predictions.csv", index=False)

    return metrics, predictions, fitted_models


# ---------------------------------------------------------------------------
# Results visualisation and feature importance
# ---------------------------------------------------------------------------
def save_model_comparison_plots(metrics: pd.DataFrame) -> None:
    for metric, title, filename in [
        ("MAE", "Model Comparison - Mean Absolute Error", "05_model_comparison_mae.png"),
        ("RMSE", "Model Comparison - Root Mean Squared Error", "06_model_comparison_rmse.png"),
        ("R2", "Model Comparison - R-squared", "07_model_comparison_r2.png"),
    ]:
        pivot = metrics.pivot(index="Model", columns="Scenario", values=metric)
        fig, ax = plt.subplots(figsize=(9, 5))
        pivot.plot(kind="bar", ax=ax)
        ax.set(title=title, xlabel="Regression model", ylabel=metric)
        ax.tick_params(axis="x", rotation=15)
        ax.legend(title="Prediction scenario")
        fig.tight_layout()
        fig.savefig(FIGURE_DIR / filename, dpi=160)
        plt.close(fig)


def save_random_forest_importance(fitted_models: dict) -> None:
    for scenario, scenario_models in fitted_models.items():
        pipeline = scenario_models["Random Forest"]
        preprocessor = pipeline.named_steps["preprocessor"]
        forest = pipeline.named_steps["model"]
        feature_names = preprocessor.get_feature_names_out()
        importance = forest.feature_importances_

        importance_df = pd.DataFrame(
            {"Feature": feature_names, "Importance": importance}
        ).sort_values("Importance", ascending=False)

        safe_name = "early" if scenario.startswith("Early") else "later"
        importance_df.to_csv(OUTPUT_DIR / f"{safe_name}_random_forest_feature_importance.csv", index=False)

        top = importance_df.head(15).sort_values("Importance", ascending=True)
        fig, ax = plt.subplots(figsize=(9, 6))
        ax.barh(top["Feature"], top["Importance"])
        ax.set(title=f"Random Forest Feature Importance - {scenario}", xlabel="Feature importance", ylabel="Encoded feature")
        fig.tight_layout()
        fig.savefig(FIGURE_DIR / f"08_{safe_name}_feature_importance.png", dpi=160)
        plt.close(fig)


def print_results_summary(metrics: pd.DataFrame) -> None:
    print("\nFINAL TEST-SET COMPARISON")
    print("-" * 78)
    print(metrics.to_string(index=False, formatters={
        "MAE": "{:.4f}".format,
        "RMSE": "{:.4f}".format,
        "R2": "{:.4f}".format,
    }))

    print("\nBest model by scenario (lowest MAE):")
    for scenario, group in metrics.groupby("Scenario"):
        best = group.sort_values("MAE").iloc[0]
        print(f"{scenario}: {best['Model']} (MAE={best['MAE']:.4f}, RMSE={best['RMSE']:.4f}, R2={best['R2']:.4f})")

    print("\nImportant interpretation:")
    print("The later-prediction scenario includes G1 and G2, which are earlier assessment grades.")
    print("Its metrics must not be presented as an early-term prediction result.")
    print("A higher R-squared or lower error indicates better performance on this held-out test set,")
    print("not proof that the model will generalise to every school or student population.")


def main() -> int:
    print("STUDENT ACADEMIC PERFORMANCE PREDICTION")
    print("UCI mathematics dataset | Regression models | Early vs later prediction")

    try:
        data = load_data()
        inspect_data(data)
        save_eda_plots(data)
        metrics, _predictions, fitted_models = run_experiments(data)
        save_model_comparison_plots(metrics)
        save_random_forest_importance(fitted_models)
        print_results_summary(metrics)

        print("\nPROJECT COMPLETED")
        print(f"Metrics: {OUTPUT_DIR / 'model_comparison_metrics.csv'}")
        print(f"Predictions: {OUTPUT_DIR / 'test_set_predictions.csv'}")
        print(f"Figures: {FIGURE_DIR}")
        return 0
    except Exception as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
