# Student Academic Performance Prediction

A reproducible Python research project based on the **mathematics subset** of the UCI Student Performance dataset. The project predicts the final mathematics grade (`G3`) and compares three regression algorithms in two prediction settings.

## Research questions implemented

1. How well can Linear Regression, Decision Tree Regression and Random Forest Regression predict final mathematics grades?
2. Which model performs best on a held-out test set?
3. How does performance change when earlier assessment grades (`G1` and `G2`) are included?
4. Which encoded input features are important in the Random Forest model?

## Two prediction scenarios

- **Early prediction:** excludes `G1` and `G2`, because those assessment grades would not yet be available at the beginning of the course.
- **Later prediction:** includes `G1` and `G2`, representing a prediction made after the first two assessment periods.

Both scenarios use the same train/test student split (80%/20%, random state 42). Categorical features are one-hot encoded. Missing-value imputation and scaling/encoding are fitted inside a scikit-learn pipeline using training data only.

## Dataset

The project uses the official UCI Student Performance dataset, specifically `student-mat.csv` for mathematics. The dataset contains student demographic, family, school, study-habit and grade information. The target is `G3`, the final grade on a 0–20 scale.

The script automatically downloads the official UCI archive on its first run. If automatic download is unavailable, download the dataset from the UCI page below, extract `student-mat.csv`, and place it in `data/raw/`.

- UCI dataset page: https://archive.ics.uci.edu/dataset/320/student+performance
- DOI: https://doi.org/10.24432/C5TG7T
- Dataset citation: Cortez, P. (2008). *Student Performance* [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5TG7T.
- UCI lists the dataset under a Creative Commons Attribution 4.0 (CC BY 4.0) licence. Retain the citation when redistributing the data.

## Repository structure

```text
student-academic-performance-prediction/
├── README.md
├── requirements.txt
├── run.bat
├── .gitignore
├── src/
│   └── student_performance_analysis.py
├── data/
│   └── raw/
│       └── student-mat.csv       # downloaded by the script; not committed by default
└── outputs/
    ├── data_inspection.txt
    ├── model_comparison_metrics.csv
    ├── test_set_predictions.csv
    ├── early_random_forest_feature_importance.csv
    ├── later_random_forest_feature_importance.csv
    └── figures/
        ├── 01_final_grade_distribution.png
        ├── 02_studytime_vs_final_grade.png
        ├── 03_absences_vs_final_grade.png
        ├── 04_correlation_heatmap.png
        ├── 05_model_comparison_mae.png
        ├── 06_model_comparison_rmse.png
        ├── 07_model_comparison_r2.png
        ├── 08_early_feature_importance.png
        └── 08_later_feature_importance.png
```

The `outputs/` files are generated when the script runs.

## Run in VS Code (Windows)

1. Open this project folder in VS Code.
2. Open **Terminal → New Terminal**.
3. Create and activate a virtual environment (recommended):

   ```powershell
   py -m venv .venv
   .venv\Scripts\Activate.ps1
   ```

4. Install the dependencies:

   ```powershell
   py -m pip install -r requirements.txt
   ```

5. Run the research workflow:

   ```powershell
   py src/student_performance_analysis.py
   ```

You can also double-click `run.bat` after Python is installed.

## Generated results

- `outputs/data_inspection.txt`: rows, columns, missing values, duplicate count, data types, summary statistics and sample records.
- `outputs/model_comparison_metrics.csv`: MAE, RMSE and R² for all 6 model/scenario combinations.
- `outputs/test_set_predictions.csv`: actual and predicted grades for every held-out test observation and model.
- `outputs/*feature_importance.csv`: Random Forest feature-importance rankings for early and later scenarios.
- `outputs/figures/`: exploratory and model-comparison plots.

## Evaluation metrics

- **MAE:** average absolute prediction error; lower is better.
- **RMSE:** square root of mean squared error; lower is better and larger errors receive more weight.
- **R²:** proportion of target variance explained on the held-out test data; higher is generally better.

Do not copy metric values into the research paper until the code has been executed. Report the values produced by the run, and keep early- and later-prediction results separate.

## Important research limitation

The later scenario has access to G1 and G2, which are strongly related to G3. It therefore represents a later prediction time and must not be described as an early-intervention model. The dataset comes from a particular historical educational context, so results should not be assumed to generalise to every school. Predictions should support—not replace—teacher judgement.

## Author

Harshita Gera  
Department of Computer Science and Engineering  
Lovely Professional University, Punjab, India
