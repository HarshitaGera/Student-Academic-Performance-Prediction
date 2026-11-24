# Student Academic Performance Prediction

A reproducible machine learning workflow for analyzing and predicting secondary school final mathematics grades (`G3`) using the UCI Student Performance dataset (`student-mat.csv`), with explicit safeguards against target leakage.

---

## Overview

Educational predictive models often report inflated accuracy by including mid-term assessment grades (`G1`, `G2`) when forecasting final grades (`G3`)—information that does not exist at the start of a semester. This study evaluates regression models across two clearly separated prediction windows to support realistic early academic intervention.

### Research Questions
1. **RQ1:** How accurately can supervised regression models predict students' final mathematics grades (`G3`, scale 0–20)?
2. **RQ2:** Which model achieves the strongest generalization performance on held-out test data?
3. **RQ3:** How does including vs. excluding earlier-period grades (`G1`, `G2`) impact predictive accuracy?
4. **RQ4:** Which demographic, behavioral, and academic variables carry the strongest predictive signal?

---

## Methodology & Leakage Prevention

* **Dataset:** UCI Machine Learning Repository — Student Performance (`student-mat.csv`)
* **Feature Configurations:**
  * **Early-Prediction Scenario:** Uses only attributes available prior to the first major assessment (demographics, family background, study time, prior failures, absences, school support) and **excludes `G1` and `G2`**.
  * **Later-Prediction Scenario:** Includes `G1` and `G2` alongside behavioral and contextual features.
* **Preprocessing Pipeline:** Train/test split performed with a fixed random seed prior to fitting `OneHotEncoder` inside a `Scikit-learn` `Pipeline` to eliminate test-set leakage.
* **Models Evaluated:**
  * Linear Regression (interpretable baseline)
  * Decision Tree Regression (depth-constrained rule-based non-linear model)
  * Random Forest Regression (ensemble variance reduction)

---

## Experimental Results

| Scenario | Model | MAE | RMSE | $R^2$ |
| :--- | :--- | :---: | :---: | :---: |
| **Early Prediction** (No `G1`/`G2`) | Linear Regression | — | — | — |
| **Early Prediction** (No `G1`/`G2`) | Decision Tree | — | — | — |
| **Early Prediction** (No `G1`/`G2`) | Random Forest | — | — | — |
| **Later Prediction** (With `G1`/`G2`) | Linear Regression | — | — | — |
| **Later Prediction** (With `G1`/`G2`) | Decision Tree | — | — | — |
| **Later Prediction** (With `G1`/`G2`) | Random Forest | — | — | — |

*(Populate with exact test-set outputs from `notebooks/student_performance_prediction.ipynb`.)*

---

## Tech Stack

* **Language:** Python 3
* **Data & Visualization:** Pandas, Matplotlib, Seaborn
* **Machine Learning:** Scikit-learn (`Pipeline`, `ColumnTransformer`, `OneHotEncoder`, `LinearRegression`, `DecisionTreeRegressor`, `RandomForestRegressor`)

---

## Limitations & Ethical Use

* Models are trained on a historical cohort from a specific educational setting and require multi-institution validation before broader deployment.
* Predictions serve strictly as early-support indicators for educators—not automated grading or student labeling tools.

---

## Authors

* **Harshita Gera** — Department of Computer Science and Engineering, Lovely Professional University, Punjab, India
* **[Add Co-Author Name(s), if applicable]** — Lovely Professional University
