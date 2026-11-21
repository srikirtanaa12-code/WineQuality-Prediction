# Machine Learning Projects: Classification & Regression

End-to-end machine learning projects on four public datasets. Each notebook follows the same workflow: **load → inspect → clean → handle outliers → encode → explore (EDA) → split → scale → select features → train 3 models → evaluate**.

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange)
![pandas](https://img.shields.io/badge/pandas-data-informational)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626)

---

## Projects at a glance

| # | Project | Problem type | Target | Best model | Best score |
|---|---|---|---|---|---|
| 1 | Credit Card Default | Classification | Default next month (0/1) | Random Forest | Accuracy 0.81, F1 0.46 |
| 2 | White Wine Quality | Classification | Good wine (quality ≥ 7) | Random Forest | Accuracy 0.83, F1 0.49 |
| 2 | White Wine Quality | Regression | Quality score (3 to 9) | Random Forest | R² 0.33 |
| 3 | Air Quality (UCI) | Regression | CO concentration (mg/m³) | Random Forest | R² 0.94 |
| 3 | Air Quality (UCI) | Classification | High CO (> 3 mg/m³) | Logistic Regression | Accuracy 0.96, F1 0.90 |
| 4 | Bike Sharing (hourly) | Regression | Bikes rented per hour | Random Forest | R² 0.94 |

---

## Algorithms used

Every project compares the same three algorithms, so results are easy to compare.

| Classification | Regression |
|---|---|
| Logistic Regression | Linear Regression |
| Decision Tree Classifier | Decision Tree Regressor |
| Random Forest Classifier | Random Forest Regressor |

**Metrics**
- Classification: accuracy, precision, recall, F1, confusion matrix
- Regression: MAE, RMSE, R²

---

## 1. Credit Card Default (Classification)

**Goal:** predict whether a client will default on their credit card payment next month.

**Highlights**
- Cleaned the header row and removed empty `Unnamed` columns
- Capped outliers with the IQR method
- One-hot encoded categorical columns (`SEX`, `EDUCATION`, `MARRIAGE`)
- Selected features with `SelectKBest` and scaled with `StandardScaler`

| Model | Accuracy | Precision | F1 | Recall |
|---|---|---|---|---|
| Logistic Regression | 0.78 | 0.62 | 0.01 | 0.01 |
| Decision Tree | 0.72 | 0.37 | 0.40 | 0.42 |
| Random Forest | 0.81 | 0.62 | 0.46 | 0.36 |

**Takeaway:** the classes are imbalanced, so accuracy looks decent while recall stays low. Logistic Regression almost never predicts a default. F1 and recall are the metrics to watch here.

---

## 2. White Wine Quality (Classification + Regression)

**Dataset:** 4,898 white wines, 11 lab measurements (acidity, sugar, chlorides, sulphates, alcohol, ...) and a quality score from 3 to 9.

**Cleaning**
- Loaded with `sep=";"`
- Found 937 duplicate rows and removed them, leaving 3,961 rows
- Capped outliers with the IQR method (the target is left untouched)
- Built a low / medium / high alcohol category and one-hot encoded it

**Classification:** a wine is "good" if quality ≥ 7 (825 good vs 3,136 not good)

| Model | Accuracy | Precision | F1 | Recall |
|---|---|---|---|---|
| Logistic Regression | 0.83 | 0.65 | 0.48 | 0.38 |
| Decision Tree | 0.76 | 0.43 | 0.44 | 0.46 |
| Random Forest | 0.83 | 0.65 | 0.49 | 0.39 |

**Regression:** predict the quality score itself

| Model | MAE | RMSE | R² |
|---|---|---|---|
| Linear Regression | 0.592 | 0.767 | 0.282 |
| Decision Tree | 0.752 | 1.065 | -0.384 |
| Random Forest | 0.566 | 0.742 | 0.328 |

**Takeaways**
- Alcohol is the strongest signal for quality, followed by density.
- Wine quality is a subjective taste score, so lab measurements explain only about a third of it.
- The unpruned Decision Tree overfits (negative R²).

---

## 3. Air Quality UCI (Regression + Classification)

**Dataset:** 9,357 hourly readings from chemical sensors in an Italian city (March 2004 to April 2005).

**Data cleaning challenges**
- Semicolon-separated file with decimal commas (`sep=";", decimal=","`)
- Two completely empty trailing columns and 114 blank rows
- `-200` is used as a "missing" marker, so it was replaced with `NaN`
- `NMHC(GT)` is about 90% missing, so the column was dropped
- Rows with no CO reading were dropped, leaving 7,674 rows
- `Date` and `Time` were turned into `Hour`, `Month` and `DayOfWeek`
- Remaining missing sensor values are filled with the median inside the preprocessing pipeline

**Regression:** predict CO concentration (`CO(GT)`)

| Model | MAE | RMSE | R² |
|---|---|---|---|
| Linear Regression | 0.321 | 0.515 | 0.873 |
| Decision Tree | 0.342 | 0.547 | 0.856 |
| Random Forest | 0.224 | 0.353 | 0.940 |

**Classification:** high CO (> 3 mg/m³) vs normal (1,715 high vs 5,959 normal)

| Model | Accuracy | Precision | F1 | Recall |
|---|---|---|---|---|
| Logistic Regression | 0.96 | 0.91 | 0.90 | 0.89 |
| Decision Tree | 0.94 | 0.86 | 0.86 | 0.86 |
| Random Forest | 0.95 | 0.90 | 0.89 | 0.89 |

**Takeaway:** the sensor readings track CO closely, so all models perform well.

---

## 4. Bike Sharing, Hourly (Regression)

**Dataset:** 17,379 hourly bike rental records with weather, season, hour and holiday information. No missing values and no duplicates.

**Highlights**
- Avoided **data leakage**: `casual + registered = cnt` exactly, so both columns were removed from the predictors
- Dropped `instant` (row number) and `dteday` (already split into year, month, hour)
- Capped outliers in the weather columns only
- One-hot encoded `season` and `weathersit`

| Model | MAE | RMSE | R² |
|---|---|---|---|
| Linear Regression | 103.720 | 137.887 | 0.400 |
| Decision Tree | 34.657 | 59.339 | 0.889 |
| Random Forest | 24.856 | 42.083 | 0.944 |

**Takeaways**
- Rentals show two commute peaks on working days, which a straight line cannot capture, so tree models win by a wide margin.
- The hour of the day is by far the most important feature, followed by temperature and year.

---

## Common workflow

1. **Load and inspect:** `head`, `tail`, `info`, `describe`, `shape`, `dtypes`
2. **Clean:** missing values, duplicates, wrong headers, placeholder values
3. **Outliers:** boxplots, then IQR capping with `clip()`
4. **Encoding:** one-hot encoding for categorical columns
5. **EDA:** 6 charts per project (distributions, group comparisons, correlation heatmap)
6. **Split:** 80/20 train/test split
7. **Preprocessing:** `ColumnTransformer` with imputer + `StandardScaler`, fitted on training data only
8. **Feature selection:** `SelectKBest`
9. **Modelling:** 3 algorithms compared on the same split
10. **Evaluation:** metrics table, confusion matrix or actual-vs-predicted plot, feature importance

---

## Repository structure

```
├── README.md
├── creditcard_full.ipynb        # Credit card default (classification)
├── winequality_full.ipynb       # White wine quality
├── airquality_full.ipynb        # Air quality (UCI)
├── bikesharing_full.ipynb       # Bike sharing (hourly)
├── pdf/                         # PDF exports of each notebook with outputs
└── data/
    ├── creditcard.xls
    ├── winequality-white.csv
    ├── AirQualityUCI.csv
    └── hour.csv
```

---

## How to run

```bash
# 1. Clone the repo
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>

# 2. Install the libraries
pip install numpy pandas matplotlib seaborn scikit-learn jupyter

# 3. Open a notebook
jupyter notebook
```

Each notebook reads its CSV from the same folder, so keep the data file next to the notebook (or update the path in the first cell).

---

## Datasets

- **Default of Credit Card Clients:** UCI Machine Learning Repository
- **Wine Quality (white):** P. Cortez et al., UCI Machine Learning Repository
- **Air Quality:** S. De Vito et al., UCI Machine Learning Repository
- **Bike Sharing Dataset:** Capital Bikeshare data, UCI Machine Learning Repository

---

## Limitations and future work

- **Time series split:** Air Quality and Bike Sharing are hourly time series, but a random 80/20 split was used, so neighbouring hours appear in both train and test and scores are slightly optimistic. A chronological split (train on earlier dates, test on later) would be a stricter test.
- **Class imbalance:** Credit Card and Wine have far more "negative" cases, so recall is low. Next steps: `class_weight="balanced"`, SMOTE, or threshold tuning.
- **Hyperparameter tuning:** all models use default settings. `GridSearchCV` or `RandomizedSearchCV` and cross-validation would likely improve results.
- **More models:** try Gradient Boosting, XGBoost and KNN.

---

## Tech stack

Python · pandas · NumPy · scikit-learn · Matplotlib · Seaborn · Jupyter Notebook

---

## Author

**<Your Name>**
[LinkedIn](https://www.linkedin.com/in/<your-profile>) · [GitHub](https://github.com/<your-username>)

If you found this useful, feel free to star the repo.
