# Health Insurance Cross-Sell Prediction

**Author:** Giacomo Piergentili  
**Course:** Project Work in Machine Learning and Data Mining  
**Date:** March 2026

## Objective

The goal of this project is to predict whether existing health insurance customers would also be interested in purchasing vehicle insurance. This is a binary classification problem based on the [Kaggle Playground Series - Season 4, Episode 7](https://www.kaggle.com/competitions/playground-series-s4e7) competition (original data by [Anmol Kumar](https://www.kaggle.com/datasets/anmolkumar/health-insurance-cross-sell-prediction)).

From a business perspective, being able to identify interested customers lets the insurer run targeted cross-sell campaigns instead of contacting everyone, saving money and improving conversion rates.

## Project structure

```
piergentili-health_insurance_cross_sell_prediction/
├── README.md               ← this file (project overview)
├── doc/
│   ├── presentation.tex    ← LaTeX Beamer slides
│   ├── presentation.pdf    ← compiled presentation
│   └── figures/            ← figures used in the presentation
└── project/
    ├── README.md           ← installation, prerequisites, how to run
    ├── preprocessing.ipynb ← data loading, inspection, type optimisation
    ├── eda.ipynb           ← exploratory data analysis
    ├── model.ipynb         ← modelling, feature engineering experiment, submission
    └── data/
        ├── old/            ← original CSV files from Kaggle
        └── processed/      ← Parquet files, submission CSV
```

## Dataset

~11.5M training records, 10 features + binary target (`Response`). Only ~12% of customers responded positively, so the dataset is significantly imbalanced.

| Column | Description |
|---|---|
| `id` | Unique customer identifier |
| `Gender` | 0 = Male, 1 = Female |
| `Age` | Customer age in years |
| `Driving_License` | 0 = No, 1 = Yes |
| `Region_Code` | Anonymised region code (0–52) |
| `Previously_Insured` | Already has vehicle insurance (0/1) |
| `Vehicle_Age` | 0 = < 1 Year, 1 = 1–2 Years, 2 = > 2 Years |
| `Vehicle_Damage` | 0 = No, 1 = Yes |
| `Annual_Premium` | Annual health insurance premium |
| `Policy_Sales_Channel` | Anonymised contact channel code |
| `Vintage` | Days associated with the company |
| `Response` | **Target**: interested in vehicle insurance (0/1) |

## Approach

The work is split across three notebooks:

1. **preprocessing.ipynb**: load the raw CSVs, inspect the data, encode categorical columns as integers, downcast numeric types, and save to Parquet.
2. **eda.ipynb**: univariate distributions, bivariate response-rate analysis, and correlation heatmap. Main findings: `Previously_Insured` and `Vehicle_Damage` are near-collinear (r ≈ −0.84); `Age` and `Vehicle_Age` are strongly correlated (r ≈ 0.78); already-insured customers have essentially zero interest.
3. **model.ipynb**: train Logistic Regression and LightGBM (both with class-weight adjustments for the imbalance), run a feature engineering experiment, evaluate, and generate the Kaggle submission.

The feature engineering experiment tested frequency encoding of high-cardinality columns, an Age×Vehicle_Age interaction term, and dropping the collinear `Vehicle_Damage` column. Models were trained on both the original and the engineered feature sets to compare.

## Results

| Model | ROC-AUC | Precision | Recall | F1 |
|---|---|---|---|---|
| Logistic Regression (original) | 0.8365 | 0.2526 | 0.9801 | 0.4017 |
| Logistic Regression (engineered) | 0.8222 | 0.2336 | 0.9739 | 0.3769 |
| **LightGBM (original)** | **0.8784** | **0.3000** | **0.9296** | **0.4536** |
| LightGBM (engineered) | 0.8693 | 0.2872 | 0.9255 | 0.4384 |

LightGBM on the original features was the best model. The feature engineering actually made things slightly worse for both models, since tree-based models already handle high-cardinality categoricals and feature interactions internally, so the manual transformations didn't add useful information.

The most important feature by far was `Previously_Insured`, which aligns with the EDA finding that already-insured customers almost never respond. `Vehicle_Age`, `Annual_Premium`, and `Policy_Sales_Channel` followed.

## Limitations

- Single train/validation split instead of k-fold cross-validation
- Default 0.5 classification threshold (not tuned for business costs)
- Only two model families tested (LR, LightGBM); CatBoost, XGBoost, or stacking could potentially improve results

## References

- [Kaggle Playground Series S4E7 (competition page)](https://www.kaggle.com/competitions/playground-series-s4e7)
- [Original dataset by Anmol Kumar](https://www.kaggle.com/datasets/anmolkumar/health-insurance-cross-sell-prediction)
- [LightGBM documentation](https://lightgbm.readthedocs.io/)
- [scikit-learn documentation](https://scikit-learn.org/stable/)
