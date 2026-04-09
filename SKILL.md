# Employee Attrition Analysis Skill

## Overview
- **Skill name:** Employee Attrition Analysis Skill
- **Version:** 1.0.0
- **Domain:** HR Analytics

This skill executes a full, production-style attrition analytics workflow from raw employee-level CSV data to validated features, model benchmarking, SHAP-based interpretation, risk segmentation, and an executive-ready HTML report. It enforces structured quality checks before modeling, compares three baseline classifiers under SMOTE-aware cross-validation, and produces reusable artifacts for both technical and business stakeholders.

**When to use this skill**
- You need to estimate attrition risk from tabular HR data.
- You need explainable model drivers (SHAP) for HR interventions.
- You need a repeatable, auditable end-to-end pipeline with reports.

**Expected runtime:** ~5-8 minutes for full pipeline (hardware dependent).

## Required Inputs

### 1. Dataset Input
- **File format:** CSV
- **Minimum rows:** 500

Required columns (31 columns; excludes `EmployeeCount`, `EmployeeNumber`, `Over18`, `StandardHours`):

| Column | Type | Valid Values | Role |
|---|---|---|---|
| Age | int | 18-60 | REQUIRED |
| Attrition | object | Yes/No | TARGET |
| BusinessTravel | object | Non-Travel / Travel_Rarely / Travel_Frequently | REQUIRED |
| DailyRate | int | 100-1500 | REQUIRED |
| Department | object | Sales / Research & Development / Human Resources | OPTIONAL |
| DistanceFromHome | int | 1-29 | REQUIRED |
| Education | int | 1-5 | REQUIRED |
| EducationField | object | Dataset categories | OPTIONAL |
| EnvironmentSatisfaction | int | 1-4 | REQUIRED |
| Gender | object | Male/Female | REQUIRED |
| HourlyRate | int | positive int | REQUIRED |
| JobInvolvement | int | 1-4 | REQUIRED |
| JobLevel | int | 1-5 | REQUIRED |
| JobRole | object | 9 categories | OPTIONAL |
| JobSatisfaction | int | 1-4 | REQUIRED |
| MaritalStatus | object | Single/Married/Divorced | OPTIONAL |
| MonthlyIncome | int | positive int | REQUIRED |
| MonthlyRate | int | positive int | REQUIRED |
| NumCompaniesWorked | int | 0-9 | REQUIRED |
| OverTime | object | Yes/No | REQUIRED |
| PercentSalaryHike | int | 11-25 | REQUIRED |
| PerformanceRating | int | 3-4 | REQUIRED |
| RelationshipSatisfaction | int | 1-4 | REQUIRED |
| StockOptionLevel | int | 0-3 | REQUIRED |
| TotalWorkingYears | int | >=0 | REQUIRED |
| TrainingTimesLastYear | int | 0-6 | REQUIRED |
| WorkLifeBalance | int | 1-4 | REQUIRED |
| YearsAtCompany | int | >=0 | REQUIRED |
| YearsInCurrentRole | int | >=0 | REQUIRED |
| YearsSinceLastPromotion | int | >=0 | REQUIRED |
| YearsWithCurrManager | int | >=0 | REQUIRED |

### 2. User Configuration Inputs

| Parameter | Type | Default | Valid Range | Description |
|---|---|---|---|---|
| --random-seed | int | 42 | any int | Reproducibility seed |
| --company-name | str | "HR Analytics" | any string | Label in report |

## Input Validation Rules
- If fewer than 500 rows: **EXIT** with error
- If fewer than 10 columns: **EXIT** with error
- If Attrition column missing: **EXIT** with error
- If OverTime or Gender missing: **EXIT** with error
- If MonthlyIncome or JobLevel missing: **EXIT** with error
- If optional column missing: **WARN** and skip dependent feature/report component
- If null percentage > 30% in required column: **EXIT** with error
- If null percentage > 10% in any column: **WARN** and continue
- If duplicate rows found: **WARN** and continue

## Pipeline Stages

### Stage 1: Data Validation & Profiling
**Script:** `python scripts/validate_data.py --input <path> --output outputs/validation_report.json`

**What it computes:**
- Row/column counts, dropped constants
- Null counts and percentages per column
- Target class distribution and imbalance ratio
- Numeric column statistics: min, max, mean, median, std, skewness, outliers (IQR)
- Categorical column profiles: unique counts, value frequencies
- Ordinal range validation against expected bounds
- Duplicate row detection

**Output:** `outputs/validation_report.json`

**Validation checks before proceeding:**
- `validation_passed` must be true
- If false: read `errors` list and report to user before stopping

### Stage 2: Feature Engineering
**Script:** `python scripts/feature_engineering.py --input <path> --output outputs/features.csv --manifest outputs/feature_manifest.json`

**What it computes:**
- Drop constant columns: `EmployeeCount`, `EmployeeNumber`, `Over18`, `StandardHours`
- Encode target: Attrition Yes=1, No=0 -> `Attrition_encoded`
- Binary encode: `OverTime` (Yes=1/No=0), `Gender` (Male=1/Female=0)
- Ordinal keep as int: `Education`, `EnvironmentSatisfaction`, `JobInvolvement`, `JobLevel`, `JobSatisfaction`, `PerformanceRating`, `RelationshipSatisfaction`, `StockOptionLevel`, `WorkLifeBalance`
- One-hot encode: `BusinessTravel`, `Department`, `EducationField`, `JobRole`, `MaritalStatus`
- Engineer 8 features:
  1. `tenure_role_ratio = YearsInCurrentRole / (YearsAtCompany + 1)`
  2. `income_per_job_level = MonthlyIncome / (JobLevel + 1)`
  3. `promotion_lag = YearsSinceLastPromotion / (YearsAtCompany + 1)`
  4. `manager_stability = YearsWithCurrManager / (YearsAtCompany + 1)`
  5. `experience_company_ratio = TotalWorkingYears / (YearsAtCompany + 1)`
  6. `overtime_satisfaction_stress = OverTime * (5 - JobSatisfaction)`
  7. `loyalty_score = (YearsAtCompany * JobSatisfaction * WorkLifeBalance) / (TotalWorkingYears + 1)`
  8. `distance_overtime_interaction = DistanceFromHome * OverTime`

**Output:** `outputs/features.csv`, `outputs/feature_manifest.json`

**Validation before proceeding:**
- Confirm `features.csv` exists and shape is `(n_rows, 50+)`
- Confirm `Attrition_encoded` is last column
- Confirm no object dtype columns remain

### Stage 3: Model Training & Evaluation
**Script:** `python scripts/run_models.py --input outputs/features.csv --output outputs/model_results.json --random-seed 42`

**What it computes:**
- Loads `features.csv`, separates X and y
- Applies SMOTE within each CV fold (not on full data before CV)
- Trains 3 models with `StandardScaler` pipeline:
  - Logistic Regression (`max_iter=1000`, `class_weight=balanced`)
  - Random Forest (`n_estimators=100`, `class_weight=balanced`)
  - Gradient Boosting (`n_estimators=100`)
- 5-fold `StratifiedKFold` cross-validation
- Metrics per fold: ROC-AUC, F1, Precision, Recall
- Selects best model by mean ROC-AUC
- Trains final model on full SMOTE-resampled data
- Evaluates on original (non-SMOTE) data
- Computes SHAP feature importances (top 15)
- Segments employees into High/Medium/Low risk tiers

**Output:** `outputs/model_results.json`

**Validation before proceeding:**
- Confirm `best_model.roc_auc_cv > 0.70` (acceptable threshold)
- If below 0.70: warn user that model performance is low; recommend checking data quality and feature completeness
- Confirm `shap_top_features` list is not empty

### Stage 4: Report Generation
**Script:** `python scripts/generate_report.py --features outputs/features.csv --model-results outputs/model_results.json --output outputs/attrition_report.html --company-name "<name>"`

**What it produces:**
- Section 1: Executive Summary with risk snapshot
- Section 2: Data Quality Summary
- Section 3: Methodology
- Section 4: Model Comparison table + ROC-AUC bar chart
- Section 5: Confusion matrix + classification report + metric cards
- Section 6: SHAP top 15 feature importance chart + interpretation table
- Section 7: Risk tier donut chart + attrition rate by OverTime/JobLevel/Department
- Section 8: Business recommendations (driven by top SHAP features)
- Section 9: Assumptions and limitations
- Section 10: Data appendix

**Output:** `outputs/attrition_report.html`

**Validation:**
- Confirm HTML file exists and size > 100KB
- Open in browser to verify all sections rendered

## Running the Full Pipeline

### Option A: Run notebook
```bash
jupyter notebook run_pipeline.ipynb
# Run all cells top to bottom
```

### Option B: Run scripts individually
```bash
python scripts/validate_data.py --input data/employee-attrition.csv --output outputs/validation_report.json

python scripts/feature_engineering.py --input data/employee-attrition.csv --output outputs/features.csv --manifest outputs/feature_manifest.json

python scripts/run_models.py --input outputs/features.csv --output outputs/model_results.json --random-seed 42

python scripts/generate_report.py --features outputs/features.csv --model-results outputs/model_results.json --output outputs/attrition_report.html --company-name "IBM HR Analytics"
```

## Expected Outputs
| File | Description | Typical Size |
|---|---|---|
| outputs/validation_report.json | Data quality checks | ~50 KB |
| outputs/features.csv | Engineered feature matrix | ~500 KB |
| outputs/feature_manifest.json | Feature documentation | ~10 KB |
| outputs/model_results.json | Model metrics + SHAP | ~50 KB |
| outputs/attrition_report.html | Final HTML report | ~500 KB |

## Error Handling
| Error | Cause | Fix |
|---|---|---|
| Dataset has N rows; minimum 500 | Too few rows | Use complete dataset |
| Missing required columns | Wrong CSV format | Check data_dictionary.md |
| Attrition_encoded missing | Wrong input to run_models | Run feature_engineering first |
| SHAP computation failed | Library version mismatch | `pip install shap --upgrade` |

## Test Scenarios
| Scenario | Input | Expected Outcome |
|---|---|---|
| Happy path | Full IBM dataset (1470 rows) | All stages pass, ROC-AUC > 0.75 |
| Bad data | CSV missing Attrition/MonthlyIncome | Validation FAILS with clear error |
| Subset analysis | Junior employees (JobLevel 1-2) | Pipeline passes, different model winner possible |
