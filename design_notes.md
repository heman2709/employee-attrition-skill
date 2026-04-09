# Design notes — Employee Attrition Skill (raw extraction for walkthrough)

Factual extraction from the codebase only. Not a design document.

---

## 1. Project Overview

**Skill package (one paragraph, from `SKILL.md`):** The Employee Attrition Analysis Skill executes a full attrition analytics workflow from raw employee-level CSV to validated features, model benchmarking, SHAP-based interpretation, risk segmentation, and an executive HTML report. It enforces quality checks before modeling, compares three baseline classifiers under SMOTE-aware cross-validation, and produces reusable artifacts.

**Domain:** HR Analytics (`SKILL.md`).

**Dataset:** IBM HR Employee Attrition. **Source (from `README.md`):** IBM HR Analytics dataset from Kaggle (public domain). **Shape:** `data/employee-attrition.csv` has **1470 rows × 35 columns** (header + 1470 data lines in repo). After dropping four columns in validation/engineering, **`cols_after_drop=31`** (`run_pipeline.ipynb` output / `validate_data.py` drop list).

**Target variable:** `Attrition` (`Yes`/`No`). Counts from `run_pipeline.ipynb` / validation output: **237 Yes, 1233 No**. Rate: **237/1470 ≈ 16.12%** (REFERENCE states **16.1%, 237/1470**).

**Primary business question (from docs, not a single explicit “question” string in code):** Predict and explain employee **attrition risk** to support **proactive retention** (`README.md` opening; `REFERENCE.md` §1 business context). `SKILL.md` frames use cases: estimate risk from tabular HR data, explain drivers (SHAP), repeatable auditable pipeline.

---

## 2. Folder Structure

Full file tree (workspace listing; omitting `.DS_Store`):

| Path | One-line description |
|------|----------------------|
| `.gitignore` | Ignore patterns (e.g. `.env`, outputs, checkpoints). |
| `README.md` | Setup, how to run validation stub, Kaggle dataset note, high-level folder overview. |
| `REFERENCE.md` | Domain reference: features, metrics, risk tiers, interventions, data-quality table. |
| `SKILL.md` | Agent/skill spec: inputs, validation rules, four stages, outputs, errors, test scenarios. |
| `design_notes.md` | This facts file. |
| `requirements.txt` | Python dependency list (unpinned Python). |
| `llm_runner.ipynb` | OpenAI function-calling agent: 4 tools, agentic loop, trace saving. |
| `run_pipeline.ipynb` | Sequential subprocess runner for all four scripts + 3 test scenarios. |
| `data/data_dictionary.md` | Column definitions, roles, drop list, modeling notes. |
| `data/employee-attrition.csv` | Main IBM attrition CSV (1470×35). |
| `scripts/validate_data.py` | Load, shape/required/null/target/profile/ordinal/duplicate checks → JSON. |
| `scripts/feature_engineering.py` | Encodings, 8 engineered features, manifest + features CSV. |
| `scripts/run_models.py` | SMOTE-in-fold CV, 3 models, best by ROC-AUC, final fit, SHAP, risk segments. |
| `scripts/generate_report.py` | Jinja2 + matplotlib charts → single-file HTML report. |
| `templates/report_template.html` | HTML scaffold for `generate_report.py`. |
| `outputs/.gitkeep` | Keeps `outputs/` in version control when empty. |
| `outputs/*` | Generated JSON/CSV/HTML/trace artifacts (see §10). |

---

## 3. Pipeline Architecture

| Stage | Script | Inputs | Outputs | What it computes (from module/`main` docstrings) |
|-------|--------|--------|---------|--------------------------------------------------|
| 1 | `scripts/validate_data.py` | `--input` CSV; `--output` JSON path | `validation_report.json` (path configurable) | **Module:** validate/profile IBM HR data; **main:** run validation workflow and write JSON report. Implements row/col minimums, required columns, null rules, target balance, numeric/categorical profiles, ordinal ranges, duplicates. |
| 2 | `scripts/feature_engineering.py` | `--input` CSV; `--output` features CSV; `--manifest` JSON | Features CSV + manifest JSON | **Module:** transform raw HR data into model-ready matrix and JSON manifest; **main:** full CLI pipeline from arguments. |
| 3 | `scripts/run_models.py` | `--input` engineered CSV; `--output` JSON; `--random-seed` (default 42) | `model_results.json` | **Module:** compare classifiers with SMOTE-aware CV, select best by ROC-AUC, SHAP, risk tiers, export JSON; **main:** run training, comparison, evaluation, SHAP, JSON export. |
| 4 | `scripts/generate_report.py` | `--features`, `--model-results`, `--output`, `--company-name` | HTML report | **Module:** read model outputs and engineered features, build matplotlib/seaborn charts as base64 PNG, write self-contained HTML. |

Default paths in `SKILL.md` examples: `outputs/validation_report.json`, `outputs/features.csv`, `outputs/feature_manifest.json`, `outputs/model_results.json`, `outputs/attrition_report.html`.

---

## 4. Design Decisions (from code / cited docs)

### 4.1 Why these 4 drop columns

**Columns:** `EmployeeCount`, `EmployeeNumber`, `Over18`, `StandardHours` (`validate_data.py` `DROP_COLUMNS`; `feature_engineering.py` `DROP_CONSTANT_COLUMNS`).

**From `data/data_dictionary.md`:**
- `EmployeeCount`: “Constant helper column; no variance.” Always `1`. Role: ID/DROP.
- `EmployeeNumber`: “Unique employee identifier.” Role: ID/DROP.
- `Over18`: “Constant legal-age indicator; no variance.” Always `Y`. Role: ID/DROP.
- `StandardHours`: “Constant standard work hours; no variance.” Always `80`. Role: ID/DROP.

**Modeling notes (same file):** Drop these four before training.

### 4.2 Why SMOTE inside CV folds

**`run_models.py`:** `cv_with_smote` docstring: *“Run manual stratified CV with SMOTE applied in each training fold.”* Implementation: for each fold, `SMOTE.fit_resample` on **train** only; fit model; predict on **validation** fold (original rows).

**`REFERENCE.md` §3.3:** *“Why SMOTE inside folds: prevents synthetic-sample leakage into validation folds.”*

(Extended “reasoning” beyond the short docstring lives primarily in `REFERENCE.md`, not in extra inline comments in `run_models.py`.)

### 4.3 Why 3 models were chosen

**`run_models.py` `define_models`:** Only defines pipelines — **no** per-algorithm rationale in docstrings.

**Narrative “what each brings” is in `REFERENCE.md` §3.1 Algorithm Guide:**

| Algorithm | Strengths | Weaknesses | Best When |
|-----------|-----------|-------------|-----------|
| Logistic Regression | Interpretable, fast, often well-calibrated | Linear decision boundary assumptions | Need explainability and stable probability outputs |
| Random Forest | Captures non-linearity, robust to outliers | Less interpretable, can be slower | Mixed feature interactions dominate |
| Gradient Boosting | Strong predictive accuracy, handles complex patterns | More sensitive to tuning, slower | Accuracy is priority and data quality is stable |

### 4.4 Why ROC-AUC as primary metric

**`REFERENCE.md` §3.2 Metric Selection Rationale:**
- **PRIMARY: ROC-AUC** — “measures discrimination across all thresholds and is robust under class imbalance.”
- **SECONDARY: Recall** — “missing a high-risk employee (false negative) is costlier than a false alarm.”

**`run_models.py`:** `select_best_model` docstring: *“Select best model by highest mean ROC-AUC.”*

### 4.5 Eight engineered features

Exact formulas and `business_meaning` from `feature_engineering.py` `engineer_features` manifest entries:

| Column | Formula | business_meaning |
|--------|---------|------------------|
| `tenure_role_ratio` | `YearsInCurrentRole / (YearsAtCompany + 1)` | Career stagnation signal - stuck in same role relative to tenure. High value = attrition risk. |
| `income_per_job_level` | `MonthlyIncome / (JobLevel + 1)` | Underpaid for seniority level. Low value = attrition risk. |
| `promotion_lag` | `YearsSinceLastPromotion / (YearsAtCompany + 1)` | Career progression slowdown signal. |
| `manager_stability` | `YearsWithCurrManager / (YearsAtCompany + 1)` | Low value = frequent manager changes = instability. |
| `experience_company_ratio` | `TotalWorkingYears / (YearsAtCompany + 1)` | High ratio = broad market experience = more likely to leave. |
| `overtime_satisfaction_stress` | `OverTime * (5 - JobSatisfaction)` | Overtime AND low satisfaction = burnout signal. |
| `loyalty_score` | `(YearsAtCompany * JobSatisfaction * WorkLifeBalance) / (TotalWorkingYears + 1)` | Composite loyalty proxy. |
| `distance_overtime_interaction` | `DistanceFromHome * OverTime` | Long commute + overtime = burnout. |

**`REFERENCE.md` §2.2** duplicates formulas and adds “High Value Interpretation” thresholds (e.g. tenure_role_ratio >0.7); those interpretive cuts are **not** enforced in `run_models.py` segmentation.

### 4.6 Risk tier thresholds

**`run_models.py` `segment_risk`:**
- **High:** `y_proba >= 0.6`
- **Medium:** `(y_proba >= 0.3) & (y_proba < 0.6)`
- **Low:** `y_proba < 0.3`

**`REFERENCE.md` §4 Risk Tier Definitions:** Same cutoffs in table form; **threshold rationale:** “0.60 prioritizes minimizing false negatives on high-cost attrition events. For high-cost roles … consider lowering threshold to 0.50.”

### 4.7 LLM orchestration (`llm_runner.ipynb`)

- **Four tools (function definitions):** `run_validation`, `run_feature_engineering`, `run_model_training`, `run_report_generation` — each wraps a `subprocess` call to the matching script (`execute_tool`).
- **Agentic loop:** `run_agentic_pipeline(user_prompt, run_name, max_iterations=20)` — `while iteration < max_iterations`, calls `client.chat.completions.create` with `model="gpt-4o"`, `tools=TOOLS`, `tool_choice="auto"`, `temperature=0`, `max_tokens=2000`. System prompt embeds full `SKILL.md` and `REFERENCE.md` text; instructs ordered tool use and validation-failure stop.
- **Assistant step:** Builds `{"role": "assistant", "content": ..., "tool_calls": [...]}` when tools are returned.
- **`role="tool"` messages:** After each tool invocation, appends `{"role": "tool", "tool_call_id": tool_call.id, "content": tool_output}` where `tool_output` is stdout/stderr string from the script.
- **Stop condition:** `if finish_reason == "stop" or not msg.tool_calls: break` (also bounded by `max_iterations`).

---

## 5. SKILL.md vs REFERENCE.md

**`SKILL.md` — how the LLM is instructed:** Overview, when to use, expected runtime; required/optional column table (note conflict with scripts — see §7); validation rules (row/column minimums, null %, duplicates); **four stages** with exact CLI, computed artifacts, and “validation before proceeding” checks (e.g. `best_model.roc_auc_cv > 0.70`, `shap_top_features` non-empty, HTML size); full pipeline commands; expected output table; error handling table; **three test scenarios** table.

**`REFERENCE.md` provides:** Business context and costs; feature meanings; engineered-feature table with interpretations; algorithm guide; **ROC-AUC / recall** rationale; imbalance and **SMOTE-in-fold** reason; **risk tier** table and playbook; SHAP reading guide; IBM-specific pattern bullets; data-quality benchmark table; reproducibility notes.

**How they work together:** `llm_runner.ipynb` injects both files into the system prompt so the model follows `SKILL.md` procedure while interpreting outputs with `REFERENCE.md` domain language.

---

## 6. Test Scenarios

| Scenario name | Input used | Expected behavior (from `SKILL.md` + `run_pipeline.ipynb`) | Tested in |
|---------------|------------|----------------------------------------------------------------|-----------|
| Happy path | Full IBM dataset `data/employee-attrition.csv` (1470 rows) | All stages pass; `SKILL.md` adds expectation ROC-AUC > 0.75 for this scenario. | `run_pipeline.ipynb` (Step 1–4 main flow); `llm_runner.ipynb` “Run 1 — Happy Path”; trace `outputs/run1_happy_path_trace.json` |
| Bad data | CSV missing **`Attrition`, `MonthlyIncome`, `OverTime`** (`outputs/bad_data_test.csv` created in notebook) | **Validation FAILS** with clear error (`SKILL.md`: “Validation FAILS with clear error”). | `run_pipeline.ipynb` bad-data cells; `llm_runner.ipynb` “Run 2”; `outputs/run2_bad_data_trace.json` |
| Subset / junior analysis | **`JobLevel` ≤ 2** subset → `outputs/junior_employees.csv`; pipeline uses `junior_*.csv/json/html` outputs | Pipeline passes; **`SKILL.md`:** “different model winner possible”. | `run_pipeline.ipynb` Scenario 3 cells; `llm_runner.ipynb` Run 3 (user prompt directs **junior** paths to avoid overwriting Run 1); `outputs/run3_junior_subset_trace.json` |

---

## 7. Guardrails and Validation

### 7.1 `validate_data.py` checks

| Check | Threshold / rule | Action |
|-------|------------------|--------|
| Row count | `< 500` | **error** (append to `errors`) |
| Column count | `< 10` | **error** |
| Load failure | file not found / read exception | **error** or empty DF path |
| Required columns | missing any of `REQUIRED_TARGET`, `REQUIRED_CATEGORICAL`, `REQUIRED_NUMERIC` | **error** |
| Null % (any column) | `> 10%` | **warn** |
| Null % (required column list) | `> 30%` | **error** |
| Target missing | no `Attrition` | **error** in `analyze_target` |
| Class imbalance ratio | `> 5.0` | **warn** (no hard error for ratio in script) |
| Numeric skewness | `\|skew\| > 2.0` | **warn** (suggest log transform) |
| Categorical cardinality | `unique_count > 50` | **warn** |
| Ordinal range | values outside `ORDINAL_RANGES` | **warn**; empty ordinal series → **warn** |
| Duplicate rows | `duplicated().sum() > 0` | **warn** |
| Final verdict | `validation_passed = (len(errors) == 0)` | Exit code **0** if pass, **1** if fail (`main`) |

**Note:** `REFERENCE.md` §7 lists “Class imbalance ratio … Error Threshold > 10.0” — **`validate_data.py` does not implement an error at ratio > 10**; only warns above 5.0.

### 7.2 `feature_engineering.py` error handling

| Location | Behavior |
|----------|----------|
| `load_and_drop` | `FileNotFoundError` → print + `sys.exit(1)`; other read `Exception` → print + `sys.exit(1)` |
| `encode_target` | Missing `Attrition` → print + `sys.exit(1)` |
| `encode_binary` | Missing `OverTime` or `Gender` → print + `sys.exit(1)` |
| `encode_ordinal` | Missing any ordinal column → print + `sys.exit(1)` |
| `encode_onehot` | Missing any of `ONE_HOT_COLUMNS` → print + `sys.exit(1)` |
| `engineer_features` | Missing any of internal `required` list (11 base columns) → print + `sys.exit(1)` |
| `final_cleanup` | Missing `Attrition_encoded` → print + `sys.exit(1)` |
| `save_outputs` | `to_csv` / manifest JSON `Exception` → print + `sys.exit(1)` |
| `compute_shap` (in `run_models.py`) | try/except → warning append, empty list |

Warnings (non-exit): unknown `Attrition` values; unknown binary values; non-numeric ordinal coerced; no constants to drop.

### 7.3 “Optional” columns vs code

**`SKILL.md`** marks `Department`, `EducationField`, `JobRole`, `MaritalStatus` as **OPTIONAL** and says missing optional → WARN/skip.

**Actual code:** `validate_data.py` includes those columns in **`REQUIRED_CATEGORICAL`** — missing → **error**. `feature_engineering.py` **`ONE_HOT_COLUMNS`** requires all five (`BusinessTravel` + those four) — missing → **`sys.exit(1)`**.

**Conclusion:** For this codebase, those columns are **required** for validation + feature engineering; behavior does not match `SKILL.md`’s optional-column rule.

---

## 8. Known Limitations

### From `generate_report.py` `generate_limitations()` (Section 9 HTML)

- SMOTE creates synthetic minority samples — results may differ on truly new data.
- SHAP values computed on training data — explanations may not fully generalize.
- Model trained on IBM HR dataset — validate before applying to other organizations.
- PerformanceRating has low variance (only 3 and 4), limiting predictive signal.
- Class imbalance (~16% attrition in this dataset family) addressed with SMOTE and balanced settings.

### `REFERENCE.md` “assumptions” section

**NOT FOUND** as a dedicated “Assumptions” section; only the word “assumptions” appears in the Logistic Regression weaknesses row (“Linear decision boundary assumptions”). Limitations-style content is mainly in `generate_limitations()` and business-context claims in §1–§6.

### Hardcoded values limiting generalizability (non-exhaustive)

- IBM-specific column schema and drops; `ORDINAL_RANGES` fixed in `validate_data.py`.
- `DROP_COLUMNS` / constant column names.
- Risk cutoffs `0.6` / `0.3` in `segment_risk`.
- Validation floors **500** rows, **10** columns.
- Null thresholds **10%** / **30%**.
- **5** CV folds; SMOTE and model hyperparameters (`n_estimators=100`, `max_iter=1000`, etc.).
- Report template and chart code tied to expected `model_results` / features schema.
- `llm_runner` fixed model **`gpt-4o`**, `max_iterations=20`.

---

## 9. Requirements

**Full `requirements.txt` (verbatim):**

```
pandas
numpy
scikit-learn
imbalanced-learn
matplotlib
seaborn
plotly
shap
xgboost
lightgbm
jinja2
argparse
scipy
python-dotenv
openai
```

**Python version:** **Not pinned** in `requirements.txt`. `llm_runner.ipynb` metadata: **`language_info.version` = `"3.11.0"`** (editor/kernel hint only).

---

## 10. Output Artifacts

Typical **byte sizes** from current `ls -la outputs/` (approximate):

| File | Description | Typical size |
|------|-------------|--------------|
| `validation_report.json` | Full validation JSON | ~10 KB |
| `feature_manifest.json` | Encoding + engineered feature manifest | ~4 KB |
| `features.csv` | Engineered matrix (happy path) | ~230 KB |
| `model_results.json` | CV metrics, best model, SHAP list, risk segments | ~4 KB |
| `attrition_report.html` | Full HTML report | ~400 KB |
| `bad_data_test.csv` | Stripped CSV for negative test | ~205 KB |
| `bad_data_validation.json` / `bad_validation.json` | Validation outputs for bad data runs | ~10 KB |
| `junior_employees.csv` | JobLevel≤2 subset | ~165 KB |
| `junior_features.csv` / `junior_manifest.json` / `junior_model_results.json` | Junior pipeline outputs | ~165 KB / ~4 KB / ~4 KB |
| `junior_validation.json` / `junior_validation_report.json` | Junior validation | ~11 KB |
| `junior_attrition_report.html` / `junior_report.html` | Junior HTML variants in repo | ~400 KB |
| `junior_results.json` | Extra JSON in repo | ~4 KB |
| `run1_happy_path_trace.json` | LLM message/tool trace Run 1 | ~9 KB |
| `run2_bad_data_trace.json` | LLM trace Run 2 | ~3 KB |
| `run3_junior_subset_trace.json` | LLM trace Run 3 | ~11 KB |
| `.gitkeep` | Placeholder | 2 B |

**Execution evidence for an assignment (from repo contents + notebook behavior):**

- **Deterministic pipeline:** `validation_report.json`, `features.csv`, `feature_manifest.json`, `model_results.json`, `attrition_report.html` (and junior/bad variants).
- **LLM orchestration traces:** `run1_happy_path_trace.json`, `run2_bad_data_trace.json`, `run3_junior_subset_trace.json` (written by `save_trace` in `llm_runner.ipynb`).
- **Notebook runs:** `run_pipeline.ipynb` and `llm_runner.ipynb` contain executed cells with stdout capturing step outcomes.

---

*End of design_notes.md*
