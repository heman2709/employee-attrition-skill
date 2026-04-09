# Employee Attrition - Domain Reference Guide

## 1. Business Context
Employee attrition is the rate at which employees voluntarily or involuntarily leave an organization over a defined period. It directly affects productivity, hiring cost, team continuity, and customer outcomes. Attrition modeling helps HR teams move from reactive backfilling to proactive retention intervention by identifying employees with elevated exit risk before resignation occurs.

- **Industry average attrition rates by sector**
  - Tech: 13-15%
  - Financial Services: 10-12%
  - Manufacturing: 8-10%
- **Cost of attrition:** typically 50-200% of annual salary per lost employee (role dependent).
- **IBM dataset context:** 16.1% attrition rate (237/1470 employees).

## 2. Feature Definitions and Business Meaning

### 2.1 Raw Features

| Feature | Business Meaning | Attrition Signal |
|---|---|---|
| Age | Employee age/life stage | Very early-career and transition-stage employees may show higher mobility |
| Attrition | Exit outcome (target) | `Yes` indicates employee left |
| BusinessTravel | Travel intensity | Frequent travel can increase fatigue and churn |
| DailyRate | Daily pay rate | Misaligned pay can increase flight risk |
| Department | Functional business unit | Some departments experience structurally higher turnover |
| DistanceFromHome | Commute burden | Long commutes correlate with burnout and resignation risk |
| Education | Formal education level | Education can influence external opportunity access |
| EducationField | Academic specialization | Market demand differences by field affect mobility |
| EnvironmentSatisfaction | Satisfaction with workplace environment | Lower satisfaction links to greater attrition risk |
| Gender | Demographic factor | May reflect policy/equity or role-distribution effects |
| HourlyRate | Hourly compensation indicator | Lower relative compensation can elevate risk |
| JobInvolvement | Engagement with job responsibilities | Low involvement is an early disengagement signal |
| JobLevel | Organizational seniority | Level 1 often has higher attrition than senior levels |
| JobRole | Specific role family | Some role types have historically higher exits |
| JobSatisfaction | Role satisfaction score | Direct inverse relationship with attrition |
| MaritalStatus | Household status | Single employees often show higher mobility |
| MonthlyIncome | Monthly compensation | Lower income relative to peers increases flight risk |
| MonthlyRate | Monthly payroll metric | Can proxy compensation structure differences |
| NumCompaniesWorked | External career mobility history | High prior switching may predict future switching |
| OverTime | Overtime workload indicator | Overtime employees are often substantially more likely to leave |
| PercentSalaryHike | Recent salary increase percent | Low hikes may weaken retention |
| PerformanceRating | Performance quality band | High performers may be poached; low variance limits signal |
| RelationshipSatisfaction | Satisfaction with workplace relationships | Poor relationship quality raises attrition likelihood |
| StockOptionLevel | Equity/retention incentive level | Higher equity can improve retention |
| TotalWorkingYears | Total career experience | High marketability can increase outside opportunities |
| TrainingTimesLastYear | Skill development exposure | Low development investment may increase churn |
| WorkLifeBalance | Work/life quality perception | Poor balance strongly relates to attrition risk |
| YearsAtCompany | Company tenure | Peak attrition risk often appears at 1-3 years |
| YearsInCurrentRole | Time in same role | Long stagnation can increase exit intent |
| YearsSinceLastPromotion | Career progression lag | Long promotion gaps are a dissatisfaction signal |
| YearsWithCurrManager | Manager continuity | Frequent manager changes can reduce stability/trust |

### 2.2 Engineered Features

| Feature | Formula | Business Meaning | High Value Interpretation |
|---|---|---|---|
| tenure_role_ratio | YearsInCurrentRole / (YearsAtCompany + 1) | Role stagnation relative to tenure | >0.7 suggests employee may be stuck in-role |
| income_per_job_level | MonthlyIncome / (JobLevel + 1) | Compensation relative to seniority | Low values (<2000) may indicate underpayment by level |
| promotion_lag | YearsSinceLastPromotion / (YearsAtCompany + 1) | Promotion velocity proxy | >0.5 indicates slow advancement trend |
| manager_stability | YearsWithCurrManager / (YearsAtCompany + 1) | Leadership continuity | Low values (<0.3) imply frequent manager changes |
| experience_company_ratio | TotalWorkingYears / (YearsAtCompany + 1) | External market experience vs firm tenure | >3 implies strong outside-market optionality |
| overtime_satisfaction_stress | OverTime * (5 - JobSatisfaction) | Burnout intensity interaction | >3 signals critical overtime+satisfaction stress |
| loyalty_score | (YearsAtCompany * JobSatisfaction * WorkLifeBalance) / (TotalWorkingYears + 1) | Composite retention propensity | <1.0 suggests weak loyalty anchor |
| distance_overtime_interaction | DistanceFromHome * OverTime | Commute-workload burden | >15 indicates severe commute+overtime risk |

## 3. Model Selection Criteria

### 3.1 Algorithm Guide
| Algorithm | Strengths | Weaknesses | Best When |
|---|---|---|---|
| Logistic Regression | Interpretable, fast, often well-calibrated | Linear decision boundary assumptions | Need explainability and stable probability outputs |
| Random Forest | Captures non-linearity, robust to outliers | Less interpretable, can be slower | Mixed feature interactions dominate |
| Gradient Boosting | Strong predictive accuracy, handles complex patterns | More sensitive to tuning, slower | Accuracy is priority and data quality is stable |

### 3.2 Metric Selection Rationale
For attrition prediction:
- **PRIMARY metric: ROC-AUC** - measures discrimination across all thresholds and is robust under class imbalance.
- **SECONDARY: Recall** - missing a high-risk employee (false negative) is costlier than a false alarm.
- **ROC-AUC interpretation**
  - < 0.70: Poor - do not use, check data quality
  - 0.70-0.80: Acceptable
  - 0.80-0.90: Good
  - > 0.90: Excellent (also check for data leakage)

### 3.3 Class Imbalance Handling
- IBM dataset class split: 83.9% No Attrition vs 16.1% Yes Attrition
- Imbalance ratio: ~5.2:1
- Strategy: SMOTE within CV folds + `class_weight=balanced`
- Why SMOTE inside folds: prevents synthetic-sample leakage into validation folds

## 4. Risk Tier Definitions
| Tier | Probability Threshold | Interpretation | Urgency |
|---|---|---|---|
| High Risk | >= 0.60 | Strong attrition signals present | Act within 2 weeks |
| Medium Risk | 0.30 - 0.59 | Some warning signs | Monitor monthly |
| Low Risk | < 0.30 | Stable engagement signals | Quarterly review |

Threshold rationale: 0.60 prioritizes minimizing false negatives on high-cost attrition events. For high-cost roles (e.g., senior engineers, top sales), consider lowering threshold to 0.50.

## 5. Intervention Playbook

### 5.1 By Risk Tier
**High Risk employees**
- Schedule 1:1 retention conversation within 2 weeks
- Review compensation against market benchmarks
- Identify promotion or role expansion opportunity
- Assign mentorship or executive sponsor

**Medium Risk employees**
- Pulse survey within 30 days
- Manager notified for increased check-in frequency
- Review workload and overtime patterns

**Low Risk employees**
- Standard quarterly engagement survey
- Recognition and development programs

### 5.2 By Top SHAP Driver
| Driver | Recommended Intervention |
|---|---|
| OverTime | Cap overtime at ~10% of workforce per quarter; add comp time policy |
| MonthlyIncome / income_per_job_level | Annual compensation review vs market bands |
| tenure_role_ratio / YearsAtCompany | Structured career ladder conversations at 2-year mark |
| JobSatisfaction | Role redesign, project rotation, manager coaching |
| WorkLifeBalance | Flexible hours, remote options, PTO planning |
| DistanceFromHome | Remote/hybrid options or relocation assistance for long commutes |
| promotion_lag | Fast-track promotion review for >2 years without promotion |

## 6. Interpretation Framework

### 6.1 Reading SHAP Values
- Positive SHAP value: pushes prediction toward attrition (`Yes=1`)
- Negative SHAP value: pushes prediction toward retention (`No=0`)
- Mean absolute SHAP: global feature influence magnitude (direction-agnostic)
- Feature with SHAP importance > 0.05 is typically a strong driver in this context

### 6.2 Model Output Interpretation Template
Use this language for stakeholders:

"Based on analysis of {n} employees, {pct_high}% are classified as High Risk. The primary drivers of attrition risk in this population are {top_3_features}. We recommend prioritizing retention efforts on the {high_risk_count} High Risk employees, particularly addressing {top_driver} which is the strongest signal."

### 6.3 Common Patterns in IBM Dataset
- Sales Representatives often show highest attrition (~40%)
- Laboratory Technicians often show elevated attrition (~24%)
- `OverTime=Yes` may show ~31% attrition vs ~10% without overtime
- Single employees often show higher attrition (~25%) than married (~13%)
- JobLevel 1 often has highest attrition (~26%) vs very low rates at JobLevel 5 (~5%)

## 7. Data Quality Benchmarks
| Check | Warning Threshold | Error Threshold |
|---|---|---|
| Null percentage | > 10% | > 30% |
| Class imbalance ratio | > 5.0 | > 10.0 |
| Skewness | > 2.0 | N/A (log transform recommended) |
| Duplicate rows | Any | N/A |
| Min rows | N/A | < 500 |

## 8. Reproducibility Notes
- All scripts accept `--random-seed` (default: 42)
- SMOTE uses same seed as model pipeline
- Two runs with identical seed + identical data should produce identical outputs
- Random seed affects CV splits, SMOTE sampling, and model initialization
