# IBM HR Employee Attrition Data Dictionary

This dictionary describes the 35-column IBM HR Employee Attrition dataset used for attrition modeling.

## Column Role Guide

- `TARGET`: `Attrition`
- `ID/DROP`: `EmployeeCount`, `EmployeeNumber`, `Over18`, `StandardHours`
- `Categorical (nominal)`: `Attrition`, `BusinessTravel`, `Department`, `EducationField`, `Gender`, `JobRole`, `MaritalStatus`, `OverTime`
- `Ordinal`: `Education`, `EnvironmentSatisfaction`, `JobInvolvement`, `JobLevel`, `JobSatisfaction`, `PerformanceRating`, `RelationshipSatisfaction`, `StockOptionLevel`, `WorkLifeBalance`
- `Continuous / numeric`: all remaining integer rate, tenure, count, and income columns

## Data Dictionary (35 columns)

| Column | Data Type | Description | Valid Values / Range | Role |
|---|---|---|---|---|
| Age | int | Employee age in years. | 18-60 (dataset integer range; often treated as numeric continuous) | Continuous |
| Attrition | object | Whether employee left the company. Prediction target. | `Yes`, `No` | TARGET, Categorical |
| BusinessTravel | object | Frequency of business travel. | `Non-Travel`, `Travel_Rarely`, `Travel_Frequently` | Categorical |
| DailyRate | int | Daily compensation rate. | 100-1500 | Continuous |
| Department | object | Functional department. | `Sales`, `Research & Development`, `Human Resources` | Categorical |
| DistanceFromHome | int | Commute distance from home (miles, dataset convention). | 1-29 | Continuous |
| Education | int | Education attainment level. | 1=Below College, 2=College, 3=Bachelor, 4=Master, 5=Doctor | Ordinal |
| EducationField | object | Academic field background. | Typical categories: `Life Sciences`, `Medical`, `Marketing`, `Technical Degree`, `Other`, `Human Resources` | Categorical |
| EmployeeCount | int | Constant helper column; no variance. | Always `1` | ID/DROP |
| EmployeeNumber | int | Unique employee identifier. | Positive integer ID (unique per employee) | ID/DROP |
| EnvironmentSatisfaction | int | Satisfaction with work environment. | 1=Low, 2=Medium, 3=High, 4=Very High | Ordinal |
| Gender | object | Employee gender. | `Male`, `Female` | Categorical |
| HourlyRate | int | Hourly compensation rate. | Positive integer (dataset commonly 30-100) | Continuous |
| JobInvolvement | int | Level of job involvement. | 1=Low, 2=Medium, 3=High, 4=Very High | Ordinal |
| JobLevel | int | Seniority / grade level of role. | 1-5 | Ordinal |
| JobRole | object | Specific job role category. | 9 categories (e.g., Sales Executive, Research Scientist, Laboratory Technician, Manufacturing Director, Healthcare Representative, Manager, Sales Representative, Research Director, Human Resources) | Categorical |
| JobSatisfaction | int | Satisfaction with job role. | 1=Low, 2=Medium, 3=High, 4=Very High | Ordinal |
| MaritalStatus | object | Marital status. | `Single`, `Married`, `Divorced` | Categorical |
| MonthlyIncome | int | Monthly income amount. | Positive integer (currency units in dataset) | Continuous |
| MonthlyRate | int | Monthly rate metric from source HR system. | Positive integer | Continuous |
| NumCompaniesWorked | int | Number of prior companies worked at. | 0-9 | Continuous (count) |
| Over18 | object | Constant legal-age indicator; no variance. | Always `Y` | ID/DROP |
| OverTime | object | Whether employee works overtime. | `Yes`, `No` | Categorical |
| PercentSalaryHike | int | Percent increase in salary from prior period. | 11-25 | Continuous |
| PerformanceRating | int | Performance review rating. | 3=Excellent, 4=Outstanding | Ordinal |
| RelationshipSatisfaction | int | Satisfaction with workplace relationships. | 1=Low, 2=Medium, 3=High, 4=Very High | Ordinal |
| StandardHours | int | Constant standard work hours; no variance. | Always `80` | ID/DROP |
| StockOptionLevel | int | Employee stock option level. | 0-3 | Ordinal |
| TotalWorkingYears | int | Total professional years of experience. | Non-negative integer | Continuous |
| TrainingTimesLastYear | int | Number of trainings attended in last year. | 0-6 | Continuous (count) |
| WorkLifeBalance | int | Perceived work-life balance level. | 1=Bad, 2=Good, 3=Better, 4=Best | Ordinal |
| YearsAtCompany | int | Tenure at current company (years). | Non-negative integer | Continuous |
| YearsInCurrentRole | int | Years in current role. | Non-negative integer | Continuous |
| YearsSinceLastPromotion | int | Years since last promotion. | Non-negative integer | Continuous |
| YearsWithCurrManager | int | Years with current manager. | Non-negative integer | Continuous |

## Modeling Notes

- Drop `EmployeeCount`, `EmployeeNumber`, `Over18`, and `StandardHours` before training.
- `Attrition` is the binary target and should be encoded (e.g., `Yes`=1, `No`=0) for many models.
- Treat ordinal columns with ordered encodings; treat nominal categorical columns with one-hot or target-safe encoding.
