"""Validate and profile IBM HR Employee Attrition data.

This script performs pre-modeling validation checks, builds column-level profiles,
and writes a structured JSON report with warnings/errors and a final verdict.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.stats import skew


REQUIRED_TARGET = ["Attrition"]
REQUIRED_CATEGORICAL = [
    "BusinessTravel",
    "Department",
    "EducationField",
    "Gender",
    "JobRole",
    "MaritalStatus",
    "OverTime",
]
REQUIRED_NUMERIC = [
    "Age",
    "DailyRate",
    "DistanceFromHome",
    "Education",
    "EnvironmentSatisfaction",
    "HourlyRate",
    "JobInvolvement",
    "JobLevel",
    "JobSatisfaction",
    "MonthlyIncome",
    "MonthlyRate",
    "NumCompaniesWorked",
    "PercentSalaryHike",
    "PerformanceRating",
    "RelationshipSatisfaction",
    "StockOptionLevel",
    "TotalWorkingYears",
    "TrainingTimesLastYear",
    "WorkLifeBalance",
    "YearsAtCompany",
    "YearsInCurrentRole",
    "YearsSinceLastPromotion",
    "YearsWithCurrManager",
]
DROP_COLUMNS = ["EmployeeCount", "EmployeeNumber", "Over18", "StandardHours"]
ORDINAL_RANGES = {
    "Education": (1, 5),
    "EnvironmentSatisfaction": (1, 4),
    "JobInvolvement": (1, 4),
    "JobLevel": (1, 5),
    "JobSatisfaction": (1, 4),
    "PerformanceRating": (3, 4),
    "RelationshipSatisfaction": (1, 4),
    "StockOptionLevel": (0, 3),
    "WorkLifeBalance": (1, 4),
}


def load_data(
    input_path: str,
    warnings: List[str],
    errors: List[str],
) -> pd.DataFrame:
    """Load CSV input and run minimum shape checks.

    Args:
        input_path: Path to CSV file.
        warnings: Mutable list used to collect warning messages.
        errors: Mutable list used to collect error messages.

    Returns:
        A pandas DataFrame. Returns an empty DataFrame if loading fails.
    """
    del warnings  # reserved for future checks

    try:
        df = pd.read_csv(input_path)
    except FileNotFoundError:
        errors.append(f"Input file not found: {input_path}")
        return pd.DataFrame()
    except Exception as exc:  # pragma: no cover - defensive branch
        errors.append(f"Failed to load CSV '{input_path}': {exc}")
        return pd.DataFrame()

    rows, cols = df.shape
    if rows < 500:
        errors.append(f"Dataset has {rows} rows; minimum required is 500.")
    if cols < 10:
        errors.append(f"Dataset has {cols} columns; minimum required is 10.")

    return df


def drop_constants(df: pd.DataFrame, drop_cols: List[str]) -> Tuple[pd.DataFrame, List[str]]:
    """Drop known ID/constant columns if present.

    Args:
        df: Input DataFrame.
        drop_cols: Candidate columns to remove.

    Returns:
        Tuple of (updated DataFrame, dropped column names).
    """
    present = [col for col in drop_cols if col in df.columns]
    return df.drop(columns=present, errors="ignore"), present


def check_required_columns(
    df: pd.DataFrame,
    warnings: List[str],
    errors: List[str],
) -> Dict[str, List[str]]:
    """Verify required target/categorical/numeric columns are available.

    Args:
        df: Working DataFrame.
        warnings: Mutable warnings list.
        errors: Mutable errors list.

    Returns:
        Dictionary containing missing columns by group.
    """
    del warnings  # not used in this check

    required_groups = {
        "target": REQUIRED_TARGET,
        "categorical": REQUIRED_CATEGORICAL,
        "numeric": REQUIRED_NUMERIC,
    }
    missing: Dict[str, List[str]] = {}
    for group_name, required_cols in required_groups.items():
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            missing[group_name] = missing_cols

    if missing:
        details = "; ".join(f"{k}: {v}" for k, v in missing.items())
        errors.append(f"Missing required columns after drop step -> {details}")
    return missing


def analyze_nulls(
    df: pd.DataFrame,
    required_columns: List[str],
    warnings: List[str],
    errors: List[str],
) -> Dict[str, Any]:
    """Compute null counts/percentages and apply null threshold rules.

    Args:
        df: Working DataFrame.
        required_columns: List of columns treated as required for hard thresholds.
        warnings: Mutable warnings list.
        errors: Mutable errors list.

    Returns:
        Null analysis dictionary for report serialization.
    """
    null_counts_series = df.isnull().sum()
    null_pct_series = (null_counts_series / len(df) * 100.0) if len(df) > 0 else null_counts_series * 0.0

    null_counts = {col: int(val) for col, val in null_counts_series.to_dict().items()}
    null_pct = {col: float(round(val, 4)) for col, val in null_pct_series.to_dict().items()}
    columns_with_nulls = [col for col, count in null_counts.items() if count > 0]

    for col, pct in null_pct.items():
        if pct > 10.0:
            warnings.append(f"Column '{col}' has high null percentage: {pct:.2f}% (>10%).")

    for col in required_columns:
        if col in null_pct and null_pct[col] > 30.0:
            errors.append(
                f"Required column '{col}' has {null_pct[col]:.2f}% nulls (>30% threshold)."
            )

    return {
        "null_counts": null_counts,
        "null_pct": null_pct,
        "columns_with_nulls": columns_with_nulls,
    }


def analyze_target(
    df: pd.DataFrame,
    warnings: List[str],
    errors: List[str],
) -> Dict[str, Any]:
    """Analyze target class counts and imbalance ratio.

    Args:
        df: Working DataFrame.
        warnings: Mutable warnings list.
        errors: Mutable errors list.

    Returns:
        Target analysis dictionary with counts and ratio.
    """
    if "Attrition" not in df.columns:
        errors.append("Cannot analyze target: 'Attrition' column missing.")
        return {"attrition_yes": 0, "attrition_no": 0, "imbalance_ratio": None}

    counts = df["Attrition"].value_counts(dropna=False)
    yes_count = int(counts.get("Yes", 0))
    no_count = int(counts.get("No", 0))

    non_zero_counts = [count for count in [yes_count, no_count] if count > 0]
    if len(non_zero_counts) < 2:
        imbalance_ratio = float("inf") if non_zero_counts else None
    else:
        imbalance_ratio = float(max(non_zero_counts) / min(non_zero_counts))

    if imbalance_ratio is not None and imbalance_ratio > 5.0:
        warnings.append(
            f"Target class imbalance ratio is high: {imbalance_ratio:.4f} (>5.0)."
        )

    return {
        "attrition_yes": yes_count,
        "attrition_no": no_count,
        "imbalance_ratio": None if imbalance_ratio is None else float(round(imbalance_ratio, 6)),
    }


def profile_numeric(
    df: pd.DataFrame,
    numeric_columns: List[str],
    warnings: List[str],
    errors: List[str],
) -> Dict[str, Dict[str, Any]]:
    """Generate descriptive statistics for numeric columns.

    Args:
        df: Working DataFrame.
        numeric_columns: Columns expected to be numeric.
        warnings: Mutable warnings list.
        errors: Mutable errors list.

    Returns:
        Numeric profile dictionary by column.
    """
    del errors  # no hard errors in this profiling step

    profiles: Dict[str, Dict[str, Any]] = {}
    for col in numeric_columns:
        if col not in df.columns:
            continue

        series = pd.to_numeric(df[col], errors="coerce").dropna()
        if series.empty:
            profiles[col] = {
                "min": None,
                "max": None,
                "mean": None,
                "median": None,
                "std": None,
                "skewness": None,
                "pct_25": None,
                "pct_75": None,
                "outlier_count": 0,
            }
            continue

        q1 = float(series.quantile(0.25))
        q3 = float(series.quantile(0.75))
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        outlier_count = int(((series < lower) | (series > upper)).sum())
        col_skew = float(skew(series.to_numpy(), bias=False, nan_policy="omit"))

        if abs(col_skew) > 2.0:
            warnings.append(
                f"Numeric column '{col}' is highly skewed (skewness={col_skew:.4f}); consider log transform."
            )

        profiles[col] = {
            "min": float(series.min()),
            "max": float(series.max()),
            "mean": float(round(series.mean(), 6)),
            "median": float(round(series.median(), 6)),
            "std": float(round(series.std(ddof=1), 6)),
            "skewness": float(round(col_skew, 6)),
            "pct_25": float(round(q1, 6)),
            "pct_75": float(round(q3, 6)),
            "outlier_count": outlier_count,
        }

    return profiles


def profile_categorical(
    df: pd.DataFrame,
    categorical_columns: List[str],
    warnings: List[str],
    errors: List[str],
) -> Dict[str, Dict[str, Any]]:
    """Generate frequency and cardinality profiles for categorical columns.

    Args:
        df: Working DataFrame.
        categorical_columns: Columns expected to be categorical.
        warnings: Mutable warnings list.
        errors: Mutable errors list.

    Returns:
        Categorical profile dictionary by column.
    """
    del errors  # no hard errors in this profiling step

    profiles: Dict[str, Dict[str, Any]] = {}
    for col in categorical_columns:
        if col not in df.columns:
            continue

        series = df[col].fillna("<<NULL>>").astype(str)
        counts = series.value_counts(dropna=False)
        unique_count = int(series.nunique(dropna=False))

        if unique_count > 50:
            warnings.append(
                f"Categorical column '{col}' has high cardinality ({unique_count} unique values)."
            )

        most_frequent = counts.index[0] if not counts.empty else None
        least_frequent = counts.index[-1] if not counts.empty else None

        profiles[col] = {
            "unique_count": unique_count,
            "value_counts": {str(k): int(v) for k, v in counts.to_dict().items()},
            "most_frequent": None if most_frequent is None else str(most_frequent),
            "least_frequent": None if least_frequent is None else str(least_frequent),
        }

    return profiles


def validate_ordinal_ranges(
    df: pd.DataFrame,
    ordinal_ranges: Dict[str, Tuple[int, int]],
    warnings: List[str],
    errors: List[str],
) -> Dict[str, Dict[str, Any]]:
    """Validate ordinal columns against expected inclusive ranges.

    Args:
        df: Working DataFrame.
        ordinal_ranges: Mapping of column name to (min, max) expected range.
        warnings: Mutable warnings list.
        errors: Mutable errors list.

    Returns:
        Ordinal validation dictionary by column.
    """
    del errors  # warnings-only policy for out-of-range values

    results: Dict[str, Dict[str, Any]] = {}
    for col, (expected_min, expected_max) in ordinal_ranges.items():
        if col not in df.columns:
            results[col] = {
                "min_found": None,
                "max_found": None,
                "valid": False,
            }
            continue

        series = pd.to_numeric(df[col], errors="coerce").dropna()
        if series.empty:
            results[col] = {
                "min_found": None,
                "max_found": None,
                "valid": False,
            }
            warnings.append(
                f"Ordinal column '{col}' has no valid numeric values for range validation."
            )
            continue

        min_found = float(series.min())
        max_found = float(series.max())
        valid = bool((min_found >= expected_min) and (max_found <= expected_max))
        if not valid:
            warnings.append(
                f"Ordinal column '{col}' out-of-range values found: "
                f"min={min_found}, max={max_found}, expected [{expected_min}, {expected_max}]."
            )

        results[col] = {
            "min_found": min_found,
            "max_found": max_found,
            "valid": valid,
        }

    return results


def check_duplicates(
    df: pd.DataFrame,
    warnings: List[str],
    errors: List[str],
) -> int:
    """Count exact duplicate rows and add warning if duplicates exist.

    Args:
        df: Working DataFrame.
        warnings: Mutable warnings list.
        errors: Mutable errors list.

    Returns:
        Number of duplicate rows.
    """
    del errors  # warnings-only check

    duplicates = int(df.duplicated().sum())
    if duplicates > 0:
        warnings.append(f"Dataset contains {duplicates} duplicate rows.")
    return duplicates


def assemble_report(
    rows: int,
    cols_after_drop: int,
    dropped_columns: List[str],
    null_analysis: Dict[str, Any],
    target_analysis: Dict[str, Any],
    numeric_profiles: Dict[str, Dict[str, Any]],
    categorical_profiles: Dict[str, Dict[str, Any]],
    ordinal_validation: Dict[str, Dict[str, Any]],
    duplicate_rows: int,
    warnings: List[str],
    errors: List[str],
) -> Dict[str, Any]:
    """Assemble final validation report payload.

    Args:
        rows: Number of rows in working data.
        cols_after_drop: Number of columns after drop step.
        dropped_columns: Columns removed before profiling.
        null_analysis: Null analysis output.
        target_analysis: Target analysis output.
        numeric_profiles: Numeric profiling output.
        categorical_profiles: Categorical profiling output.
        ordinal_validation: Ordinal range validation output.
        duplicate_rows: Number of exact duplicate rows.
        warnings: Collected warnings.
        errors: Collected errors.

    Returns:
        Report dictionary.
    """
    return {
        "dataset_summary": {
            "rows": rows,
            "cols_after_drop": cols_after_drop,
            "dropped_columns": dropped_columns,
        },
        "null_analysis": null_analysis,
        "target_analysis": target_analysis,
        "numeric_profiles": numeric_profiles,
        "categorical_profiles": categorical_profiles,
        "ordinal_validation": ordinal_validation,
        "duplicate_rows": duplicate_rows,
        "warnings": warnings,
        "errors": errors,
        "validation_passed": len(errors) == 0,
    }


def print_summary(report: Dict[str, Any]) -> None:
    """Print human-readable validation summary to stdout.

    Args:
        report: Validation report dictionary.
    """
    def mark(condition: bool) -> str:
        return "✅" if condition else "❌"

    errors = report.get("errors", [])
    warnings = report.get("warnings", [])
    dataset_summary = report.get("dataset_summary", {})
    target_analysis = report.get("target_analysis", {})

    rows = dataset_summary.get("rows", 0)
    cols_after_drop = dataset_summary.get("cols_after_drop", 0)
    dropped_columns = dataset_summary.get("dropped_columns", [])

    null_analysis = report.get("null_analysis", {})
    required_null_error_present = any(">30% threshold" in err for err in errors)
    shape_ok = rows >= 500 and cols_after_drop >= 10
    target_ok = "attrition_yes" in target_analysis and "attrition_no" in target_analysis
    duplicates = int(report.get("duplicate_rows", 0))

    print("\n=== Employee Attrition Data Validation Summary ===")
    print(
        f"{mark(shape_ok)} Basic checks: rows={rows}, cols_after_drop={cols_after_drop}, "
        f"dropped={dropped_columns}"
    )
    print(f"{mark(not required_null_error_present)} Null checks: analyzed all columns")
    print(
        f"{mark(target_ok)} Target analysis: "
        f"Yes={target_analysis.get('attrition_yes', 'N/A')}, "
        f"No={target_analysis.get('attrition_no', 'N/A')}, "
        f"Ratio={target_analysis.get('imbalance_ratio', 'N/A')}"
    )
    print(f"{mark(True)} Numeric profiling: {len(report.get('numeric_profiles', {}))} columns")
    print(
        f"{mark(True)} Categorical profiling: {len(report.get('categorical_profiles', {}))} columns"
    )
    print(
        f"{mark(duplicates == 0)} Duplicate check: {duplicates} duplicate row(s)"
    )
    print(
        f"{mark(all(v.get('valid', False) for v in report.get('ordinal_validation', {}).values()))} "
        "Ordinal range validation completed"
    )

    if warnings:
        print("\nWarnings:")
        for item in warnings:
            print(f"- {item}")
    else:
        print("\nWarnings: None")

    if errors:
        print("\nErrors:")
        for item in errors:
            print(f"- {item}")

    final_verdict = "PASSED" if report.get("validation_passed", False) else "FAILED"
    if final_verdict == "PASSED":
        print("\nFinal verdict: PASSED")
    else:
        reason = errors[0] if errors else "Validation error(s) present."
        print(f"\nFinal verdict: FAILED - {reason}")


def main() -> int:
    """Run data validation workflow and write JSON report."""
    parser = argparse.ArgumentParser(
        description="Validate and profile IBM HR Employee Attrition dataset."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to input CSV file.",
    )
    parser.add_argument(
        "--output",
        default="outputs/validation_report.json",
        help="Path to output JSON report.",
    )
    args = parser.parse_args()

    warnings: List[str] = []
    errors: List[str] = []

    df = load_data(args.input, warnings, errors)
    if not df.empty:
        df, dropped_columns = drop_constants(df, DROP_COLUMNS)
    else:
        dropped_columns = []

    rows = int(df.shape[0])
    cols_after_drop = int(df.shape[1])

    if not df.empty:
        check_required_columns(df, warnings, errors)
        required_all = REQUIRED_TARGET + REQUIRED_CATEGORICAL + REQUIRED_NUMERIC
        null_analysis = analyze_nulls(df, required_all, warnings, errors)
        target_analysis = analyze_target(df, warnings, errors)
        numeric_profiles = profile_numeric(df, REQUIRED_NUMERIC, warnings, errors)
        categorical_profiles = profile_categorical(
            df,
            REQUIRED_CATEGORICAL + REQUIRED_TARGET,
            warnings,
            errors,
        )
        ordinal_validation = validate_ordinal_ranges(df, ORDINAL_RANGES, warnings, errors)
        duplicate_rows = check_duplicates(df, warnings, errors)
    else:
        null_analysis = {"null_counts": {}, "null_pct": {}, "columns_with_nulls": []}
        target_analysis = {"attrition_yes": 0, "attrition_no": 0, "imbalance_ratio": None}
        numeric_profiles = {}
        categorical_profiles = {}
        ordinal_validation = {}
        duplicate_rows = 0

    report = assemble_report(
        rows=rows,
        cols_after_drop=cols_after_drop,
        dropped_columns=dropped_columns,
        null_analysis=null_analysis,
        target_analysis=target_analysis,
        numeric_profiles=numeric_profiles,
        categorical_profiles=categorical_profiles,
        ordinal_validation=ordinal_validation,
        duplicate_rows=duplicate_rows,
        warnings=warnings,
        errors=errors,
    )

    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    try:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
    except Exception as exc:  # pragma: no cover - defensive I/O branch
        print(f"Failed to write report to '{args.output}': {exc}", file=sys.stderr)
        return 1

    print_summary(report)
    return 0 if report["validation_passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
