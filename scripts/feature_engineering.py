"""Engineer model-ready features for IBM HR Employee Attrition data.

This script transforms raw HR attrition data into a clean feature matrix and
produces a JSON manifest documenting all encoding and engineered features.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd


DROP_CONSTANT_COLUMNS = ["EmployeeCount", "EmployeeNumber", "Over18", "StandardHours"]
ORDINAL_COLUMNS = [
    "Education",
    "EnvironmentSatisfaction",
    "JobInvolvement",
    "JobLevel",
    "JobSatisfaction",
    "PerformanceRating",
    "RelationshipSatisfaction",
    "StockOptionLevel",
    "WorkLifeBalance",
]
ONE_HOT_COLUMNS = [
    "BusinessTravel",
    "Department",
    "EducationField",
    "JobRole",
    "MaritalStatus",
]


def load_and_drop(
    input_path: str,
    warnings: List[str],
) -> pd.DataFrame:
    """Load input CSV and drop known constant/ID columns.

    Args:
        input_path: Path to source CSV file.
        warnings: Mutable warnings list.

    Returns:
        DataFrame with constant/ID columns removed when present.
    """
    try:
        df = pd.read_csv(input_path)
    except FileNotFoundError:
        print(f"Error: Input file not found at '{input_path}'.", file=sys.stderr)
        sys.exit(1)
    except Exception as exc:  # pragma: no cover - defensive branch
        print(f"Error: Failed to load CSV '{input_path}': {exc}", file=sys.stderr)
        sys.exit(1)

    dropped = [col for col in DROP_CONSTANT_COLUMNS if col in df.columns]
    if dropped:
        df = df.drop(columns=dropped, errors="ignore")
    else:
        warnings.append("No constant/ID columns were present to drop.")

    print(f"Loaded data: {df.shape[0]} rows, {df.shape[1]} columns after dropping constants.")
    return df


def encode_target(
    df: pd.DataFrame,
    warnings: List[str],
) -> pd.DataFrame:
    """Encode target column Attrition into Attrition_encoded.

    Args:
        df: Input DataFrame.
        warnings: Mutable warnings list.

    Returns:
        DataFrame with encoded target and original Attrition removed.
    """
    if "Attrition" not in df.columns:
        print("Error: Required target column 'Attrition' is missing.", file=sys.stderr)
        sys.exit(1)

    mapping = {"Yes": 1, "No": 0}
    encoded = df["Attrition"].map(mapping)

    unknown_values = sorted(set(df["Attrition"].dropna().astype(str)) - set(mapping.keys()))
    if unknown_values:
        warnings.append(
            f"Unexpected Attrition values encountered and encoded to 0: {unknown_values}"
        )

    df["Attrition_encoded"] = encoded.fillna(0).astype(int)
    df = df.drop(columns=["Attrition"])

    attrition_one = int((df["Attrition_encoded"] == 1).sum())
    attrition_zero = int((df["Attrition_encoded"] == 0).sum())
    print(f"Attrition=1: {attrition_one}, Attrition=0: {attrition_zero}")
    return df


def encode_binary(
    df: pd.DataFrame,
    manifest: List[Dict[str, Any]],
    warnings: List[str],
) -> pd.DataFrame:
    """Apply binary encoding to OverTime and Gender columns.

    Args:
        df: Input DataFrame.
        manifest: Mutable manifest feature list.
        warnings: Mutable warnings list.

    Returns:
        DataFrame with binary encoded columns.
    """
    binary_maps: Dict[str, Dict[str, int]] = {
        "OverTime": {"Yes": 1, "No": 0},
        "Gender": {"Male": 1, "Female": 0},
    }

    for col, mapping in binary_maps.items():
        if col not in df.columns:
            print(f"Error: Required binary column '{col}' is missing.", file=sys.stderr)
            sys.exit(1)

        encoded = df[col].map(mapping)
        unknown_values = sorted(set(df[col].dropna().astype(str)) - set(mapping.keys()))
        if unknown_values:
            warnings.append(
                f"Unexpected values in '{col}' encoded to 0: {unknown_values}"
            )
        df[col] = encoded.fillna(0).astype(int)
        manifest.append(
            {
                "column": col,
                "type": "binary_encode",
                "mapping": mapping,
            }
        )

    return df


def encode_ordinal(
    df: pd.DataFrame,
    manifest: List[Dict[str, Any]],
    warnings: List[str],
) -> pd.DataFrame:
    """Cast ordinal columns to integer type and record manifest entries.

    Args:
        df: Input DataFrame.
        manifest: Mutable manifest feature list.
        warnings: Mutable warnings list.

    Returns:
        DataFrame with ordinal columns cast to int.
    """
    for col in ORDINAL_COLUMNS:
        if col not in df.columns:
            print(f"Error: Required ordinal column '{col}' is missing.", file=sys.stderr)
            sys.exit(1)

        if not pd.api.types.is_numeric_dtype(df[col]):
            warnings.append(
                f"Ordinal column '{col}' is non-numeric and was coerced to integer."
            )
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)
        manifest.append({"column": col, "type": "ordinal_keep"})

    return df


def encode_onehot(
    df: pd.DataFrame,
    manifest: List[Dict[str, Any]],
    warnings: List[str],
) -> pd.DataFrame:
    """One-hot encode nominal categorical columns.

    Args:
        df: Input DataFrame.
        manifest: Mutable manifest feature list.
        warnings: Mutable warnings list.

    Returns:
        DataFrame with one-hot encoded categorical columns.
    """
    del warnings  # reserved for future one-hot warnings

    for col in ONE_HOT_COLUMNS:
        if col not in df.columns:
            print(f"Error: Required one-hot column '{col}' is missing.", file=sys.stderr)
            sys.exit(1)

        dummies = pd.get_dummies(df[col], prefix=col, drop_first=False).astype(int)
        resulting_columns = list(dummies.columns)
        df = pd.concat([df.drop(columns=[col]), dummies], axis=1)

        manifest.append(
            {
                "column": col,
                "type": "one_hot",
                "resulting_columns": resulting_columns,
            }
        )

    return df


def engineer_features(
    df: pd.DataFrame,
    manifest: List[Dict[str, Any]],
    warnings: List[str],
) -> pd.DataFrame:
    """Create engineered interaction and ratio features.

    Args:
        df: Working DataFrame after encoding.
        manifest: Mutable manifest feature list.
        warnings: Mutable warnings list.

    Returns:
        DataFrame with new engineered features.
    """
    required = [
        "YearsInCurrentRole",
        "YearsAtCompany",
        "MonthlyIncome",
        "JobLevel",
        "YearsSinceLastPromotion",
        "YearsWithCurrManager",
        "TotalWorkingYears",
        "OverTime",
        "JobSatisfaction",
        "WorkLifeBalance",
        "DistanceFromHome",
    ]
    missing = [col for col in required if col not in df.columns]
    if missing:
        print(
            f"Error: Missing required columns for feature engineering: {missing}",
            file=sys.stderr,
        )
        sys.exit(1)

    df["tenure_role_ratio"] = df["YearsInCurrentRole"] / (df["YearsAtCompany"] + 1)
    manifest.append(
        {
            "column": "tenure_role_ratio",
            "type": "engineered",
            "formula": "YearsInCurrentRole / (YearsAtCompany + 1)",
            "business_meaning": (
                "Career stagnation signal - stuck in same role relative to tenure. "
                "High value = attrition risk."
            ),
        }
    )

    df["income_per_job_level"] = df["MonthlyIncome"] / (df["JobLevel"] + 1)
    manifest.append(
        {
            "column": "income_per_job_level",
            "type": "engineered",
            "formula": "MonthlyIncome / (JobLevel + 1)",
            "business_meaning": "Underpaid for seniority level. Low value = attrition risk.",
        }
    )

    df["promotion_lag"] = df["YearsSinceLastPromotion"] / (df["YearsAtCompany"] + 1)
    manifest.append(
        {
            "column": "promotion_lag",
            "type": "engineered",
            "formula": "YearsSinceLastPromotion / (YearsAtCompany + 1)",
            "business_meaning": "Career progression slowdown signal.",
        }
    )

    df["manager_stability"] = df["YearsWithCurrManager"] / (df["YearsAtCompany"] + 1)
    manifest.append(
        {
            "column": "manager_stability",
            "type": "engineered",
            "formula": "YearsWithCurrManager / (YearsAtCompany + 1)",
            "business_meaning": "Low value = frequent manager changes = instability.",
        }
    )

    df["experience_company_ratio"] = df["TotalWorkingYears"] / (df["YearsAtCompany"] + 1)
    manifest.append(
        {
            "column": "experience_company_ratio",
            "type": "engineered",
            "formula": "TotalWorkingYears / (YearsAtCompany + 1)",
            "business_meaning": "High ratio = broad market experience = more likely to leave.",
        }
    )

    df["overtime_satisfaction_stress"] = df["OverTime"] * (5 - df["JobSatisfaction"])
    manifest.append(
        {
            "column": "overtime_satisfaction_stress",
            "type": "engineered",
            "formula": "OverTime * (5 - JobSatisfaction)",
            "business_meaning": "Overtime AND low satisfaction = burnout signal.",
        }
    )

    df["loyalty_score"] = (
        (df["YearsAtCompany"] * df["JobSatisfaction"] * df["WorkLifeBalance"])
        / (df["TotalWorkingYears"] + 1)
    )
    manifest.append(
        {
            "column": "loyalty_score",
            "type": "engineered",
            "formula": (
                "(YearsAtCompany * JobSatisfaction * WorkLifeBalance) / "
                "(TotalWorkingYears + 1)"
            ),
            "business_meaning": "Composite loyalty proxy.",
        }
    )

    df["distance_overtime_interaction"] = df["DistanceFromHome"] * df["OverTime"]
    manifest.append(
        {
            "column": "distance_overtime_interaction",
            "type": "engineered",
            "formula": "DistanceFromHome * OverTime",
            "business_meaning": "Long commute + overtime = burnout.",
        }
    )

    return df


def drop_remaining_objects(
    df: pd.DataFrame,
    warnings: List[str],
) -> Tuple[pd.DataFrame, List[str]]:
    """Drop any remaining object-dtype columns after encoding.

    Args:
        df: Working DataFrame.
        warnings: Mutable warnings list.

    Returns:
        Tuple of (updated DataFrame, dropped object columns).
    """
    object_cols = df.select_dtypes(include=["object"]).columns.tolist()
    if object_cols:
        df = df.drop(columns=object_cols, errors="ignore")
        warnings.append(f"Dropped remaining object columns: {object_cols}")
        print(f"Dropped remaining object columns: {object_cols}")
    else:
        print("Dropped remaining object columns: []")
    return df, object_cols


def final_cleanup(df: pd.DataFrame) -> pd.DataFrame:
    """Apply final cleanup and schema ordering rules.

    Args:
        df: Working DataFrame.

    Returns:
        Cleaned DataFrame with target as last column.
    """
    df = df.fillna(0)

    float_cols = df.select_dtypes(include=["float", "float64", "float32"]).columns.tolist()
    if float_cols:
        df[float_cols] = df[float_cols].round(6)

    if "Attrition_encoded" not in df.columns:
        print(
            "Error: 'Attrition_encoded' missing during final cleanup.",
            file=sys.stderr,
        )
        sys.exit(1)

    ordered_cols = [col for col in df.columns if col != "Attrition_encoded"] + ["Attrition_encoded"]
    df = df[ordered_cols]
    return df


def save_outputs(
    df: pd.DataFrame,
    manifest_features: List[Dict[str, Any]],
    dropped_object_columns: List[str],
    warnings: List[str],
    output_path: str,
    manifest_path: str,
) -> None:
    """Save engineered feature matrix and feature manifest JSON.

    Args:
        df: Final feature matrix.
        manifest_features: Collected manifest entries.
        dropped_object_columns: Columns removed for object dtype.
        warnings: Collected warning messages.
        output_path: Destination CSV path.
        manifest_path: Destination manifest JSON path.
    """
    output_dir = os.path.dirname(output_path)
    manifest_dir = os.path.dirname(manifest_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    if manifest_dir:
        os.makedirs(manifest_dir, exist_ok=True)

    try:
        df.to_csv(output_path, index=False)
    except Exception as exc:  # pragma: no cover - defensive branch
        print(f"Error: Failed to save features CSV '{output_path}': {exc}", file=sys.stderr)
        sys.exit(1)

    manifest_payload = {
        "total_features": int(max(df.shape[1] - 1, 0)),
        "target_column": "Attrition_encoded",
        "features": manifest_features,
        "dropped_object_columns": dropped_object_columns,
        "warnings": warnings,
        "output_shape": {"rows": int(df.shape[0]), "cols": int(df.shape[1])},
    }

    try:
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest_payload, f, indent=2)
    except Exception as exc:  # pragma: no cover - defensive branch
        print(f"Error: Failed to save manifest JSON '{manifest_path}': {exc}", file=sys.stderr)
        sys.exit(1)


def main() -> None:
    """Run complete feature engineering pipeline from CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Engineer model-ready features for IBM HR Attrition dataset."
    )
    parser.add_argument("--input", required=True, help="Input CSV path.")
    parser.add_argument("--output", required=True, help="Output features CSV path.")
    parser.add_argument("--manifest", required=True, help="Output manifest JSON path.")
    args = parser.parse_args()

    warnings: List[str] = []
    manifest_features: List[Dict[str, Any]] = []

    df = load_and_drop(args.input, warnings)
    df = encode_target(df, warnings)
    df = encode_binary(df, manifest_features, warnings)
    df = encode_ordinal(df, manifest_features, warnings)
    df = encode_onehot(df, manifest_features, warnings)
    df = engineer_features(df, manifest_features, warnings)
    df, dropped_object_columns = drop_remaining_objects(df, warnings)
    df = final_cleanup(df)

    print(f"Feature matrix shape: {df.shape[0]} x {df.shape[1]}")
    print(f"Columns: {df.columns.tolist()}")

    save_outputs(
        df=df,
        manifest_features=manifest_features,
        dropped_object_columns=dropped_object_columns,
        warnings=warnings,
        output_path=args.output,
        manifest_path=args.manifest,
    )


if __name__ == "__main__":
    main()
