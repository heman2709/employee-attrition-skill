"""Train and evaluate attrition models on engineered HR features.

This script compares multiple classifiers with SMOTE-aware cross-validation,
selects the best model by ROC-AUC, computes SHAP importances, segments risk
tiers, and exports a structured JSON summary.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import warnings
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    import shap

    SHAP_AVAILABLE = True
except Exception:  # pragma: no cover - environment dependent
    shap = None
    SHAP_AVAILABLE = False


def load_features(input_path: str) -> Tuple[pd.DataFrame, pd.Series]:
    """Load engineered feature matrix and separate X/y.

    Args:
        input_path: Path to engineered features CSV.

    Returns:
        Tuple of features DataFrame X and target Series y.
    """
    try:
        df = pd.read_csv(input_path)
    except FileNotFoundError:
        print(f"Error: Input feature file not found at '{input_path}'.", file=sys.stderr)
        sys.exit(1)
    except Exception as exc:  # pragma: no cover - defensive branch
        print(f"Error: Failed to read features file '{input_path}': {exc}", file=sys.stderr)
        sys.exit(1)

    if "Attrition_encoded" not in df.columns:
        print("Error: Required target column 'Attrition_encoded' is missing.", file=sys.stderr)
        sys.exit(1)

    X = df.drop(columns=["Attrition_encoded"])
    y = df["Attrition_encoded"].astype(int)

    rows, n_features = X.shape
    class_0 = int((y == 0).sum())
    class_1 = int((y == 1).sum())
    minority = max(min(class_0, class_1), 1)
    majority = max(class_0, class_1)
    ratio = majority / minority

    print(f"Loaded feature matrix: {rows} rows, {n_features} features")
    print(f"Class 0: {class_0}, Class 1: {class_1}, Imbalance ratio: {ratio:.2f}")
    return X, y


def define_models(random_seed: int) -> Dict[str, Pipeline]:
    """Define candidate classification pipelines.

    Args:
        random_seed: Seed for deterministic model behavior.

    Returns:
        Mapping from model name to sklearn Pipeline.
    """
    models: Dict[str, Pipeline] = {
        "Logistic Regression": Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "clf",
                    LogisticRegression(
                        max_iter=1000,
                        random_state=random_seed,
                        class_weight="balanced",
                    ),
                ),
            ]
        ),
        "Random Forest": Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "clf",
                    RandomForestClassifier(
                        n_estimators=100,
                        random_state=random_seed,
                        class_weight="balanced",
                    ),
                ),
            ]
        ),
        "Gradient Boosting": Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "clf",
                    GradientBoostingClassifier(
                        n_estimators=100,
                        random_state=random_seed,
                    ),
                ),
            ]
        ),
    }
    return models


def cv_with_smote(
    model: Pipeline,
    X: pd.DataFrame,
    y: pd.Series,
    cv: StratifiedKFold,
    random_seed: int,
) -> Dict[str, float]:
    """Run manual stratified CV with SMOTE applied in each training fold.

    Args:
        model: Candidate pipeline to evaluate.
        X: Feature matrix.
        y: Target vector.
        cv: Stratified CV splitter.
        random_seed: Seed for SMOTE.

    Returns:
        Dictionary of mean/std metrics for ROC-AUC, F1, precision, and recall.
    """
    roc_aucs: List[float] = []
    f1s: List[float] = []
    precisions: List[float] = []
    recalls: List[float] = []

    for train_idx, val_idx in cv.split(X, y):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        smote = SMOTE(random_state=random_seed)
        X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

        model.fit(X_train_res, y_train_res)

        y_pred = model.predict(X_val)
        y_proba = model.predict_proba(X_val)[:, 1]

        roc_aucs.append(float(roc_auc_score(y_val, y_proba)))
        f1s.append(float(f1_score(y_val, y_pred, average="binary", zero_division=0)))
        precisions.append(float(precision_score(y_val, y_pred, average="binary", zero_division=0)))
        recalls.append(float(recall_score(y_val, y_pred, average="binary", zero_division=0)))

    return {
        "roc_auc_mean": float(np.mean(roc_aucs)),
        "roc_auc_std": float(np.std(roc_aucs, ddof=1)) if len(roc_aucs) > 1 else 0.0,
        "f1_mean": float(np.mean(f1s)),
        "f1_std": float(np.std(f1s, ddof=1)) if len(f1s) > 1 else 0.0,
        "precision_mean": float(np.mean(precisions)),
        "precision_std": float(np.std(precisions, ddof=1)) if len(precisions) > 1 else 0.0,
        "recall_mean": float(np.mean(recalls)),
        "recall_std": float(np.std(recalls, ddof=1)) if len(recalls) > 1 else 0.0,
    }


def compare_models(
    models: Dict[str, Pipeline],
    X: pd.DataFrame,
    y: pd.Series,
    random_seed: int,
) -> Dict[str, Dict[str, float]]:
    """Compare all models using SMOTE within CV folds.

    Args:
        models: Candidate model pipelines.
        X: Feature matrix.
        y: Target vector.
        random_seed: Seed for CV and SMOTE.

    Returns:
        Model comparison dictionary keyed by model name.
    """
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_seed)
    results: Dict[str, Dict[str, float]] = {}

    print("\nModel               | ROC-AUC | F1    | Precision | Recall")
    print("-" * 60)
    for model_name, model in models.items():
        metrics = cv_with_smote(model, X, y, cv, random_seed)
        results[model_name] = metrics
        print(
            f"{model_name:<19} | "
            f"{metrics['roc_auc_mean']:.3f}   | "
            f"{metrics['f1_mean']:.3f} | "
            f"{metrics['precision_mean']:.3f}     | "
            f"{metrics['recall_mean']:.3f}"
        )
    return results


def select_best_model(
    model_comparison: Dict[str, Dict[str, float]],
) -> Tuple[str, float]:
    """Select best model by highest mean ROC-AUC.

    Args:
        model_comparison: Cross-validation metric summary by model.

    Returns:
        Tuple of best model name and its mean ROC-AUC.
    """
    best_name = max(
        model_comparison.keys(),
        key=lambda name: model_comparison[name]["roc_auc_mean"],
    )
    best_score = float(model_comparison[best_name]["roc_auc_mean"])
    print(f"\nBest model: {best_name} (ROC-AUC: {best_score:.4f})")
    return best_name, best_score


def train_final_model(
    best_model: Pipeline,
    X: pd.DataFrame,
    y: pd.Series,
    random_seed: int,
) -> Tuple[Pipeline, np.ndarray]:
    """Train final best model on full data with SMOTE.

    Args:
        best_model: Selected model pipeline.
        X: Feature matrix.
        y: Target vector.
        random_seed: Seed for SMOTE.

    Returns:
        Tuple of fitted model and predicted class-1 probabilities on original X.
    """
    smote = SMOTE(random_state=random_seed)
    X_res, y_res = smote.fit_resample(X, y)
    class_0 = int((y_res == 0).sum())
    class_1 = int((y_res == 1).sum())
    print(f"After SMOTE — Class 0: {class_0}, Class 1: {class_1}")

    best_model.fit(X_res, y_res)
    y_proba = best_model.predict_proba(X)[:, 1]
    return best_model, y_proba


def evaluate_model(
    model: Pipeline,
    X: pd.DataFrame,
    y: pd.Series,
) -> Dict[str, Any]:
    """Evaluate fitted model on original non-SMOTE data.

    Args:
        model: Fitted pipeline.
        X: Original feature matrix.
        y: Original target vector.

    Returns:
        Dictionary with confusion matrix, report, and key metrics.
    """
    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)[:, 1]

    cm = confusion_matrix(y, y_pred).tolist()
    report = classification_report(y, y_pred, output_dict=True, zero_division=0)
    roc_auc = float(roc_auc_score(y, y_proba))
    f1 = float(f1_score(y, y_pred, average="binary", zero_division=0))
    precision = float(precision_score(y, y_pred, average="binary", zero_division=0))
    recall = float(recall_score(y, y_pred, average="binary", zero_division=0))

    return {
        "confusion_matrix": cm,
        "classification_report": report,
        "final_roc_auc": roc_auc,
        "final_f1": f1,
        "final_precision": precision,
        "final_recall": recall,
    }


def compute_shap(
    model: Pipeline,
    X: pd.DataFrame,
    warnings_list: List[str],
) -> List[Dict[str, float]]:
    """Compute top SHAP feature importances for fitted model.

    Args:
        model: Fitted best model pipeline.
        X: Original feature matrix.
        warnings_list: Mutable warning list.

    Returns:
        Top 15 SHAP feature importance records.
    """
    if not SHAP_AVAILABLE:
        warnings_list.append("SHAP is not installed; shap_top_features set to [].")
        return []

    try:
        scaler = model.named_steps["scaler"]
        clf = model.named_steps["clf"]
        X_scaled = scaler.transform(X)
        feature_names = list(X.columns)

        if isinstance(clf, (RandomForestClassifier, GradientBoostingClassifier)):
            explainer = shap.TreeExplainer(clf)
            shap_values = explainer.shap_values(X_scaled)
        elif isinstance(clf, LogisticRegression):
            explainer = shap.LinearExplainer(clf, X_scaled)
            shap_values = explainer.shap_values(X_scaled)
        else:
            warnings_list.append("Unsupported model type for SHAP; shap_top_features set to [].")
            return []

        values = np.array(shap_values)
        if values.ndim == 3:
            values = values[:, :, 1]
        elif values.ndim == 1:
            values = values.reshape(-1, 1)
        elif isinstance(shap_values, list) and len(shap_values) == 2:
            values = np.array(shap_values[1])

        mean_abs = np.mean(np.abs(values), axis=0)
        shap_df = pd.DataFrame({"feature": feature_names, "shap_importance": mean_abs})
        shap_df = shap_df.sort_values("shap_importance", ascending=False).head(15)
        return [
            {
                "feature": str(row["feature"]),
                "shap_importance": float(row["shap_importance"]),
            }
            for _, row in shap_df.iterrows()
        ]
    except Exception as exc:  # pragma: no cover - library/runtime dependent
        warnings_list.append(f"SHAP computation failed: {exc}. shap_top_features set to [].")
        return []


def segment_risk(y_proba: np.ndarray) -> Dict[str, Dict[str, float]]:
    """Segment prediction probabilities into risk tiers.

    Args:
        y_proba: Predicted probabilities for class 1.

    Returns:
        Risk segment counts and percentages.
    """
    total = max(len(y_proba), 1)
    high = int(np.sum(y_proba >= 0.6))
    medium = int(np.sum((y_proba >= 0.3) & (y_proba < 0.6)))
    low = int(np.sum(y_proba < 0.3))

    return {
        "High Risk": {"count": high, "percentage": float((high / total) * 100.0)},
        "Medium Risk": {"count": medium, "percentage": float((medium / total) * 100.0)},
        "Low Risk": {"count": low, "percentage": float((low / total) * 100.0)},
    }


def assemble_results(
    model_comparison: Dict[str, Dict[str, float]],
    best_model_name: str,
    best_cv_roc_auc: float,
    evaluation: Dict[str, Any],
    shap_top_features: List[Dict[str, float]],
    risk_segments: Dict[str, Dict[str, float]],
    random_seed: int,
    n_features: int,
    n_samples: int,
) -> Dict[str, Any]:
    """Assemble final result payload for JSON serialization.

    Args:
        model_comparison: CV metrics for all candidate models.
        best_model_name: Name of selected best model.
        best_cv_roc_auc: Best model CV ROC-AUC.
        evaluation: Final evaluation metrics.
        shap_top_features: Top SHAP feature list.
        risk_segments: Predicted risk segmentation.
        random_seed: Global random seed.
        n_features: Number of model features.
        n_samples: Number of samples evaluated.

    Returns:
        Structured results dictionary.
    """
    return {
        "model_comparison": model_comparison,
        "best_model": {
            "name": best_model_name,
            "roc_auc_cv": float(best_cv_roc_auc),
            "final_roc_auc": float(evaluation["final_roc_auc"]),
            "final_f1": float(evaluation["final_f1"]),
            "final_precision": float(evaluation["final_precision"]),
            "final_recall": float(evaluation["final_recall"]),
            "confusion_matrix": evaluation["confusion_matrix"],
            "classification_report": evaluation["classification_report"],
        },
        "shap_top_features": shap_top_features,
        "risk_segments": risk_segments,
        "metadata": {
            "random_seed": int(random_seed),
            "n_features": int(n_features),
            "n_samples": int(n_samples),
            "cv_folds": 5,
            "smote_applied": True,
        },
    }


def main() -> None:
    """Run training, comparison, evaluation, SHAP analysis, and JSON export."""
    parser = argparse.ArgumentParser(
        description="Train and evaluate attrition classification models."
    )
    parser.add_argument("--input", required=True, help="Path to engineered features CSV.")
    parser.add_argument("--output", required=True, help="Path to model results JSON.")
    parser.add_argument("--random-seed", type=int, default=42, help="Random seed.")
    args = parser.parse_args()

    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    warnings.filterwarnings("ignore", category=UserWarning)

    X, y = load_features(args.input)
    models = define_models(args.random_seed)
    comparison = compare_models(models, X, y, args.random_seed)
    best_name, best_cv_score = select_best_model(comparison)

    best_pipeline = models[best_name]
    best_pipeline, y_proba = train_final_model(best_pipeline, X, y, args.random_seed)
    evaluation = evaluate_model(best_pipeline, X, y)

    runtime_warnings: List[str] = []
    shap_top_features = compute_shap(best_pipeline, X, runtime_warnings)
    risk_segments = segment_risk(y_proba)

    results = assemble_results(
        model_comparison=comparison,
        best_model_name=best_name,
        best_cv_roc_auc=best_cv_score,
        evaluation=evaluation,
        shap_top_features=shap_top_features,
        risk_segments=risk_segments,
        random_seed=args.random_seed,
        n_features=X.shape[1],
        n_samples=X.shape[0],
    )
    if runtime_warnings:
        results["warnings"] = runtime_warnings

    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    try:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
    except Exception as exc:  # pragma: no cover - defensive branch
        print(f"Error: Failed to write model results '{args.output}': {exc}", file=sys.stderr)
        sys.exit(1)

    print(f"Model results saved to {args.output}")


if __name__ == "__main__":
    main()
