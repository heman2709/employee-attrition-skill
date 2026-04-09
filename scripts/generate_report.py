"""Generate a self-contained HTML attrition analytics report.

This script reads model outputs and engineered features, generates charts using
matplotlib/seaborn, embeds them as base64 PNG images, and writes a polished
single-file HTML report with no external dependencies.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import os
import sys
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
import seaborn as sns
from jinja2 import Environment, FileSystemLoader, select_autoescape


def fig_to_base64(fig: plt.Figure) -> str:
    """Convert a matplotlib figure to an embedded base64 PNG HTML image tag.

    Args:
        fig: Matplotlib figure object.

    Returns:
        HTML img tag with data URI source.
    """
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=150)
    buf.seek(0)
    encoded = base64.b64encode(buf.read()).decode("utf-8")
    plt.close(fig)
    return (
        f'<img src="data:image/png;base64,{encoded}" '
        'style="max-width:100%;height:auto;" alt="Chart">'
    )


def load_inputs(
    features_path: str,
    model_results_path: str,
) -> Dict[str, Any]:
    """Load required and optional input files for the report.

    Args:
        features_path: Path to features CSV.
        model_results_path: Path to model results JSON.

    Returns:
        Dictionary containing loaded objects, presence flags, and warnings.
    """
    data: Dict[str, Any] = {
        "features_df": None,
        "model_results": None,
        "validation_report": None,
        "feature_manifest": None,
        "warnings": [],
    }

    if os.path.exists(features_path):
        try:
            data["features_df"] = pd.read_csv(features_path)
        except Exception as exc:
            data["warnings"].append(f"Could not read features file: {exc}")
    else:
        data["warnings"].append(f"Features file missing: {features_path}")

    if os.path.exists(model_results_path):
        try:
            with open(model_results_path, "r", encoding="utf-8") as f:
                data["model_results"] = json.load(f)
        except Exception as exc:
            data["warnings"].append(f"Could not read model results file: {exc}")
    else:
        data["warnings"].append(f"Model results file missing: {model_results_path}")

    base_dir = os.path.dirname(model_results_path) or "."
    validation_path = os.path.join(base_dir, "validation_report.json")
    manifest_path = os.path.join(base_dir, "feature_manifest.json")

    if os.path.exists(validation_path):
        try:
            with open(validation_path, "r", encoding="utf-8") as f:
                data["validation_report"] = json.load(f)
        except Exception as exc:
            data["warnings"].append(f"Could not read validation report: {exc}")

    if os.path.exists(manifest_path):
        try:
            with open(manifest_path, "r", encoding="utf-8") as f:
                data["feature_manifest"] = json.load(f)
        except Exception as exc:
            data["warnings"].append(f"Could not read feature manifest: {exc}")

    return data


def generate_executive_summary(
    company_name: str,
    model_results: Optional[Dict[str, Any]],
    features_df: Optional[pd.DataFrame],
    runtime_warnings: List[str],
) -> str:
    """Build the Executive Summary section HTML.

    Args:
        company_name: Company label used in report.
        model_results: Parsed model results payload.
        features_df: Features DataFrame if available.
        runtime_warnings: Runtime warning messages.

    Returns:
        HTML content for section 1.
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    n_rows = int(features_df.shape[0]) if features_df is not None else "N/A"
    n_features = (
        model_results.get("metadata", {}).get("n_features", "N/A")
        if model_results
        else "N/A"
    )
    best_model = model_results.get("best_model", {}) if model_results else {}
    best_name = best_model.get("name", "N/A")
    best_auc = best_model.get("roc_auc_cv", "N/A")

    risk_segments = model_results.get("risk_segments", {}) if model_results else {}
    risk_rows = ""
    for tier in ["High Risk", "Medium Risk", "Low Risk"]:
        item = risk_segments.get(tier, {"count": 0, "percentage": 0.0})
        risk_rows += (
            f"<tr><td>{tier}</td><td>{item.get('count', 0)}</td>"
            f"<td>{float(item.get('percentage', 0.0)):.2f}%</td></tr>"
        )

    shap_items = model_results.get("shap_top_features", []) if model_results else []
    top3 = shap_items[:3]
    top3_html = "".join(
        [f"<li><strong>{x['feature']}</strong> (SHAP: {x['shap_importance']:.6f})</li>" for x in top3]
    ) or "<li>No SHAP features available.</li>"

    class_note = ""
    if features_df is not None and "Attrition_encoded" in features_df.columns:
        y = features_df["Attrition_encoded"]
        class1 = int((y == 1).sum())
        class0 = int((y == 0).sum())
        rate = (class1 / max(len(y), 1)) * 100.0
        class_note = (
            f"<p><strong>Class imbalance note:</strong> Attrition positive rate is "
            f"{rate:.2f}% ({class1}/{len(y)}), handled in modeling via SMOTE and balanced settings.</p>"
        )

    warnings_html = ""
    if runtime_warnings:
        warnings_html = (
            "<div class='warning-box'><strong>Runtime warnings:</strong><ul>"
            + "".join([f"<li>{w}</li>" for w in runtime_warnings])
            + "</ul></div>"
        )

    return f"""
    <section id="executive-summary">
      <h2>1. Executive Summary</h2>
      <p><strong>Company:</strong> {company_name}<br>
         <strong>Generated:</strong> {timestamp}</p>
      <p><strong>Dataset Size:</strong> {n_rows} rows, {n_features} features</p>
      <p><strong>Best Model:</strong> {best_name} (CV ROC-AUC: {best_auc})</p>
      <h3>Risk Segment Summary</h3>
      <table>
        <thead><tr><th>Risk Tier</th><th>Count</th><th>Percentage</th></tr></thead>
        <tbody>{risk_rows}</tbody>
      </table>
      <h3>Key Findings (Top SHAP Drivers)</h3>
      <ul>{top3_html}</ul>
      {class_note}
      {warnings_html}
    </section>
    """


def generate_data_quality(validation_report: Optional[Dict[str, Any]]) -> str:
    """Build Data Quality Summary section, or placeholder when missing.

    Args:
        validation_report: Validation JSON payload if available.

    Returns:
        HTML content for section 2.
    """
    if not validation_report:
        return """
        <section id="data-quality">
          <h2>2. Data Quality Summary</h2>
          <p class="placeholder">Validation report not found in outputs/. Section skipped.</p>
        </section>
        """

    ds = validation_report.get("dataset_summary", {})
    null_analysis = validation_report.get("null_analysis", {})
    duplicate_rows = validation_report.get("duplicate_rows", "N/A")
    warnings_list = validation_report.get("warnings", [])
    ordinal_validation = validation_report.get("ordinal_validation", {})

    null_count_total = int(sum(null_analysis.get("null_counts", {}).values()))
    dropped_cols = ds.get("dropped_columns", [])
    ord_rows = ""
    for col, info in ordinal_validation.items():
        ord_rows += (
            "<tr>"
            f"<td>{col}</td><td>{info.get('min_found')}</td><td>{info.get('max_found')}</td>"
            f"<td>{'Yes' if info.get('valid') else 'No'}</td>"
            "</tr>"
        )

    warning_box = ""
    if warnings_list:
        warning_box = (
            "<div class='warning-box'><strong>Validation Warnings</strong><ul>"
            + "".join([f"<li>{w}</li>" for w in warnings_list])
            + "</ul></div>"
        )

    return f"""
    <section id="data-quality">
      <h2>2. Data Quality Summary</h2>
      <p><strong>Rows:</strong> {ds.get("rows", "N/A")} &nbsp;|&nbsp;
         <strong>Columns:</strong> {ds.get("cols_after_drop", "N/A")} &nbsp;|&nbsp;
         <strong>Dropped columns:</strong> {dropped_cols}</p>
      <p><strong>Total null values:</strong> {null_count_total} &nbsp;|&nbsp;
         <strong>Duplicate rows:</strong> {duplicate_rows}</p>
      {warning_box}
      <h3>Ordinal Validation</h3>
      <table>
        <thead><tr><th>Column</th><th>Min Found</th><th>Max Found</th><th>Valid</th></tr></thead>
        <tbody>{ord_rows}</tbody>
      </table>
    </section>
    """


def generate_methodology() -> str:
    """Build methodology section HTML.

    Returns:
        HTML content for section 3.
    """
    return """
    <section id="methodology">
      <h2>3. Methodology</h2>
      <p>Pipeline stages:
      <strong>Data Validation → Feature Engineering → SMOTE Balancing → Model Training (3 models)
      → Cross Validation (5-fold) → SHAP Analysis → Risk Segmentation</strong></p>
      <h3>Model Hyperparameters</h3>
      <table>
        <thead><tr><th>Model</th><th>Key Parameters</th></tr></thead>
        <tbody>
          <tr><td>Logistic Regression</td><td>max_iter=1000, class_weight=balanced</td></tr>
          <tr><td>Random Forest</td><td>n_estimators=100, class_weight=balanced</td></tr>
          <tr><td>Gradient Boosting</td><td>n_estimators=100</td></tr>
        </tbody>
      </table>
      <p><strong>SMOTE:</strong> Applied within each CV training fold and on full training data for final fit.</p>
    </section>
    """


def generate_model_comparison(model_results: Optional[Dict[str, Any]]) -> str:
    """Build model comparison section with table and ROC-AUC chart.

    Args:
        model_results: Parsed model results payload.

    Returns:
        HTML content for section 4.
    """
    if not model_results:
        return """
        <section id="model-comparison">
          <h2>4. Model Comparison</h2>
          <p class="placeholder">Model results JSON unavailable. Section skipped.</p>
        </section>
        """

    comparison = model_results.get("model_comparison", {})
    best_name = model_results.get("best_model", {}).get("name", "")

    table_rows = ""
    names: List[str] = []
    means: List[float] = []
    stds: List[float] = []

    for model_name, m in comparison.items():
        row_class = "best-row" if model_name == best_name else ""
        table_rows += (
            f"<tr class='{row_class}'>"
            f"<td>{model_name}</td>"
            f"<td>{m.get('roc_auc_mean', 0.0):.4f} ± {m.get('roc_auc_std', 0.0):.4f}</td>"
            f"<td>{m.get('f1_mean', 0.0):.4f} ± {m.get('f1_std', 0.0):.4f}</td>"
            f"<td>{m.get('precision_mean', 0.0):.4f}</td>"
            f"<td>{m.get('recall_mean', 0.0):.4f}</td>"
            "</tr>"
        )
        names.append(model_name)
        means.append(float(m.get("roc_auc_mean", 0.0)))
        stds.append(float(m.get("roc_auc_std", 0.0)))

    sns.set_style("whitegrid")
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = sns.color_palette("Set2", n_colors=max(len(names), 3))
    ax.bar(names, means, yerr=stds, capsize=6, color=colors[: len(names)])
    ax.set_title("Model Comparison — Cross-Validation ROC-AUC")
    ax.set_ylim(0, 1)
    ax.set_ylabel("ROC-AUC")
    ax.yaxis.set_major_locator(ticker.MultipleLocator(0.1))
    ax.tick_params(axis="x", rotation=15)
    roc_chart = fig_to_base64(fig)

    return f"""
    <section id="model-comparison">
      <h2>4. Model Comparison</h2>
      <table>
        <thead>
          <tr><th>Model</th><th>ROC-AUC (mean±std)</th><th>F1 (mean±std)</th><th>Precision</th><th>Recall</th></tr>
        </thead>
        <tbody>{table_rows}</tbody>
      </table>
      <div class="chart-wrap">{roc_chart}</div>
    </section>
    """


def generate_model_performance(model_results: Optional[Dict[str, Any]]) -> str:
    """Build best-model performance section with confusion matrix and metrics.

    Args:
        model_results: Parsed model results payload.

    Returns:
        HTML content for section 5.
    """
    if not model_results:
        return """
        <section id="model-performance">
          <h2>5. Model Performance</h2>
          <p class="placeholder">Model results JSON unavailable. Section skipped.</p>
        </section>
        """

    best = model_results.get("best_model", {})
    best_name = best.get("name", "Best Model")
    cm = np.array(best.get("confusion_matrix", [[0, 0], [0, 0]]))
    report = best.get("classification_report", {})

    sns.set_style("whitegrid")
    fig, ax = plt.subplots(figsize=(8, 6))
    cm_total = max(int(cm.sum()), 1)
    annot_labels = np.array(
        [
            [f"{int(v)}\n({(int(v) / cm_total) * 100:.1f}%)" for v in row]
            for row in cm
        ]
    )
    sns.heatmap(
        cm,
        annot=annot_labels,
        fmt="",
        cmap="YlGnBu",
        cbar=True,
        linewidths=1.0,
        linecolor="#ffffff",
        annot_kws={"fontsize": 12, "fontweight": "bold", "color": "#0f172a"},
        cbar_kws={"shrink": 0.9, "label": "Count"},
        xticklabels=["Predicted No Attrition", "Predicted Attrition"],
        yticklabels=["Actual No Attrition", "Actual Attrition"],
        ax=ax,
    )
    ax.set_title(f"Confusion Matrix — {best_name}")
    conf_img = fig_to_base64(fig)

    report_rows = ""
    for row_name in ["0", "1", "macro avg", "weighted avg"]:
        row = report.get(row_name, {})
        report_rows += (
            "<tr>"
            f"<td>{row_name}</td>"
            f"<td>{float(row.get('precision', 0.0)):.4f}</td>"
            f"<td>{float(row.get('recall', 0.0)):.4f}</td>"
            f"<td>{float(row.get('f1-score', 0.0)):.4f}</td>"
            f"<td>{int(row.get('support', 0))}</td>"
            "</tr>"
        )

    cards = f"""
    <div class="metric-grid">
      <div class="metric-card"><div class="metric-value">{best.get("final_roc_auc", 0.0):.4f}</div><div class="metric-label">ROC-AUC</div></div>
      <div class="metric-card"><div class="metric-value">{best.get("final_f1", 0.0):.4f}</div><div class="metric-label">F1 Score</div></div>
      <div class="metric-card"><div class="metric-value">{best.get("final_precision", 0.0):.4f}</div><div class="metric-label">Precision</div></div>
      <div class="metric-card"><div class="metric-value">{best.get("final_recall", 0.0):.4f}</div><div class="metric-label">Recall</div></div>
    </div>
    """

    return f"""
    <section id="model-performance">
      <h2>5. Model Performance</h2>
      {cards}
      <div class="chart-wrap">{conf_img}</div>
      <h3>Classification Report</h3>
      <table>
        <thead><tr><th>Class</th><th>Precision</th><th>Recall</th><th>F1-Score</th><th>Support</th></tr></thead>
        <tbody>{report_rows}</tbody>
      </table>
    </section>
    """


def generate_shap_section(
    model_results: Optional[Dict[str, Any]],
    feature_manifest: Optional[Dict[str, Any]],
) -> str:
    """Build SHAP feature-importance section with chart and interpretation.

    Args:
        model_results: Parsed model results payload.
        feature_manifest: Feature manifest JSON if available.

    Returns:
        HTML content for section 6.
    """
    if not model_results:
        return """
        <section id="shap-importance">
          <h2>6. Feature Importance (SHAP)</h2>
          <p class="placeholder">Model results JSON unavailable. Section skipped.</p>
        </section>
        """

    shap_features = model_results.get("shap_top_features", [])
    if not shap_features:
        return """
        <section id="shap-importance">
          <h2>6. Feature Importance (SHAP)</h2>
          <p class="placeholder">SHAP values unavailable in model results.</p>
        </section>
        """

    shap_df = pd.DataFrame(shap_features).head(15)
    shap_df = shap_df.sort_values("shap_importance", ascending=True)

    sns.set_style("whitegrid")
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = sns.color_palette("Blues", n_colors=len(shap_df))
    ax.barh(shap_df["feature"], shap_df["shap_importance"], color=colors)
    ax.set_title("Top 15 Features by SHAP Importance")
    ax.set_xlabel("mean |SHAP value|")
    shap_img = fig_to_base64(fig)

    meaning_map: Dict[str, str] = {}
    if feature_manifest and "features" in feature_manifest:
        for item in feature_manifest["features"]:
            col = item.get("column")
            meaning = item.get("business_meaning") or item.get("formula") or item.get("type", "")
            if col:
                meaning_map[str(col)] = str(meaning)

    interp_rows = ""
    for _, row in shap_df.sort_values("shap_importance", ascending=False).head(5).iterrows():
        f_name = str(row["feature"])
        meaning = meaning_map.get(f_name, "No feature-manifest business meaning available.")
        interp_rows += (
            "<tr>"
            f"<td>{f_name}</td>"
            f"<td>{float(row['shap_importance']):.6f}</td>"
            f"<td>{meaning}</td>"
            "</tr>"
        )

    return f"""
    <section id="shap-importance">
      <h2>6. Feature Importance (SHAP)</h2>
      <div class="chart-wrap">{shap_img}</div>
      <h3>Top 5 Feature Interpretation</h3>
      <table>
        <thead><tr><th>Feature</th><th>SHAP Importance</th><th>Business Meaning</th></tr></thead>
        <tbody>{interp_rows}</tbody>
      </table>
    </section>
    """


def generate_risk_segmentation(
    model_results: Optional[Dict[str, Any]],
    features_df: Optional[pd.DataFrame],
) -> str:
    """Build risk segmentation section with donut and attrition-rate charts.

    Args:
        model_results: Parsed model results payload.
        features_df: Features DataFrame if available.

    Returns:
        HTML content for section 7.
    """
    if not model_results:
        return """
        <section id="risk-segmentation">
          <h2>7. Risk Segmentation</h2>
          <p class="placeholder">Model results JSON unavailable. Section skipped.</p>
        </section>
        """

    segments = model_results.get("risk_segments", {})
    labels = ["High Risk", "Medium Risk", "Low Risk"]
    counts = [int(segments.get(k, {}).get("count", 0)) for k in labels]
    perc = [float(segments.get(k, {}).get("percentage", 0.0)) for k in labels]

    sns.set_style("whitegrid")
    fig, ax = plt.subplots(figsize=(6, 6))
    colors = ["#d9534f", "#f0ad4e", "#5cb85c"]
    wedges, texts, autotexts = ax.pie(
        counts,
        labels=labels,
        autopct="%1.1f%%",
        startangle=90,
        colors=colors,
        wedgeprops={"width": 0.45, "edgecolor": "white"},
    )
    del wedges, texts, autotexts
    ax.set_title("Employee Risk Tier Distribution")
    donut_img = fig_to_base64(fig)

    action_map = {
        "High Risk": "Immediate retention interview",
        "Medium Risk": "Engagement check-in within 30 days",
        "Low Risk": "Standard engagement monitoring",
    }
    rows = ""
    for i, tier in enumerate(labels):
        rows += (
            "<tr>"
            f"<td>{tier}</td><td>{counts[i]}</td><td>{perc[i]:.2f}%</td><td>{action_map[tier]}</td>"
            "</tr>"
        )

    extra_charts = ""
    if features_df is not None and "Attrition_encoded" in features_df.columns:
        chart_html_list: List[str] = []
        for col in ["OverTime", "JobLevel", "Department"]:
            if col not in features_df.columns:
                continue
            grp = (
                features_df.groupby(col)["Attrition_encoded"]
                .mean()
                .sort_values(ascending=False)
                .reset_index()
            )
            fig2, ax2 = plt.subplots(figsize=(10, 6))
            sns.barplot(data=grp, x=col, y="Attrition_encoded", ax=ax2, color="#4C72B0")
            ax2.set_title(f"Attrition Rate by {col}")
            ax2.set_ylabel("Attrition Rate")
            ax2.yaxis.set_major_formatter(ticker.PercentFormatter(1.0))
            chart_html_list.append(fig_to_base64(fig2))
        if chart_html_list:
            extra_charts = "<div class='chart-row'>" + "".join(
                [f"<div class='chart-cell'>{c}</div>" for c in chart_html_list]
            ) + "</div>"
        else:
            extra_charts = "<p class='placeholder'>No compatible columns found for attrition-rate breakdown charts.</p>"

    return f"""
    <section id="risk-segmentation">
      <h2>7. Risk Segmentation</h2>
      <div class="chart-wrap">{donut_img}</div>
      <table>
        <thead><tr><th>Risk Tier</th><th>Count</th><th>Percentage</th><th>Recommended Action</th></tr></thead>
        <tbody>{rows}</tbody>
      </table>
      <h3>Attrition Rate by Key Features</h3>
      {extra_charts}
    </section>
    """


def generate_recommendations(model_results: Optional[Dict[str, Any]]) -> str:
    """Build business recommendations based on top SHAP features and risk tiers.

    Args:
        model_results: Parsed model results payload.

    Returns:
        HTML content for section 8.
    """
    if not model_results:
        return """
        <section id="recommendations">
          <h2>8. Business Recommendations</h2>
          <p class="placeholder">Model results JSON unavailable. Section skipped.</p>
        </section>
        """

    top = [x.get("feature", "") for x in model_results.get("shap_top_features", [])[:5]]
    top_set = set(top)
    recommendations: List[str] = []

    if "OverTime" in top_set:
        recommendations.append(
            "Review overtime policies — high overtime correlates strongly with attrition risk."
        )
    if "MonthlyIncome" in top_set or "income_per_job_level" in top_set:
        recommendations.append(
            "Conduct compensation benchmarking for at-risk roles."
        )
    if any(x in top_set for x in ["YearsAtCompany", "tenure_role_ratio", "promotion_lag"]):
        recommendations.append(
            "Implement structured career pathing for employees at 2-5 year tenure mark."
        )
    if "JobSatisfaction" in top_set:
        recommendations.append(
            "Deploy quarterly pulse surveys to monitor satisfaction trends."
        )
    if "WorkLifeBalance" in top_set:
        recommendations.append(
            "Introduce flexible work arrangements for high-risk departments."
        )

    high = model_results.get("risk_segments", {}).get("High Risk", {})
    recommendations.append(
        f"Focus retention budget on {high.get('count', 0)} High Risk employees "
        f"({float(high.get('percentage', 0.0)):.2f}% of workforce)."
    )

    while len(recommendations) < 5:
        recommendations.append(
            "Track intervention outcomes monthly and recalibrate risk thresholds by department."
        )

    cards = "".join(
        [
            f"<div class='rec-card'><div class='rec-num'>{idx + 1}</div><div>{txt}</div></div>"
            for idx, txt in enumerate(recommendations[:6])
        ]
    )
    return f"""
    <section id="recommendations">
      <h2>8. Business Recommendations</h2>
      <div class="rec-grid">{cards}</div>
    </section>
    """


def generate_limitations() -> str:
    """Build assumptions and limitations section.

    Returns:
        HTML content for section 9.
    """
    return """
    <section id="limitations">
      <h2>9. Assumptions and Limitations</h2>
      <ul>
        <li>SMOTE creates synthetic minority samples — results may differ on truly new data.</li>
        <li>SHAP values are computed on training data — explanations may not fully generalize.</li>
        <li>Model is trained on IBM HR dataset — validate before applying to other organizations.</li>
        <li>PerformanceRating has low variance (only 3 and 4), limiting predictive signal.</li>
        <li>Class imbalance (~16% attrition in this dataset family) is addressed with SMOTE and balanced settings.</li>
      </ul>
    </section>
    """


def generate_appendix(
    feature_manifest: Optional[Dict[str, Any]],
    model_results: Optional[Dict[str, Any]],
) -> str:
    """Build data appendix section with feature list and metadata.

    Args:
        feature_manifest: Feature manifest JSON if available.
        model_results: Parsed model results payload.

    Returns:
        HTML content for section 10.
    """
    feature_rows = ""
    if feature_manifest and "features" in feature_manifest:
        for item in feature_manifest["features"]:
            col = item.get("column", "")
            typ = item.get("type", "")
            notes = item.get("formula") or item.get("business_meaning") or item.get("mapping") or ""
            feature_rows += f"<tr><td>{col}</td><td>{typ}</td><td>{notes}</td></tr>"
    else:
        feature_rows = "<tr><td colspan='3'>Feature manifest not available.</td></tr>"

    metadata = model_results.get("metadata", {}) if model_results else {}
    meta_html = (
        f"<p><strong>Dataset shape:</strong> {metadata.get('n_samples', 'N/A')} samples, "
        f"{metadata.get('n_features', 'N/A')} features<br>"
        f"<strong>Random seed:</strong> {metadata.get('random_seed', 'N/A')}<br>"
        f"<strong>CV folds:</strong> {metadata.get('cv_folds', 'N/A')}</p>"
    )

    return f"""
    <section id="appendix">
      <h2>10. Data Appendix</h2>
      {meta_html}
      <h3>Feature Manifest</h3>
      <table>
        <thead><tr><th>Feature</th><th>Type</th><th>Formula/Notes</th></tr></thead>
        <tbody>{feature_rows}</tbody>
      </table>
    </section>
    """


def assemble_html(
    company_name: str,
    sections: Dict[str, str],
) -> str:
    """Assemble full single-file HTML with inline CSS and top navigation.

    Args:
        company_name: Company label in report header.
        sections: Rendered HTML section fragments.

    Returns:
        Complete HTML document string.
    """
    generated = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    nav = """
    <nav class="top-nav">
      <a href="#executive-summary">Executive Summary</a>
      <a href="#data-quality">Data Quality</a>
      <a href="#methodology">Methodology</a>
      <a href="#model-comparison">Model Comparison</a>
      <a href="#model-performance">Model Performance</a>
      <a href="#shap-importance">SHAP</a>
      <a href="#risk-segmentation">Risk Segmentation</a>
      <a href="#recommendations">Recommendations</a>
      <a href="#limitations">Limitations</a>
      <a href="#appendix">Appendix</a>
    </nav>
    """

    css = """
    body { font-family: system-ui, -apple-system, sans-serif; margin: 0; background: #f4f7fb; color: #1f2937; }
    .container { max-width: 1100px; margin: 24px auto; background: #ffffff; padding: 28px; border-radius: 10px; box-shadow: 0 2px 12px rgba(0,0,0,0.08);}
    .banner { background: linear-gradient(90deg, #1a2744 0%, #2f4b85 100%); color: #fff; padding: 22px; border-radius: 10px; }
    .banner h1 { margin: 0 0 8px 0; font-size: 30px; }
    .banner p { margin: 4px 0; }
    .top-nav { display: flex; flex-wrap: wrap; gap: 8px; margin: 16px 0 20px 0; }
    .top-nav a { text-decoration: none; background: #e6eefc; color: #1a2744; padding: 7px 10px; border-radius: 18px; font-size: 13px; }
    section { margin: 28px 0; }
    h2 { color: #1a2744; border-bottom: 2px solid #d6e2f4; padding-bottom: 6px; }
    h3 { color: #243b67; margin-top: 18px; }
    table { width: 100%; border-collapse: collapse; margin: 10px 0; font-size: 14px; }
    th, td { border: 1px solid #d9e2ef; padding: 8px 10px; text-align: left; vertical-align: top; }
    th { background: #ecf3ff; }
    tr:nth-child(even) { background: #fafcff; }
    .best-row { background: #d9f7e8 !important; }
    .warning-box { background: #fff3cd; border: 1px solid #ffe69c; color: #6b5a00; border-radius: 8px; padding: 10px 12px; margin-top: 10px; }
    .metric-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; margin: 10px 0 18px 0; }
    .metric-card { background: #e9f2ff; border: 1px solid #cfe1ff; border-radius: 10px; padding: 12px; text-align: center; }
    .metric-value { font-size: 28px; font-weight: 700; color: #1a2744; line-height: 1.1; }
    .metric-label { font-size: 13px; color: #415879; margin-top: 4px; }
    .rec-grid { display: grid; grid-template-columns: 1fr; gap: 10px; }
    .rec-card { display: grid; grid-template-columns: 34px 1fr; gap: 10px; align-items: start; background: #f0f7ff; border: 1px solid #d4e6ff; border-radius: 10px; padding: 10px 12px; }
    .rec-num { width: 28px; height: 28px; border-radius: 50%; background: #1a2744; color: #fff; display: flex; align-items: center; justify-content: center; font-weight: 700; }
    .chart-wrap { margin: 14px 0; }
    .chart-row { display: grid; grid-template-columns: repeat(3, 1fr); gap: 12px; }
    .chart-cell { background: #fff; border: 1px solid #e4ebf5; border-radius: 8px; padding: 8px; }
    .placeholder { color: #6b7280; font-style: italic; }
    @media (max-width: 900px) {
      .metric-grid { grid-template-columns: repeat(2, 1fr); }
      .chart-row { grid-template-columns: 1fr; }
    }
    """

    # We intentionally use jinja2 rendering while keeping a fully in-code template string.
    env = Environment(
        loader=FileSystemLoader("."),
        autoescape=select_autoescape(["html", "xml"]),
    )
    template = env.from_string(
        """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>Employee Attrition Analysis Report</title>
  <style>{{ css }}</style>
</head>
<body>
  <div class="container">
    <header class="banner">
      <h1>Employee Attrition Analysis Report</h1>
      <p><strong>{{ company_name }}</strong></p>
      <p>Generated on {{ generated }}</p>
    </header>
    {{ nav | safe }}
    {{ sections.executive_summary | safe }}
    {{ sections.data_quality | safe }}
    {{ sections.methodology | safe }}
    {{ sections.model_comparison | safe }}
    {{ sections.model_performance | safe }}
    {{ sections.shap_section | safe }}
    {{ sections.risk_segmentation | safe }}
    {{ sections.recommendations | safe }}
    {{ sections.limitations | safe }}
    {{ sections.appendix | safe }}
  </div>
</body>
</html>
        """
    )
    return template.render(
        css=css,
        company_name=company_name,
        generated=generated,
        nav=nav,
        sections=sections,
    )


def main() -> None:
    """Parse arguments, generate sections, and save the final HTML report."""
    parser = argparse.ArgumentParser(
        description="Generate self-contained HTML attrition report."
    )
    parser.add_argument("--features", required=True, help="Path to features CSV.")
    parser.add_argument("--model-results", required=True, help="Path to model results JSON.")
    parser.add_argument("--output", required=True, help="Path to output HTML report.")
    parser.add_argument("--company-name", required=True, help="Company name label.")
    args = parser.parse_args()

    sns.set_style("whitegrid")

    loaded = load_inputs(args.features, args.model_results)
    features_df: Optional[pd.DataFrame] = loaded.get("features_df")
    model_results: Optional[Dict[str, Any]] = loaded.get("model_results")
    validation_report: Optional[Dict[str, Any]] = loaded.get("validation_report")
    feature_manifest: Optional[Dict[str, Any]] = loaded.get("feature_manifest")
    runtime_warnings: List[str] = loaded.get("warnings", [])

    sections = {
        "executive_summary": generate_executive_summary(
            company_name=args.company_name,
            model_results=model_results,
            features_df=features_df,
            runtime_warnings=runtime_warnings,
        ),
        "data_quality": generate_data_quality(validation_report),
        "methodology": generate_methodology(),
        "model_comparison": generate_model_comparison(model_results),
        "model_performance": generate_model_performance(model_results),
        "shap_section": generate_shap_section(model_results, feature_manifest),
        "risk_segmentation": generate_risk_segmentation(model_results, features_df),
        "recommendations": generate_recommendations(model_results),
        "limitations": generate_limitations(),
        "appendix": generate_appendix(feature_manifest, model_results),
    }

    html = assemble_html(args.company_name, sections)

    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    try:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(html)
    except Exception as exc:
        print(f"Error: Could not write report to '{args.output}': {exc}", file=sys.stderr)
        sys.exit(1)

    print(f"HTML report saved to {args.output}")


if __name__ == "__main__":
    main()
