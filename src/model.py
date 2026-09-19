"""
model.py — Decision Tree Classifier
=====================================
Trains a Decision Tree to predict whether a student passes (avg score ≥ 60).

Why a Decision Tree for this demo?
  - Interpretable: the tree can be visualised and explained to non-technical
    audiences, which matters in an educational context.
  - No hyperparameter tuning required for a first baseline.
  - Feature importances are directly readable.

Author  : Oscar León
"""

import os
import logging
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, roc_auc_score, ConfusionMatrixDisplay
)

logger = logging.getLogger(__name__)

FIGURES_DIR = "outputs/figures"
RESULTS_DIR = "outputs"


def train(X_train, y_train, max_depth: int = 4, random_state: int = 42) -> DecisionTreeClassifier:
    """
    Fits a Decision Tree classifier.

    max_depth=4 keeps the tree shallow enough to visualise clearly while
    avoiding overfitting on this small dataset.
    """
    clf = DecisionTreeClassifier(
        max_depth=max_depth,
        class_weight="balanced",  # handles slight class imbalance
        random_state=random_state,
    )
    clf.fit(X_train, y_train)
    logger.info(f"Model trained — depth: {clf.get_depth()}, leaves: {clf.get_n_leaves()}")
    return clf


def evaluate(clf: DecisionTreeClassifier, X_test, y_test, feature_names: list) -> dict:
    """
    Evaluates the model and saves:
      - Classification report (console + JSON)
      - Confusion matrix figure
      - Feature importance bar chart
      - Tree visualisation

    Returns a dict with summary metrics.
    """
    os.makedirs(FIGURES_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test)[:, 1]

    acc     = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_prob)
    report  = classification_report(y_test, y_pred, target_names=["Failed", "Passed"], output_dict=True)

    # ── Console output ─────────────────────────────────────────────────────────
    print("\n── Model Evaluation ──────────────────────────────")
    print(f"  Accuracy : {acc:.4f}")
    print(f"  ROC-AUC  : {roc_auc:.4f}")
    print("\n── Classification Report ─────────────────────────")
    print(classification_report(y_test, y_pred, target_names=["Failed", "Passed"]))

    # ── Save JSON report ───────────────────────────────────────────────────────
    report["accuracy_score"] = round(acc, 4)
    report["roc_auc"]        = round(roc_auc, 4)
    with open(os.path.join(RESULTS_DIR, "metrics.json"), "w") as f:
        json.dump(report, f, indent=2)
    logger.info("Saved → outputs/metrics.json")

    # ── Confusion matrix ───────────────────────────────────────────────────────
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Failed", "Passed"])
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title("Confusion Matrix", fontweight="bold")
    plt.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "07_confusion_matrix.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved → outputs/figures/07_confusion_matrix.png")

    # ── Feature importances ────────────────────────────────────────────────────
    importances = pd.Series(clf.feature_importances_, index=feature_names)
    importances = importances[importances > 0].sort_values(ascending=True)

    fig, ax = plt.subplots(figsize=(8, max(4, len(importances) * 0.35)))
    bars = ax.barh(importances.index, importances.values,
                   color=sns.color_palette("muted", len(importances)))
    ax.set_xlabel("Importance (Gini)")
    ax.set_title("Feature Importances — Decision Tree", fontweight="bold")
    ax.xaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    for bar, val in zip(bars, importances.values):
        ax.text(val + 0.002, bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}", va="center", fontsize=9)
    plt.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "08_feature_importances.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved → outputs/figures/08_feature_importances.png")

    # ── Tree visualisation ─────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(18, 8))
    plot_tree(
        clf, feature_names=feature_names,
        class_names=["Failed", "Passed"],
        filled=True, rounded=True, fontsize=8, ax=ax,
        impurity=False, proportion=True,
    )
    ax.set_title("Decision Tree (max depth = 4)", fontweight="bold", fontsize=13)
    plt.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "09_decision_tree.png"), dpi=120, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved → outputs/figures/09_decision_tree.png")

    # ── Text representation (for README / docs) ────────────────────────────────
    tree_text = export_text(clf, feature_names=feature_names, max_depth=3)
    with open(os.path.join(RESULTS_DIR, "tree_rules.txt"), "w") as f:
        f.write(tree_text)
    logger.info("Saved → outputs/tree_rules.txt")

    return {"accuracy": acc, "roc_auc": roc_auc}


# Needed for the formatter import used in evaluate()
import matplotlib.ticker as mticker
