"""
analysis.py — Main Runner
==========================
Executes the full analysis pipeline in four steps:
  1. Load data
  2. Exploratory Data Analysis (EDA)
  3. Feature engineering & preprocessing
  4. Model training & evaluation

Usage:
    python src/analysis.py
    python src/analysis.py --data data/raw/StudentsPerformance.csv --depth 5

Author  : Oscar León
"""

import argparse
import logging
import sys
import os

import pandas as pd

# Allow imports from src/ regardless of working directory
sys.path.insert(0, os.path.dirname(__file__))

from eda      import run_eda
from features import prepare_data
from model    import train, evaluate

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def main(data_path: str, max_depth: int) -> None:
    logger.info("=" * 55)
    logger.info("STUDENT PERFORMANCE ANALYSIS — STARTED")
    logger.info("=" * 55)

    # ── 1. Load ────────────────────────────────────────────────
    logger.info("[1/4] Loading dataset...")
    df = pd.read_csv(data_path)
    logger.info(f"      {len(df):,} rows × {len(df.columns)} columns loaded")

    # ── 2. EDA ─────────────────────────────────────────────────
    logger.info("[2/4] Running EDA...")
    run_eda(df)

    # ── 3. Preprocessing ───────────────────────────────────────
    logger.info("[3/4] Feature engineering & preprocessing...")
    X_train, X_test, y_train, y_test, feature_names = prepare_data(df)

    # ── 4. Model ───────────────────────────────────────────────
    logger.info("[4/4] Training and evaluating Decision Tree...")
    clf     = train(X_train, y_train, max_depth=max_depth)
    metrics = evaluate(clf, X_test, y_test, feature_names)

    logger.info("=" * 55)
    logger.info(f"DONE — Accuracy: {metrics['accuracy']:.4f}  |  ROC-AUC: {metrics['roc_auc']:.4f}")
    logger.info("Outputs saved to: outputs/")
    logger.info("=" * 55)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the student performance analysis.")
    parser.add_argument(
        "--data", default="data/raw/StudentsPerformance.csv",
        help="Path to the CSV file (default: data/raw/StudentsPerformance.csv)"
    )
    parser.add_argument(
        "--depth", type=int, default=4,
        help="Maximum depth for the Decision Tree (default: 4)"
    )
    args = parser.parse_args()
    main(args.data, args.depth)
