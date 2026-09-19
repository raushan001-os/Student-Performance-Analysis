"""
features.py — Feature Engineering & Preprocessing
====================================================
Prepares the raw DataFrame for modelling.

Steps:
  1. Derive composite features (average score, pass/fail label)
  2. Encode categorical variables with OrdinalEncoder / OneHotEncoder
  3. Split into train / test sets (stratified on the target)

Author  : Oscar León
"""

import logging
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

logger = logging.getLogger(__name__)

# Ordered levels for parental education (used by OrdinalEncoder)
EDU_LEVELS = [
    ["some high school", "high school", "some college",
     "associate's degree", "bachelor's degree", "master's degree"]
]

ORDINAL_COLS    = ["parental level of education"]
ONEHOT_COLS     = ["gender", "race/ethnicity", "lunch", "test preparation course"]
NUMERIC_TARGETS = ["math score", "reading score", "writing score"]


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds derived columns to the raw DataFrame:
      - average_score  : mean of the three subject scores
      - passed         : 1 if average_score >= 60 else 0  (binary target)
      - performance    : categorical label (Low / Medium / High)
    """
    df = df.copy()
    df["average_score"] = df[NUMERIC_TARGETS].mean(axis=1).round(2)
    df["passed"]        = (df["average_score"] >= 60).astype(int)
    df["performance"]   = pd.cut(
        df["average_score"],
        bins=[0, 50, 70, 100],
        labels=["Low", "Medium", "High"],
        right=True,
    )
    logger.info("Derived features added: average_score, passed, performance")
    return df


def build_preprocessor() -> ColumnTransformer:
    """
    Returns a ColumnTransformer that:
      - OrdinalEncodes  parental level of education (respects natural order)
      - OneHotEncodes   gender, race/ethnicity, lunch, test preparation course
    """
    ordinal_pipe = Pipeline([
        ("enc", OrdinalEncoder(categories=EDU_LEVELS, handle_unknown="use_encoded_value", unknown_value=-1))
    ])
    onehot_pipe = Pipeline([
        ("enc", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])
    preprocessor = ColumnTransformer(
        transformers=[
            ("ordinal", ordinal_pipe, ORDINAL_COLS),
            ("onehot",  onehot_pipe,  ONEHOT_COLS),
        ],
        remainder="drop",   # drop raw score columns; target is 'passed'
    )
    return preprocessor


def prepare_data(df: pd.DataFrame, test_size: float = 0.20, random_state: int = 42):
    """
    Full preparation pipeline.

    Parameters
    ----------
    df           : Raw DataFrame (output of engineer_features)
    test_size    : Proportion for the test split (default 20 %)
    random_state : Seed for reproducibility

    Returns
    -------
    X_train, X_test, y_train, y_test : numpy arrays ready for sklearn
    feature_names                    : list of column names after encoding
    """
    df = engineer_features(df)

    X = df[ORDINAL_COLS + ONEHOT_COLS]
    y = df["passed"]

    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    preprocessor = build_preprocessor()
    X_train = preprocessor.fit_transform(X_train_raw)
    X_test  = preprocessor.transform(X_test_raw)

    # Recover feature names for interpretability
    ohe_names = preprocessor.named_transformers_["onehot"]["enc"].get_feature_names_out(ONEHOT_COLS)
    feature_names = ORDINAL_COLS + list(ohe_names)

    logger.info(f"Train size: {len(X_train):,}   Test size: {len(X_test):,}")
    logger.info(f"Class balance — train pass rate: {y_train.mean():.2%}")

    return X_train, X_test, y_train, y_test, feature_names
