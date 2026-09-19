"""
eda.py — Exploratory Data Analysis
====================================
Generates descriptive statistics and saves all figures to outputs/figures/.

Figures produced:
  01_score_distributions.png   — histograms for math, reading, writing
  02_scores_by_gender.png      — boxplots split by gender
  03_scores_by_lunch.png       — boxplots split by lunch type
  04_scores_by_test_prep.png   — boxplots split by test preparation
  05_parental_edu_heatmap.png  — mean scores by parental education level
  06_correlation_matrix.png    — Pearson correlation between numeric features

Author  : Oscar León
Dataset : Kaggle — Students Performance in Exams
"""

import os
import logging
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

FIGURES_DIR = "outputs/figures"
SCORE_COLS  = ["math score", "reading score", "writing score"]
PALETTE     = "muted"

# Ordered list for parental education (low → high)
EDU_ORDER = [
    "some high school", "high school", "some college",
    "associate's degree", "bachelor's degree", "master's degree"
]

sns.set_theme(style="whitegrid", palette=PALETTE, font_scale=1.1)


def _save(fig: plt.Figure, filename: str) -> None:
    os.makedirs(FIGURES_DIR, exist_ok=True)
    path = os.path.join(FIGURES_DIR, filename)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved → {path}")


def plot_score_distributions(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    fig.suptitle("Score Distributions", fontweight="bold")
    colors = sns.color_palette(PALETTE, 3)

    for ax, col, color in zip(axes, SCORE_COLS, colors):
        ax.hist(df[col], bins=20, color=color, edgecolor="white", linewidth=0.6)
        ax.axvline(df[col].mean(), color="black", linestyle="--", linewidth=1.2, label=f"Mean: {df[col].mean():.1f}")
        ax.set_title(col.replace(" score", "").capitalize())
        ax.set_xlabel("Score (0–100)")
        ax.set_ylabel("Count")
        ax.legend(fontsize=9)

    plt.tight_layout()
    _save(fig, "01_score_distributions.png")


def plot_boxplots_by_group(df: pd.DataFrame, groupby: str, filename: str, title: str) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    fig.suptitle(title, fontweight="bold")

    for ax, col in zip(axes, SCORE_COLS):
        order = sorted(df[groupby].unique())
        sns.boxplot(data=df, x=groupby, y=col, order=order, ax=ax,
                    hue=groupby, palette=PALETTE, legend=False)
        ax.set_title(col.replace(" score", "").capitalize())
        ax.set_xlabel("")
        ax.set_ylabel("Score (0–100)")
        ax.tick_params(axis="x", rotation=15)

    plt.tight_layout()
    _save(fig, filename)


def plot_parental_education_heatmap(df: pd.DataFrame) -> None:
    order = [e for e in EDU_ORDER if e in df["parental level of education"].unique()]
    means = (
        df.groupby("parental level of education")[SCORE_COLS]
        .mean()
        .reindex(order)
        .round(1)
    )

    fig, ax = plt.subplots(figsize=(9, 5))
    sns.heatmap(
        means, annot=True, fmt=".1f", cmap="YlOrRd",
        linewidths=0.5, ax=ax, vmin=50, vmax=85,
        cbar_kws={"label": "Mean score"}
    )
    ax.set_title("Mean Scores by Parental Education Level", fontweight="bold", pad=12)
    ax.set_xlabel("")
    ax.set_ylabel("")
    plt.tight_layout()
    _save(fig, "05_parental_edu_heatmap.png")


def plot_correlation_matrix(df: pd.DataFrame) -> None:
    # Encode categoricals for correlation
    df_enc = df.copy()
    df_enc["gender_num"]   = (df_enc["gender"] == "female").astype(int)
    df_enc["lunch_num"]    = (df_enc["lunch"] == "standard").astype(int)
    df_enc["prep_num"]     = (df_enc["test preparation course"] == "completed").astype(int)

    corr_cols = SCORE_COLS + ["gender_num", "lunch_num", "prep_num"]
    labels    = ["Math", "Reading", "Writing", "Female", "Std. Lunch", "Test Prep"]
    corr = df_enc[corr_cols].corr()
    corr.index   = labels
    corr.columns = labels

    fig, ax = plt.subplots(figsize=(7, 6))
    mask = pd.DataFrame(False, index=corr.index, columns=corr.columns)
    for i in range(len(mask)):
        for j in range(i):
            mask.iloc[i, j] = True  # lower triangle only

    sns.heatmap(
        corr, mask=mask, annot=True, fmt=".2f", cmap="coolwarm",
        center=0, linewidths=0.5, ax=ax, vmin=-1, vmax=1,
        cbar_kws={"shrink": 0.8}
    )
    ax.set_title("Feature Correlation Matrix", fontweight="bold", pad=12)
    plt.tight_layout()
    _save(fig, "06_correlation_matrix.png")


def run_eda(df: pd.DataFrame) -> None:
    """Run the full EDA suite."""
    logger.info("Starting EDA...")

    # --- Descriptive stats to console ---
    print("\n── Dataset Overview ──────────────────────────────")
    print(f"  Rows: {len(df):,}   |   Columns: {len(df.columns)}")
    print(f"  Missing values: {df.isnull().sum().sum()}")
    print("\n── Score Statistics ──────────────────────────────")
    print(df[SCORE_COLS].describe().round(2).to_string())

    print("\n── Pass Rate (score ≥ 60) ────────────────────────")
    for col in SCORE_COLS:
        rate = (df[col] >= 60).mean() * 100
        print(f"  {col:<20}: {rate:.1f}%")

    # --- Figures ---
    plot_score_distributions(df)
    plot_boxplots_by_group(df, "gender",  "02_scores_by_gender.png",   "Scores by Gender")
    plot_boxplots_by_group(df, "lunch",   "03_scores_by_lunch.png",    "Scores by Lunch Type")
    plot_boxplots_by_group(df, "test preparation course",
                           "04_scores_by_test_prep.png", "Scores by Test Preparation")
    plot_parental_education_heatmap(df)
    plot_correlation_matrix(df)

    logger.info("EDA complete. All figures saved to outputs/figures/")
