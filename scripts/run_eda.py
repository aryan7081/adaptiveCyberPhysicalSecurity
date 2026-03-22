#!/usr/bin/env python3
"""
Exploratory Data Analysis for NSL-KDD.
Generates figures and summary statistics for the report.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.loader import NSLKDDLoader, NSL_KDD_COLUMNS, LABEL_COL


def main():
    data_dir = Path(__file__).parent.parent / "data" / "raw"
    figures_dir = Path(__file__).parent.parent / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    loader = NSLKDDLoader(str(data_dir))
    try:
        train_df, test_df = loader.load()
    except FileNotFoundError as e:
        print(e)
        print("Run: python scripts/download_data.py")
        return 1

    # Combine for overall EDA
    train_df["split"] = "train"
    test_df["split"] = "test"
    if "difficulty" in test_df.columns:
        test_df = test_df.drop(columns=["difficulty"])
    df = pd.concat([train_df, test_df], ignore_index=True)

    # 1. Label distribution
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    label_counts = df[LABEL_COL].value_counts()
    # Top 15 for readability
    top_labels = label_counts.head(15)
    axes[0].barh(range(len(top_labels)), top_labels.values)
    axes[0].set_yticks(range(len(top_labels)))
    axes[0].set_yticklabels(top_labels.index, fontsize=8)
    axes[0].set_xlabel("Count")
    axes[0].set_title("Attack Type Distribution (Top 15)")
    axes[0].invert_yaxis()

    # Binary: normal vs attack
    df["is_attack"] = (df[LABEL_COL].astype(str).str.lower() != "normal").astype(int)
    vc = df["is_attack"].value_counts().sort_index()
    labels = ["Normal" if i == 0 else "Attack" for i in vc.index]
    axes[1].pie(
        vc.values,
        labels=labels,
        autopct="%1.1f%%",
        startangle=90,
    )
    axes[1].set_title("Normal vs Attack (Class Imbalance)")
    plt.tight_layout()
    plt.savefig(figures_dir / "eda_label_distribution.pdf", bbox_inches="tight")
    plt.close()
    print("Saved eda_label_distribution.pdf")

    # 2. Numeric feature distributions (key heavy-tailed)
    numeric_cols = [c for c in NSL_KDD_COLUMNS if c not in ["protocol_type", "service", "flag"]]
    key_cols = ["duration", "src_bytes", "dst_bytes", "count", "srv_count"]
    key_cols = [c for c in key_cols if c in df.columns]
    nk = len(key_cols)
    fig, axes = plt.subplots(2, (nk + 1) // 2, figsize=(12, 8))
    axes = axes.flatten()
    for i, col in enumerate(key_cols):
        vals = pd.to_numeric(df[col], errors="coerce").dropna()
        vals = vals[vals < vals.quantile(0.99)]  # Clip extremes for viz
        axes[i].hist(vals, bins=50, edgecolor="black", alpha=0.7)
        axes[i].set_title(col)
        axes[i].set_ylabel("Count")
    for j in range(i + 1, len(axes)):
        axes[j].axis("off")
    plt.suptitle("Distribution of Key Numeric Features (99th percentile clipped)")
    plt.tight_layout()
    plt.savefig(figures_dir / "eda_numeric_distributions.pdf", bbox_inches="tight")
    plt.close()
    print("Saved eda_numeric_distributions.pdf")

    # 3. Correlation matrix (sample for speed)
    num_df = df[numeric_cols].apply(pd.to_numeric, errors="coerce").fillna(0)
    if len(num_df) > 5000:
        num_df = num_df.sample(5000, random_state=42)
    corr = num_df.corr()
    plt.figure(figsize=(14, 12))
    sns.heatmap(corr, cmap="RdBu_r", center=0, square=True, linewidths=0.5)
    plt.title("Correlation Matrix of Numeric Features")
    plt.tight_layout()
    plt.savefig(figures_dir / "eda_correlation_matrix.pdf", bbox_inches="tight")
    plt.close()
    print("Saved eda_correlation_matrix.pdf")

    # 4. Summary statistics to CSV
    processed_dir = figures_dir.parent / "data" / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)
    summary = df.describe(include="all").T
    summary.to_csv(processed_dir / "eda_summary.csv")
    print("Saved eda_summary.csv")

    # 5. Missing/duplicate report
    missing = df.isnull().sum()
    dupes = df.duplicated().sum()
    with open(figures_dir / "eda_data_quality.txt", "w") as f:
        f.write(f"Missing values per column:\n{missing[missing > 0]}\n\n")
        f.write(f"Duplicate rows: {dupes}\n")
        f.write(f"Total rows: {len(df)}\n")
    print("Saved eda_data_quality.txt")
    return 0


if __name__ == "__main__":
    sys.exit(main())
