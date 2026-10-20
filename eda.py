"""Exploratory data analysis -> PNGs in reports/figures + summary JSON.

Run:  python -m heartml.eda
"""
from __future__ import annotations

import json
import logging

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402

from heartml import config  # noqa: E402

log = logging.getLogger("heartml.eda")
PALETTE = {0: "#4C9F70", 1: "#C8553D"}  # healthy green / disease rust


def _save(fig, name: str) -> str:
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    path = config.FIGURES_DIR / name
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return str(path)


def class_balance(df: pd.DataFrame) -> str:
    fig, ax = plt.subplots(figsize=(5, 4))
    counts = df[config.TARGET].value_counts().sort_index()
    ax.bar(["No disease (0)", "Disease (1)"], counts.values, color=[PALETTE[0], PALETTE[1]])
    for i, v in enumerate(counts.values):
        ax.text(i, v + 2, f"{v} ({v / counts.sum():.0%})", ha="center")
    ax.set_title("Class balance")
    ax.set_ylabel("Patients")
    return _save(fig, "01_class_balance.png")


def numeric_histograms(df: pd.DataFrame) -> str:
    cols = config.NUMERIC_FEATURES
    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    for ax, col in zip(axes.ravel(), cols):
        for t in (0, 1):
            sns.histplot(df.loc[df[config.TARGET] == t, col].dropna(), ax=ax, kde=True,
                         color=PALETTE[t], label=f"target={t}", stat="density", alpha=0.45)
        ax.set_title(col)
        ax.legend(fontsize=8)
    fig.suptitle("Numeric feature distributions by outcome")
    return _save(fig, "02_numeric_histograms.png")


def categorical_rates(df: pd.DataFrame) -> str:
    cols = config.CATEGORICAL_FEATURES + config.BINARY_FEATURES
    fig, axes = plt.subplots(2, 4, figsize=(15, 7))
    for ax, col in zip(axes.ravel(), cols):
        rate = df.groupby(col)[config.TARGET].mean()
        ax.bar(rate.index.astype(str), rate.values, color="#3B6E8F")
        ax.set_ylim(0, 1)
        ax.set_title(f"P(disease) by {col}")
    fig.suptitle("Disease rate per category")
    return _save(fig, "03_categorical_disease_rate.png")


def correlation_heatmap(df: pd.DataFrame) -> str:
    fig, ax = plt.subplots(figsize=(10, 8))
    corr = df.corr(numeric_only=True)
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="vlag", center=0, ax=ax, square=True,
                cbar_kws={"shrink": 0.8})
    ax.set_title("Pearson correlation (incl. target)")
    return _save(fig, "04_correlation_heatmap.png")


def boxplots(df: pd.DataFrame) -> str:
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, col in zip(axes, ["thalach", "oldpeak", "age"]):
        sns.boxplot(data=df, x=config.TARGET, y=col, hue=config.TARGET, palette=PALETTE,
                    legend=False, ax=ax)
        ax.set_title(f"{col} vs target")
    return _save(fig, "05_boxplots.png")


def run(df: pd.DataFrame | None = None) -> dict:
    if df is None:
        df = pd.read_csv(config.CLEAN_FILE)
    figures = [class_balance(df), numeric_histograms(df), categorical_rates(df),
               correlation_heatmap(df), boxplots(df)]
    corr_t = df.corr(numeric_only=True)[config.TARGET].drop(config.TARGET)
    summary = {
        "rows": int(len(df)),
        "positive_rate": round(float(df[config.TARGET].mean()), 4),
        "missing": {k: int(v) for k, v in df.isna().sum().items() if v},
        "top_correlations_with_target": corr_t.abs().sort_values(ascending=False)
        .head(6).round(3).to_dict(),
        "figures": figures,
    }
    (config.REPORTS_DIR / "eda_summary.json").write_text(json.dumps(summary, indent=2))
    df.describe().T.round(2).to_csv(config.REPORTS_DIR / "eda_describe.csv")
    return summary


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    s = run()
    log.info("EDA done: %s", json.dumps({k: v for k, v in s.items() if k != "figures"}))


if __name__ == "__main__":
    main()
