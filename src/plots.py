import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pathlib import Path
from sklearn.metrics import confusion_matrix, roc_curve

def generate_report_plots(df: pd.DataFrame, output_dir: Path) -> dict:
    """Assembles descriptive EDA visuals and stores target artifacts safely onto disk storage."""
    output_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")
    paths = {}

    # 1. Balance Plot
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.histplot(df["balance"], kde=True, bins=40, ax=ax)
    ax.set_title("Balance Distribution Profile")
    paths["balance_distribution"] = output_dir / "balance_distribution.png"
    fig.savefig(paths["balance_distribution"], bbox_inches="tight")
    plt.close(fig)

    # 2. Income Plot
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.histplot(df["income"], kde=True, bins=40, color="green", ax=ax)
    ax.set_title("Income Distribution Profile")
    paths["income_distribution"] = output_dir / "income_distribution.png"
    fig.savefig(paths["income_distribution"], bbox_inches="tight")
    plt.close(fig)

    # 3. Default Pie
    fig, ax = plt.subplots(figsize=(5, 5))
    df["default"].value_counts().plot.pie(autopct="%1.1f%%", labels=["No Default", "Default"], colors=["#4CAF50", "#F44336"], ax=ax)
    ax.set_ylabel("")
    ax.set_title("Global Ledger Default Mix")
    paths["default_rate"] = output_dir / "default_rate.png"
    fig.savefig(paths["default_rate"], bbox_inches="tight")
    plt.close(fig)

    # 4. Student Demographics
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.barplot(x="student", y="default", data=df, ax=ax, palette="muted")
    ax.set_xticklabels(["Non-Student", "Student Class"])
    ax.set_title("Default Trajectory Across Cohorts")
    paths["default_by_student"] = output_dir / "default_by_student.png"
    fig.savefig(paths["default_by_student"], bbox_inches="tight")
    plt.close(fig)

    # 5. Financial Boxplots Combo
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    sns.boxplot(x="default", y="balance", data=df, ax=axes[0], palette="Set2")
    axes[0].set_title("Balance vs Default Risk State")
    sns.boxplot(x="default", y="income", data=df, ax=axes[1], palette="Set2")
    axes[1].set_title("Income vs Default Risk State")
    paths["financial_boxplots"] = output_dir / "financial_boxplots.png"
    fig.savefig(paths["financial_boxplots"], bbox_inches="tight")
    plt.close(fig)

    # 6. Matrix heatmaps
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.heatmap(df.select_dtypes(include='number').corr(), annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
    ax.set_title("Features Core Correlation Matrix Grid")
    paths["correlation_heatmap"] = output_dir / "correlation_heatmap.png"
    fig.savefig(paths["correlation_heatmap"], bbox_inches="tight")
    plt.close(fig)

    return {k: str(v) for k, v in paths.items()}

def generate_model_plots(model_results: dict, output_dir: Path) -> dict:
    """Compiles precision machine learning verification profiles to output directories."""
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    
    y_test = model_results["y_test"]
    y_pred = model_results["y_pred"]
    y_proba = model_results["y_proba"]
    
    # 1. Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)  # Fixed your missing 'a' bug
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    ax.set_title("Confusion Validation Matrix")
    paths["confusion_matrix"] = output_dir / "confusion_matrix.png"
    fig.savefig(paths["confusion_matrix"], bbox_inches="tight")  # Fixed your dot accessor syntax bug
    plt.close(fig)
    
    # 2. ROC-AUC curve mapping
    fpr, tpr, _ = roc_curve(y_test, y_proba)  # Fixed your missing 'e' bug
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(fpr, tpr, label=f"Logistic Regression (AUC = {model_results['metrics']['roc_auc']:.2f})", color="darkorange", linewidth=2)
    ax.plot([0, 1], [0, 1], linestyle="--", color="grey")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("Receiver Operating Characteristic Curve")
    ax.legend(loc="lower right")
    paths["roc_curve"] = output_dir / "roc_curve.png"
    fig.savefig(paths["roc_curve"], bbox_inches="tight")
    plt.close(fig)
    
    return {k: str(v) for k, v in paths.items()}