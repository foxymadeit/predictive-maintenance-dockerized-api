import matplotlib.pyplot as plt
import pandas as pd


def plot_precision_recall_tradeoff(models_thresholds: pd.DataFrame) -> None:
    """
    Plots Precision-Recall curves for multiple models based on probability thresholds.

    Args:
        models_thresholds (pd.DataFrame): Data containing threshold iterations. 
            Must include 'run_dir', 'threshold', 'precision', and 'recall' columns.

    """
    plt.figure(figsize=(8,6))
    for run_dir, g in models_thresholds.groupby("run_dir"):
        g = g.sort_values("threshold")
        plt.plot(g["recall"], g["precision"], label=run_dir)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision–Recall trade-off across thresholds")
    plt.legend()
    plt.show()