import matplotlib.pyplot as plt
from typing import Optional
import pandas as pd


def plot_model_comparison(
    models_results: pd.DataFrame,
    metric: str = "pr_auc",
    top_n: Optional[int] = None,
    ascending: bool = False,
    annotate: bool = True
) -> None:
    """
    Visualizes a comparison of different models with chosen metrics using a bar chart.

    Args:
        models_results (pd.DataFrame): DataFrame containing model metrics and 'run_dir' names.
        metric (str): The column name to use for comparison. Defaults to "pr_auc".
        top_n (int, optional): Number of top models to display. Shows all if None.
        ascending (bool): Sort order. False for descending (best models first).
        annotate (bool): Whether to display value labels on top of the bars.
    """

    df = models_results.copy()

    if metric not in df.columns:
        raise ValueError(f"Metric '{metric}' not found")

    df = df.sort_values(by=metric, ascending=ascending)
    if top_n is not None:
        df = df.head(top_n)

    plt.figure(figsize=(10, 6))
    bars = plt.bar(df["run_dir"], df[metric])

    plt.xticks(rotation=45, ha="right")
    plt.ylabel(metric)
    plt.title(f"Models comparison by {metric}")

    if annotate:
        for bar in bars:
            height = bar.get_height()
            plt.text(
                bar.get_x() + bar.get_width() / 2,
                height,
                f"{height:.3f}",
                ha="center",
                va="bottom",
                fontsize=9
            )

    plt.tight_layout()
    plt.show()



def plot_precision_vs_recall_at_chosen_threshold(
        models_results: pd.DataFrame
        ) -> None:
    
    """
    Plots a scatter diagram of Precision vs Recall for specific model thresholds.
    
    This helps to evaluate how close each model's chosen threshold is to the 
    ideal (1, 1) point.

    Args:
        models_results (pd.DataFrame): DataFrame containing 'recall', 'precision', 
            and 'run_dir' columns.
    """
    
    plt.figure()
    plt.scatter(models_results["recall"], models_results["precision"], s=90)
    for _, r in models_results.iterrows():
        plt.text(
            r["recall"] + 0.005,
            r["precision"] + 0.005,
            r["run_dir"], fontsize=8)
    plt.grid(True, alpha=0.3)
    plt.xlim((0,1))
    plt.ylim((0,1))
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Models at chosen thresholds: Precision vs Recall")
    plt.tight_layout()
    plt.show()