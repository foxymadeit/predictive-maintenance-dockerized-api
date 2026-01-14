from sklearn.metrics import (precision_score,
                            recall_score,
                            confusion_matrix,
                            roc_auc_score,
                            average_precision_score,
                            accuracy_score,
                            f1_score
                            )
from typing import Dict, Any, Union
import numpy as np
import pandas as pd
ArrayLike = Union[np.ndarray, pd.Series]

def evaluate_classifier(
        y_true: ArrayLike,
        y_preds: ArrayLike,
        y_proba: ArrayLike
        ) -> Dict[str, Any]:
    """Evaluate classifier results.

    Args:
        y_true (ArrayLike): ArrayLike true y values.
        y_preds (ArrayLike): ArrayLike y preds values.
        y_proba (ArrayLike): ArrayLike y proba values.

    Returns:
        dict with a following information:
        - confusion matrix (tn, fp, fn, tp)
        - accuracy score
        - precision score
        - recall score
        - f1_score
        - roc_auc score
        - pr_auc score
    """
    cm = confusion_matrix(y_true, y_preds)
    tn, fp, fn, tp = cm.ravel()

    results = {
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        "accuracy": float(accuracy_score(y_true, y_preds)),
        "precision": float(precision_score(y_true, y_preds, zero_division=0)),
        "recall": float(recall_score(y_true, y_preds, zero_division=0)),
        "f1_score": float(f1_score(y_true, y_preds, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_proba)),
        "pr_auc": float(average_precision_score(y_true, y_proba)),
    }

    return results


def print_results(results: dict) -> None:
    "Prints the evaluated classifier results."
    for metric, value in results.items():
        if isinstance(value, int):
            print(f"{metric:<20} | {value}")
        else:
            print(f"{metric:<20} | {value:.4f}")
        



