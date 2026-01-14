import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score

def apply_threshold(proba: np.ndarray, threshold: float) -> np.ndarray:
    """Applies a threshold to probability scores to get binary predictions.

    Args:
        proba (ArrayLike): Array-like object containing probability scores for the 
            positive class.
        threshold (float): The value above which a sample is classified as 1.

    Returns:
        An integer array of binary predictions (0 or 1).
    """
    return (proba >= threshold).astype(int)



def sweep_thresholds(y_true: np.ndarray, proba: np.ndarray, thresholds=None) -> pd.DataFrame:
    """Calculates classification metrics across a range of probability thresholds.

    Args:
        y_true (ArrayLike): Ground truth binary labels.
        proba (ArrayLike): Predicted probabilities for the positive class.
        thresholds: List or array of threshold values to evaluate. 
            If None, evaluates 101 points from 0 to 1. Defaults to None.

    Returns:
        A pandas DataFrame where each row contains metrics (Precision, Recall, 
        F1, TP, FP, FN, TN) for a specific threshold.
    """
    if thresholds is None:
        thresholds = np.linspace(0, 1, 101)

    rows = []
    for t in thresholds:
        y_pred = apply_threshold(proba, t)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0,1]).ravel()

        rows.append({
                "threshold": float(t),
                "precision": float(precision_score(y_true, y_pred, zero_division=0)),
                "recall": float(recall_score(y_true, y_pred, zero_division=0)),
                "f1": float(f1_score(y_true, y_pred, zero_division=0)),
                "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
            })
    return pd.DataFrame(rows)