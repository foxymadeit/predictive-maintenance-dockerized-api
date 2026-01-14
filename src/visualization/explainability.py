import matplotlib.pyplot as plt
from typing import Optional, Any, Tuple, List
import pandas as pd
import shap
import numpy as np
from scipy import sparse


def to_dense(a):
    return a.toarray() if sparse.issparse(a) else a



def get_shap_values(
        pipeline: Any,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        ) -> Tuple[shap.Explanation, pd.DataFrame, List[str]]:
    """
    Extracts SHAP values from a scikit-learn pipeline.

    Args:
        pipeline (Any): Fitted sklearn Pipeline containing 'preprocessor' and 'model' steps.
        X_train (pd.DataFrame): Training data used as a background distribution for SHAP.
        X_test (pd.DataFrame): Test data to calculate SHAP values for.

    Returns:
        shap.Explanation: Calculated SHAP values for X_test.

    Raises:
        KeyError: If pipeline is missing specified steps
    """
    
    try:
        preproc_shap = pipeline.named_steps['preprocessor']
        model_shap = pipeline.named_steps['model']
    except KeyError as e:
        raise KeyError(
            f"Step {e} not found in pipeline. "
            "Ensure steps are named 'preprocessor' and 'model'."
        )

    X_train_tr = preproc_shap.transform(X_train)
    X_test_tr = preproc_shap.transform(X_test)

    feature_names = list(preproc_shap.get_feature_names_out())


    X_train_tr = to_dense(X_train_tr)
    X_test_tr = to_dense(X_test_tr)

    X_train_shap = pd.DataFrame(X_train_tr, columns=feature_names)
    X_test_shap = pd.DataFrame(X_test_tr, columns=feature_names)

    explainer = shap.Explainer(model_shap, X_train_shap)
    shap_values = explainer(X_test_shap, check_additivity=False)
    return shap_values, X_test_shap, feature_names



def plot_shap_summary(shap_values: shap.Explanation, max_display: int = 20) -> None:
    """Plots shap summary-type plot.

    Args:
        shap_values (shap.Explanation): shap.Explanation object.

        max_display (int): Maximum number of features to display in the plot.
            Defaults to 20.
    
    """
    shap.plots.beeswarm(shap_values, max_display=max_display, show=False)


def plot_shap_importance_bar(shap_values: shap.Explanation, max_display: int = 20) -> None:
    """Plots shap importance bar type plot.

    Args:
        shap_values (shap.Explanation): shap.Explanation object.

        max_display (int): Maximum number of features to display in the plot.
            Defaults to 20.
    """
    shap.plots.bar(shap_values, max_display=max_display, show=False)


def get_top_shap_features(
    shap_values: shap.Explanation,
    feature_names: List[str],
    top_k: int = 3
) -> List[str]:
    """Returns sorted list with shap.Explanation features in a descending order.

    Args:
        shap_values (shap.Explanation): shap.Explanation object.

        feature_names (List[str]): List with a specified feature names.

        top_k (int): Limiting returned top featues.

    Returns:
        List[str]: Sorted list with features in a descending order.
    """
    vals = shap_values.values
    mean_abs = np.abs(vals).mean(axis=0)
    order = np.argsort(mean_abs)[::-1]
    return [feature_names[i] for i in order[:top_k]]


def plot_shap_dependence(
    shap_values: shap.Explanation,
    features: List[str],
) -> None:
    """Plots shap dependency type plot.

    Args:
        shap_values (shap.Explanation): shap.Explanation object.

        features (List[str]): List with a specified features for plotting.
    """
    for f in features:
        shap.plots.scatter(shap_values[:, f], color=shap_values[:, f], show=False)


def pick_example_index(
    y_true: Optional[np.ndarray],
    proba: Optional[np.ndarray],
    threshold: float,
    kind: str = "tp",   # "tp", "fp", "fn", "high_risk"
) -> Optional[int]:
    """Picks a specific sample index based on prediction performance type.

    From the identified group (TP, FP, or FN), it selects the example with 
    the highest predicted probability.

    Args:
        y_true (Optional[np.ndarray]): Ground truth binary labels.
        proba (Optional[np.ndarray]): Predicted probabilities for the positive class.
        threshold (float): Classification threshold to determine predictions.
        kind (str): Type of example to pick. Must be one of: 'tp' (True Positive), 
            'fp' (False Positive), 'fn' (False Negative), or 
            'high_risk' (Highest probability regardless of label). 
            Defaults to "tp".

    Returns:
        The index of the selected example, or None if the probability 
        array is empty or no examples match the criteria.

    Raises:
        ValueError: If an unsupported 'kind' is provided.
    """
    if proba is None:
        return None

    if kind == "high_risk" or y_true is None:
        return int(np.argmax(proba))

    y_true = np.asarray(y_true).astype(int)
    preds = (proba >= threshold).astype(int)

    if kind == "tp":
        idx = np.where((preds == 1) & (y_true == 1))[0]
    elif kind == "fp":
        idx = np.where((preds == 1) & (y_true == 0))[0]
    elif kind == "fn":
        idx = np.where((preds == 0) & (y_true == 1))[0]
    else:
        raise ValueError("kind must be one of: 'tp', 'fp', 'fn', 'high_risk'")

    if len(idx) == 0:
        return None
    return int(idx[np.argmax(proba[idx])])


def plot_shap_waterfall(
    shap_values: shap.Explanation,
    index: int,
    *,
    kind: str = "case",
    proba: Optional[np.ndarray] = None,
    threshold: Optional[float] = None,
    max_display: int = 15
) -> None:
    """Plots a SHAP waterfall chart for a specific instance.

    Visualizes how individual features contributed to a specific prediction
    relative to the base value.

    Args:
        shap_values (shap.Explanation): A SHAP Explanation object containing calculated values.
        index (int): The index of the specific sample to visualize.
        kind (str): Descriptive label for the case type (e.g., 'tp', 'fp'). 
            Used only for the plot title. Defaults to "case".
        proba (Optional[np.ndarray]): Array of all predicted probabilities. If provided, 
            shows the probability for the selected index in the title.
        threshold (Optional[float]): The classification threshold used. If provided, 
            shows it in the plot title.
        max_display (int): Maximum number of features to show in the plot. 
            Defaults to 15.
    """

    title = f"SHAP Waterfall — {kind.upper()} case"
    if proba is not None:
        title += f"\nPredicted probability = {float(proba[index]):.3f}"
    if threshold is not None:
        title += f", threshold = {float(threshold):.2f}"


    plt.figure()
    shap.plots.waterfall(shap_values[index], max_display=max_display, show=False)
    plt.title(title)
    plt.tight_layout()



def make_shap_report(
    pipeline: Any,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    *,
    y_true: Optional[np.ndarray] = None,
    proba: Optional[np.ndarray] = None,
    threshold: float = 0.10,
    max_display: int = 20,
    top_dependence: int = 3,
    waterfall_kinds: Optional[List[str]] = None,
) -> dict:
    """
    Generates a SHAP report for a fitted pipeline:
      - beeswarm
      - bar (global importance)
      - dependence plots for top features
      - waterfall plots for selected examples (tp/fp/fn or high_risk)

    Returns: 
        dict: A dict with useful objects (shap_values, X_test_shap, top_features, picked_indices).
    """
    shap_values, X_test_shap, feature_names = get_shap_values(pipeline, X_train, X_test)

    if waterfall_kinds is None:
        waterfall_kinds = ["tp", "fp", "fn"]
    
    # 1) Global summary plots
    plot_shap_summary(shap_values, max_display=max_display)
    plot_shap_importance_bar(shap_values, max_display=max_display)

    # 2) Dependence for top-K features
    top_feats = get_top_shap_features(shap_values, feature_names, top_k=top_dependence)
    plot_shap_dependence(shap_values, top_feats)

    # 3) Waterfalls
    picked = {}
    for kind in waterfall_kinds:
        idx = pick_example_index(y_true=y_true, proba=proba, threshold=threshold, kind=kind)
        picked[kind] = idx
        if idx is not None:
            plot_shap_waterfall(
                shap_values,
                idx,
                kind=kind,
                proba=proba,
                threshold=threshold,
                max_display=15
            )
        else:
            print(f"[SHAP] No sample found for waterfall kind='{kind}' at threshold={threshold}")

    return {
        "shap_values": shap_values,
        "X_test_shap": X_test_shap,
        "feature_names": feature_names,
        "top_dependence_features": top_feats,
        "picked_indices": picked,
    }