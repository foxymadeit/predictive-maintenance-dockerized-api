from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.base import BaseEstimator
from typing import Any


def build_model(model_name: str, **params: Any) -> BaseEstimator:
    """Factory function that builds a model with a given set of parameters.

    Args:
        model_name (str): Name of the required model. Must be one of:
            'logreg', 'random_forest', 'rf', 'hist_gb', 'hgb' or 'gradient_boosting.
        params (Any): Keyword arguments passed to the model constructor.


    Returns:
        BaseEstimator: An initialized Scikit-learn estimator object.

    Raises:
        ValueError: If model_name is not recognized.
        TypeError: If invalid parameters are passed for the chosen model.
    """
    model_name = model_name.lower().strip()

    try:
        if model_name == 'logreg':
            return LogisticRegression(**params)
        
        if model_name in ("random_forest", "rf"):
            return RandomForestClassifier(**params)
            
        if model_name in ("hist_gb", "hgb", "gradient_boosting"):
            return HistGradientBoostingClassifier(**params)
            
    except TypeError as e:
        raise TypeError(
            f"Invalid parameters for model '{model_name}'. "
            f"Original error: {e}"
        )

    raise ValueError(
        f"Unknown model_name='{model_name}'. "
        "Use one of: 'logreg', 'rf', or 'hgb' / 'hist_gb' / 'gradient_boosting'."
    )