from sklearn.pipeline import Pipeline
import numpy as np

def train_and_predict(pipeline: Pipeline, X_train: np.ndarray,
                     y_train: np.ndarray, X_test: np.ndarray):
    """Fit the X_train and y_train data to the pipeline and returns prediction and probability values.

    Args:
        pipeline (Pipeline): sklearn Pipeline instance object.
        X_train (np.ndarray): X_train values.
        y_train (np.ndarray): y_train values.
        X_train (np.ndarray): X_test values.

    Returns:
        Prediction and Probability of failure.
    """
    pipeline.fit(X_train, y_train)
    preds = pipeline.predict(X_test)
    proba = pipeline.predict_proba(X_test)[:, 1]
    return preds, proba