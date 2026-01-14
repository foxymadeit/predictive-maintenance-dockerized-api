from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Union, List

import joblib
import numpy as np
import pandas as pd


PathLike = Union[str, Path]


@dataclass
class PredictiveMaintenanceModel:
    """Production-ready wrapper around a trained sklearn Pipeline and chosen threshold.

    Args:
        pipeline (Any): sklearn Pipeline object.
        threshold (float): Chosen threshold value.
        feature_cols (Sequence[str]): Feature columns.
        artifacts_dir (Path): Directory with artifacts. Expected files inside:
            - pipeline.joblib
            - threshold.joblib
            - (optional) metrics.json
            - (optional) threshold_sweep.csv

        _shap_explainer (Any): shap.Explainer object. Defaults to None.
        _X_background (Optional[pd.DataFrame]): X split of the data for SHAP cached operations. Defaults to None. 
        _feature_names_out (Optional[List[str]]): Features names for SHAP explanations.

    Returns:
        PredictiveMaintenanceModel instance

    
    """
    pipeline: Any
    threshold: float
    feature_cols: Sequence[str]
    artifacts_dir: Path

    # Optional cached SHAP objects
    _shap_explainer: Any = None
    _X_background: Optional[pd.DataFrame] = None
    _feature_names_out: Optional[List[str]] = None

    def __post_init__(self):
        """Post-initialization checks.
        """
        if not (0.0 <= self.threshold <= 1.0):
            raise ValueError(
                f"Invalid threshold={self.threshold}. Must be in [0, 1]."
            )
        
        if self.pipeline is None:
            raise ValueError("pipeline must not be None.")

        if not hasattr(self.pipeline, "predict_proba"):
            raise TypeError("pipeline must implement predict_proba(X). "
                        "Your pipeline/model does not support probability output.")
        
        if not self.feature_cols or len(self.feature_cols) == 0:
            raise ValueError("feature_cols must be a non-empty sequence of column names.")


    @classmethod
    def load(
        cls,
        artifacts_dir: PathLike,
        *,
        feature_cols: Sequence[str],
        pipeline_name: str = "pipeline.joblib",
        threshold_name: str = "threshold.joblib",
    ) -> "PredictiveMaintenanceModel":
        """
        Load pipeline and threshold from artifacts_dir.
        feature_cols must be the raw input columns expected by the pipeline (pre-preprocessing).

        Args:
            artifacts_dir (PathLike): PathLike object with artifacts.
            feature_cols (Sequency[str]): Feature columns.
            pipeline_name (str): sklearn Pipeline object name. Defaults to 'pipeline.joblib'.
            threshold_name (str): Threshold object name. Defaults to 'threshold.joblib'

        Returns:
            PredictiveMaintenanceModel instance

        Raises:
            FileNotFoundError: If missing pipeline and/or threshold artifact.
        """
        artifacts_dir = Path(artifacts_dir)
        pipeline_path = artifacts_dir / pipeline_name
        threshold_path = artifacts_dir / threshold_name

        if not pipeline_path.exists():
            raise FileNotFoundError(f"Missing pipeline artifact: {pipeline_path}")
        if not threshold_path.exists():
            raise FileNotFoundError(f"Missing threshold artifact: {threshold_path}")

        pipeline = joblib.load(pipeline_path)
        threshold = float(joblib.load(threshold_path))


        return cls(
            pipeline=pipeline,
            threshold=threshold,
            feature_cols=list(feature_cols),
            artifacts_dir=artifacts_dir,
        )


    # Validation / preparation
    def _ensure_dataframe(self, X: Union[pd.DataFrame, Dict[str, Any]]) -> pd.DataFrame:
        if isinstance(X, dict):
            X = pd.DataFrame([X])
        if not isinstance(X, pd.DataFrame):
            raise TypeError("X must be a pandas DataFrame or a dict-like single record.")
        return X

    def validate_input(self, X: Union[pd.DataFrame, Dict[str, Any]]) -> pd.DataFrame:
        """
        Ensures required columns exist and returns a safe copy with only feature_cols.

        Args:
            X (DictLike): dict-like object.

        Returns:
            Validated data.

        Raises:
            ValueError: If missing required feature columns.
        """
        X = self._ensure_dataframe(X)

        missing = [c for c in self.feature_cols if c not in X.columns]
        if missing:
            raise ValueError(f"Missing required feature columns: {missing}")

        # Keep only required cols
        Xv = X.loc[:, self.feature_cols].copy()

        return Xv


    # Inference
    def predict_proba(self, X: Union[pd.DataFrame, Dict[str, Any]]) -> np.ndarray:
        """Calculates failure probabilities.

        Args: 
            X (DictLike): dict-like object with data to be predicted.

        Returns:
            np.asarray with failure probabilities.
        """
        Xv = self.validate_input(X)
        proba = self.pipeline.predict_proba(Xv)[:, 1]
        return np.asarray(proba)

    def predict(self, X: Union[pd.DataFrame, Dict[str, Any]]) -> pd.DataFrame:
        """
        Returns a DataFrame with proba + alert label using stored threshold.
        """
        Xv = self.validate_input(X)
        proba = self.pipeline.predict_proba(Xv)[:, 1]
        alert = (proba >= self.threshold).astype(int)

        out = pd.DataFrame(
            {
                "proba_failure": proba.astype(float),
                "alert": alert.astype(int),
                "threshold": float(self.threshold),
            },
            index=Xv.index,
        )
        return out



    # Explainability (SHAP)
    def _get_preprocessor_and_model(self):
        try:
            preproc = self.pipeline.named_steps["preprocessor"]
            model = self.pipeline.named_steps["model"]
        except Exception as e:
            raise KeyError(
                "Pipeline must contain named steps: 'preprocessor' and 'model'."
            ) from e
        return preproc, model

    def _build_shap_background(
        self,
        X_background: pd.DataFrame,
        *,
        max_background: int = 500,
        random_state: int = 0,
    ) -> pd.DataFrame:
        """
        SHAP background is used to estimate expected values.
        We subsample to keep it lightweight.
        """
        Xb = self.validate_input(X_background)
        if len(Xb) > max_background:
            Xb = Xb.sample(n=max_background, random_state=random_state)
        return Xb

    def explain(
        self,
        X: Union[pd.DataFrame, Dict[str, Any]],
        *,
        X_background: Optional[pd.DataFrame] = None,
        max_background: int = 500,
        check_additivity: bool = False,
    ):
        """Calculates SHAP values for the provided input data.

        This method uses a lazy-loading strategy for the SHAP explainer and 
        caches the processed background distribution to speed up subsequent calls.

        Args:
            X: Input data (DataFrame or Dictionary) to explain.
            X_background: Representative training data used to initialize 
                the SHAP explainer (required on the first call). Defaults to None.
            max_background: Maximum number of samples to use for the 
                background distribution to optimize performance. Defaults to 500.
            check_additivity: Whether to check if the sum of SHAP values 
                matches the model output. Defaults to False.

        Returns:
            A shap.Explanation object containing the feature contributions 
            for the input data.

        Raises:
            ValueError: If X_background is not provided during the first 
                call and no cached background exists.
            KeyError: If the internal pipeline is missing necessary steps.
        """
        import shap  # lazy import
        from scipy import sparse

        Xv = self.validate_input(X)
        preproc, model = self._get_preprocessor_and_model()

        # Prepare / cache background
        if X_background is not None:
            self._X_background = self._build_shap_background(
                X_background, max_background=max_background
            )

        if self._X_background is None:
            raise ValueError(
                "X_background is required on first call to explain(). "
                "Pass X_train (or a representative sample) once."
            )

        # Transform background + X
        Xb_tr = preproc.transform(self._X_background)
        Xt_tr = preproc.transform(Xv)

        feature_names = list(preproc.get_feature_names_out())
        self._feature_names_out = feature_names

        # Convert to dense if needed
        if sparse.issparse(Xb_tr):
            Xb_tr = Xb_tr.toarray()
        if sparse.issparse(Xt_tr):
            Xt_tr = Xt_tr.toarray()

        Xb_shap = pd.DataFrame(Xb_tr, columns=feature_names)
        Xt_shap = pd.DataFrame(Xt_tr, columns=feature_names, index=Xv.index)

        # Build explainer once (cached)
        if self._shap_explainer is None:
            self._shap_explainer = shap.Explainer(model, Xb_shap)

        if isinstance(self._shap_explainer, shap.explainers.Linear):
            shap_values = self._shap_explainer(Xt_shap)
        else:
            shap_values = self._shap_explainer(Xt_shap, check_additivity=check_additivity)
        return shap_values

    # Convenience
    def explain_one(
        self,
        X_row: Union[pd.DataFrame, Dict[str, Any]],
        *,
        X_background: Optional[pd.DataFrame] = None,
        max_display: int = 15,
    ) -> None:
        """Generates and displays a SHAP waterfall plot for a single observation.

        This is a convenience method that combines explanation calculation 
        and visualization into one step.

        Args:
            X_row: A single observation (as a DataFrame row or dictionary).
            X_background: Training data for the explainer initialization 
                (if not already cached). Defaults to None.
            max_display: Maximum number of features to show in the 
                waterfall plot. Defaults to 15.

        Returns:
            None: Renders a matplotlib figure directly.

        Raises:
            ValueError: If X_row contains more than one observation.
        """
        import shap

        Xv = self._ensure_dataframe(X_row)
        if len(Xv) != 1:
            raise ValueError("explain_one expects exactly one row.")

        shap_values = self.explain(
            Xv, X_background=X_background, max_background=500, check_additivity=False
        )
        shap.plots.waterfall(shap_values[0], max_display=max_display, show=True)