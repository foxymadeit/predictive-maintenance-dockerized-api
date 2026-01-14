import joblib
import numpy as np
import json
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Optional

def save_split(
    split_dir: Path,
    test_idx: np.ndarray,
    y_test: np.ndarray,
    X_test: Optional[pd.DataFrame] = None,
    meta: Optional[dict] = None
) -> None:
    """Saves data split to specified directory.

    Args:
        split_dir (Path): Directory Path.
        test_idx (np.ndarray): Index object of the test data split.
        y_test (np.ndarray): y_test split of the data.
        X_test (Optional[pd.DataFrame]): X_test split of the data. Deafults to None.
        meta (Optional[dict]): Any additional information. Defaults to None.

    Returns:
        None: Creates specified folder by given Path and saves given data.
    """
    
    split_dir.mkdir(parents=True, exist_ok=True)

    np.save(split_dir / "test_idx.npy", test_idx)
    np.save(split_dir / "y_test.npy", y_test)

    if X_test is not None:
        X_test.to_parquet(split_dir / "X_test.parquet")

    if meta is not None:
        with open(split_dir / "meta.json", "w") as f:
            json.dump(meta, f, indent=4)



def load_split(split_dir: Path) -> Dict[str, Any]:
    """Load saved test split artifacts.

    Args:
        split_dir (Path): Path to the directory from which to load the data.

    Returns:
        dict with keys:
            - test_idx: np.ndarray
            - y_test: np.ndarray
            - meta: dict
    """
    split_dir = Path(split_dir)

    test_idx = np.load(split_dir / "test_idx.npy")
    y_test = np.load(split_dir / "y_test.npy")

    meta_path = split_dir / "meta.json"
    if meta_path.exists():
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
    else:
        meta = {}

    return {
        "test_idx": test_idx,
        "y_test": y_test,
        "meta": meta
    }



def save_artifacts(
        model_dir: Path,
        model_name: str,
        threshold: float,
        pipeline: Any,
        classifier_results: Dict[str, Any],
        proba_test: np.ndarray,
        threshold_sweep: Optional[pd.DataFrame] = None,
) -> None:
    """Save all inference-time artifacts for a trained classifier.

    Args:
        model_dir (Path): New directory Path.
        model_name (str): Name of the model.
        threshold (float): Selected model threshold.
        pipeline (Any): sklearn.Pipeline object.
        classifier_results (Dict[str, Any]): dict with classifier results.
        proba_test (np.ndarray): NumPy ndarray containing probability values.
        theshold_sweep (Optinal[pd.DataFrame]): Metrics at different thresholds.

    Saves:
      - pipeline.joblib
      - threshold.joblib
      - metrics.json
      - proba_test.npy
      - threshold_sweep.csv (optional)
    """
    
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    metrics_json = {
    "model": model_name, "threshold": float(threshold)}

    for k, v in classifier_results.items():
        if isinstance(v, (np.integer,)):
            metrics_json[k] = int(v)
        elif isinstance(v, (np.floating,)):
            metrics_json[k] = float(v)
        elif isinstance(v, np.ndarray):
            metrics_json[k] = v.tolist()
        elif isinstance(v, Path):
            metrics_json[k] = str(v)
        else:
            metrics_json[k] = v


    joblib.dump(pipeline, model_dir /  "pipeline.joblib")
    joblib.dump(float(threshold), model_dir /  "threshold.joblib")

    with open(model_dir /  "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics_json, f, indent=4)

    np.save(model_dir / "proba_test.npy", np.asarray(proba_test))
    
    if threshold_sweep is not None:
        threshold_sweep.to_csv(model_dir / "threshold_sweep.csv", index=False)