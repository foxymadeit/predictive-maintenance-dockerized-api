import pandas as pd
import numpy as np
import joblib
from src.artifacts_io import save_artifacts, save_split, load_split

def test_save_artifacts_ok(temp_artifacts_path, tmp_path):
    # 1) Load pipeline from fixture
    pipeline = joblib.load(temp_artifacts_path / "pipeline.joblib")
    

    # Create temp folder for the test
    test_model_dir = tmp_path / "new_model_save"
    
    classifier_results = {
        "f1": 0.85,
        "tp": np.int64(10),
        "recall": 0.9
    }
    proba_test = np.array([0.1, 0.8, 0.4])
    sweep_df = pd.DataFrame({"threshold": [0.1, 0.5], "f1": [0.4, 0.8]})

    # 2) Test `save_artifacts` fn
    save_artifacts(
        model_dir=test_model_dir,
        model_name="test_logreg",
        threshold=0.41,
        pipeline=pipeline,
        classifier_results=classifier_results,
        proba_test=proba_test,
        threshold_sweep=sweep_df
    )

    # 3) Check if the files exist in the temp folder
    assert (test_model_dir / "pipeline.joblib").exists()
    assert (test_model_dir / "metrics.json").exists()
    assert (test_model_dir / "proba_test.npy").exists()
    assert (test_model_dir / "threshold_sweep.csv").exists()


    # Additional check to know if JSON is capable of reading
    import json
    with open(test_model_dir / "metrics.json", "r") as f:
        metrics = json.load(f)
    assert metrics["model"] == "test_logreg"
    assert isinstance(metrics["tp"], int) 


def test_save_load_split_cycle(tmp_path):
    # Test combination of save & load for data split
    split_dir = tmp_path / "data_split"
    test_idx = np.array([1, 2, 3])
    y_test = np.array([0, 1, 0])
    X_test = pd.DataFrame({"feat": [1, 2, 3]})
    meta = {"source": "unit_test"}


    save_split(split_dir, test_idx, y_test, X_test=X_test, meta=meta)
    loaded = load_split(split_dir)

    assert np.array_equal(loaded["test_idx"], test_idx)
    assert np.array_equal(loaded["y_test"], y_test)
    assert loaded["meta"]["source"] == "unit_test"
    assert (split_dir / "X_test.parquet").exists()