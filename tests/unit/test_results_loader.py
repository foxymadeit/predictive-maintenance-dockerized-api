import pandas as pd
import json
from src.results_loader import load_all_metrics, load_all_threshold_sweeps

def test_load_all_metrics_integration(tmp_path):
    # Temp model dir 1
    m1_dir = tmp_path / "model_v1"
    m1_dir.mkdir()
    with open(m1_dir / "metrics.json", "w") as f:
        json.dump({"model": "rf", "f1_score": 0.8}, f)

    # Temp model dir 2
    m2_dir = tmp_path / "model_v2"
    m2_dir.mkdir()
    with open(m2_dir / "metrics.json", "w") as f:
        json.dump({"model": "lr", "f1_score": 0.7}, f)


    df = load_all_metrics(tmp_path)


    assert isinstance(df, pd.DataFrame)
    assert len(df) == 2
    assert "run_dir" in df.columns
    assert "f1_score" in df.columns
    assert set(df["run_dir"]) == {"model_v1", "model_v2"}



def test_load_all_metrics_missing_json(tmp_path):
    # Check that if the dir missing metrics.json
    empty_dir = tmp_path / "empty_model"
    empty_dir.mkdir()
    
    df = load_all_metrics(tmp_path)
    assert len(df) == 0



def test_load_all_threshold_sweeps_ok(tmp_path):
    # Create model dir and `threshold_sweep.csv` file
    m_dir = tmp_path / "model_1"
    m_dir.mkdir()
    sweep_data = pd.DataFrame({"threshold": [0.1, 0.5], "f1": [0.8, 0.9]})
    sweep_data.to_csv(m_dir / "threshold_sweep.csv", index=False)

    # Check if function is working properly
    df = load_all_threshold_sweeps(tmp_path)
    assert len(df) == 2
    assert "run_dir" in df.columns



def test_load_all_threshold_sweeps_empty(tmp_path):
    df = load_all_threshold_sweeps(tmp_path)
    assert df.empty
    assert isinstance(df, pd.DataFrame)