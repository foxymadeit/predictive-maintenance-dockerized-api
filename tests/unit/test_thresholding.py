import numpy as np
import pandas as pd
from src.thresholding import apply_threshold, sweep_thresholds

def test_apply_threshold_ok():
    proba = np.array([0.1, 0.6, 0.8])
    threshold = 0.5
    result = apply_threshold(proba, threshold)
    
    assert np.array_equal(result, [0, 1, 1])


def test_sweep_manual_threshold_ok():
    y_true = np.array([1, 0, 1])
    proba = np.array([0.1, 0.6, 0.8])
    thresholds = np.array([0.1, 0.2, 0.3, 1.05])

    results = sweep_thresholds(y_true, proba, thresholds)
    assert isinstance(results, pd.DataFrame)
    assert "threshold" in results 
    assert isinstance(results["threshold"][0], float)
    assert "precision" in results
    assert isinstance(results["precision"][0], float)
    assert "recall" in results
    assert isinstance(results["recall"][0], float)
    assert "f1" in results
    assert isinstance(results["f1"][0], float)
    assert "tp" in results
    assert isinstance(results["tp"][0], np.int64)
    assert "fp" in results
    assert isinstance(results["fp"][0], np.int64)
    assert "fn" in results
    assert isinstance(results["fn"][0], np.int64)
    assert "tn" in results
    assert isinstance(results["tn"][0], np.int64)


def test_sweep_default_threshold_ok():
    y_true = np.array([1, 0, 1])
    proba = np.array([0.1, 0.6, 0.8])


    results = sweep_thresholds(y_true, proba)
    assert isinstance(results, pd.DataFrame)
    assert isinstance(results, pd.DataFrame)
    assert "threshold" in results 
    assert isinstance(results["threshold"][0], float)
    assert "precision" in results
    assert isinstance(results["precision"][0], float)
    assert "recall" in results
    assert isinstance(results["recall"][0], float)
    assert "f1" in results
    assert isinstance(results["f1"][0], float)
    assert "tp" in results
    assert isinstance(results["tp"][0], np.int64)
    assert "fp" in results
    assert isinstance(results["fp"][0], np.int64)
    assert "fn" in results
    assert isinstance(results["fn"][0], np.int64)
    assert "tn" in results
    assert isinstance(results["tn"][0], np.int64)


def test_sweep_thresholds_logic():
    y_true = np.array([1, 0])
    proba = np.array([0.8, 0.1])
    results = sweep_thresholds(y_true, proba, thresholds=[0.9])
    
    assert results.iloc[0]["tp"] == 0
    assert results.iloc[0]["fp"] == 0
    assert results.iloc[0]["fn"] == 1
    assert results.iloc[0]["tn"] == 1



def test_sweep_thresholds_zero_division():
    y_true = np.array([0, 0])
    proba = np.array([0.1, 0.2])
    results = sweep_thresholds(y_true, proba, thresholds=[0.5])
    assert results.iloc[0]["precision"] == 0.0