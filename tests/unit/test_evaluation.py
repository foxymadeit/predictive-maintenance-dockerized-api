import numpy as np
from src.evaluation import evaluate_classifier,print_results

def test_evaluate_classifier():
    y_true = np.array([1, 0, 1, 0, 0])
    y_preds = np.array([1, 1, 1, 0, 0])
    y_proba = np.array([0.53, 0.67, 0.83, 0.34, 0])

    results = evaluate_classifier(y_true, y_preds, y_proba)
    assert "accuracy" in results
    assert "precision" in results
    assert "recall" in results
    assert results["recall"] == 1.0
    assert "f1_score" in results
    assert "roc_auc" in results
    assert "pr_auc" in results
    assert "tn" in results
    assert "fp" in results
    assert "fn" in results
    assert "tp" in results



def test_print_results(capsys):
    y_true = np.array([1, 0, 1, 0, 0])
    y_preds = np.array([1, 1, 1, 0, 0])
    y_proba = np.array([0.53, 0.67, 0.83, 0.34, 0])
    results = evaluate_classifier(y_true, y_preds, y_proba)
    print_results(results)

    captured = capsys.readouterr()
    assert "accuracy" in captured.out
    assert "precision" in captured.out
    assert "recall" in captured.out
    assert "f1_score" in captured.out
    assert "roc_auc" in captured.out
    assert "pr_auc" in captured.out
    assert "tn" in captured.out
    assert "fp" in captured.out
    assert "fn" in captured.out
    assert "tp" in captured.out