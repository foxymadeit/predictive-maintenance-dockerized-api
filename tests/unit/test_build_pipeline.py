from src.models import build_model
from src.preprocessing import build_preprocessor
from src.pipeline_builder import build_pipeline
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.base import BaseEstimator
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
import pytest


@pytest.mark.parametrize("name, expected_class", 
                         [
                            ('logreg', LogisticRegression),
                            ('random_forest', RandomForestClassifier),
                            ('rf', RandomForestClassifier),
                            ('hist_gb', HistGradientBoostingClassifier),
                            ('hgb', HistGradientBoostingClassifier),
                            ('gradient_boosting', HistGradientBoostingClassifier)
                         ])

def test_build_pipeline_ok(cfg, name, expected_class):
    """
    Test pipeline integration
    """
    model = build_model(name, random_state=42)
    preprocessor = build_preprocessor(cfg, True)
    pipeline = build_pipeline(preprocessor, model)

    assert isinstance(pipeline, Pipeline)
    assert 'preprocessor' in pipeline.named_steps
    assert 'model' in pipeline.named_steps
    assert isinstance(pipeline.named_steps['preprocessor'], ColumnTransformer)
    assert isinstance(pipeline.named_steps['model'], expected_class)
    