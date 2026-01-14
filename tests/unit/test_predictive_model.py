from src.predictive_model import PredictiveMaintenanceModel
from sklearn.pipeline import Pipeline
import pytest
from unittest.mock import MagicMock
from pathlib import Path
import pandas as pd



def test_model_load_and_predict(temp_artifacts_path, cfg):
    """
    Test building the model exemplar of PredictiveMaintenanceModel
    ==============================================================
    Make sure the pipeline is trained before asserting the `predict` method results
    """
    feature_cols = cfg['features']['numerical'] + cfg['features']['categorical']

    model = PredictiveMaintenanceModel.load(
        artifacts_dir=temp_artifacts_path,
        feature_cols=feature_cols
        )
    
    assert model.threshold == 0.41
    assert isinstance(model.pipeline, Pipeline)
    assert hasattr(model.pipeline, "predict_proba")
    assert len(model.feature_cols) > 0
    assert model.artifacts_dir == temp_artifacts_path

    sample_input = {col: 1.0 for col in feature_cols}
    sample_input['Type'] = 'L'
    
    result = model.predict(sample_input)
    assert "proba_failure" in result.columns
    assert "alert" in result.columns
    assert "threshold" in result.columns



def test_model_predict_missing_column(temp_artifacts_path, cfg):
    feature_cols = cfg['features']['numerical'] + cfg['features']['categorical']
    model = PredictiveMaintenanceModel.load(temp_artifacts_path, feature_cols=feature_cols)

    # Removing one required column
    incomplete_input = {col: 1.0 for col in feature_cols[:-1]} 
    
    with pytest.raises(ValueError, match="Missing required feature columns"):
        model.predict(incomplete_input)




def test_model_invalid_threshold_post_init():
    with pytest.raises(ValueError, match='Invalid threshold'):
        PredictiveMaintenanceModel(
            pipeline=MagicMock(),
            threshold=1.05,
            feature_cols=['test'],
            artifacts_dir=Path('.')
        )


def test_model_explain_smoke(temp_artifacts_path, cfg):
    # 1) Loading model
    feature_cols = cfg['features']['numerical'] + cfg['features']['categorical']
    model = PredictiveMaintenanceModel.load(temp_artifacts_path, feature_cols=feature_cols)

    # 2. Create backgroung X
    bg_data = []
    for _ in range(5):
        row = {col: 1.0 for col in cfg['features']['numerical']}
        row['Type'] = 'L'
        bg_data.append(row)
    X_bg = pd.DataFrame(bg_data)

    # 3. Example data from DataFrame
    sample = X_bg.iloc[[0]].copy()
    


    # Initialize explainer and cache X_background
    shap_values = model.explain(sample, X_background=X_bg)


    assert shap_values is not None
    assert model._shap_explainer is not None
    assert model._X_background is not None
    

    # Ensure that quantity of SHAP-values are matching columns quantity
    # after OneHotEncoding (5 numerical + categorical columns)
    n_features_out = len(model._feature_names_out)
    assert shap_values.values.shape[1] == n_features_out


    # Check second init (without X_background)
    shap_values_2 = model.explain(sample)
    assert shap_values_2 is not None