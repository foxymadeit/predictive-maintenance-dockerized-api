import pytest
from fastapi.testclient import TestClient
from api.main import app
from src.models import build_model
from src.preprocessing import build_preprocessor
from src.pipeline_builder import build_pipeline
import joblib
import pandas as pd
import numpy as np

@pytest.fixture(scope='session')
def client():
    return TestClient(app)


@pytest.fixture
def valid_record():
    return {
        "Air temperature [K]": 300,
        "Process temperature [K]": 310,
        "Rotational speed [rpm]": 1500,
        "Torque [Nm]": 40,
        "Tool wear [min]": 100,
        "Type": "M",
    }


@pytest.fixture
def invalid_record():
    return {
        "Air temperature [K]": -1,
        "Process temperature [K]": 310,
        "Rotational speed [rpm]": 1500,
        "Torque [Nm]": 40,
        "Tool wear [min]": 100,
        "Type": "I",
    }



@pytest.fixture
def valid_columns():
    return ['Air temperature [K]',
            'Process temperature [K]',
            'Rotational speed [rpm]',
            'Torque [Nm]',
            'Tool wear [min]',
            'Type']

@pytest.fixture
def invalid_columns():
    return ['Air temperature [C]',
            'Process temperature [C]',
            'Rotational speed [rps]',
            'Torque [Nm]',
            'Tool wear [min]',
            'Types']

@pytest.fixture(scope="session")
def cfg():
    return {
        "features": {"numerical": ['Air temperature [K]',
                    'Process temperature [K]',
                    'Rotational speed [rpm]',
                    'Torque [Nm]',
                    'Tool wear [min]'],
                    "categorical": ['Type']
                    }
    }

@pytest.fixture
def invalid_cfg():
    return {
        "features": {"num": ['Air temperature [K]',
                    'Process temperature [K]',
                    'Rotational speed [rpm]',
                    'Torque [Nm]',
                    'Tool wear [min]'],
                    "cat": ['Type']
                    }
    }




@pytest.fixture(scope='session')
def temp_artifacts_path(tmp_path_factory, cfg):
    temp_dir = tmp_path_factory.mktemp("temp_artifacts")
    
    # 1) Build and train pipeline
    model = build_model('logreg', random_state=42)
    preprocessor = build_preprocessor(cfg=cfg, scale_numeric=True)
    pipeline = build_pipeline(preprocessor, model)

    X_train = pd.DataFrame([
        {col: 1.0 for col in cfg['features']['numerical']} | {'Type': 'L'},
        {col: 2.0 for col in cfg['features']['numerical']} | {'Type': 'M'}
    ])
    y_train = np.array([0, 1])
    pipeline.fit(X_train, y_train)


    # 2) Generate test prediction
    y_test = np.array([0, 1, 0])
    X_test = pd.DataFrame([
        {col: 1.1 for col in cfg['features']['numerical']} | {'Type': 'L'},
        {col: 2.1 for col in cfg['features']['numerical']} | {'Type': 'M'},
        {col: 1.2 for col in cfg['features']['numerical']} | {'Type': 'L'}
    ])
    y_proba = pipeline.predict_proba(X_test)[:, 1]


    # 3) Save all temp files in a folder
    joblib.dump(pipeline, temp_dir / "pipeline.joblib")
    joblib.dump(0.41, temp_dir / "threshold.joblib")
    joblib.dump(y_test, temp_dir / "y_test.joblib")
    joblib.dump(y_proba, temp_dir / "y_proba.joblib")
    

    return temp_dir