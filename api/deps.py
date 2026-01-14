from functools import lru_cache
import pandas as pd

from src.config import load_config
from src.paths import PROJECT_ROOT
from src.predictive_model import PredictiveMaintenanceModel
from src.data_loader import load_processed_data
from src.artifacts_io import load_split


@lru_cache(maxsize=1)
def get_model() -> PredictiveMaintenanceModel:
    """Used to keep PredictiveMaintenaceModel instance object in cash.
    """
    cfg = load_config()

    final_dir = PROJECT_ROOT / cfg['paths']['final_dir']
    feature_cols = cfg["features"]["numerical"] + cfg["features"]["categorical"]

    return PredictiveMaintenanceModel.load(
        artifacts_dir=final_dir,
        feature_cols=feature_cols,
    )



@lru_cache(maxsize=1)
def get_background_df() -> pd.DataFrame:
    """
    Returns X_train (raw features) to be used as SHAP background.
    Cached in memory.
    """
    cfg = load_config()
    df = load_processed_data()

    split_dir = PROJECT_ROOT / cfg["paths"]["split_dir"]
    split = load_split(split_dir)
    test_idx = split["test_idx"]

    feature_cols = cfg["features"]["numerical"] + cfg["features"]["categorical"]
    X_train = df.drop(index=test_idx).loc[:, feature_cols].copy()
    
    return X_train.sample(n=min(500, len(X_train)), random_state=0)