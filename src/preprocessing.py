from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

def build_preprocessor(cfg: dict, scale_numeric: bool = True) ->ColumnTransformer:
    """Builds preprocessor object.

    Args:
        cfg (dict): dict with features information.
        scale_numeric (bool): Flag used for LogisticRegressionClassifier only. Defaults to True.

    Returns:
        sklearn Preprocessor object.
    """
    num_cols = cfg['features']['numerical']
    cat_cols = cfg['features']['categorical']

    categorical_transformer = OneHotEncoder(handle_unknown='ignore')
    if scale_numeric:
        numerical_transformer = StandardScaler()
    else:
        numerical_transformer = "passthrough"

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numerical_transformer, num_cols),
            ("cat", categorical_transformer, cat_cols),
        ]
    )
    
    return preprocessor