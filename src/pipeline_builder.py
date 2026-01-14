from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.base import BaseEstimator



def build_pipeline(preprocessor: ColumnTransformer, model: BaseEstimator) -> Pipeline:
    """sklearn.Pipeline builder with a specified ColumnTransformer object and specified sklearn model.

    Args:
        preprocessor (ColumnTransformer): sklearn ColumnTransfomrer instance.
        model (BaseEstimator): sklearn BaseEstimator (LogisticRegression, RandomForest etc.)
    
    Returns:
        sklearn.Pipeline object.
    """
    return Pipeline(steps=[('preprocessor', preprocessor),
                          ('model', model)
                    ])
