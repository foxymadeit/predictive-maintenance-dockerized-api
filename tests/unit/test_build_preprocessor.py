from src.preprocessing import build_preprocessor
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import pytest



def test_preprocessor_scaling_true(cfg):
    """
    Test creation of preprocessor object
    with 'scale_numeric=True'
    """
    
    preprocessor = build_preprocessor(cfg, scale_numeric=True)

    # Getting name, used transformer and numerical cols
    name, transformer ,columns = preprocessor.transformers[0]
    assert isinstance(preprocessor, ColumnTransformer)
    assert name == 'num'
    assert isinstance(transformer, StandardScaler)
    assert columns == cfg['features']['numerical']

    # Getting name, used transformer and categorical cols
    name, transformer, columns = preprocessor.transformers[1]
    assert isinstance(preprocessor, ColumnTransformer)
    assert name == 'cat'
    assert isinstance(transformer, OneHotEncoder)
    assert columns == cfg['features']['categorical']



def test_preprocessor_scaling_false(cfg):
    """
    Test creation of preprocessor object
    with 'scale_numeric=False'
    """
    preprocessor = build_preprocessor(cfg, scale_numeric=False)

    # Getting name, used transformer and numerical cols
    name, transformer, columns = preprocessor.transformers[0]
    assert isinstance(preprocessor, ColumnTransformer)
    assert name == 'num'
    assert transformer == 'passthrough'
    assert columns == cfg['features']['numerical']

    # Getting name, used transformer and categorical cols
    name, transformer, columns = preprocessor.transformers[1]
    assert isinstance(preprocessor, ColumnTransformer)
    assert name == 'cat'
    assert isinstance(transformer, OneHotEncoder)
    assert columns == cfg['features']['categorical']




def test_preprocessor_invalid_key_in_cfg_error(invalid_cfg):
    """
    Test creation of preprocessor object with
    invalid KeyName for numerical column
    ------
    KeyName must be 'numerical'
    """
    with pytest.raises(KeyError, match='numerical'):
        build_preprocessor(invalid_cfg)

