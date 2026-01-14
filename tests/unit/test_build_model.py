from src.models import build_model
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
import pytest

@pytest.mark.parametrize("name, expected_class", [
    ("logreg", LogisticRegression),
    ("rf", RandomForestClassifier),
    ("random_forest", RandomForestClassifier),
    ("hist_gb", HistGradientBoostingClassifier),
    ("hgb", HistGradientBoostingClassifier),
])


def test_build_model_ok(name, expected_class):
    model = build_model(name, random_state=42)
    assert isinstance(model, expected_class)
    assert model.random_state == 42




@pytest.mark.parametrize("invalid_name", [
    "logregression",
    "rf_classifier",
    "random_tree_forest",
    "hist_gboost",
    " ",           
    "unknown_ml",  
])
def test_build_model_not_ok(invalid_name):
    # ValueError is expected for any of the given parameters
    with pytest.raises(ValueError, match="Unknown model_name"):
        build_model(invalid_name, random_state=42)




def test_build_model_invalid_params():
    # Trying to set `max_depth` parametr for LogisticRegressor Classifier
    # instead of forest-type model
    with pytest.raises(TypeError, match="Invalid parameters for model 'logreg'"):
        build_model("logreg", max_depth=10)