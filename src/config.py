import yaml
from src.paths import CONFIG_DIR


def load_config() -> dict:
    """Loads YAML config file
    """
    with open(CONFIG_DIR / "config.yaml", "r") as f:
        return yaml.safe_load(f)
    