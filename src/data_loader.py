import pandas as pd
from typing import Optional
from pathlib import Path
from src.config import load_config
from src.paths import PROJECT_ROOT

def load_raw_data(path: Optional[Path] = None) -> pd.DataFrame:
    """
    Reads and loads csv data from given path
    or from specified file path from config if path is not specified

    Args:
        path (Optional[Path]): Specified path from where to load the data.

    Returns:
        pd.DataFrame

    Raises:
        FileNotFoundError: If data is not found at specified Path
        RuntimeError: If there is any error during loading the data.
    """
    if path is None:
        cfg = load_config()
        path = PROJECT_ROOT / cfg['paths']['data_raw']
    
    try:
        return pd.read_csv(path)
    except FileNotFoundError:
        raise FileNotFoundError(f"Raw data not found at: {path}")
    except Exception as e:
        raise RuntimeError(f"Error loading raw data: {e}")



def load_processed_data(path: Optional[Path] = None) -> pd.DataFrame:
    """
    Reads and loads parquet data from given path
    or from specified file path from config if path is not specified

    Args:
        path (Optional[Path]): Specified path from where to load the data.

    Returns:
        pd.DataFrame

    Raises:
        FileNotFoundError: If data is not found at specified Path
        RuntimeError: If there is any error during loading the data.
    """
    if path is None:
        cfg = load_config()
        path = PROJECT_ROOT / cfg["paths"]["data_processed"]
        
    try:
        return pd.read_parquet(path)
    except FileNotFoundError:
        raise FileNotFoundError(f"Processed data not found at: {path}")
    except Exception as e:
        raise RuntimeError(f"Error loading processed data: {e}")