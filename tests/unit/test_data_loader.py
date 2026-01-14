import pytest
from pathlib import Path
import pandas as pd
from src.data_loader import load_raw_data, load_processed_data


def test_load_raw_data_ok(tmp_path):
    # Create temp csv file
    d = tmp_path / "data"
    d.mkdir()
    file_path = d / "test.csv"
    df_original = pd.DataFrame({"a": [1], "b": [2]})
    df_original.to_csv(file_path, index=False)


    df_loaded = load_raw_data(path=file_path)


    pd.testing.assert_frame_equal(df_original, df_loaded)

def test_load_processed_data_ok(tmp_path):
    # Create temp parquet file
    file_path = tmp_path / "test.parquet"
    df_original = pd.DataFrame({"a": [1], "b": [2]})
    df_original.to_parquet(file_path)


    df_loaded = load_processed_data(path=file_path)


    pd.testing.assert_frame_equal(df_original, df_loaded)



def test_load_raw_data_not_found():
    # Create invalid path
    fake_path = Path("non_existent_path.csv")
    
    with pytest.raises(FileNotFoundError, match="Raw data not found"):
        load_raw_data(path=fake_path)

def test_load_raw_data_corrupted(tmp_path):
    # Create corrupted file
    file_path = tmp_path / "broken.csv"
    file_path.write_text("not,a,csv\n1,2,3,4,5")
    


    file_path.write_bytes(b'\x00\xFF\x00\xFF')
    
    with pytest.raises(RuntimeError, match="Error loading raw data"):
        load_raw_data(path=file_path)



def test_load_processed_data_not_found():
    # Create invalid path
    fake_path = Path("non_existent_path.csv")
    
    with pytest.raises(FileNotFoundError, match="Processed data not found"):
        load_processed_data(path=fake_path)

def test_load_processed_data_corrupted(tmp_path):
    # Create corrupted file
    file_path = tmp_path / "broken.csv"
    file_path.write_text("not,a,csv\n1,2,3,4,5")
    


    file_path.write_bytes(b'\x00\xFF\x00\xFF')
    
    with pytest.raises(RuntimeError, match="Error loading processed data"):
        load_processed_data(path=file_path)