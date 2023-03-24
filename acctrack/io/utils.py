from pathlib import Path
from typing import Union
import pickle

import pandas as pd

def save_data(df, filename: Union[str, Path]):
    """Save the dataframe to a file"""
    if isinstance(df, pd.DataFrame):
        df.to_parquet(filename, compression='gzip')
    else:
        with open(filename, "wb") as f:
            pickle.dump(df, f)

def load_data(filename: Union[str, Path]):
    """Load the dataframe from a file"""
    if isinstance(filename, Path):
        filename = str(filename)

    if filename.endswith(".pkl"):
        with open(filename, "rb") as f:
            return pickle.load(f)
    else:
        return pd.read_parquet(filename)
