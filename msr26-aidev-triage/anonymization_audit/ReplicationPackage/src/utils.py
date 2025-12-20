import logging
import pandas as pd
from pathlib import Path
from typing import Union

def setup_logger(name: str = "aidev_triage") -> logging.Logger:
    """Sets up a console logger with standard formatting."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger

def load_data(path: Union[str, Path], file_type: str = "parquet") -> pd.DataFrame:
    """Robust data loader supporting parquet and csv."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Data validation failed: {path} not found.")
    
    if file_type == "parquet" or str(path).endswith(".parquet"):
        return pd.read_parquet(path)
    elif file_type == "csv" or str(path).endswith(".csv"):
        return pd.read_csv(path)
    else:
        raise ValueError(f"Unsupported file type: {path.suffix}")

def save_data(df: pd.DataFrame, path: Union[str, Path]) -> None:
    """Saves dataframe ensuring parent directory exists."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    if str(path).endswith(".parquet"):
        df.to_parquet(path, index=False)
    elif str(path).endswith(".csv"):
        df.to_csv(path, index=False)
    else:
        # Default to parquet if unspecified
        df.to_parquet(path, index=False)
