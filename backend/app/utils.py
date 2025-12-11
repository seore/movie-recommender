from __future__ import annotations

from pathlib import Path
from typing import Iterable, Set

import numpy as np
import pandas as pd
from pandas.errors import EmptyDataError


def normalize_scores(arr: np.ndarray) -> np.ndarray:
    """Clamp negatives to 0 and scale to [0, 1] if possible."""
    arr = arr.copy()
    arr[arr < 0] = 0
    max_val = arr.max()
    if max_val == 0:
        return arr
    return arr / max_val


def load_csv_with_required_columns(
    path: Path, required_columns: Iterable[str], friendly_name: str
) -> pd.DataFrame:
    """
    Load a CSV file and verify that required columns are present.
    Raises a helpful error if the file is empty or malformed.
    """
    required: Set[str] = set(required_columns)
    try:
        df = pd.read_csv(path)
    except FileNotFoundError:
        raise FileNotFoundError(f"{friendly_name} file not found at {path}")
    except EmptyDataError:
        raise ValueError(
            f"{friendly_name} at {path} is empty. "
            f"Make sure you have generated it from the MovieLens dataset."
        )

    if not required.issubset(df.columns):
        raise ValueError(
            f"{friendly_name} at {path} must contain columns {required}, "
            f"but found {list(df.columns)}"
        )

    return df
