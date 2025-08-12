from pathlib import Path
from typing import Union
import logging
import pandas as pd

import config

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Alias mapping for column normalization
_ALIAS_MAP = {
    "platfrom": "platform",
    "product_name": "product",
    "review_text": "review",
    "comments": "review",
    "content": "review",
}


def load_reviews(
    filepath: Union[str, Path]
) -> pd.DataFrame:
    """
    Read an Excel review file and return a cleaned DataFrame.

    Uses config.REQUIRED_COLUMNS and config.KEEP_META to enforce schema.
    """
    # 1) Load
    df = pd.read_excel(filepath, engine="openpyxl")

    # 2) Standardize column names
    df.columns = [c.lower().strip() for c in df.columns]

    # 3) Drop unnamed columns
    df = df.loc[:, ~df.columns.str.contains(r"^unnamed", regex=True)]

    # 4) Rename aliases
    df = df.rename(columns=_ALIAS_MAP)

    # 5) Validate required columns
    missing = [c for c in config.REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # 6) Optionally drop metadata columns
    if not config.KEEP_META:
        df = df[config.REQUIRED_COLUMNS].copy()

    logger.info(
        "✅ Loaded %s rows • columns: %s",
        f"{len(df):,}",
        ", ".join(df.columns),
    )
    return df
