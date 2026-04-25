from __future__ import annotations

import logging
import re
from pathlib import Path

import pandas as pd

from config import ACCOUNT_NAME_ALIASES, REQUIRED_FILES


def setup_logging(level: int = logging.INFO) -> None:
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def normalize_account_name(name: str) -> str:
    normalized = re.sub(r"[^a-z0-9 ]", "", str(name).strip().lower())
    normalized = re.sub(r"\s+", " ", normalized)
    return ACCOUNT_NAME_ALIASES.get(normalized, normalized)


def validate_input_dir(input_dir: Path) -> None:
    missing = [f for f in REQUIRED_FILES if not (input_dir / f).exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing required file(s) in {input_dir}: {', '.join(missing)}"
        )


def robust_to_datetime(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce", utc=True)


def robust_to_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")
