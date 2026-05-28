"""Shared utilities for YouTube spam detection experiments."""

from __future__ import annotations

import re
import string
from dataclasses import dataclass

import pandas as pd


TEXT_COLUMN = "CONTENT"
LABEL_COLUMN = "CLASS"


@dataclass(frozen=True)
class DatasetColumns:
    text: str = TEXT_COLUMN
    label: str = LABEL_COLUMN


def clean_comment(text: object) -> str:
    """Normalize a YouTube comment while preserving spam-relevant tokens."""
    if pd.isna(text):
        return ""

    value = str(text).lower()
    value = re.sub(r"https?://\S+|www\.\S+", " urltoken ", value)
    value = value.translate(str.maketrans("", "", string.punctuation))
    value = re.sub(r"\s+", " ", value)
    return value.strip()


def load_comment_dataset(path: str, columns: DatasetColumns | None = None) -> pd.DataFrame:
    """Load a CSV dataset and return normalized CONTENT and CLASS columns."""
    selected = columns or DatasetColumns()
    frame = pd.read_csv(path)

    missing = {selected.text, selected.label}.difference(frame.columns)
    if missing:
        expected = ", ".join(sorted(missing))
        raise ValueError(f"Missing required column(s): {expected}")

    result = frame[[selected.text, selected.label]].copy()
    result[selected.text] = result[selected.text].map(clean_comment)
    result[selected.label] = result[selected.label].astype(int)
    return result.rename(columns={selected.text: TEXT_COLUMN, selected.label: LABEL_COLUMN})
