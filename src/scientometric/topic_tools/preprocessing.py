from __future__ import annotations

import re
from typing import Iterable, List, Optional

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS


CUSTOM_STOPWORDS = {
    "study", "research", "paper", "article", "result", "results",
    "method", "methods", "analysis", "data", "using", "based",
    "approach", "model", "models", "proposed", "application",
    "applications", "effect", "effects", "finding", "findings",
    "objective", "objectives", "conclusion", "conclusions",
    "author", "authors", "et", "al",
}

STOPWORDS = set(ENGLISH_STOP_WORDS).union(CUSTOM_STOPWORDS)


def clean_text(text: object) -> str:
    if text is None or (isinstance(text, float) and np.isnan(text)):
        return ""

    value = str(text).lower()
    value = re.sub(r"https?://\S+|www\.\S+", " ", value)
    value = re.sub(r"\bdoi\s*:\s*\S+", " ", value)
    value = re.sub(r"[^a-zA-Z\s-]", " ", value)
    value = value.replace("-", " ")
    value = re.sub(r"\s+", " ", value).strip()

    tokens = [
        token for token in value.split()
        if len(token) >= 3 and token not in STOPWORDS
    ]
    return " ".join(tokens)


def combine_text_columns(
    df: pd.DataFrame,
    text_columns: List[str]
) -> pd.Series:
    safe_df = df[text_columns].fillna("").astype(str)
    return safe_df.apply(
        lambda row: " ".join(v for v in row if v and v != "nan"),
        axis=1,
    )


def extract_year(series: pd.Series) -> pd.Series:
    years = (
        series.astype(str)
        .str.extract(r"((?:19|20)\d{2})", expand=False)
    )
    return pd.to_numeric(years, errors="coerce").astype("Int64")


def prepare_dataframe(
    df: pd.DataFrame,
    text_columns: List[str],
    year_column: Optional[str],
) -> pd.DataFrame:
    result = df.copy()
    result["combined_text"] = combine_text_columns(result, text_columns)
    result["clean_text"] = result["combined_text"].map(clean_text)

    result = result[
        result["clean_text"].str.split().str.len().fillna(0) >= 3
    ].copy()

    if year_column:
        result["publication_year"] = extract_year(result[year_column])
    else:
        result["publication_year"] = pd.Series(
            [pd.NA] * len(result), dtype="Int64"
        )

    result.reset_index(drop=True, inplace=True)
    result["document_id"] = np.arange(1, len(result) + 1)
    return result
