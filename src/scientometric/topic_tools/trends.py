from __future__ import annotations

from typing import List, Tuple

import numpy as np
import pandas as pd
from scipy.stats import linregress

from .models import TopicModelResult


def create_document_topic_table(
    result: TopicModelResult,
    source_df: pd.DataFrame,
) -> pd.DataFrame:
    topic_columns = [
        f"topic_{i + 1}_weight"
        for i in range(result.document_topic_matrix.shape[1])
    ]
    weights = pd.DataFrame(
        result.document_topic_matrix,
        columns=topic_columns,
    )

    output = pd.concat(
        [
            source_df[["document_id", "publication_year"]].reset_index(drop=True),
            weights,
        ],
        axis=1,
    )
    output["dominant_topic"] = (
        np.argmax(result.document_topic_matrix, axis=1) + 1
    )
    output["dominant_topic_weight"] = np.max(
        result.document_topic_matrix, axis=1
    )
    return output


def calculate_yearly_topic_prevalence(
    document_topics: pd.DataFrame,
) -> pd.DataFrame:
    valid = document_topics.dropna(subset=["publication_year"]).copy()
    if valid.empty:
        return pd.DataFrame()

    topic_columns = [
        c for c in valid.columns
        if c.startswith("topic_") and c.endswith("_weight")
    ]

    yearly = (
        valid.groupby("publication_year")[topic_columns]
        .mean()
        .reset_index()
        .sort_values("publication_year")
    )
    return yearly


def calculate_trending_topics(
    yearly_prevalence: pd.DataFrame,
    topic_labels: List[str],
    latest_year_window: int,
) -> pd.DataFrame:
    if yearly_prevalence.empty:
        return pd.DataFrame(
            columns=[
                "topic_id", "topic_label", "slope",
                "recent_prevalence", "trend_score", "trend_direction"
            ]
        )

    topic_columns = [
        c for c in yearly_prevalence.columns
        if c.startswith("topic_") and c.endswith("_weight")
    ]

    recent = yearly_prevalence.tail(latest_year_window)
    records = []

    for idx, column in enumerate(topic_columns):
        x = recent["publication_year"].astype(float).to_numpy()
        y = recent[column].astype(float).to_numpy()

        slope = linregress(x, y).slope if len(x) >= 2 else 0.0
        recent_prevalence = float(y.mean()) if len(y) else 0.0
        trend_score = float(slope * recent_prevalence)

        if slope > 0.001:
            direction = "Increasing"
        elif slope < -0.001:
            direction = "Declining"
        else:
            direction = "Stable"

        records.append({
            "topic_id": idx + 1,
            "topic_label": topic_labels[idx],
            "slope": slope,
            "recent_prevalence": recent_prevalence,
            "trend_score": trend_score,
            "trend_direction": direction,
        })

    return pd.DataFrame(records).sort_values(
        "trend_score", ascending=False
    ).reset_index(drop=True)
