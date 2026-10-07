from __future__ import annotations

from typing import Dict, List, Sequence
import itertools

import numpy as np
import pandas as pd
from scipy.spatial.distance import cosine
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)
from sklearn.preprocessing import normalize

from .models import TopicModelResult


def extract_top_words(
    result: TopicModelResult,
    top_n: int,
) -> List[List[str]]:
    topics = []
    for weights in result.topic_term_matrix:
        indices = np.argsort(weights)[::-1][:top_n]
        topics.append([str(result.feature_names[i]) for i in indices])
    return topics


def topic_diversity(topics: List[List[str]]) -> float:
    words = [word for topic in topics for word in topic]
    return len(set(words)) / max(len(words), 1)


def dominant_topic_assignments(result: TopicModelResult) -> np.ndarray:
    return np.argmax(result.document_topic_matrix, axis=1)


def calculate_coherence(
    texts: Sequence[str],
    topics: List[List[str]],
) -> float:
    try:
        from gensim.corpora import Dictionary
        from gensim.models import CoherenceModel
    except ImportError:
        return float("nan")

    tokenized = [text.split() for text in texts]
    dictionary = Dictionary(tokenized)

    valid_topics = [
        [word for word in topic if word in dictionary.token2id]
        for topic in topics
    ]
    valid_topics = [topic for topic in valid_topics if len(topic) >= 2]

    if not valid_topics:
        return float("nan")

    coherence_model = CoherenceModel(
        topics=valid_topics,
        texts=tokenized,
        dictionary=dictionary,
        coherence="c_v",
        processes=1,
    )
    return float(coherence_model.get_coherence())


def topic_separation(result: TopicModelResult) -> float:
    matrix = normalize(
        np.maximum(result.topic_term_matrix, 0),
        norm="l2",
        axis=1,
    )
    similarities = matrix @ matrix.T
    mask = ~np.eye(similarities.shape[0], dtype=bool)
    if not mask.any():
        return 0.0
    return float(1.0 - similarities[mask].mean())


def clustering_metrics(result: TopicModelResult) -> Dict[str, float]:
    labels = dominant_topic_assignments(result)
    unique = np.unique(labels)

    if len(unique) < 2 or len(unique) >= len(labels):
        return {
            "silhouette": float("nan"),
            "davies_bouldin": float("nan"),
            "calinski_harabasz": float("nan"),
        }

    features = result.document_topic_matrix
    return {
        "silhouette": float(
            silhouette_score(features, labels, metric="euclidean")
        ),
        "davies_bouldin": float(
            davies_bouldin_score(features, labels)
        ),
        "calinski_harabasz": float(
            calinski_harabasz_score(features, labels)
        ),
    }


def topic_confidence(result: TopicModelResult) -> float:
    return float(
        np.mean(np.max(result.document_topic_matrix, axis=1))
    )


def evaluate_model(
    result: TopicModelResult,
    texts: Sequence[str],
    top_n: int,
) -> Dict[str, float]:
    topics = extract_top_words(result, top_n)
    cluster_scores = clustering_metrics(result)

    metrics = {
        "model": result.name,
        "num_topics": len(topics),
        "coherence_cv": calculate_coherence(texts, topics),
        "topic_diversity": topic_diversity(topics),
        "topic_separation": topic_separation(result),
        "mean_topic_confidence": topic_confidence(result),
        "silhouette": cluster_scores["silhouette"],
        "davies_bouldin": cluster_scores["davies_bouldin"],
        "calinski_harabasz": cluster_scores["calinski_harabasz"],
        "runtime_seconds": result.runtime_seconds,
    }
    return metrics


def rank_models(metrics_df: pd.DataFrame) -> pd.DataFrame:
    df = metrics_df.copy()

    higher_better = [
        "coherence_cv",
        "topic_diversity",
        "topic_separation",
        "mean_topic_confidence",
        "silhouette",
        "calinski_harabasz",
    ]
    lower_better = ["davies_bouldin", "runtime_seconds"]

    rank_columns = []

    for column in higher_better:
        if column in df:
            rank_col = f"{column}_rank"
            df[rank_col] = df[column].rank(
                ascending=False, method="average", na_option="bottom"
            )
            rank_columns.append(rank_col)

    for column in lower_better:
        if column in df:
            rank_col = f"{column}_rank"
            df[rank_col] = df[column].rank(
                ascending=True, method="average", na_option="bottom"
            )
            rank_columns.append(rank_col)

    df["overall_rank_score"] = df[rank_columns].mean(axis=1)
    df["overall_rank"] = df["overall_rank_score"].rank(
        ascending=True, method="dense"
    ).astype(int)

    return df.sort_values(
        ["overall_rank", "overall_rank_score"]
    ).reset_index(drop=True)
