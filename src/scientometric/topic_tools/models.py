from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import time

import numpy as np
from scipy.special import softmax
from sklearn.cluster import KMeans
from sklearn.decomposition import (
    LatentDirichletAllocation,
    NMF,
    TruncatedSVD,
)
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import normalize


@dataclass
class TopicModelResult:
    name: str
    topic_term_matrix: np.ndarray
    document_topic_matrix: np.ndarray
    feature_names: np.ndarray
    runtime_seconds: float
    model_object: object
    vector_matrix: object
    vectorizer: object


def _safe_components(num_topics: int, matrix) -> int:
    upper = min(matrix.shape[0] - 1, matrix.shape[1] - 1)
    return max(2, min(num_topics, upper))


def fit_lda(
    texts: List[str],
    num_topics: int,
    min_df: int,
    max_df: float,
    max_features: int,
    random_state: int,
) -> TopicModelResult:
    vectorizer = CountVectorizer(
        min_df=min_df,
        max_df=max_df,
        max_features=max_features,
        ngram_range=(1, 2),
    )
    matrix = vectorizer.fit_transform(texts)
    components = _safe_components(num_topics, matrix)

    model = LatentDirichletAllocation(
        n_components=components,
        learning_method="batch",
        max_iter=30,
        random_state=random_state,
        evaluate_every=-1,
        n_jobs=-1,
    )

    start = time.perf_counter()
    doc_topic = model.fit_transform(matrix)
    runtime = time.perf_counter() - start

    return TopicModelResult(
        name="LDA",
        topic_term_matrix=model.components_,
        document_topic_matrix=doc_topic,
        feature_names=vectorizer.get_feature_names_out(),
        runtime_seconds=runtime,
        model_object=model,
        vector_matrix=matrix,
        vectorizer=vectorizer,
    )


def fit_nmf(
    texts: List[str],
    num_topics: int,
    min_df: int,
    max_df: float,
    max_features: int,
    random_state: int,
) -> TopicModelResult:
    vectorizer = TfidfVectorizer(
        min_df=min_df,
        max_df=max_df,
        max_features=max_features,
        ngram_range=(1, 2),
        sublinear_tf=True,
    )
    matrix = vectorizer.fit_transform(texts)
    components = _safe_components(num_topics, matrix)

    model = NMF(
        n_components=components,
        init="nndsvda",
        random_state=random_state,
        max_iter=500,
        l1_ratio=0.1,
    )

    start = time.perf_counter()
    doc_topic = model.fit_transform(matrix)
    runtime = time.perf_counter() - start

    return TopicModelResult(
        name="NMF",
        topic_term_matrix=model.components_,
        document_topic_matrix=doc_topic,
        feature_names=vectorizer.get_feature_names_out(),
        runtime_seconds=runtime,
        model_object=model,
        vector_matrix=matrix,
        vectorizer=vectorizer,
    )


def fit_lsa(
    texts: List[str],
    num_topics: int,
    min_df: int,
    max_df: float,
    max_features: int,
    random_state: int,
) -> TopicModelResult:
    vectorizer = TfidfVectorizer(
        min_df=min_df,
        max_df=max_df,
        max_features=max_features,
        ngram_range=(1, 2),
        sublinear_tf=True,
    )
    matrix = vectorizer.fit_transform(texts)
    components = _safe_components(num_topics, matrix)

    model = TruncatedSVD(
        n_components=components,
        random_state=random_state,
        n_iter=10,
    )

    start = time.perf_counter()
    transformed = model.fit_transform(matrix)
    runtime = time.perf_counter() - start

    doc_topic = softmax(np.abs(transformed), axis=1)
    topic_term = np.abs(model.components_)

    return TopicModelResult(
        name="LSA",
        topic_term_matrix=topic_term,
        document_topic_matrix=doc_topic,
        feature_names=vectorizer.get_feature_names_out(),
        runtime_seconds=runtime,
        model_object=model,
        vector_matrix=matrix,
        vectorizer=vectorizer,
    )


def fit_kmeans(
    texts: List[str],
    num_topics: int,
    min_df: int,
    max_df: float,
    max_features: int,
    random_state: int,
) -> TopicModelResult:
    vectorizer = TfidfVectorizer(
        min_df=min_df,
        max_df=max_df,
        max_features=max_features,
        ngram_range=(1, 2),
        sublinear_tf=True,
    )
    matrix = vectorizer.fit_transform(texts)
    components = _safe_components(num_topics, matrix)

    model = KMeans(
        n_clusters=components,
        random_state=random_state,
        n_init=20,
        max_iter=500,
    )

    start = time.perf_counter()
    labels = model.fit_predict(matrix)
    runtime = time.perf_counter() - start

    distances = model.transform(matrix)
    similarities = 1.0 / (1.0 + distances)
    doc_topic = normalize(similarities, norm="l1", axis=1)
    topic_term = np.maximum(model.cluster_centers_, 0)

    return TopicModelResult(
        name="KMeans",
        topic_term_matrix=topic_term,
        document_topic_matrix=doc_topic,
        feature_names=vectorizer.get_feature_names_out(),
        runtime_seconds=runtime,
        model_object=model,
        vector_matrix=matrix,
        vectorizer=vectorizer,
    )


def fit_bertopic(
    texts: List[str],
    num_topics: int,
    random_state: int,
) -> TopicModelResult:
    try:
        from bertopic import BERTopic
        from sklearn.feature_extraction.text import CountVectorizer
    except ImportError as exc:
        raise ImportError(
            "BERTopic is optional. Install requirements-optional.txt."
        ) from exc

    vectorizer = CountVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        min_df=2,
    )

    model = BERTopic(
        nr_topics=num_topics,
        vectorizer_model=vectorizer,
        calculate_probabilities=True,
        verbose=False,
    )

    start = time.perf_counter()
    topics, probabilities = model.fit_transform(texts)
    runtime = time.perf_counter() - start

    topic_ids = sorted(t for t in set(topics) if t != -1)
    if not topic_ids:
        raise RuntimeError("BERTopic did not produce valid topics.")

    words = sorted({
        word
        for topic_id in topic_ids
        for word, _ in model.get_topic(topic_id)
    })
    word_index = {word: idx for idx, word in enumerate(words)}

    topic_term = np.zeros((len(topic_ids), len(words)))
    id_index = {topic_id: idx for idx, topic_id in enumerate(topic_ids)}

    for topic_id in topic_ids:
        row = id_index[topic_id]
        for word, score in model.get_topic(topic_id):
            topic_term[row, word_index[word]] = score

    if probabilities is not None and probabilities.shape[1] == len(topic_ids):
        doc_topic = probabilities
    else:
        doc_topic = np.zeros((len(texts), len(topic_ids)))
        for i, topic_id in enumerate(topics):
            if topic_id in id_index:
                doc_topic[i, id_index[topic_id]] = 1.0
            else:
                doc_topic[i] = 1.0 / len(topic_ids)

    return TopicModelResult(
        name="BERTopic",
        topic_term_matrix=topic_term,
        document_topic_matrix=doc_topic,
        feature_names=np.array(words),
        runtime_seconds=runtime,
        model_object=model,
        vector_matrix=None,
        vectorizer=vectorizer,
    )


MODEL_FUNCTIONS = {
    "lda": fit_lda,
    "nmf": fit_nmf,
    "lsa": fit_lsa,
    "kmeans": fit_kmeans,
    "bertopic": fit_bertopic,
}
