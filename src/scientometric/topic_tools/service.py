from __future__ import annotations

from typing import Iterable, List
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA

from .preprocessing import prepare_dataframe
from .models import MODEL_FUNCTIONS
from .evaluation import evaluate_model, extract_top_words, rank_models
from .trends import create_document_topic_table, calculate_yearly_topic_prevalence, calculate_trending_topics


def _topic_label(words: List[str], count: int = 4) -> str:
    return " | ".join(words[:count])


def _topics_dataframe(result, top_words: int) -> pd.DataFrame:
    topics = extract_top_words(result, top_words)
    prevalence = result.document_topic_matrix.mean(axis=0)
    rows = []
    for idx, words in enumerate(topics):
        rows.append({
            "Topic": idx + 1,
            "Topic_Label": _topic_label(words),
            "Top_Words": ", ".join(words),
            "Mean_Prevalence": float(prevalence[idx]),
        })
    return pd.DataFrame(rows)


def run_topic_models(
    df: pd.DataFrame,
    models: Iterable[str],
    text_columns: Iterable[str],
    num_topics: int = 8,
    top_words: int = 10,
    min_df: int = 2,
    max_df: float = 0.95,
    max_features: int = 10000,
    trend_window: int = 5,
    random_state: int = 42,
) -> dict:
    text_columns = [c for c in text_columns if c in df.columns]
    if not text_columns:
        raise ValueError("Select at least one available text field for topic modeling.")

    prepared = prepare_dataframe(df, text_columns=text_columns, year_column="year" if "year" in df.columns else None)
    if len(prepared) < 5:
        raise ValueError("At least five documents with usable text are required for topic modeling.")

    texts = prepared["clean_text"].tolist()
    model_outputs = {}
    metric_rows = []
    errors = []

    for key in models:
        key = str(key).lower().strip()
        if key not in MODEL_FUNCTIONS:
            errors.append(f"Unknown topic model: {key}")
            continue
        try:
            if key == "bertopic":
                result = MODEL_FUNCTIONS[key](texts=texts, num_topics=num_topics, random_state=random_state)
            else:
                result = MODEL_FUNCTIONS[key](
                    texts=texts,
                    num_topics=num_topics,
                    min_df=min_df,
                    max_df=max_df,
                    max_features=max_features,
                    random_state=random_state,
                )
            topics = _topics_dataframe(result, top_words)
            doc_topics = create_document_topic_table(result, prepared)
            # Keep useful source metadata next to the topic assignment.
            for col in ["title", "authors", "source", "year"]:
                if col in prepared.columns:
                    doc_topics[col] = prepared[col].reset_index(drop=True)
            yearly = calculate_yearly_topic_prevalence(doc_topics)
            labels = topics["Topic_Label"].tolist()
            trending = calculate_trending_topics(yearly, labels, trend_window)
            metrics = evaluate_model(result, texts=texts, top_n=top_words)
            metric_rows.append(metrics)

            # Topic-to-topic similarity and a compact 2-D topic map for interactive display.
            term_matrix = np.maximum(np.asarray(result.topic_term_matrix, dtype=float), 0)
            similarity = cosine_similarity(term_matrix)
            similarity_df = pd.DataFrame(
                similarity,
                index=[f"Topic {i+1}" for i in range(similarity.shape[0])],
                columns=[f"Topic {i+1}" for i in range(similarity.shape[1])],
            )
            if term_matrix.shape[0] >= 2 and term_matrix.shape[1] >= 2:
                coords = PCA(n_components=2, random_state=random_state).fit_transform(term_matrix)
            else:
                coords = np.column_stack([np.arange(term_matrix.shape[0]), np.zeros(term_matrix.shape[0])])
            topic_map = topics[["Topic", "Topic_Label", "Mean_Prevalence"]].copy()
            topic_map["X"] = coords[:, 0]
            topic_map["Y"] = coords[:, 1]

            model_outputs[result.name] = {
                "topics": topics,
                "document_topics": doc_topics,
                "yearly_prevalence": yearly,
                "trending_topics": trending,
                "topic_similarity": similarity_df,
                "topic_map": topic_map,
            }
        except Exception as exc:
            errors.append(f"{key}: {exc}")

    if not metric_rows:
        raise RuntimeError("No selected topic model completed successfully. " + " | ".join(errors))

    comparison = rank_models(pd.DataFrame(metric_rows))
    best_model = str(comparison.iloc[0]["model"])
    return {
        "prepared": prepared,
        "comparison": comparison,
        "best_model": best_model,
        "models": model_outputs,
        "errors": errors,
    }
