from pathlib import Path
import yaml

from src.scientometric.io import read_one
from src.scientometric.clean import clean_data
from src.scientometric.forecasting_tools import run_forecasting
from src.scientometric.topic_tools import run_topic_models


def sample_df():
    root = Path(__file__).resolve().parents[1]
    raw = read_one(root / "examples" / "sample_scopus.csv")
    cfg = yaml.safe_load((root / "config" / "config.yaml").read_text())
    df, _ = clean_data(raw, cfg.get("cleaning", {}))
    return df


def test_forecasting_integration():
    r = run_forecasting(sample_df(), ["naive", "linear_regression"], test_size=2, horizon=2, lags=1)
    assert not r["annual"].empty
    assert "best_forecast" in r["publications"].forecasts
    assert "best_forecast" in r["citations"].forecasts


def test_topic_modeling_integration():
    r = run_topic_models(
        sample_df(), ["lda", "nmf"], ["title", "abstract", "author_keywords"],
        num_topics=3, top_words=5, min_df=1, max_features=1000,
    )
    assert r["best_model"] in r["models"]
    detail = r["models"][r["best_model"]]
    assert not detail["topics"].empty
    assert detail["topic_similarity"].shape[0] == detail["topic_similarity"].shape[1]
