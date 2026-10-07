from __future__ import annotations

from typing import Iterable
import pandas as pd

from .forecasting import evaluate_and_forecast


def aggregate_annual_series(df: pd.DataFrame) -> pd.DataFrame:
    """Build a complete annual publication/citation series from canonical records."""
    if "year" not in df.columns:
        return pd.DataFrame(columns=["year", "publication_count", "citation_count"])
    work = df.copy()
    work["year"] = pd.to_numeric(work["year"], errors="coerce")
    work = work.dropna(subset=["year"])
    if work.empty:
        return pd.DataFrame(columns=["year", "publication_count", "citation_count"])
    work["year"] = work["year"].astype(int)
    citations = pd.to_numeric(work.get("citations", 0), errors="coerce").fillna(0)
    work = work.assign(_citations=citations)
    annual = (
        work.groupby("year", as_index=False)
        .agg(publication_count=("year", "size"), citation_count=("_citations", "sum"))
        .sort_values("year")
    )
    full_years = pd.DataFrame({"year": range(int(annual.year.min()), int(annual.year.max()) + 1)})
    annual = full_years.merge(annual, on="year", how="left").fillna(0)
    annual["publication_count"] = annual["publication_count"].astype(int)
    annual["citation_count"] = annual["citation_count"].round(0).astype(int)
    return annual


def run_forecasting(
    df: pd.DataFrame,
    models: Iterable[str],
    test_size: int = 5,
    horizon: int = 5,
    lags: int = 3,
    random_state: int = 42,
) -> dict:
    annual = aggregate_annual_series(df)
    minimum = test_size + lags + 1
    if len(annual) < minimum:
        raise ValueError(
            f"Forecasting needs at least {minimum} annual observations for the selected "
            f"validation window ({test_size}) and lag count ({lags}). Found {len(annual)}."
        )
    common = dict(
        models=list(models),
        test_size=int(test_size),
        horizon=int(horizon),
        lags=int(lags),
        random_state=int(random_state),
    )
    publications = evaluate_and_forecast(annual, "publication_count", **common)
    citations = evaluate_and_forecast(annual, "citation_count", **common)
    return {"annual": annual, "publications": publications, "citations": citations}
