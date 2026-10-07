import pandas as pd
from .utils import norm_text, norm_key


def clean_data(df, cfg):
    x = df.copy()
    for c in x.columns:
        x[c] = x[c].map(norm_text)
    x["year"] = pd.to_numeric(x["year"].str.extract(r"(\d{4})")[0], errors="coerce").astype("Int64")
    x["citations"] = pd.to_numeric(x["citations"].str.replace(",", "", regex=False), errors="coerce").fillna(0).astype(int)
    x["doi_norm"] = x["doi"].map(norm_key)
    x["title_norm"] = x["title"].map(norm_key)

    sy, ey = cfg.get("start_year"), cfg.get("end_year")
    if sy:
        x = x[x.year >= int(sy)]
    if ey:
        x = x[x.year <= int(ey)]

    min_title = int(cfg.get("minimum_title_length", 1))
    if min_title > 0:
        # Keep records without titles because some exports omit title metadata.
        x = x[(x.title_norm.str.len() >= min_title) | (x.title_norm == "")].copy()

    x["database_sources"] = x["database"]
    if not cfg.get("deduplicate", True):
        return x.reset_index(drop=True), {
            "raw_records": len(df),
            "clean_records": len(x),
            "duplicates_removed": 0,
        }

    x["dedup_key"] = x.apply(
        lambda r: "doi:" + r.doi_norm if r.doi_norm else ("title:" + r.title_norm if r.title_norm else f"row:{r.name}"),
        axis=1,
    )
    rows = []
    for _, g in x.groupby("dedup_key", sort=False):
        score = g.astype(str).map(len).sum(axis=1)
        r = g.loc[score.idxmax()].copy()
        r["database_sources"] = "; ".join(sorted(set(g.database)))
        r["input_file"] = "; ".join(sorted(set(g.input_file)))
        r["citations"] = g.citations.max()
        rows.append(r)
    out = pd.DataFrame(rows).reset_index(drop=True)
    return out, {"raw_records": len(df), "clean_records": len(out), "duplicates_removed": len(x) - len(out)}
