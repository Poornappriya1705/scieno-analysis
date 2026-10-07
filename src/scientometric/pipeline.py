from pathlib import Path
import yaml
import pandas as pd

from .utils import ensure_dir
from .io import load_folder
from .clean import clean_data
from .metrics import (
    overview, main_information_table, author_table, affiliation_table, sources, countries,
    keyword_table, doc_types, languages, collaboration, bradford, lotka, network,
    topic_model, database_comparison,
)
from .plots import bar, line, network_plot
from .report import build_report


def analyze_dataframe(df, cfg):
    o, annual = overview(df)
    a = author_table(df)
    aff = affiliation_table(df)
    s = sources(df)
    c = countries(df)
    k = keyword_table(df)
    dt = doc_types(df)
    lg = languages(df)
    _, collab = collaboration(df)
    br = bradford(s)
    lo = lotka(a)
    Gk, kn, ke = network(df, "author_keywords", cfg["analysis"]["min_keyword_frequency"], cfg["analysis"]["network_top_nodes"])
    Ga, an, ae = network(df, "authors", cfg["analysis"]["min_author_documents"], cfg["analysis"]["network_top_nodes"])
    topics, _ = topic_model(df, cfg["analysis"]["topics"], cfg["analysis"]["topic_top_words"], cfg["analysis"]["random_state"])
    dbc = database_comparison(df)
    return {
        "overview": o,
        "main_information": main_information_table(df),
        "annual": annual,
        "authors": a,
        "affiliations": aff,
        "sources": s,
        "countries": c,
        "keywords": k,
        "document_types": dt,
        "languages": lg,
        "collaboration_summary": collab,
        "bradford": br,
        "lotka": lo,
        "keyword_nodes": kn,
        "keyword_edges": ke,
        "author_nodes": an,
        "author_edges": ae,
        "topics": topics,
        "database_comparison": dbc,
        "_keyword_graph": Gk,
        "_author_graph": Ga,
    }


def run_pipeline(config_path, input_override=None, output_override=None):
    config_path = Path(config_path).resolve()
    root = config_path.parent.parent
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    inp = Path(input_override) if input_override else root / cfg["input"]["folder"]
    out = ensure_dir(Path(output_override) if output_override else root / cfg["report"]["output_folder"])
    figs = ensure_dir(out / "figures")
    tabs = ensure_dir(out / "tables")

    raw, files = load_folder(inp, set(cfg["input"]["allowed_extensions"]), cfg["input"].get("recursive", True))
    df, cleaning = clean_data(raw, cfg["cleaning"])
    df.to_csv(out / "cleaned_bibliographic_data.csv", index=False, encoding="utf-8-sig")

    results = analyze_dataframe(df, cfg)
    Gk = results.pop("_keyword_graph")
    Ga = results.pop("_author_graph")
    results.update({"cleaning": cleaning, "databases": set(raw.database)})

    for name, d in results.items():
        if isinstance(d, pd.DataFrame):
            d.to_excel(tabs / f"{name}.xlsx", index=False)

    figures = {
        "annual_production": str(figs / "annual_production.png"),
        "top_authors": str(figs / "top_authors.png"),
        "top_affiliations": str(figs / "top_affiliations.png"),
        "top_sources": str(figs / "top_sources.png"),
        "top_countries": str(figs / "top_countries.png"),
        "top_keywords": str(figs / "top_keywords.png"),
        "document_types": str(figs / "document_types.png"),
        "languages": str(figs / "languages.png"),
        "keyword_network": str(figs / "keyword_network.png"),
        "author_network": str(figs / "author_network.png"),
    }
    line(results["annual"], "year", "Documents", "Annual Scientific Production", figures["annual_production"])
    bar(results["authors"], "Author", "Documents", "Most Productive Authors", figures["top_authors"])
    bar(results["affiliations"], "Institution", "Documents", "Most Productive Colleges / Universities", figures["top_affiliations"])
    bar(results["sources"], "Source", "Documents", "Most Relevant Sources", figures["top_sources"])
    bar(results["countries"], "Country", "Documents", "Country Scientific Production", figures["top_countries"])
    bar(results["keywords"], "Keyword", "Frequency", "Most Frequent Keywords", figures["top_keywords"])
    bar(results["document_types"], "Document_Type", "Documents", "Document Types", figures["document_types"])
    bar(results["languages"], "Language", "Documents", "Publication Languages", figures["languages"])
    network_plot(Gk, figures["keyword_network"], "Keyword Co-occurrence Network")
    network_plot(Ga, figures["author_network"], "Co-authorship Network")

    outfile = out / cfg["report"]["filename"]
    build_report(cfg, results, figures, outfile)
    print(f"Completed. Report: {outfile}")
    return outfile
