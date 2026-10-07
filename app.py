from pathlib import Path
from io import BytesIO
import tempfile
import zipfile

import pandas as pd
import plotly.express as px
import streamlit as st
import yaml

from src.scientometric.io import load_uploaded_files
from src.scientometric.clean import clean_data
from src.scientometric.metrics import (
    overview,
    main_information_table,
    sources,
    author_table,
    affiliation_table,
    production_over_time,
    extract_institutions,
)
from src.scientometric.pipeline import analyze_dataframe
from src.scientometric.plots import bar, line, network_plot
from src.scientometric.report import build_report
from src.scientometric.utils import ensure_dir
from src.scientometric.forecasting_tools import run_forecasting
from src.scientometric.topic_tools import run_topic_models

ROOT = Path(__file__).resolve().parent
CONFIG_PATH = ROOT / "config" / "config.yaml"

st.set_page_config(page_title="Scientometric Study Generator", page_icon="📊", layout="wide")


@st.cache_data(show_spinner=False)
def load_config():
    return yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))


def to_csv_bytes(df):
    return df.to_csv(index=False).encode("utf-8-sig")


def chart_download_data(df, key):
    st.download_button(
        "Download table (CSV)",
        data=to_csv_bytes(df),
        file_name=f"{key}.csv",
        mime="text/csv",
        key=f"download_{key}",
    )



def dataframes_zip(files: dict[str, pd.DataFrame], prefix: str = "analysis") -> bytes:
    """Package named DataFrames as UTF-8 CSV files."""
    buf = BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for name, frame in files.items():
            if isinstance(frame, pd.DataFrame):
                zf.writestr(f"{prefix}/{name}.csv", frame.to_csv(index=False).encode("utf-8-sig"))
    return buf.getvalue()


def forecast_plot(annual, result, target_label):
    hist_col = result.target
    historical = annual[["year", hist_col]].rename(columns={hist_col: "Value"}).copy()
    historical["Series"] = "Historical"
    future = result.forecasts[["year", "best_forecast"]].rename(columns={"best_forecast": "Value"}).copy()
    future["Series"] = f"Forecast ({result.best_model})"
    combined = pd.concat([historical, future], ignore_index=True)
    fig = px.line(combined, x="year", y="Value", color="Series", markers=True, title=f"{target_label}: Historical and Forecast")
    fig.update_layout(xaxis_title="Year", yaxis_title=target_label)
    return fig


def topic_prevalence_long(yearly: pd.DataFrame, topics: pd.DataFrame) -> pd.DataFrame:
    if yearly.empty:
        return pd.DataFrame()
    topic_map = {f"topic_{int(r.Topic)}_weight": r.Topic_Label for _, r in topics.iterrows()}
    value_cols = [c for c in yearly.columns if c.startswith("topic_") and c.endswith("_weight")]
    long = yearly.melt(id_vars=["publication_year"], value_vars=value_cols, var_name="Topic_Key", value_name="Prevalence")
    long["Topic"] = long["Topic_Key"].map(topic_map).fillna(long["Topic_Key"])
    return long

def main_info_cards(o):
    cols = st.columns(4)
    cards = [
        ("Documents", f"{o['documents']:,}"),
        ("Sources", f"{o['sources']:,}"),
        ("Authors", f"{o['authors']:,}"),
        ("Timespan", o["timespan"]),
        ("Annual Growth Rate", f"{o['annual_growth_rate']:.2f}%" if pd.notna(o["annual_growth_rate"]) else "NA"),
        ("Average Citations / Doc", f"{o['mean_citations']:.2f}"),
        ("Co-Authors / Doc", f"{o['coauthors_per_doc']:.2f}"),
        ("International Co-authorship", f"{o['international_coauthorship_pct']:.2f}%"),
    ]
    for i, (label, value) in enumerate(cards):
        cols[i % 4].metric(label, value)


def generate_report_package(df, raw, cleaning, cfg):
    """Create the existing full Word report + tables/figures in a ZIP."""
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "scientometric_output"
        figs = ensure_dir(out / "figures")
        tabs = ensure_dir(out / "tables")

        results = analyze_dataframe(df, cfg)
        Gk = results.pop("_keyword_graph")
        Ga = results.pop("_author_graph")
        results.update({"cleaning": cleaning, "databases": set(raw.database)})

        df.to_csv(out / "cleaned_bibliographic_data.csv", index=False, encoding="utf-8-sig")
        for name, obj in results.items():
            if isinstance(obj, pd.DataFrame):
                obj.to_excel(tabs / f"{name}.xlsx", index=False)

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

        report_path = out / cfg["report"]["filename"]
        build_report(cfg, results, figures, report_path)

        buf = BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
            for p in out.rglob("*"):
                if p.is_file():
                    zf.write(p, p.relative_to(out))
        return buf.getvalue()


st.title("Scientometric Study Generator")
st.caption("Upload Scopus or Web of Science exports and generate report-ready scientometric indicators, tables and charts.")

cfg = load_config()

with st.sidebar:
    st.header("1. Upload Data")
    uploaded_files = st.file_uploader(
        "Upload Scopus / Web of Science files",
        type=["csv", "txt", "bib", "xlsx", "xls"],
        accept_multiple_files=True,
        help="You can upload one or multiple Scopus/WoS export files.",
    )
    deduplicate = st.checkbox("Deduplicate by DOI/title", value=False, help="Turn this on when combining overlapping WoS and Scopus exports.")
    min_title = st.number_input("Minimum title length", min_value=0, max_value=200, value=0, step=1)

    st.header("2. Analysis")
    page = st.radio(
        "View",
        [
            "Main Information",
            "Annual Publication Growth",
            "Source Analysis",
            "Author Profile Analysis",
            "Affiliation Analysis",
            "Time Series Forecasting",
            "Topic Modeling",
            "Generate Report",
        ],
    )

if not uploaded_files:
    st.info("Upload at least one .csv, .txt, .bib, .xlsx or .xls file using the sidebar to begin.")
    st.stop()

try:
    raw, errors = load_uploaded_files(uploaded_files)
except Exception as exc:
    st.error(f"Unable to read the uploaded data: {exc}")
    st.stop()

if errors:
    with st.expander("Files skipped during import"):
        for e in errors:
            st.warning(e)

clean_cfg = dict(cfg.get("cleaning", {}))
clean_cfg.update({"deduplicate": deduplicate, "minimum_title_length": int(min_title)})
df, cleaning = clean_data(raw, clean_cfg)

if df.empty:
    st.error("No usable bibliographic records remain after cleaning/filtering.")
    st.stop()

# Use the latest publication year as the stable reference for historical datasets.
latest_year = int(df.year.dropna().max()) if df.year.notna().any() else None
o, annual = overview(df, reference_year=latest_year)

st.success(f"Loaded {cleaning['clean_records']:,} usable records from {len(uploaded_files)} file(s).")

if page == "Main Information":
    st.subheader("Main Information of the Data")
    main_info_cards(o)
    st.caption(f"Document Average Age uses {o['reference_year']} as the reference year so a historical dataset produces reproducible results.")
    info = main_information_table(df, reference_year=latest_year)
    st.dataframe(info, use_container_width=True, hide_index=True)
    chart_download_data(info, "main_information")

    with st.expander("Preview cleaned bibliographic data"):
        st.dataframe(df.head(100), use_container_width=True)
        st.download_button("Download cleaned data (CSV)", to_csv_bytes(df), "cleaned_bibliographic_data.csv", "text/csv")

elif page == "Annual Publication Growth":
    st.subheader("Annual Publication Growth")
    if annual.empty:
        st.warning("Publication year information is not available in the uploaded data.")
    else:
        annual_show = annual.copy()
        annual_show["year"] = annual_show["year"].astype(int)
        metric_cols = st.columns(3)
        metric_cols[0].metric("Timespan", o["timespan"])
        metric_cols[1].metric("Annual Growth Rate (CAGR)", f"{o['annual_growth_rate']:.2f}%")
        metric_cols[2].metric("Peak Annual Output", f"{int(annual_show.Documents.max()):,}")

        fig = px.line(annual_show, x="year", y="Documents", markers=True, title="Annual Scientific Production")
        fig.update_layout(xaxis_title="Year", yaxis_title="Number of Publications")
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(annual_show, use_container_width=True, hide_index=True)
        chart_download_data(annual_show, "annual_publication_growth")

elif page == "Source Analysis":
    st.subheader("Source Analysis")
    source_df = sources(df)
    top_n = st.slider("Number of sources to display", 5, min(50, max(5, len(source_df))), min(20, max(5, len(source_df)))) if len(source_df) else 5
    metric = st.selectbox("Rank sources by", ["Documents", "Citations", "Mean_Citations"])
    ranked = source_df.sort_values([metric, "Documents"], ascending=False).head(top_n)
    if ranked.empty:
        st.warning("Source information is not available in the uploaded data.")
    else:
        fig = px.bar(ranked.sort_values(metric), x=metric, y="Source", orientation="h", title=f"Top {top_n} Sources by {metric.replace('_', ' ')}")
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(ranked, use_container_width=True, hide_index=True)
        chart_download_data(source_df, "source_analysis")

        growth = production_over_time(df, "source", "Source", top_n=min(10, top_n))
        if not growth.empty:
            st.markdown("#### Source Production Over Time")
            gf = px.line(growth, x="Year", y="Documents", color="Source", markers=True)
            st.plotly_chart(gf, use_container_width=True)

elif page == "Author Profile Analysis":
    st.subheader("Author Profile Analysis")
    authors = author_table(df)
    top_n = st.slider("Number of authors to display", 5, min(50, max(5, len(authors))), min(20, max(5, len(authors)))) if len(authors) else 5
    metric = st.selectbox("Rank authors by", ["Documents", "Citations", "h_index", "Mean_Citations"])
    ranked = authors.sort_values([metric, "Documents"], ascending=False).head(top_n)
    if ranked.empty:
        st.warning("Author information is not available in the uploaded data.")
    else:
        fig = px.bar(ranked.sort_values(metric), x=metric, y="Author", orientation="h", title=f"Top {top_n} Authors by {metric.replace('_', ' ')}")
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(ranked, use_container_width=True, hide_index=True)
        chart_download_data(authors, "author_profile_analysis")

        growth = production_over_time(df, "authors", "Author", top_n=min(10, top_n))
        if not growth.empty:
            st.markdown("#### Author Production Over Time")
            gf = px.line(growth, x="Year", y="Documents", color="Author", markers=True)
            st.plotly_chart(gf, use_container_width=True)

elif page == "Affiliation Analysis":
    st.subheader("Affiliation Analysis — College / University of Authors")
    st.caption("Affiliation addresses are normalized to the institutional level. Departments, cities, postal addresses and countries are not counted as separate affiliations. When both a college and its parent university are present in the same address, the university is preferred.")
    affiliations = affiliation_table(df)
    top_n = st.slider("Number of institutions to display", 5, min(50, max(5, len(affiliations))), min(20, max(5, len(affiliations)))) if len(affiliations) else 5
    metric = st.selectbox("Rank institutions by", ["Documents", "Citations", "Mean_Citations"])
    ranked = affiliations.sort_values([metric, "Documents"], ascending=False).head(top_n)
    if ranked.empty:
        st.warning("No college/university-level institution could be identified from the affiliation fields in the uploaded data.")
    else:
        # New affiliation analysis returns `Institution`. Keep compatibility
        # with older cached/results tables that may still use `Affiliation`.
        institution_col = "Institution" if "Institution" in ranked.columns else "Affiliation"
        color_col = "Institution_Type" if "Institution_Type" in ranked.columns else None
        fig = px.bar(
            ranked.sort_values(metric),
            x=metric,
            y=institution_col,
            color=color_col,
            orientation="h",
            title=f"Top {top_n} Colleges / Universities by {metric.replace('_', ' ')}",
        )
        fig.update_layout(yaxis_title="College / University", xaxis_title=metric.replace("_", " "))
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(ranked, use_container_width=True, hide_index=True)
        chart_download_data(affiliations, "affiliation_institution_analysis")

        growth = production_over_time(df, "affiliations", "Institution", splitter=extract_institutions, top_n=min(10, top_n))
        if not growth.empty:
            st.markdown("#### College / University Production Over Time")
            growth_label = "Institution" if "Institution" in growth.columns else "Affiliation"
            gf = px.line(growth, x="Year", y="Documents", color=growth_label, markers=True)
            st.plotly_chart(gf, use_container_width=True)

elif page == "Time Series Forecasting":
    st.subheader("Bibliometric Time Series Forecasting")
    st.caption("Forecast annual publication and citation counts using statistical and machine-learning models from the uploaded forecasting project.")

    available_models = {
        "Naive": "naive",
        "Exponential Smoothing": "exponential_smoothing",
        "ARIMA (1,1,1)": "arima",
        "Linear Regression": "linear_regression",
        "Random Forest": "random_forest",
        "Gradient Boosting": "gradient_boosting",
        "Support Vector Regression": "svr",
    }
    c1, c2, c3 = st.columns(3)
    horizon = c1.slider("Forecast horizon (years)", 1, 15, 5)
    test_size = c2.slider("Validation window (years)", 2, 10, 5)
    lags = c3.slider("Lag count for ML models", 1, 8, 3)
    selected_labels = st.multiselect(
        "Forecasting models",
        list(available_models),
        default=["Naive", "Exponential Smoothing", "ARIMA (1,1,1)", "Linear Regression", "Random Forest", "Gradient Boosting"],
    )

    if st.button("Run Forecasting Analysis", type="primary"):
        if not selected_labels:
            st.warning("Select at least one forecasting model.")
        else:
            with st.spinner("Evaluating forecasting models and generating future estimates..."):
                try:
                    fres = run_forecasting(
                        df,
                        models=[available_models[x] for x in selected_labels],
                        test_size=test_size,
                        horizon=horizon,
                        lags=lags,
                    )
                    st.session_state["forecast_results"] = fres
                except Exception as exc:
                    st.exception(exc)

    if "forecast_results" in st.session_state:
        fres = st.session_state["forecast_results"]
        annual_f = fres["annual"]
        pub = fres["publications"]
        cit = fres["citations"]

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Historical years", len(annual_f))
        m2.metric("Publication best model", pub.best_model.replace("_", " ").title())
        m3.metric("Citation best model", cit.best_model.replace("_", " ").title())
        m4.metric("Forecast horizon", len(pub.forecasts))

        st.markdown("#### Annual Bibliometric Series")
        st.dataframe(annual_f, use_container_width=True, hide_index=True)
        annual_long = annual_f.melt(id_vars="year", value_vars=["publication_count", "citation_count"], var_name="Metric", value_name="Count")
        st.plotly_chart(px.line(annual_long, x="year", y="Count", color="Metric", markers=True, title="Historical Publications and Citations"), use_container_width=True)

        st.markdown("#### Publication Forecast")
        st.plotly_chart(forecast_plot(annual_f, pub, "Publications"), use_container_width=True)
        st.dataframe(pub.metrics, use_container_width=True, hide_index=True)
        st.dataframe(pub.forecasts, use_container_width=True, hide_index=True)

        st.markdown("#### Citation Forecast")
        st.plotly_chart(forecast_plot(annual_f, cit, "Citations"), use_container_width=True)
        st.dataframe(cit.metrics, use_container_width=True, hide_index=True)
        st.dataframe(cit.forecasts, use_container_width=True, hide_index=True)

        forecast_zip = dataframes_zip({
            "annual_bibliometric_series": annual_f,
            "publication_model_metrics": pub.metrics,
            "publication_forecasts": pub.forecasts,
            "publication_validation_predictions": pub.test_predictions,
            "citation_model_metrics": cit.metrics,
            "citation_forecasts": cit.forecasts,
            "citation_validation_predictions": cit.test_predictions,
        }, "forecasting")
        st.download_button("Download forecasting results (ZIP)", forecast_zip, "bibliometric_forecasting_results.zip", "application/zip")

elif page == "Topic Modeling":
    st.subheader("Bibliometric Topic Modeling")
    st.caption("Discover latent research themes, compare topic models, examine topic prevalence, and identify increasing or declining themes over time.")

    text_options = {
        "Title": "title",
        "Abstract": "abstract",
        "Author Keywords": "author_keywords",
        "Index Keywords / Keywords Plus": "index_keywords",
    }
    default_text = [label for label, col in text_options.items() if col in df.columns and df[col].astype(str).str.strip().ne("").any()]
    selected_text_labels = st.multiselect("Text fields to analyze", list(text_options), default=default_text)

    model_options = {"LDA": "lda", "NMF": "nmf", "LSA": "lsa", "K-Means": "kmeans"}
    selected_topic_models = st.multiselect("Topic models", list(model_options), default=["LDA", "NMF", "LSA", "K-Means"])
    t1, t2, t3, t4 = st.columns(4)
    num_topics = t1.slider("Requested topics", 2, 30, 8)
    top_words = t2.slider("Top words per topic", 5, 25, 10)
    min_df = t3.slider("Minimum document frequency", 1, 10, 2)
    trend_window = t4.slider("Recent trend window (years)", 2, 15, 5)

    with st.expander("Advanced topic-model settings"):
        max_df = st.slider("Maximum document frequency proportion", 0.50, 1.00, 0.95, 0.01)
        max_features = st.number_input("Maximum vocabulary features", 1000, 50000, 10000, 1000)
        st.caption("BERTopic is intentionally optional because it requires additional transformer dependencies. The integrated core models run with the standard requirements file.")

    if st.button("Run Topic Modeling", type="primary"):
        if not selected_topic_models:
            st.warning("Select at least one topic model.")
        elif not selected_text_labels:
            st.warning("Select at least one text field.")
        else:
            with st.spinner("Preprocessing documents, training topic models, and evaluating topic structure..."):
                try:
                    tres = run_topic_models(
                        df,
                        models=[model_options[x] for x in selected_topic_models],
                        text_columns=[text_options[x] for x in selected_text_labels],
                        num_topics=num_topics,
                        top_words=top_words,
                        min_df=min_df,
                        max_df=max_df,
                        max_features=int(max_features),
                        trend_window=trend_window,
                    )
                    st.session_state["topic_results"] = tres
                except Exception as exc:
                    st.exception(exc)

    if "topic_results" in st.session_state:
        tres = st.session_state["topic_results"]
        comparison = tres["comparison"]
        st.success(f"Best-ranked topic model: {tres['best_model']}")
        if tres.get("errors"):
            with st.expander("Models that could not be completed"):
                for err in tres["errors"]:
                    st.warning(err)

        st.markdown("#### Model Comparison")
        st.dataframe(comparison, use_container_width=True, hide_index=True)
        if "overall_rank_score" in comparison.columns:
            st.plotly_chart(px.bar(comparison.sort_values("overall_rank_score", ascending=False), x="overall_rank_score", y="model", orientation="h", title="Topic Model Comparison (lower rank score is better)"), use_container_width=True)

        model_names = list(tres["models"].keys())
        selected_model_name = st.selectbox("Inspect model", model_names, index=model_names.index(tres["best_model"]) if tres["best_model"] in model_names else 0)
        detail = tres["models"][selected_model_name]
        topics = detail["topics"]
        st.markdown(f"#### {selected_model_name} Topics")
        st.dataframe(topics, use_container_width=True, hide_index=True)
        st.plotly_chart(px.bar(topics.sort_values("Mean_Prevalence"), x="Mean_Prevalence", y="Topic_Label", orientation="h", title=f"{selected_model_name} Topic Prevalence"), use_container_width=True)

        yearly = detail["yearly_prevalence"]
        prevalence_long = topic_prevalence_long(yearly, topics)
        if not prevalence_long.empty:
            st.markdown("#### Topic Evolution Over Time")
            st.plotly_chart(px.line(prevalence_long, x="publication_year", y="Prevalence", color="Topic", markers=True), use_container_width=True)

        trending = detail["trending_topics"]
        if not trending.empty:
            st.markdown("#### Trending Topics")
            st.dataframe(trending, use_container_width=True, hide_index=True)
            st.plotly_chart(px.bar(trending.sort_values("trend_score"), x="trend_score", y="topic_label", color="trend_direction", orientation="h", title="Recent Topic Trend Scores"), use_container_width=True)

        st.markdown("#### Topic Similarity Heatmap")
        sim = detail["topic_similarity"]
        sim_fig = px.imshow(sim, x=sim.columns, y=sim.index, zmin=0, zmax=1, aspect="auto", title=f"{selected_model_name} Topic Similarity")
        st.plotly_chart(sim_fig, use_container_width=True)

        st.markdown("#### Inter-Topic / Cluster Map")
        topic_map = detail["topic_map"]
        map_fig = px.scatter(
            topic_map, x="X", y="Y", size="Mean_Prevalence", hover_name="Topic_Label", text="Topic",
            title=f"{selected_model_name} 2-D Topic Map", size_max=45
        )
        map_fig.update_traces(textposition="top center")
        st.plotly_chart(map_fig, use_container_width=True)

        st.markdown("#### Document–Topic Assignments")
        st.dataframe(detail["document_topics"].head(250), use_container_width=True, hide_index=True)

        topic_files = {"model_comparison": comparison, "cleaned_topic_text": tres["prepared"]}
        for model_name, values in tres["models"].items():
            safe = model_name.lower().replace(" ", "_")
            topic_files[f"{safe}_topics"] = values["topics"]
            topic_files[f"{safe}_document_topics"] = values["document_topics"]
            topic_files[f"{safe}_yearly_prevalence"] = values["yearly_prevalence"]
            topic_files[f"{safe}_trending_topics"] = values["trending_topics"]
            topic_files[f"{safe}_topic_similarity"] = values["topic_similarity"].reset_index().rename(columns={"index": "Topic"})
            topic_files[f"{safe}_topic_map"] = values["topic_map"]
        topic_zip = dataframes_zip(topic_files, "topic_modeling")
        st.download_button("Download topic-modeling results (ZIP)", topic_zip, "bibliometric_topic_modeling_results.zip", "application/zip")

elif page == "Generate Report":
    st.subheader("Generate Scientometric Report")

    st.write(
        "This runs the complete scientometric analysis pipeline and packages "
        "the Word report, cleaned data, Excel tables, and 300-DPI figures "
        "into one ZIP file."
    )

    if st.button("Generate Report", type="primary"):
        with st.spinner("Generating scientometric tables, figures, and Word report..."):
            try:
                package = generate_report_package(df, raw, cleaning, cfg)

                st.session_state["report_package"] = package

                st.success("Scientometric report package generated successfully.")

            except Exception as exc:
                st.exception(exc)

    if "report_package" in st.session_state:
        st.download_button(
            "Download Report Package",
            data=st.session_state["report_package"],
            file_name="scientometric_report_output.zip",
            mime="application/zip",
        )


# elif page == "Generate Chapter 4":
#     st.subheader("Generate Chapter 4 Package")
#     st.write("This runs the complete existing scientometric pipeline and packages the Word report, cleaned data, Excel tables and 300-DPI figures into one ZIP file.")
#     if st.button("Generate Chapter 4", type="primary"):
#         with st.spinner("Generating scientometric tables, figures and Word report..."):
#             try:
#                 package = generate_report_package(df, raw, cleaning, cfg)
#                 st.session_state["chapter4_package"] = package
#                 st.success("Chapter 4 package generated successfully.")
#             except Exception as exc:
#                 st.exception(exc)
#     if "chapter4_package" in st.session_state:
#         st.download_button(
#             "Download Chapter 4 package",
#             data=st.session_state["chapter4_package"],
#             file_name="scientometric_chapter4_output.zip",
#             mime="application/zip",
#         )
