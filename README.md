# Scientometric Chapter 4 Generator

A complete Python project for generating a thesis-ready **Chapter 4: Scientometric Analysis** from Web of Science and Scopus exports.

## Supported inputs
- WoS plain-text `.txt` exports
- Scopus/WoS `.csv`, `.xlsx`, `.xls`
- BibTeX `.bib`
- Multiple files in the same input folder

## Analyses produced
Dataset cleaning and DOI/title deduplication; annual scientific production; citation indicators and h-index; leading authors, sources, countries, document types and languages; collaboration index; Bradford and Lotka distributions; keyword co-occurrence and co-authorship networks with centrality and Louvain communities; LDA topic modelling; and WoS–Scopus provenance comparison.

Every metric is exported as an Excel table, relevant figures are saved at 300 DPI, and the Word report includes a description and interpretation after each table/figure.

## Installation
```bash
cd scientometric_chapter4
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Input preparation
Copy all exported files into:
```text
data/input/
```
For best metadata completeness, export full records and cited references. WoS plain text and Scopus CSV are recommended.

## Run
```bash
python run.py
```
Or:
```bash
python run.py --input /path/to/data --output /path/to/results
```

## Outputs
- `outputs/Chapter_4_Scientometric_Analysis.docx`
- `outputs/cleaned_bibliographic_data.csv`
- `outputs/tables/*.xlsx`
- `outputs/figures/*.png`

## Configuration
Edit `config/config.yaml` to change thesis title, year filters, top-N limits, topic count and network thresholds.

## Important methodological note
WoS and Scopus differ in source coverage and citation indexing. The program retains database provenance and deduplicates records, but citation totals should still be discussed with the database and export date stated in the thesis methodology.

## Streamlit dashboard
The project now includes an interactive Streamlit application for direct Scopus/WoS upload and analysis.

```bash
pip install -r requirements.txt
streamlit run app.py
```

Dashboard modules:
- Main Information of the Data (Bibliometrix-style summary)
- Annual Publication Growth with interactive chart
- Source Analysis with ranking and production-over-time chart
- Author Profile Analysis with productivity/citation/h-index views
- Affiliation Analysis based on the authors’ **college/university/institute name** (department, city, postal address and country are excluded), with ranking, institution type, citations and production-over-time chart
- Full Chapter 4 ZIP generation (Word report, Excel tables, figures and cleaned data)

Supported upload formats: `.csv`, `.txt`, `.bib`, `.xlsx`, `.xls`.

## Integrated forecasting and topic modeling (v3)

The Streamlit dashboard now also includes two modules merged from the supplied projects.

### Time Series Forecasting
Uses the already uploaded/cleaned bibliographic records to build complete annual publication and citation series. Available models:
- Naive baseline
- Exponential Smoothing
- ARIMA(1,1,1)
- Linear Regression with lag features
- Random Forest
- Gradient Boosting
- Support Vector Regression

The dashboard supports configurable hold-out validation years, lag count, and forecast horizon. It reports MAE, RMSE, MAPE, sMAPE and R², selects the lowest-RMSE valid model, plots historical + forecast values, and exports the annual series, validation predictions, model metrics and forecasts as ZIP/CSV.

### Topic Modeling and Trend Analysis
Uses any combination of Title, Abstract, Author Keywords and Index Keywords/Keywords Plus. Built-in models:
- LDA
- NMF
- LSA
- K-Means

Outputs include model-comparison metrics, topic labels and top terms, document-topic assignments, topic prevalence, yearly topic evolution, trending-topic ranking, topic-similarity heatmap and a 2-D inter-topic/cluster map. Results can be downloaded as a ZIP of CSV files.

BERTopic remains optional because it requires transformer/embedding dependencies. Core topic modeling does not require BERTopic or gensim; if gensim is absent the coherence metric is shown as unavailable while the remaining evaluation metrics continue to work.
