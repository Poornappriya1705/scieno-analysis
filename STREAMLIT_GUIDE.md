# Streamlit Scientometric Dashboard – Quick Start

## 1. Install dependencies
```bash
pip install -r requirements.txt
```

## 2. Start the application
```bash
streamlit run app.py
```

Windows users can also double-click `START_STREAMLIT.bat`.

## 3. Upload bibliographic data
Supported formats:
- Scopus CSV / Excel / text exports
- Web of Science CSV / Excel / plain-text exports
- BibTeX `.bib`
- Multiple files at once

## 4. Dashboard modules
1. **Main Information** – timespan, sources, documents, annual growth rate, document age, citations, references, keywords, authors, collaboration, and document types.
2. **Annual Publication Growth** – annual publication table and interactive trend chart.
3. **Source Analysis** – source productivity/citation ranking and source production over time.
4. **Author Profile Analysis** – author productivity, citations, mean citations, h-index, and production over time.
5. **Affiliation Analysis** – extracts the authors’ college/university/institute from Scopus/WoS affiliation addresses, then shows institutional productivity, citations, institution type and production over time.
6. **Generate Chapter 4** – generates a ZIP containing the Word report, cleaned data, Excel tables, and 300-DPI figures.

## Deduplication
When uploading a single database export, leave **Deduplicate by DOI/title** off if you want the dashboard totals to reproduce the source export exactly. Turn it on when combining overlapping Scopus and Web of Science datasets.

## Document Average Age
For reproducibility of historical datasets, the dashboard uses the latest publication year in the uploaded collection as the reference year. This prevents the same historical dataset from changing its average age every calendar year.

## Time Series Forecasting
Select **Time Series Forecasting** after uploading the Scopus/WoS data. Choose the models, validation window, forecast horizon and lag count, then click **Run Forecasting Analysis**. A sufficiently long annual series is required; the application will tell you the minimum number of years for the selected settings.

## Topic Modeling
Select **Topic Modeling**, choose the bibliographic text fields and one or more core models (LDA, NMF, LSA, K-Means), then choose the requested number of topics. Click **Run Topic Modeling** to view model comparison, topic terms, prevalence, yearly trends, similarity heatmap and 2-D topic map.
