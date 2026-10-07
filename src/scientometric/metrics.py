from collections import Counter, defaultdict
from itertools import combinations
import re
from datetime import date

import numpy as np
import pandas as pd
import networkx as nx
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

from .utils import split_multi, split_keywords, norm_text


def h_index(vals):
    s = sorted([int(v) for v in vals if pd.notna(v)], reverse=True)
    return max([i for i, v in enumerate(s, 1) if v >= i], default=0)


def _unique_items(series, keyword=False):
    vals = set()
    splitter = split_keywords if keyword else split_multi
    for x in series.fillna(""):
        vals.update(v for v in splitter(x) if v)
    return vals


def _reference_items(x):
    """Split cited-reference strings conservatively without splitting commas inside citations."""
    s = norm_text(x)
    if not s:
        return []
    # WoS and most Scopus full-record exports separate references with semicolons.
    parts = re.split(r"\s*;\s*|\s*\|\s*", s)
    return [p.strip() for p in parts if p.strip()]


def _clean_affiliation(a):
    a = norm_text(a)
    # WoS C1 may begin with an author-address block: [SMITH J; DOE A] University...
    a = re.sub(r"^\[[^\]]+\]\s*", "", a)
    return a.strip(" ;,")


def split_affiliations(x):
    """Return raw affiliation/address strings from Scopus/WoS exports."""
    return [a for a in (_clean_affiliation(v) for v in split_multi(x)) if a]


# Institution-level affiliation analysis ---------------------------------------
# The order is intentional: when an address contains both a college and its
# parent university, the university is the preferred organization.
_INSTITUTION_PATTERNS = [
    ("University", re.compile(r"\b(?:university|universit(?:y|é)|univ\.?)\b", re.I)),
    ("College", re.compile(r"\b(?:college|coll\.?)\b", re.I)),
    ("Institute", re.compile(r"\b(?:institute|institut|inst\.?)\b|\b(?:IIT|NIT|IIIT)\b", re.I)),
    ("School/Academy", re.compile(r"\b(?:academy|school|polytechnic)\b", re.I)),
]

# Administrative/address components that should not become institutions on
# their own.  They are still retained when they are part of a real institution
# name such as "College of Engineering" or "Institute of Technology".
_NON_INSTITUTION_PREFIX = re.compile(
    r"^(?:department|dept\.?|faculty|division|centre|center|laboratory|lab\.?|"
    r"school of [^,]+(?:,|$)|unit|section|program(?:me)?|chair)\b", re.I
)


def _normalise_institution_name(name):
    name = norm_text(name).strip(" .;,:-")
    name = re.sub(r"^the\s+", "", name, flags=re.I)
    # Expand common export abbreviations only when they clearly denote the
    # organization type; do not otherwise rewrite the institution name.
    name = re.sub(r"\bUniv\.?\b", "University", name, flags=re.I)
    name = re.sub(r"\bColl\.?\b", "College", name, flags=re.I)
    name = re.sub(r"\bInst\.?\b", "Institute", name, flags=re.I)
    name = re.sub(r"\s+", " ", name).strip()
    return name


def institution_type(name):
    for typ, pat in _INSTITUTION_PATTERNS:
        if pat.search(norm_text(name)):
            return typ
    return "Other Institution"


def extract_institutions(x):
    """Extract college/university-level organizations from affiliation text.

    Examples
    --------
    ``Dept. of CSE, Bharathidasan University, Tiruchirappalli, India``
        -> ``Bharathidasan University``
    ``PSG College of Technology, Coimbatore, India``
        -> ``PSG College of Technology``

    The function is designed for Scopus ``Affiliations`` and WoS ``C1``/
    ``Addresses`` fields.  A university is preferred over a college/institute
    when both occur in the same affiliation address.
    """
    institutions = []
    for raw_aff in split_affiliations(x):
        # Remove a WoS author block again in case semicolon splitting exposed it.
        aff = re.sub(r"^\[[^\]]+\]\s*", "", raw_aff).strip()
        if not aff:
            continue

        # Most Scopus/WoS affiliation addresses are comma-delimited.
        parts = [_normalise_institution_name(p) for p in re.split(r"\s*,\s*", aff) if norm_text(p)]
        candidates = []
        for idx, part in enumerate(parts):
            for priority, (typ, pat) in enumerate(_INSTITUTION_PATTERNS):
                if pat.search(part):
                    # Avoid treating a pure administrative school/department
                    # as the institution when a parent organization follows.
                    if typ == "School/Academy" and _NON_INSTITUTION_PREFIX.search(part):
                        continue
                    candidates.append((priority, idx, part, typ))
                    break

        if candidates:
            # University > College > Institute > School/Academy.  For the same
            # type choose the first occurrence in the address.
            candidates.sort(key=lambda z: (z[0], z[1]))
            institutions.append(candidates[0][2])
            continue

        # Fallback for exports with no commas, e.g. "National Institute of
        # Technology Tiruchirappalli India".  Keep the full string only when
        # an institution keyword is present.
        for _, pat in _INSTITUTION_PATTERNS:
            if pat.search(aff):
                institutions.append(_normalise_institution_name(aff))
                break

    # Deduplicate case-insensitively while preserving the first spelling.
    seen = set()
    out = []
    for inst in institutions:
        key = re.sub(r"[^a-z0-9]+", " ", inst.lower()).strip()
        if inst and key and key not in seen:
            seen.add(key)
            out.append(inst)
    return out


def _country_aliases():
    aliases = {
        "usa": "United States", "u.s.a.": "United States", "united states of america": "United States",
        "uk": "United Kingdom", "u.k.": "United Kingdom", "england": "United Kingdom",
        "scotland": "United Kingdom", "wales": "United Kingdom", "northern ireland": "United Kingdom",
        "peoples r china": "China", "people's republic of china": "China", "pr china": "China",
        "south korea": "South Korea", "republic of korea": "South Korea", "korea": "South Korea",
        "russian federation": "Russia", "viet nam": "Vietnam", "uae": "United Arab Emirates",
        "taiwan": "Taiwan", "hong kong": "Hong Kong",
    }
    try:
        import pycountry
        for c in pycountry.countries:
            aliases[c.name.lower()] = c.name
            if hasattr(c, "official_name"):
                aliases[c.official_name.lower()] = c.name
    except Exception:
        pass
    return aliases


_COUNTRY_ALIASES = _country_aliases()


def extract_countries(row):
    supplied = split_multi(row.get("countries", ""))
    found = []
    for c in supplied:
        key = c.strip(" .").lower()
        found.append(_COUNTRY_ALIASES.get(key, c.strip()))
    if found:
        return list(dict.fromkeys(found))

    text = norm_text(row.get("affiliations", ""))
    low = text.lower()
    # Longest names first prevents partial alias collisions.
    for alias in sorted(_COUNTRY_ALIASES, key=len, reverse=True):
        if len(alias) < 3:
            continue
        if re.search(r"(?<![a-z])" + re.escape(alias) + r"(?![a-z])", low):
            found.append(_COUNTRY_ALIASES[alias])
    return list(dict.fromkeys(found))


def overview(df, reference_year=None):
    yrs = df.year.dropna().astype(int)
    annual = (
        df.dropna(subset=["year"])
        .groupby("year", dropna=True)
        .agg(Documents=("title", "count"), Citations=("citations", "sum"), Mean_Citations=("citations", "mean"))
        .reset_index()
        .sort_values("year")
    )
    if not annual.empty:
        annual["Annual_Growth_%"] = annual.Documents.pct_change() * 100
        y0, y1 = int(annual.year.iloc[0]), int(annual.year.iloc[-1])
        span = max(1, y1 - y0)
        first, last = float(annual.Documents.iloc[0]), float(annual.Documents.iloc[-1])
        cagr = ((last / first) ** (1 / span) - 1) * 100 if first > 0 else np.nan
    else:
        cagr = np.nan

    author_lists = [list(dict.fromkeys(split_multi(x))) for x in df.authors]
    author_counts = np.array([len(x) for x in author_lists], dtype=float)
    unique_authors = set(a for items in author_lists for a in items)
    single_docs_mask = author_counts == 1
    single_authors = set()
    for is_single, authors in zip(single_docs_mask, author_lists):
        if is_single and authors:
            single_authors.add(authors[0])

    countries_per_doc = [extract_countries(r) for _, r in df.iterrows()]
    intl_docs = sum(len(set(c)) > 1 for c in countries_per_doc)

    refs = set()
    for x in df.references:
        refs.update(_reference_items(x))

    if reference_year is None:
        # Stable for completed historical datasets; current year for current datasets.
        reference_year = int(yrs.max()) if len(yrs) else date.today().year
    ages = [max(0, reference_year - int(y)) for y in yrs]

    o = {
        "documents": len(df),
        "timespan": f"{yrs.min()}:{yrs.max()}" if len(yrs) else "Unknown",
        "sources": int(df.source.replace("", np.nan).nunique()),
        "authors": len(unique_authors),
        "author_appearances": int(author_counts.sum()),
        "authors_single_authored_docs": len(single_authors),
        "single_authored_docs": int(single_docs_mask.sum()),
        "coauthors_per_doc": float(np.mean(author_counts)) if len(author_counts) else 0.0,
        "international_coauthorship_pct": float(intl_docs / len(df) * 100) if len(df) else 0.0,
        "citations": int(df.citations.sum()),
        "mean_citations": float(df.citations.mean()) if len(df) else 0.0,
        "h_index": h_index(df.citations),
        "annual_growth_rate": float(cagr) if pd.notna(cagr) else np.nan,
        "cagr": float(cagr) if pd.notna(cagr) else np.nan,
        "document_average_age": float(np.mean(ages)) if ages else np.nan,
        "references": len(refs),
        "keywords_plus": len(_unique_items(df.index_keywords, keyword=True)),
        "author_keywords": len(_unique_items(df.author_keywords, keyword=True)),
        "reference_year": reference_year,
    }
    return o, annual


def main_information_table(df, reference_year=None):
    o, _ = overview(df, reference_year=reference_year)
    rows = [
        ("MAIN INFORMATION ABOUT DATA", ""),
        ("Timespan", o["timespan"]),
        ("Sources (Journals, Books, etc)", o["sources"]),
        ("Documents", o["documents"]),
        ("Annual Growth Rate %", round(o["annual_growth_rate"], 2) if pd.notna(o["annual_growth_rate"]) else "NA"),
        ("Document Average Age", round(o["document_average_age"], 2) if pd.notna(o["document_average_age"]) else "NA"),
        ("Average citations per doc", round(o["mean_citations"], 2)),
        ("References", o["references"]),
        ("DOCUMENT CONTENTS", ""),
        ("Keywords Plus (ID)", o["keywords_plus"]),
        ("Author's Keywords (DE)", o["author_keywords"]),
        ("AUTHORS", ""),
        ("Authors", o["authors"]),
        ("Authors of single-authored docs", o["authors_single_authored_docs"]),
        ("AUTHORS COLLABORATION", ""),
        ("Single-authored docs", o["single_authored_docs"]),
        ("Co-Authors per Doc", round(o["coauthors_per_doc"], 2)),
        ("International co-authorships %", round(o["international_coauthorship_pct"], 2)),
        ("DOCUMENT TYPES", ""),
    ]
    dt = document_type_counts(df)
    rows.extend((r.Document_Type, int(r.Documents)) for r in dt.itertuples(index=False))
    return pd.DataFrame(rows, columns=["Description", "Results"])


def document_type_counts(df):
    vals = df.document_type.fillna("").map(lambda s: norm_text(s).lower())
    vals = vals[vals != ""]
    if vals.empty:
        return pd.DataFrame(columns=["Document_Type", "Documents"])
    return vals.value_counts().rename_axis("Document_Type").reset_index(name="Documents")


def entity_table(df, col, label, split=True, splitter=split_multi):
    rows = []
    if split:
        mp = defaultdict(lambda: {"docs": 0, "cit": 0, "years": []})
        for _, r in df.iterrows():
            for e in set(splitter(r[col])):
                if not e:
                    continue
                mp[e]["docs"] += 1
                mp[e]["cit"] += r.citations
                if pd.notna(r.year):
                    mp[e]["years"].append(int(r.year))
        for e, v in mp.items():
            rows.append([e, v["docs"], v["cit"], v["cit"] / v["docs"], min(v["years"]) if v["years"] else None, max(v["years"]) if v["years"] else None])
    else:
        g = df[df[col] != ""].groupby(col).agg(
            Documents=("title", "count"), Citations=("citations", "sum"), Mean_Citations=("citations", "mean"),
            First_Year=("year", "min"), Last_Year=("year", "max")
        ).reset_index()
        g.columns = [label, "Documents", "Citations", "Mean_Citations", "First_Year", "Last_Year"]
        return g.sort_values(["Documents", "Citations"], ascending=False)
    return pd.DataFrame(rows, columns=[label, "Documents", "Citations", "Mean_Citations", "First_Year", "Last_Year"]).sort_values(["Documents", "Citations"], ascending=False)


def author_table(df):
    t = entity_table(df, "authors", "Author", True)
    citations_by = defaultdict(list)
    for _, r in df.iterrows():
        for a in set(split_multi(r.authors)):
            citations_by[a].append(r.citations)
    if not t.empty:
        t["h_index"] = t.Author.map(lambda a: h_index(citations_by[a]))
    else:
        t["h_index"] = []
    return t.sort_values(["Documents", "h_index", "Citations"], ascending=False)


def affiliation_table(df):
    """Institution-level affiliation table (college/university of authors)."""
    t = entity_table(df, "affiliations", "Institution", True, splitter=extract_institutions)
    if not t.empty:
        t.insert(1, "Institution_Type", t["Institution"].map(institution_type))
    else:
        t.insert(1, "Institution_Type", pd.Series(dtype=str))
    return t


def production_over_time(df, field, item_label, splitter=split_multi, top_n=10):
    totals = Counter()
    per_year = Counter()
    for _, r in df.iterrows():
        if pd.isna(r.year):
            continue
        items = set(splitter(r[field]))
        for item in items:
            if item:
                totals[item] += 1
                per_year[(item, int(r.year))] += 1
    top = [x for x, _ in totals.most_common(top_n)]
    rows = [{item_label: item, "Year": year, "Documents": count} for (item, year), count in per_year.items() if item in top]
    return pd.DataFrame(rows).sort_values([item_label, "Year"]) if rows else pd.DataFrame(columns=[item_label, "Year", "Documents"])


def keyword_table(df):
    c = Counter()
    for _, r in df.iterrows():
        c.update(set(split_keywords(r.author_keywords) + split_keywords(r.index_keywords)))
    return pd.DataFrame(c.most_common(), columns=["Keyword", "Frequency"])


def doc_types(df):
    return entity_table(df, "document_type", "Document_Type", False)


def languages(df):
    return entity_table(df, "language", "Language", False)


def sources(df):
    return entity_table(df, "source", "Source", False)


def countries(df):
    vals = ["; ".join(extract_countries(r)) for _, r in df.iterrows()]
    z = df.copy()
    z["_countries"] = vals
    return entity_table(z, "_countries", "Country", True)


def collaboration(df):
    rows = []
    for _, r in df.iterrows():
        a = len(set(split_multi(r.authors)))
        c = len(set(extract_countries(r)))
        rows.append([a, c, "Single-authored" if a == 1 else "Multi-authored", "International" if c > 1 else "Domestic/Unknown"])
    z = pd.DataFrame(rows, columns=["Authors_Count", "Countries_Count", "Authorship", "Collaboration"])
    summary = pd.DataFrame({
        "Metric": ["Mean authors per document", "Single-authored documents", "Multi-authored documents", "International collaboration documents", "Collaboration index"],
        "Value": [z.Authors_Count.mean(), (z.Authorship == "Single-authored").sum(), (z.Authorship == "Multi-authored").sum(), (z.Collaboration == "International").sum(), z.loc[z.Authorship == "Multi-authored", "Authors_Count"].mean()],
    })
    return z, summary


def bradford(source_table):
    t = source_table.copy().sort_values("Documents", ascending=False).reset_index(drop=True)
    total = t.Documents.sum()
    zone_target = total / 3
    cum = 0
    zone = 1
    zones = []
    for n in t.Documents:
        cum += n
        zones.append(zone)
        if cum >= zone * zone_target and zone < 3:
            zone += 1
    t["Bradford_Zone"] = zones
    return t


def lotka(authors_df):
    return authors_df.groupby("Documents").size().reset_index(name="Authors").rename(columns={"Documents": "Publications"})


def network(df, field, min_freq=2, top_nodes=75):
    counter = Counter()
    edges = Counter()
    for x in df[field]:
        items = list(dict.fromkeys(split_keywords(x) if "keyword" in field else split_multi(x)))
        counter.update(items)
        for a, b in combinations(sorted(items), 2):
            edges[(a, b)] += 1
    nodes = {n for n, _ in counter.most_common(top_nodes) if counter[n] >= min_freq}
    G = nx.Graph()
    for n in nodes:
        G.add_node(n, frequency=counter[n])
    for (a, b), w in edges.items():
        if a in nodes and b in nodes:
            G.add_edge(a, b, weight=w)
    if G.number_of_nodes():
        deg = nx.degree_centrality(G)
        bet = nx.betweenness_centrality(G, weight="weight") if G.number_of_nodes() < 300 else {n: 0 for n in G}
        try:
            comm = nx.community.louvain_communities(G, weight="weight", seed=42)
        except Exception:
            comm = list(nx.community.greedy_modularity_communities(G, weight="weight"))
        cmap = {n: i + 1 for i, c in enumerate(comm) for n in c}
    else:
        deg = bet = cmap = {}
    nt = pd.DataFrame([{"Node": n, "Frequency": counter[n], "Degree": G.degree(n), "Degree_Centrality": deg.get(n, 0), "Betweenness": bet.get(n, 0), "Cluster": cmap.get(n, 0)} for n in G.nodes()]).sort_values(["Frequency", "Degree"], ascending=False) if G.number_of_nodes() else pd.DataFrame(columns=["Node", "Frequency", "Degree", "Degree_Centrality", "Betweenness", "Cluster"])
    et = pd.DataFrame([{"Source": a, "Target": b, "Weight": d["weight"]} for a, b, d in G.edges(data=True)]).sort_values("Weight", ascending=False) if G.number_of_edges() else pd.DataFrame(columns=["Source", "Target", "Weight"])
    return G, nt, et


def topic_model(df, n_topics=8, top_words=10, random_state=42):
    corpus = (df.title.fillna("") + " " + df.abstract.fillna("") + " " + df.author_keywords.fillna("")).tolist()
    corpus = [x for x in corpus if len(x.split()) >= 5]
    if len(corpus) < max(5, n_topics):
        return pd.DataFrame(), pd.DataFrame()
    vec = CountVectorizer(stop_words="english", max_df=.95, min_df=2, max_features=5000)
    X = vec.fit_transform(corpus)
    if X.shape[1] < 2:
        return pd.DataFrame(), pd.DataFrame()
    k = min(n_topics, max(2, X.shape[0] // 3), X.shape[1])
    lda = LatentDirichletAllocation(n_components=k, random_state=random_state, learning_method="batch").fit(X)
    names = vec.get_feature_names_out()
    topics = []
    for i, comp in enumerate(lda.components_):
        topics.append([i + 1, ", ".join(names[comp.argsort()[-top_words:][::-1]])])
    doc_topic = lda.transform(X)
    prevalence = doc_topic.mean(axis=0)
    t = pd.DataFrame(topics, columns=["Topic", "Top_Terms"])
    t["Prevalence"] = prevalence
    return t, pd.DataFrame(doc_topic, columns=[f"Topic_{i+1}" for i in range(k)])


def database_comparison(df):
    rows = []
    for db, g in df.groupby("database_sources"):
        rows.append([db, len(g), g.citations.sum(), g.citations.mean(), g.source.nunique(), h_index(g.citations)])
    return pd.DataFrame(rows, columns=["Database_Source", "Documents", "Citations", "Mean_Citations", "Sources", "h_index"]).sort_values("Documents", ascending=False)
