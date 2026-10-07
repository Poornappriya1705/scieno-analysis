from pathlib import Path
import io
import re
import pandas as pd

CANON = {
    "title": ["TI", "Title", "Article Title", "Document Title", "title"],
    "authors": ["AU", "Authors", "Author full names", "Authors with affiliations", "Author", "author"],
    "year": ["PY", "Year", "Publication Year", "year"],
    "source": ["SO", "Source title", "Source Title", "Publication Name", "Journal", "journal", "Booktitle", "booktitle"],
    "doi": ["DI", "DOI", "doi"],
    "citations": ["TC", "Cited by", "Times Cited, All Databases", "Times Cited", "Citations", "citedby", "cited_by"],
    "abstract": ["AB", "Abstract", "abstract"],
    "author_keywords": ["DE", "Author Keywords", "Keywords", "keywords", "keyword"],
    "index_keywords": ["ID", "Index Keywords", "Keywords Plus", "Keywords Plus®"],
    "affiliations": ["C1", "Affiliations", "Author Affiliations", "Affiliation", "Addresses", "address"],
    "countries": ["Countries", "Country/Region", "Country", "AU_CO"],
    "references": ["CR", "References", "Cited References", "references"],
    "document_type": ["DT", "Document Type", "Document Type(s)", "Type", "ENTRYTYPE", "entrytype"],
    "language": ["LA", "Language of Original Document", "Language", "language"],
    "open_access": ["OA", "Open Access"],
    "funding": ["FU", "Funding Details", "Funding Texts", "Funding", "Acknowledgements"],
    "issn": ["SN", "ISSN", "issn"],
    "eid": ["EID", "UT", "Accession Number", "Unique ID", "ID"],
}


def detect_db(cols, filename=""):
    cs = {str(c).strip() for c in cols}
    if "EID" in cs or "Cited by" in cs or "Source title" in cs:
        return "Scopus"
    if "UT" in cs or "TI" in cs or "SO" in cs or "C1" in cs:
        return "WoS"
    n = str(filename).lower()
    return "Scopus" if "scopus" in n else ("WoS" if "wos" in n or "webofscience" in n else "Unknown")


def map_columns(df):
    out = pd.DataFrame(index=df.index)
    lower = {str(c).strip().lower(): c for c in df.columns}
    for target, aliases in CANON.items():
        found = None
        for a in aliases:
            if a.lower() in lower:
                found = lower[a.lower()]
                break
        out[target] = df[found] if found is not None else ""
    return out


def read_wos_plain_stream(stream):
    records = []
    cur = {}
    for raw in stream:
        line = raw.rstrip("\n\r")
        if line == "ER":
            if cur:
                records.append(cur)
                cur = {}
            continue
        m = re.match(r"^([A-Z0-9]{2})\s(.*)$", line)
        if m:
            k, v = m.groups()
            cur[k] = v if k not in cur else cur[k] + "; " + v
        elif line.startswith("   ") and cur:
            k = list(cur)[-1]
            cur[k] += " " + line.strip()
    if cur:
        records.append(cur)
    return pd.DataFrame(records)


def read_wos_plain(path):
    with open(path, "r", encoding="utf-8-sig", errors="replace") as f:
        return read_wos_plain_stream(f)


def read_bib_text(text):
    import bibtexparser

    db = bibtexparser.loads(text)
    return pd.DataFrame(db.entries)


def read_bib(path):
    with open(path, encoding="utf-8-sig", errors="replace") as f:
        return read_bib_text(f.read())


def _read_csv_flexible(source, encoding="utf-8-sig", sep=None):
    kwargs = dict(encoding=encoding)
    if sep is None:
        # Python's sniffer gracefully handles comma/tab/semicolon delimited text exports.
        kwargs.update(sep=None, engine="python")
    else:
        kwargs.update(sep=sep, low_memory=False)
        return pd.read_csv(source, **kwargs)
    return pd.read_csv(source, **kwargs)


def _standardize(df, filename):
    db = detect_db(df.columns, filename)
    x = map_columns(df)
    x["database"] = db
    x["input_file"] = filename
    return x


def read_one(path):
    path = Path(path)
    ext = path.suffix.lower()
    if ext == ".csv":
        try:
            df = pd.read_csv(path, encoding="utf-8-sig", low_memory=False)
        except UnicodeDecodeError:
            df = pd.read_csv(path, encoding="latin1", low_memory=False)
    elif ext in {".xlsx", ".xls"}:
        df = pd.read_excel(path)
    elif ext == ".bib":
        df = read_bib(path)
    elif ext == ".txt":
        head = path.read_text(encoding="utf-8-sig", errors="replace")[:500]
        if "FN " in head or "VR " in head or "PT " in head:
            df = read_wos_plain(path)
        else:
            try:
                df = _read_csv_flexible(path, encoding="utf-8-sig")
            except UnicodeDecodeError:
                df = _read_csv_flexible(path, encoding="latin1")
    else:
        raise ValueError(f"Unsupported: {path}")
    return _standardize(df, path.name)


def read_uploaded(uploaded_file):
    """Read one Streamlit UploadedFile (or compatible object) into canonical fields."""
    filename = getattr(uploaded_file, "name", "uploaded_file")
    ext = Path(filename).suffix.lower()
    data = uploaded_file.getvalue() if hasattr(uploaded_file, "getvalue") else uploaded_file.read()

    if ext == ".csv":
        try:
            df = pd.read_csv(io.BytesIO(data), encoding="utf-8-sig", low_memory=False)
        except UnicodeDecodeError:
            df = pd.read_csv(io.BytesIO(data), encoding="latin1", low_memory=False)
    elif ext in {".xlsx", ".xls"}:
        df = pd.read_excel(io.BytesIO(data))
    elif ext == ".bib":
        text = data.decode("utf-8-sig", errors="replace")
        df = read_bib_text(text)
    elif ext == ".txt":
        text = data.decode("utf-8-sig", errors="replace")
        head = text[:500]
        if "FN " in head or "VR " in head or "PT " in head:
            df = read_wos_plain_stream(io.StringIO(text))
        else:
            try:
                df = pd.read_csv(io.StringIO(text), sep=None, engine="python")
            except Exception:
                df = pd.read_csv(io.StringIO(text), sep="\t", low_memory=False)
    else:
        raise ValueError(f"Unsupported file type: {ext}. Use CSV, TXT, BIB, XLSX or XLS.")
    return _standardize(df, filename)


def load_uploaded_files(uploaded_files):
    frames = []
    errors = []
    for f in uploaded_files:
        try:
            frames.append(read_uploaded(f))
        except Exception as exc:
            errors.append(f"{getattr(f, 'name', 'file')}: {exc}")
    if not frames:
        raise RuntimeError("No uploaded file could be read. " + " | ".join(errors))
    return pd.concat(frames, ignore_index=True), errors


def load_folder(folder, allowed, recursive=True):
    folder = Path(folder)
    pattern = "**/*" if recursive else "*"
    files = [p for p in folder.glob(pattern) if p.is_file() and p.suffix.lower() in allowed]
    if not files:
        raise FileNotFoundError(f"No supported files found in {folder}")
    frames = []
    for p in sorted(files):
        try:
            frames.append(read_one(p))
        except Exception as e:
            print(f"WARNING: skipped {p.name}: {e}")
    if not frames:
        raise RuntimeError("All input files failed to load")
    return pd.concat(frames, ignore_index=True), files
