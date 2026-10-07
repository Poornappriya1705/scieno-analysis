import re, unicodedata
from pathlib import Path
from collections import Counter
import pandas as pd

EMPTY = {"", "nan", "none", "null", "n/a", "na"}

def norm_text(x):
    if pd.isna(x): return ""
    s = unicodedata.normalize("NFKC", str(x)).strip()
    return re.sub(r"\s+", " ", s)

def norm_key(x):
    s = norm_text(x).lower()
    s = re.sub(r"https?://(dx\.)?doi\.org/", "", s)
    s = re.sub(r"[^a-z0-9]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()

def split_multi(x):
    s = norm_text(x)
    if not s: return []
    parts = re.split(r"\s*;\s*|\s*\|\s*", s)
    return [p.strip(" ,") for p in parts if p.strip(" ,").lower() not in EMPTY]

def split_keywords(x):
    return [re.sub(r"\s+", " ", p.lower()).strip() for p in split_multi(x)]

def ensure_dir(p):
    p = Path(p); p.mkdir(parents=True, exist_ok=True); return p

def safe_filename(s):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(s)).strip("_")

def explode_count(series, splitter=split_multi):
    c = Counter()
    for x in series.fillna(""):
        c.update(splitter(x))
    return pd.DataFrame(c.most_common(), columns=["Item", "Frequency"])
