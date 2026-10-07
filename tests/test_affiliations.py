import pandas as pd

from src.scientometric.metrics import extract_institutions, affiliation_table


def test_extract_college_or_university():
    assert extract_institutions(
        "Department of Computer Science, Bharathidasan University, Tiruchirappalli, India"
    ) == ["Bharathidasan University"]
    assert extract_institutions(
        "Department of CSE, PSG College of Technology, Coimbatore, India"
    ) == ["PSG College of Technology"]


def test_parent_university_is_preferred():
    assert extract_institutions(
        "College of Engineering, Anna University, Chennai, India"
    ) == ["Anna University"]


def test_same_institution_counted_once_per_document():
    df = pd.DataFrame({
        "affiliations": [
            "Dept A, University X, India; Dept B, University X, India",
            "University X, India; University Y, USA",
        ],
        "citations": [5, 10],
        "year": [2024, 2025],
        "title": ["Paper 1", "Paper 2"],
    })
    t = affiliation_table(df).set_index("Institution")
    assert int(t.loc["University X", "Documents"]) == 2
    assert int(t.loc["University Y", "Documents"]) == 1
