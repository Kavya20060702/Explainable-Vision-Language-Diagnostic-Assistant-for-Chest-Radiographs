"""
Builds a retrieval index over real IU/OpenI radiology reports so the report
generator can produce clinically-worded text instead of a rigid template.

Written for the common Kaggle version of this dataset with two files:
  indiana_reports.csv     columns: uid, MeSH, Problems, image, indication,
                                    comparison, findings, impression
  indiana_projections.csv columns: uid, filename, projection

If your actual column names differ (check with a quick pandas .columns
print), adjust the two `df["..."]` lookups in load_openi_reports below --
everything downstream only depends on the final DataFrame having
"text", "labels", and "filename" columns.
"""

import argparse
import os
import pickle

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer


def load_openi_reports(reports_csv, projections_csv):
    reports = pd.read_csv(reports_csv)
    projections = pd.read_csv(projections_csv)

    required_reports = {"uid", "findings", "impression", "Problems"}
    missing = required_reports - set(reports.columns)
    if missing:
        raise ValueError(
            f"indiana_reports.csv is missing columns {missing}. "
            f"Found columns: {reports.columns.tolist()}. "
            f"Adjust load_openi_reports() to match your actual file."
        )

    required_proj = {"uid", "filename"}
    missing = required_proj - set(projections.columns)
    if missing:
        raise ValueError(
            f"indiana_projections.csv is missing columns {missing}. "
            f"Found columns: {projections.columns.tolist()}. "
            f"Adjust load_openi_reports() to match your actual file."
        )

    merged = projections.merge(reports, on="uid", how="left")
    merged["text"] = (
        merged["findings"].fillna("") + " " + merged["impression"].fillna("")
    ).str.strip()
    # "Problems" holds MeSH-style condition terms (e.g. "Cardiomegaly", "Effusion",
    # or "normal") -- used later to match against our predicted condition labels.
    merged["labels"] = merged["Problems"].fillna("normal")

    merged = merged[merged["text"].str.len() > 0].reset_index(drop=True)
    return merged


def build_index(reports_csv, projections_csv, out_path="models/report_index.pkl"):
    df = load_openi_reports(reports_csv, projections_csv)
    vectorizer = TfidfVectorizer(max_features=5000, stop_words="english")
    tfidf_matrix = vectorizer.fit_transform(df["text"])

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "wb") as f:
        pickle.dump(
            {"vectorizer": vectorizer, "tfidf_matrix": tfidf_matrix, "reports": df},
            f,
        )
    print(f"Built index over {len(df)} reports -> {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--reports", default="data/openi/indiana_reports.csv")
    parser.add_argument("--projections", default="data/openi/indiana_projections.csv")
    parser.add_argument("--out", default="models/report_index.pkl")
    args = parser.parse_args()
    build_index(args.reports, args.projections, args.out)