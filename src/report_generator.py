"""
Turns (predicted condition, confidence, Grad-CAM region) into a readable
report. Tries retrieval from real OpenI report text first (if the index
exists); falls back to a clean template if the index isn't built yet or
nothing relevant is found. Always keep the template fallback -- it's what
guarantees the demo works even if the retrieval index has issues.
"""

import os
import pickle
import re

import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


def clean_report_text(text):
    """
    IU/OpenI reports use 'XXXX' as a de-identification placeholder for
    redacted terms. Strip it out for readability in the final report.
    """
    text = re.sub(r"\bXXXX\b", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"\s+([.,])", r"\1", text)
    return text


TEMPLATE = (
    "Findings suggest {condition_phrase}, most notably in the {region}. "
    "Model confidence: {confidence_pct}%. {urgency_note}"
)

CONDITION_PHRASES = {
    "No Finding": "no significant abnormality",
    "Infiltration": "an area of pulmonary infiltration",
    "Effusion": "findings consistent with pleural effusion",
    "Atelectasis": "an area of atelectasis (partial lung collapse)",
    "Nodule": "a pulmonary nodule",
    "Mass": "a pulmonary mass requiring further evaluation",
    "Pneumothorax": "findings suggestive of pneumothorax",
    "Consolidation": "an area of pulmonary consolidation",
    "Pleural_Thickening": "pleural thickening",
    "Cardiomegaly": "cardiac silhouette enlargement (cardiomegaly)",
    "Pneumonia": "signs consistent with pneumonia",
}

# IU/OpenI's "Problems" field uses MeSH-style terms that don't always match
# our classifier's label names exactly (e.g. "Pleural Effusion" vs "Effusion").
# Map each of our labels to the search terms likely to appear in that field.
CONDITION_SYNONYMS = {
    "No Finding": ["normal"],
    "Infiltration": ["infiltrate", "infiltration", "opacity"],
    "Effusion": ["effusion", "pleural effusion"],
    "Atelectasis": ["atelectasis", "atelectatic"],
    "Nodule": ["nodule", "nodular"],
    "Mass": ["mass", "neoplasm"],
    "Pneumothorax": ["pneumothorax"],
    "Consolidation": ["consolidation", "consolidative"],
    "Pleural_Thickening": ["pleural thickening", "thickening", "pleural"],
    "Cardiomegaly": ["cardiomegaly", "cardiac"],
    "Pneumonia": ["pneumonia"],
}


def load_index(index_path="models/report_index.pkl"):
    if not os.path.exists(index_path):
        return None
    with open(index_path, "rb") as f:
        return pickle.load(f)


def retrieve_similar_report(index, condition, top_k=1):
    """Find the most similar real report mentioning this condition, if any."""
    if index is None:
        return None
    reports = index["reports"]

    synonyms = CONDITION_SYNONYMS.get(condition, [condition])
    mask = pd.Series(False, index=reports.index)
    for term in synonyms:
        mask = mask | reports["labels"].str.contains(term, case=False, na=False)

    if not mask.any():
        return None

    query_vec = index["vectorizer"].transform([" ".join(synonyms)])
    subset_matrix = index["tfidf_matrix"][mask.values]
    sims = cosine_similarity(query_vec, subset_matrix)[0]
    best_idx = sims.argmax()
    matched_reports = reports[mask].reset_index(drop=True)
    return matched_reports.iloc[best_idx]["text"]


def generate_report(condition, confidence, region, urgency_note="", index=None):
    condition_phrase = CONDITION_PHRASES.get(condition, condition.lower())
    confidence_pct = round(confidence * 100, 1)

    retrieved = retrieve_similar_report(index, condition) if index else None

    if retrieved:
        # Ground the retrieved real report text with our own structured facts
        # up front, so the output stays tied to this specific prediction.
        cleaned = clean_report_text(retrieved)
        report = (
            f"AI-assisted finding: {condition_phrase}, focused in the {region} "
            f"(confidence {confidence_pct}%).\n\n"
            f"Similar documented finding pattern: {cleaned[:400]}\n\n"
            f"{urgency_note}"
        )
    else:
        report = TEMPLATE.format(
            condition_phrase=condition_phrase,
            region=region,
            confidence_pct=confidence_pct,
            urgency_note=urgency_note,
        )

    return report


if __name__ == "__main__":
    idx = load_index()
    print(generate_report("Pneumonia", 0.81, "lower right lung field", index=idx))