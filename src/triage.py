"""
Rule-based urgency scoring. Deliberately NOT a learned model -- triage
decisions need to be auditable and explainable to clinical staff, and a
simple rule table is easier to validate with a clinician than a black box.

Adjust SEVERITY and the thresholds below in consultation with whatever
clinical guidance you have access to for the project; the values here are
placeholders for demo purposes, not vetted medical thresholds.
"""

SEVERITY = {
    "No Finding": 0,
    "Nodule": 1,
    "Atelectasis": 1,
    "Pleural_Thickening": 1,
    "Infiltration": 1,
    "Mass": 2,
    "Consolidation": 2,
    "Effusion": 2,
    "Pneumonia": 2,
    "Cardiomegaly": 2,
    "Pneumothorax": 3,
}

URGENCY_TIERS = {
    0: ("Routine", "No urgent action indicated based on this image alone."),
    1: ("Soon", "Please have a clinician review within the next few days."),
    2: ("Urgent", "Please see a doctor within 24 hours."),
    3: ("Immediate", "Seek immediate medical attention."),
}


def assess(condition, confidence):
    base_severity = SEVERITY.get(condition, 1)

    # Low-confidence high-severity predictions get bumped down a tier --
    # we don't want an uncertain model call causing false alarm fatigue,
    # but we also flag it as needing human review either way.
    if confidence < 0.5 and base_severity > 0:
        effective_severity = max(base_severity - 1, 1)
        note_suffix = " (Note: model confidence is low -- please verify.)"
    else:
        effective_severity = base_severity
        note_suffix = ""

    tier_label, tier_note = URGENCY_TIERS[effective_severity]
    return {
        "tier": tier_label,
        "message": tier_note + note_suffix,
        "severity_score": effective_severity,
    }


if __name__ == "__main__":
    print(assess("Pneumonia", 0.81))
    print(assess("Pneumonia", 0.4))
    print(assess("No Finding", 0.95))