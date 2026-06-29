# ─────────────────────────────────────────────────────────────────────────────
# data/clinical_notes.py
#
# Hardcoded sample clinical notes used as input to the agent pipeline.
# In a real system these would come from an EHR or HL7/FHIR feed.
#
# Each note is a dict with three keys:
#   id    — patient identifier (used as the JSON output filename)
#   title — short display label shown in the dashboard
#   note  — raw free-text clinical note sent to Agent 1 (Extraction)
#
# Three cases are intentionally chosen to stress-test different paths:
#   P001 — rich, complete cardiac note  → expected CRITICAL risk
#   P002 — rich, complete obstetric note → expected CRITICAL risk
#   P003 — sparse, incomplete note       → tests documentation gap detection
# ─────────────────────────────────────────────────────────────────────────────

CLINICAL_NOTES = [
    {
        "id": "P001",
        "title": "Cardiac Emergency",
        # Classic ACS presentation: elevated troponin + ST changes + multiple
        # cardiac risk factors (HTN, T2DM, smoking). Designed to trigger
        # CRITICAL risk and ICD-10 codes in the I20-I25 range.
        "note": (
            "58 year old male presents with chest pain radiating to the left arm, "
            "onset 2 hours ago. Patient has a history of hypertension and Type 2 diabetes. "
            "Smoker for 20 years, approximately 1 pack per day. "
            "Vital signs: BP 155/95 mmHg, HR 102 bpm, RR 18, O2 sat 96% on room air. "
            "Troponin elevated at 0.8 ng/mL. EKG shows ST changes in leads II, III, aVF. "
            "Current medications: metformin 1000mg BID, lisinopril 10mg daily. "
            "No known drug allergies. Patient is diaphoretic and appears in moderate distress."
        ),
    },
    {
        "id": "P002",
        "title": "Obstetric Emergency",
        # Preeclampsia / suspected HELLP syndrome: BP ≥160/110, proteinuria 3+,
        # low platelets. Designed to trigger CRITICAL risk and O14.x ICD-10 codes.
        "note": (
            "34 year old pregnant female at 28 weeks gestation presents with severe headache "
            "and visual changes described as blurry vision with floaters, onset 6 hours ago. "
            "This is her first pregnancy. "
            "Vital signs: BP 168/110 mmHg, HR 88 bpm, RR 16, O2 sat 99% on room air. "
            "Urine dipstick shows protein 3+. "
            "Lab results: platelets 95,000 per microliter (low), LFTs mildly elevated. "
            "No current medications. No prior obstetric complications. "
            "Patient denies fever, chest pain, or abdominal pain. "
            "Fetal heart tones present and regular."
        ),
    },
    {
        "id": "P003",
        "title": "Incomplete Documentation Case",
        # Intentionally sparse: no vitals, no labs, no history, no meds.
        # Used to demonstrate Agent 2's documentation gap detection feature.
        # Expected to show 5+ gaps in the dashboard's Documentation Gaps tab.
        "note": (
            "Patient presents with fever and cough. "
            "No vitals recorded. "
            "No lab results available. "
            "No medical history documented. "
            "No current medications listed. "
            "Duration and severity of symptoms not noted."
        ),
    },
]
