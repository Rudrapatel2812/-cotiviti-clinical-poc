# ─────────────────────────────────────────────────────────────────────────────
# agents/extraction_agent.py  —  Agent 1: Information Extraction
#
# PURPOSE:
#   Convert a raw, free-text clinical note into a structured Python dict.
#   This is the first step in the 3-agent pipeline. Its output feeds
#   directly into Agent 2 (pattern_risk_agent).
#
# INPUT:  note_text (str) — raw clinical note from data/clinical_notes.py
# OUTPUT: dict with keys:
#           patient_info   — {age, gender, chief_complaint}
#           diagnoses      — list of identified conditions
#           medications    — list of current medications
#           vitals         — {BP: "155/95 mmHg", HR: "102 bpm", ...}
#           lab_results    — {troponin: "0.8 ng/mL", ...}
#           symptoms       — list of reported symptoms
#           (any missing field is returned as null by GPT-4o)
#
# WHY response_format={"type": "json_object"}:
#   Without this flag, GPT-4o sometimes wraps JSON in markdown code fences
#   (```json ... ```), which breaks json.loads(). This flag guarantees a
#   raw JSON string so no post-processing is needed.
# ─────────────────────────────────────────────────────────────────────────────

import json
import os

from openai import OpenAI


def run(note_text: str) -> dict:
    """Send a clinical note to GPT-4o and return structured extracted fields."""

    # Build client fresh per call — API key read from environment, never hardcoded
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    response = client.chat.completions.create(
        model="gpt-4o",
        # Force a JSON response — prevents markdown wrapping around the JSON
        response_format={"type": "json_object"},
        messages=[
            {
                # System prompt defines the agent's role and output contract
                "role": "system",
                "content": (
                    "You are a clinical information extraction specialist. "
                    "Extract all structured information from clinical notes. "
                    "Always respond with valid JSON only, no explanation."
                ),
            },
            {
                # User prompt specifies the exact fields to extract and the note to process
                "role": "user",
                "content": (
                    "Extract from this clinical note:\n"
                    "- patient_info: age, gender, chief complaint\n"
                    "- diagnoses: list of identified conditions\n"
                    "- medications: list of current medications\n"
                    "- vitals: dict of vital signs with values\n"
                    "- lab_results: dict of lab values\n"
                    "- symptoms: list of reported symptoms\n\n"
                    f"Clinical Note: {note_text}\n\n"
                    "Respond with JSON only. If any field is absent from the note, use null."
                ),
            },
        ],
    )

    # Parse the guaranteed-JSON response string into a Python dict
    return json.loads(response.choices[0].message.content)
