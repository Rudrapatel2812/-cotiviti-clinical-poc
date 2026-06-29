import json
import os

from openai import OpenAI


def run(extraction: dict, pattern_risk: dict) -> dict:
    """Synthesize clinical findings into ICD-10/CPT codes and explainable summary."""
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    response = client.chat.completions.create(
        model="gpt-4o",
        response_format={"type": "json_object"},
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a clinical documentation and coding specialist. "
                    "Synthesize clinical findings into structured explainable summaries. "
                    "Always respond with valid JSON only."
                ),
            },
            {
                "role": "user",
                "content": (
                    "Given these clinical findings and risk assessment:\n\n"
                    f"EXTRACTED DATA: {json.dumps(extraction, indent=2)}\n"
                    f"PATTERN ANALYSIS: {json.dumps(pattern_risk, indent=2)}\n\n"
                    "Generate:\n"
                    "- clinical_summary: 2-3 sentence plain English summary\n"
                    "- icd10_codes: list of dicts with code and description\n"
                    "- cpt_codes: list of dicts with code and description\n"
                    "- recommended_actions: list of immediate next steps\n"
                    "- explainability: plain English explanation of WHY this case was classified "
                    "at this risk level, referencing specific data points from the note\n"
                    "- payment_flags: list of any billing concerns or upcoding risks "
                    "(empty list if none)\n\n"
                    "Respond with JSON only."
                ),
            },
        ],
    )
    return json.loads(response.choices[0].message.content)
