# ─────────────────────────────────────────────────────────────────────────────
# agents/synthesis_agent.py  —  Agent 3: Synthesis & Explainability
#
# PURPOSE:
#   The final agent in the pipeline. Receives output from both Agent 1
#   (extraction) and Agent 2 (pattern_risk) and produces the clinician-facing
#   and billing-facing deliverables: ICD-10/CPT codes, recommended actions,
#   a plain-English explainability statement, and payment flags.
#
# INPUT:
#   extraction   (dict) — output of extraction_agent.run()
#   pattern_risk (dict) — output of pattern_risk_agent.run()
#
# OUTPUT: dict with keys:
#   clinical_summary     — 2-3 sentence plain English summary of the case
#   icd10_codes          — list of {code, description} dicts
#                          e.g. [{"code": "I21.9", "description": "Acute MI"}]
#   cpt_codes            — list of {code, description} dicts for billing
#   recommended_actions  — ordered list of immediate clinical next steps
#   explainability       — plain English WHY this risk level was assigned,
#                          citing specific data points (troponin value, BP, etc.)
#   payment_flags        — list of billing concerns / upcoding risks
#                          (empty list [] if none identified)
#
# WHY both upstream outputs are passed:
#   ICD-10/CPT coding requires both the raw clinical facts (extraction) AND
#   the interpreted risk context (pattern_risk). Passing both lets GPT-4o
#   correlate specific lab values with the pattern-level diagnosis to produce
#   more accurate codes than either input alone would yield.
# ─────────────────────────────────────────────────────────────────────────────

import json
import os

from openai import OpenAI


def run(extraction: dict, pattern_risk: dict) -> dict:
    """Synthesize Agent 1 + Agent 2 outputs into codes, actions, and explainability."""

    # Build client fresh per call — API key read from environment, never hardcoded
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    response = client.chat.completions.create(
        model="gpt-4o",
        # Force a JSON response — prevents markdown wrapping around the JSON
        response_format={"type": "json_object"},
        messages=[
            {
                # System prompt positions GPT-4o as a medical coding specialist
                "role": "system",
                "content": (
                    "You are a clinical documentation and coding specialist. "
                    "Synthesize clinical findings into structured explainable summaries. "
                    "Always respond with valid JSON only."
                ),
            },
            {
                # Both upstream agent outputs are embedded as labelled JSON blocks
                # so GPT-4o can cross-reference extracted vitals with risk patterns
                # when choosing ICD-10/CPT codes and writing the explainability text
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

    # Parse the guaranteed-JSON response string into a Python dict
    return json.loads(response.choices[0].message.content)
