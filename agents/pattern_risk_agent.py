# ─────────────────────────────────────────────────────────────────────────────
# agents/pattern_risk_agent.py  —  Agent 2: Pattern Recognition & Risk
#
# PURPOSE:
#   Receive the structured extraction from Agent 1 and identify clinical
#   patterns, assign a risk level, and flag documentation gaps.
#   Its output feeds directly into Agent 3 (synthesis_agent).
#
# INPUT:  extraction (dict) — output of extraction_agent.run()
# OUTPUT: dict with keys:
#           clinical_patterns     — list of recognized medical patterns
#                                   e.g. ["Acute Coronary Syndrome", "Hypertensive Crisis"]
#           risk_level            — one of: CRITICAL / HIGH / MEDIUM / LOW
#           risk_reasons          — list explaining WHY that risk level was chosen
#           documentation_gaps    — list of missing info required for proper documentation
#                                   (will be long for P003 — incomplete note case)
#           recommended_urgency   — one of: immediate / urgent / routine
#
# KEY DESIGN NOTE:
#   The extraction dict is serialized to JSON before being embedded in the prompt.
#   This keeps the prompt structured and machine-readable rather than prose,
#   which helps GPT-4o reason more reliably about specific field values.
# ─────────────────────────────────────────────────────────────────────────────

import json
import os

from openai import OpenAI


def run(extraction: dict) -> dict:
    """Identify clinical patterns and classify risk level from extracted note data."""

    # Build client fresh per call — API key read from environment, never hardcoded
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    response = client.chat.completions.create(
        model="gpt-4o",
        # Force a JSON response — prevents markdown wrapping around the JSON
        response_format={"type": "json_object"},
        messages=[
            {
                # System prompt establishes the agent as a clinical risk specialist
                "role": "system",
                "content": (
                    "You are a clinical pattern recognition specialist. "
                    "Identify patterns and classify risk from extracted clinical data. "
                    "Always respond with valid JSON only."
                ),
            },
            {
                # Pass Agent 1's full output as structured JSON so GPT-4o
                # can reason precisely over field values (e.g. troponin level,
                # blood pressure numbers) rather than reparsing free text
                "role": "user",
                "content": (
                    "Given these extracted clinical findings:\n"
                    f"{json.dumps(extraction, indent=2)}\n\n"
                    "Identify:\n"
                    "- clinical_patterns: list of recognized medical patterns\n"
                    "- risk_level: one of CRITICAL / HIGH / MEDIUM / LOW\n"
                    "- risk_reasons: list of specific reasons for risk level\n"
                    "- documentation_gaps: list of missing information that should be present "
                    "for proper clinical documentation\n"
                    "- recommended_urgency: immediate / urgent / routine\n\n"
                    "Respond with JSON only."
                ),
            },
        ],
    )

    # Parse the guaranteed-JSON response string into a Python dict
    return json.loads(response.choices[0].message.content)
