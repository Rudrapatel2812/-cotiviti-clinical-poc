import json
import os

from openai import OpenAI


def run(extraction: dict) -> dict:
    """Identify clinical patterns and classify risk from extracted data."""
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    response = client.chat.completions.create(
        model="gpt-4o",
        response_format={"type": "json_object"},
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a clinical pattern recognition specialist. "
                    "Identify patterns and classify risk from extracted clinical data. "
                    "Always respond with valid JSON only."
                ),
            },
            {
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
    return json.loads(response.choices[0].message.content)
