import json
import os

from openai import OpenAI


def run(note_text: str) -> dict:
    """Extract structured clinical information from a raw note string."""
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    response = client.chat.completions.create(
        model="gpt-4o",
        response_format={"type": "json_object"},
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a clinical information extraction specialist. "
                    "Extract all structured information from clinical notes. "
                    "Always respond with valid JSON only, no explanation."
                ),
            },
            {
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
    return json.loads(response.choices[0].message.content)
