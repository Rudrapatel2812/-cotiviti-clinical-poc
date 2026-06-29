import json
from unittest.mock import MagicMock, patch

import agents.extraction_agent as extraction_agent


FAKE_EXTRACTION = {
    "patient_info": {"age": 58, "gender": "male", "chief_complaint": "chest pain"},
    "diagnoses": ["hypertension", "type 2 diabetes"],
    "medications": ["metformin 1000mg BID", "lisinopril 10mg daily"],
    "vitals": {"BP": "155/95 mmHg", "HR": "102 bpm"},
    "lab_results": {"troponin": "0.8 ng/mL"},
    "symptoms": ["chest pain radiating to left arm", "diaphoresis"],
}


def _mock_openai(content: dict):
    """Return a patched OpenAI client whose create() returns content as JSON."""
    mock_response = MagicMock()
    mock_response.choices[0].message.content = json.dumps(content)
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = mock_response
    return mock_client


def test_run_returns_dict():
    with patch("agents.extraction_agent.OpenAI") as MockOpenAI:
        MockOpenAI.return_value = _mock_openai(FAKE_EXTRACTION)
        result = extraction_agent.run("58 year old male with chest pain...")
    assert isinstance(result, dict)


def test_run_returns_all_expected_keys():
    with patch("agents.extraction_agent.OpenAI") as MockOpenAI:
        MockOpenAI.return_value = _mock_openai(FAKE_EXTRACTION)
        result = extraction_agent.run("58 year old male with chest pain...")
    expected_keys = {"patient_info", "diagnoses", "medications", "vitals", "lab_results", "symptoms"}
    assert expected_keys.issubset(result.keys())


def test_run_passes_note_text_to_api():
    note = "unique clinical note text for testing"
    with patch("agents.extraction_agent.OpenAI") as MockOpenAI:
        mock_client = _mock_openai(FAKE_EXTRACTION)
        MockOpenAI.return_value = mock_client
        extraction_agent.run(note)
    call_kwargs = mock_client.chat.completions.create.call_args
    user_message = call_kwargs.kwargs["messages"][1]["content"]
    assert note in user_message


def test_run_uses_gpt4o_model():
    with patch("agents.extraction_agent.OpenAI") as MockOpenAI:
        mock_client = _mock_openai(FAKE_EXTRACTION)
        MockOpenAI.return_value = mock_client
        extraction_agent.run("test note")
    call_kwargs = mock_client.chat.completions.create.call_args
    assert call_kwargs.kwargs["model"] == "gpt-4o"


def test_run_requests_json_object_format():
    with patch("agents.extraction_agent.OpenAI") as MockOpenAI:
        mock_client = _mock_openai(FAKE_EXTRACTION)
        MockOpenAI.return_value = mock_client
        extraction_agent.run("test note")
    call_kwargs = mock_client.chat.completions.create.call_args
    assert call_kwargs.kwargs["response_format"] == {"type": "json_object"}
