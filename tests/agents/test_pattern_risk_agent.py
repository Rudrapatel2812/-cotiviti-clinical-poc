import json
from unittest.mock import MagicMock, patch

import agents.pattern_risk_agent as pattern_risk_agent


FAKE_EXTRACTION = {
    "patient_info": {"age": 58, "gender": "male", "chief_complaint": "chest pain"},
    "diagnoses": ["hypertension", "type 2 diabetes"],
    "medications": ["metformin", "lisinopril"],
    "vitals": {"BP": "155/95", "HR": "102"},
    "lab_results": {"troponin": "0.8"},
    "symptoms": ["chest pain", "diaphoresis"],
}

FAKE_PATTERN_RISK = {
    "clinical_patterns": ["acute coronary syndrome", "hypertensive emergency"],
    "risk_level": "CRITICAL",
    "risk_reasons": ["elevated troponin", "ST changes", "multiple cardiac risk factors"],
    "documentation_gaps": [],
    "recommended_urgency": "immediate",
}


def _mock_openai(content: dict):
    mock_response = MagicMock()
    mock_response.choices[0].message.content = json.dumps(content)
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = mock_response
    return mock_client


def test_run_returns_dict():
    with patch("agents.pattern_risk_agent.OpenAI") as MockOpenAI:
        MockOpenAI.return_value = _mock_openai(FAKE_PATTERN_RISK)
        result = pattern_risk_agent.run(FAKE_EXTRACTION)
    assert isinstance(result, dict)


def test_run_returns_all_expected_keys():
    with patch("agents.pattern_risk_agent.OpenAI") as MockOpenAI:
        MockOpenAI.return_value = _mock_openai(FAKE_PATTERN_RISK)
        result = pattern_risk_agent.run(FAKE_EXTRACTION)
    expected = {"clinical_patterns", "risk_level", "risk_reasons", "documentation_gaps", "recommended_urgency"}
    assert expected.issubset(result.keys())


def test_run_serializes_extraction_into_prompt():
    with patch("agents.pattern_risk_agent.OpenAI") as MockOpenAI:
        mock_client = _mock_openai(FAKE_PATTERN_RISK)
        MockOpenAI.return_value = mock_client
        pattern_risk_agent.run(FAKE_EXTRACTION)
    call_kwargs = mock_client.chat.completions.create.call_args
    user_message = call_kwargs.kwargs["messages"][1]["content"]
    assert "hypertension" in user_message


def test_run_uses_gpt4o_model():
    with patch("agents.pattern_risk_agent.OpenAI") as MockOpenAI:
        mock_client = _mock_openai(FAKE_PATTERN_RISK)
        MockOpenAI.return_value = mock_client
        pattern_risk_agent.run(FAKE_EXTRACTION)
    call_kwargs = mock_client.chat.completions.create.call_args
    assert call_kwargs.kwargs["model"] == "gpt-4o"


def test_run_requests_json_object_format():
    with patch("agents.pattern_risk_agent.OpenAI") as MockOpenAI:
        mock_client = _mock_openai(FAKE_PATTERN_RISK)
        MockOpenAI.return_value = mock_client
        pattern_risk_agent.run(FAKE_EXTRACTION)
    call_kwargs = mock_client.chat.completions.create.call_args
    assert call_kwargs.kwargs["response_format"] == {"type": "json_object"}
