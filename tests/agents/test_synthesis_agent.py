import json
from unittest.mock import MagicMock, patch

import agents.synthesis_agent as synthesis_agent


FAKE_EXTRACTION = {
    "patient_info": {"age": 58, "gender": "male", "chief_complaint": "chest pain"},
    "diagnoses": ["hypertension", "type 2 diabetes"],
    "medications": ["metformin", "lisinopril"],
    "vitals": {"BP": "155/95", "HR": "102"},
    "lab_results": {"troponin": "0.8"},
    "symptoms": ["chest pain", "diaphoresis"],
}

FAKE_PATTERN_RISK = {
    "clinical_patterns": ["acute coronary syndrome"],
    "risk_level": "CRITICAL",
    "risk_reasons": ["elevated troponin"],
    "documentation_gaps": [],
    "recommended_urgency": "immediate",
}

FAKE_SYNTHESIS = {
    "clinical_summary": "58-year-old male presenting with STEMI. Immediate cath lab activation required.",
    "icd10_codes": [{"code": "I21.9", "description": "Acute myocardial infarction, unspecified"}],
    "cpt_codes": [{"code": "92941", "description": "Percutaneous transluminal revascularization"}],
    "recommended_actions": ["Activate cath lab", "Administer aspirin 325mg", "Start heparin drip"],
    "explainability": "CRITICAL risk due to elevated troponin 0.8 and ST changes.",
    "payment_flags": [],
}


def _mock_openai(content: dict):
    mock_response = MagicMock()
    mock_response.choices[0].message.content = json.dumps(content)
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = mock_response
    return mock_client


def test_run_returns_dict():
    with patch("agents.synthesis_agent.OpenAI") as MockOpenAI:
        MockOpenAI.return_value = _mock_openai(FAKE_SYNTHESIS)
        result = synthesis_agent.run(FAKE_EXTRACTION, FAKE_PATTERN_RISK)
    assert isinstance(result, dict)


def test_run_returns_all_expected_keys():
    with patch("agents.synthesis_agent.OpenAI") as MockOpenAI:
        MockOpenAI.return_value = _mock_openai(FAKE_SYNTHESIS)
        result = synthesis_agent.run(FAKE_EXTRACTION, FAKE_PATTERN_RISK)
    expected = {"clinical_summary", "icd10_codes", "cpt_codes", "recommended_actions", "explainability", "payment_flags"}
    assert expected.issubset(result.keys())


def test_run_includes_both_inputs_in_prompt():
    with patch("agents.synthesis_agent.OpenAI") as MockOpenAI:
        mock_client = _mock_openai(FAKE_SYNTHESIS)
        MockOpenAI.return_value = mock_client
        synthesis_agent.run(FAKE_EXTRACTION, FAKE_PATTERN_RISK)
    call_kwargs = mock_client.chat.completions.create.call_args
    user_message = call_kwargs.kwargs["messages"][1]["content"]
    assert "hypertension" in user_message
    assert "acute coronary syndrome" in user_message


def test_run_uses_gpt4o_model():
    with patch("agents.synthesis_agent.OpenAI") as MockOpenAI:
        mock_client = _mock_openai(FAKE_SYNTHESIS)
        MockOpenAI.return_value = mock_client
        synthesis_agent.run(FAKE_EXTRACTION, FAKE_PATTERN_RISK)
    call_kwargs = mock_client.chat.completions.create.call_args
    assert call_kwargs.kwargs["model"] == "gpt-4o"


def test_icd10_codes_are_list_of_dicts():
    with patch("agents.synthesis_agent.OpenAI") as MockOpenAI:
        MockOpenAI.return_value = _mock_openai(FAKE_SYNTHESIS)
        result = synthesis_agent.run(FAKE_EXTRACTION, FAKE_PATTERN_RISK)
    assert isinstance(result["icd10_codes"], list)
    assert all("code" in c and "description" in c for c in result["icd10_codes"])
