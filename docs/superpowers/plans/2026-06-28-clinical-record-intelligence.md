# Multi-Agent Clinical Record Intelligence — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a working 3-agent GPT-4o pipeline over hardcoded clinical notes with a Streamlit dashboard that renders extraction, risk, coding, and gap analysis results.

**Architecture:** Three plain Python functions (one per agent file) called sequentially by `main.py`. Each agent calls GPT-4o with `response_format={"type": "json_object"}` and returns a parsed dict. `dashboard.py` loads pre-computed JSON from `results/all_results.json` and re-runs the pipeline only when cache is missing.

**Tech Stack:** Python 3.10+, openai, streamlit, python-dotenv, pytest (tests only), unittest.mock (stdlib)

## Global Constraints

- All agent files: single exported `run()` function, no classes
- All OpenAI calls: model `gpt-4o`, `response_format={"type": "json_object"}`
- API key: `os.environ["OPENAI_API_KEY"]` — never hardcoded
- No third-party libraries beyond openai, streamlit, python-dotenv
- Every agent returns a `dict` (not a string)
- Files: max 800 lines; functions: max 50 lines
- `main.py` must be importable (logic in `main()` function, `if __name__ == "__main__": main()` guard)

---

## File Map

| File | Responsibility |
|------|---------------|
| `data/__init__.py` | Package marker |
| `data/clinical_notes.py` | 3 hardcoded sample notes as `CLINICAL_NOTES: list[dict]` |
| `agents/__init__.py` | Package marker |
| `agents/extraction_agent.py` | Agent 1: structured info extraction from note text |
| `agents/pattern_risk_agent.py` | Agent 2: pattern recognition + risk classification |
| `agents/synthesis_agent.py` | Agent 3: ICD-10/CPT coding + explainability |
| `main.py` | Orchestrator: runs all 3 agents per patient, writes JSON to `results/` |
| `dashboard.py` | Streamlit UI: 4-tab display, cache-first pipeline trigger |
| `requirements.txt` | Runtime dependencies |
| `README.md` | Setup and usage docs |
| `tests/__init__.py` | Package marker |
| `tests/data/__init__.py` | Package marker |
| `tests/data/test_clinical_notes.py` | Tests for data module |
| `tests/agents/__init__.py` | Package marker |
| `tests/agents/test_extraction_agent.py` | Agent 1 unit tests (mocked OpenAI) |
| `tests/agents/test_pattern_risk_agent.py` | Agent 2 unit tests (mocked OpenAI) |
| `tests/agents/test_synthesis_agent.py` | Agent 3 unit tests (mocked OpenAI) |
| `tests/test_main.py` | Orchestrator unit tests (mocked agents + tmp filesystem) |

---

## Task 1: Project Scaffold

**Files:**
- Create: `requirements.txt`
- Create: `data/__init__.py`
- Create: `agents/__init__.py`
- Create: `tests/__init__.py`
- Create: `tests/data/__init__.py`
- Create: `tests/agents/__init__.py`
- Create: `.env.example`
- Create: `.gitignore`

**Interfaces:**
- Produces: runnable Python package structure; `pytest` can discover all test files

- [ ] **Step 1: Create requirements.txt**

```
openai
streamlit
python-dotenv
pytest
```

- [ ] **Step 2: Create all __init__.py files and support files**

Create the following empty files (each is just an empty file):
- `data/__init__.py`
- `agents/__init__.py`
- `tests/__init__.py`
- `tests/data/__init__.py`
- `tests/agents/__init__.py`

Create `.env.example`:
```
OPENAI_API_KEY=your_openai_key_here
```

Create `.gitignore`:
```
.env
results/
__pycache__/
*.pyc
.pytest_cache/
```

- [ ] **Step 3: Install dependencies**

Run: `pip install -r requirements.txt`
Expected: All packages install without errors.

- [ ] **Step 4: Verify pytest discovers tests**

Run: `pytest tests/ --collect-only`
Expected: "no tests ran" (no tests written yet) — but no import errors.

- [ ] **Step 5: Commit**

```bash
git init
git add requirements.txt data/__init__.py agents/__init__.py tests/__init__.py tests/data/__init__.py tests/agents/__init__.py .env.example .gitignore
git commit -m "chore: project scaffold"
```

---

## Task 2: Clinical Notes Data Module

**Files:**
- Create: `data/clinical_notes.py`
- Create: `tests/data/test_clinical_notes.py`

**Interfaces:**
- Produces: `CLINICAL_NOTES: list[dict]` — each dict has keys `id: str`, `title: str`, `note: str`
- Consumed by: `main.py` (imports `CLINICAL_NOTES`)

- [ ] **Step 1: Write the failing test**

Create `tests/data/test_clinical_notes.py`:
```python
from data.clinical_notes import CLINICAL_NOTES


def test_has_three_notes():
    assert len(CLINICAL_NOTES) == 3


def test_each_note_has_required_keys():
    required = {"id", "title", "note"}
    for note in CLINICAL_NOTES:
        assert required.issubset(note.keys()), f"Missing keys in {note.get('id')}"


def test_patient_ids_are_correct():
    ids = [n["id"] for n in CLINICAL_NOTES]
    assert ids == ["P001", "P002", "P003"]


def test_notes_are_non_empty_strings():
    for note in CLINICAL_NOTES:
        assert isinstance(note["note"], str) and len(note["note"]) > 50
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/data/test_clinical_notes.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'data.clinical_notes'`

- [ ] **Step 3: Write implementation**

Create `data/clinical_notes.py`:
```python
CLINICAL_NOTES = [
    {
        "id": "P001",
        "title": "Cardiac Emergency",
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/data/test_clinical_notes.py -v`
Expected:
```
PASSED tests/data/test_clinical_notes.py::test_has_three_notes
PASSED tests/data/test_clinical_notes.py::test_each_note_has_required_keys
PASSED tests/data/test_clinical_notes.py::test_patient_ids_are_correct
PASSED tests/data/test_clinical_notes.py::test_notes_are_non_empty_strings
4 passed
```

- [ ] **Step 5: Commit**

```bash
git add data/clinical_notes.py tests/data/test_clinical_notes.py
git commit -m "feat: add hardcoded clinical notes data module"
```

---

## Task 3: Extraction Agent (Agent 1)

**Files:**
- Create: `agents/extraction_agent.py`
- Create: `tests/agents/test_extraction_agent.py`

**Interfaces:**
- Produces: `run(note_text: str) -> dict` with keys: `patient_info`, `diagnoses`, `medications`, `vitals`, `lab_results`, `symptoms`
- Consumed by: `main.py` (step 1 per patient), `pattern_risk_agent` indirectly via main

- [ ] **Step 1: Write the failing tests**

Create `tests/agents/test_extraction_agent.py`:
```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/agents/test_extraction_agent.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'agents.extraction_agent'` (or similar import error)

- [ ] **Step 3: Write implementation**

Create `agents/extraction_agent.py`:
```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/agents/test_extraction_agent.py -v`
Expected:
```
PASSED tests/agents/test_extraction_agent.py::test_run_returns_dict
PASSED tests/agents/test_extraction_agent.py::test_run_returns_all_expected_keys
PASSED tests/agents/test_extraction_agent.py::test_run_passes_note_text_to_api
PASSED tests/agents/test_extraction_agent.py::test_run_uses_gpt4o_model
PASSED tests/agents/test_extraction_agent.py::test_run_requests_json_object_format
5 passed
```

- [ ] **Step 5: Commit**

```bash
git add agents/extraction_agent.py tests/agents/test_extraction_agent.py
git commit -m "feat: add extraction agent (Agent 1)"
```

---

## Task 4: Pattern Risk Agent (Agent 2)

**Files:**
- Create: `agents/pattern_risk_agent.py`
- Create: `tests/agents/test_pattern_risk_agent.py`

**Interfaces:**
- Consumes: `extraction: dict` — output of `extraction_agent.run()`
- Produces: `run(extraction: dict) -> dict` with keys: `clinical_patterns`, `risk_level`, `risk_reasons`, `documentation_gaps`, `recommended_urgency`
- Consumed by: `main.py` (step 2 per patient), `synthesis_agent.run()`

- [ ] **Step 1: Write the failing tests**

Create `tests/agents/test_pattern_risk_agent.py`:
```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/agents/test_pattern_risk_agent.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'agents.pattern_risk_agent'`

- [ ] **Step 3: Write implementation**

Create `agents/pattern_risk_agent.py`:
```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/agents/test_pattern_risk_agent.py -v`
Expected:
```
PASSED tests/agents/test_pattern_risk_agent.py::test_run_returns_dict
PASSED tests/agents/test_pattern_risk_agent.py::test_run_returns_all_expected_keys
PASSED tests/agents/test_pattern_risk_agent.py::test_run_serializes_extraction_into_prompt
PASSED tests/agents/test_pattern_risk_agent.py::test_run_uses_gpt4o_model
PASSED tests/agents/test_pattern_risk_agent.py::test_run_requests_json_object_format
5 passed
```

- [ ] **Step 5: Commit**

```bash
git add agents/pattern_risk_agent.py tests/agents/test_pattern_risk_agent.py
git commit -m "feat: add pattern risk agent (Agent 2)"
```

---

## Task 5: Synthesis Agent (Agent 3)

**Files:**
- Create: `agents/synthesis_agent.py`
- Create: `tests/agents/test_synthesis_agent.py`

**Interfaces:**
- Consumes: `extraction: dict` (from `extraction_agent.run()`), `pattern_risk: dict` (from `pattern_risk_agent.run()`)
- Produces: `run(extraction: dict, pattern_risk: dict) -> dict` with keys: `clinical_summary`, `icd10_codes`, `cpt_codes`, `recommended_actions`, `explainability`, `payment_flags`
- Consumed by: `main.py` (step 3 per patient)

- [ ] **Step 1: Write the failing tests**

Create `tests/agents/test_synthesis_agent.py`:
```python
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
    assert "hypertension" in user_message       # from extraction
    assert "acute coronary syndrome" in user_message  # from pattern_risk


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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/agents/test_synthesis_agent.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'agents.synthesis_agent'`

- [ ] **Step 3: Write implementation**

Create `agents/synthesis_agent.py`:
```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/agents/test_synthesis_agent.py -v`
Expected:
```
PASSED tests/agents/test_synthesis_agent.py::test_run_returns_dict
PASSED tests/agents/test_synthesis_agent.py::test_run_returns_all_expected_keys
PASSED tests/agents/test_synthesis_agent.py::test_run_includes_both_inputs_in_prompt
PASSED tests/agents/test_synthesis_agent.py::test_run_uses_gpt4o_model
PASSED tests/agents/test_synthesis_agent.py::test_icd10_codes_are_list_of_dicts
5 passed
```

- [ ] **Step 5: Commit**

```bash
git add agents/synthesis_agent.py tests/agents/test_synthesis_agent.py
git commit -m "feat: add synthesis agent (Agent 3)"
```

---

## Task 6: Main Orchestrator

**Files:**
- Create: `main.py`
- Create: `tests/test_main.py`

**Interfaces:**
- Consumes: `CLINICAL_NOTES` from `data.clinical_notes`, `run()` from all 3 agent modules
- Produces: `results/P001.json`, `results/P002.json`, `results/P003.json`, `results/all_results.json`
- Each output JSON has shape: `{patient_id, title, original_note, extraction, pattern_risk, synthesis}`
- Exports: `main()` function (importable)

- [ ] **Step 1: Write the failing tests**

Create `tests/test_main.py`:
```python
import json
import os
from pathlib import Path
from unittest.mock import patch

import pytest

import main as main_module


FAKE_EXTRACTION = {"patient_info": {"age": 58}, "diagnoses": ["hypertension"]}
FAKE_PATTERN_RISK = {"risk_level": "CRITICAL", "clinical_patterns": []}
FAKE_SYNTHESIS = {"clinical_summary": "Test summary", "icd10_codes": []}


@pytest.fixture
def results_dir(tmp_path, monkeypatch):
    """Redirect results/ writes to a tmp directory and set a fake API key."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("OPENAI_API_KEY", "test-key-123")
    (tmp_path / "data").mkdir()
    return tmp_path


def test_main_creates_results_directory(results_dir):
    with (
        patch("main.extraction_agent.run", return_value=FAKE_EXTRACTION),
        patch("main.pattern_risk_agent.run", return_value=FAKE_PATTERN_RISK),
        patch("main.synthesis_agent.run", return_value=FAKE_SYNTHESIS),
    ):
        main_module.main()
    assert (results_dir / "results").is_dir()


def test_main_writes_individual_patient_files(results_dir):
    with (
        patch("main.extraction_agent.run", return_value=FAKE_EXTRACTION),
        patch("main.pattern_risk_agent.run", return_value=FAKE_PATTERN_RISK),
        patch("main.synthesis_agent.run", return_value=FAKE_SYNTHESIS),
    ):
        main_module.main()
    for pid in ["P001", "P002", "P003"]:
        assert (results_dir / "results" / f"{pid}.json").exists()


def test_main_writes_all_results_json(results_dir):
    with (
        patch("main.extraction_agent.run", return_value=FAKE_EXTRACTION),
        patch("main.pattern_risk_agent.run", return_value=FAKE_PATTERN_RISK),
        patch("main.synthesis_agent.run", return_value=FAKE_SYNTHESIS),
    ):
        main_module.main()
    all_results_path = results_dir / "results" / "all_results.json"
    assert all_results_path.exists()
    data = json.loads(all_results_path.read_text())
    assert len(data) == 3


def test_main_output_has_required_keys(results_dir):
    with (
        patch("main.extraction_agent.run", return_value=FAKE_EXTRACTION),
        patch("main.pattern_risk_agent.run", return_value=FAKE_PATTERN_RISK),
        patch("main.synthesis_agent.run", return_value=FAKE_SYNTHESIS),
    ):
        main_module.main()
    p001 = json.loads((results_dir / "results" / "P001.json").read_text())
    required = {"patient_id", "title", "original_note", "extraction", "pattern_risk", "synthesis"}
    assert required.issubset(p001.keys())


def test_main_exits_without_api_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(SystemExit):
        main_module.main()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_main.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'main'`

- [ ] **Step 3: Write implementation**

Create `main.py`:
```python
import json
import os
import sys
from pathlib import Path

from data.clinical_notes import CLINICAL_NOTES
from agents import extraction_agent, pattern_risk_agent, synthesis_agent


def main():
    """Run the full 3-agent pipeline over all clinical notes and save results."""
    if not os.environ.get("OPENAI_API_KEY"):
        sys.exit("Error: OPENAI_API_KEY environment variable is not set.")

    Path("results").mkdir(exist_ok=True)
    all_results = []

    for note in CLINICAL_NOTES:
        patient_id = note["id"]
        print(f"\nAnalyzing {patient_id}: {note['title']}...")

        extraction = extraction_agent.run(note["note"])
        print("  Extraction complete")

        pattern_risk = pattern_risk_agent.run(extraction)
        print("  Pattern analysis complete")

        synthesis = synthesis_agent.run(extraction, pattern_risk)
        print("  Synthesis complete")

        result = {
            "patient_id": patient_id,
            "title": note["title"],
            "original_note": note["note"],
            "extraction": extraction,
            "pattern_risk": pattern_risk,
            "synthesis": synthesis,
        }

        with open(f"results/{patient_id}.json", "w") as f:
            json.dump(result, f, indent=2)

        all_results.append(result)

    with open("results/all_results.json", "w") as f:
        json.dump(all_results, f, indent=2)

    print("\nAll patients processed")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_main.py -v`
Expected:
```
PASSED tests/test_main.py::test_main_creates_results_directory
PASSED tests/test_main.py::test_main_writes_individual_patient_files
PASSED tests/test_main.py::test_main_writes_all_results_json
PASSED tests/test_main.py::test_main_output_has_required_keys
PASSED tests/test_main.py::test_main_exits_without_api_key
5 passed
```

- [ ] **Step 5: Run the full test suite**

Run: `pytest tests/ -v`
Expected: All 19 tests pass.

- [ ] **Step 6: Commit**

```bash
git add main.py tests/test_main.py
git commit -m "feat: add main orchestrator with full pipeline"
```

---

## Task 7: Streamlit Dashboard

**Files:**
- Create: `dashboard.py`

**Interfaces:**
- Consumes: `results/all_results.json` (runtime file written by `main.py`)
- No unit tests (visual component); manual verification steps provided below

- [ ] **Step 1: Write dashboard.py**

Create `dashboard.py`:
```python
import json
import subprocess
import sys
from pathlib import Path

import streamlit as st

RESULTS_FILE = Path("results/all_results.json")

RISK_COLORS = {
    "CRITICAL": "#ff4b4b",
    "HIGH": "#ffa500",
    "MEDIUM": "#ffd700",
    "LOW": "#00cc44",
}


def load_results():
    """Load cached results from disk. Returns list or None."""
    if RESULTS_FILE.exists():
        with open(RESULTS_FILE) as f:
            return json.load(f)
    return None


def run_pipeline():
    """Execute main.py as a subprocess and reload results."""
    with st.spinner("Running analysis pipeline... (~60–90 seconds)"):
        proc = subprocess.run(
            [sys.executable, "main.py"],
            capture_output=True,
            text=True,
        )
    if proc.returncode != 0:
        st.error(f"Pipeline failed:\n\n{proc.stderr}")
        return None
    return load_results()


def risk_badge(risk_level: str) -> str:
    color = RISK_COLORS.get(risk_level, "#888888")
    return (
        f'<span style="background:{color};color:white;padding:6px 16px;'
        f'border-radius:4px;font-weight:bold;font-size:1.1em;">'
        f'{risk_level}</span>'
    )


# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(page_title="Clinical Record Intelligence", layout="wide")
st.title("Multi-Agent Clinical Record Intelligence")
st.caption("Cotiviti POC — Agentic AI for TPO")

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Controls")
    results = load_results()

    if results:
        patient_ids = [r["patient_id"] for r in results]
        labels = {r["patient_id"]: f"{r['patient_id']} — {r['title']}" for r in results}
    else:
        patient_ids = ["P001", "P002", "P003"]
        labels = {p: p for p in patient_ids}

    selected_id = st.selectbox(
        "Select Patient",
        patient_ids,
        format_func=lambda x: labels.get(x, x),
    )

    btn_label = "Re-analyze All Patients" if results else "Analyze All Patients"
    if st.button(btn_label, use_container_width=True):
        results = run_pipeline()

    if results is None:
        st.info("Click 'Analyze All Patients' to run the pipeline.")
        st.stop()

# ── Resolve selected patient ──────────────────────────────────────────────────
patient_data = next((r for r in results if r["patient_id"] == selected_id), None)
if patient_data is None:
    st.error(f"No data found for {selected_id}. Try re-analyzing.")
    st.stop()

extraction = patient_data.get("extraction", {})
pattern_risk = patient_data.get("pattern_risk", {})
synthesis = patient_data.get("synthesis", {})
risk_level = pattern_risk.get("risk_level", "UNKNOWN")

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs(
    ["Patient Overview", "Agent Reasoning Chain", "Billing & Coding", "Documentation Gaps"]
)

# ── TAB 1: Patient Overview ───────────────────────────────────────────────────
with tab1:
    st.subheader(patient_data["title"])
    st.markdown(f"**Risk Level:** {risk_badge(risk_level)}", unsafe_allow_html=True)
    st.markdown("---")

    with st.expander("Original Clinical Note"):
        st.text(patient_data.get("original_note", ""))

    col_left, col_right = st.columns(2)

    with col_left:
        st.markdown("**Diagnoses**")
        for d in extraction.get("diagnoses") or []:
            st.write(f"• {d}")

        st.markdown("**Symptoms**")
        for s in extraction.get("symptoms") or []:
            st.write(f"• {s}")

        st.markdown("**Medications**")
        for m in extraction.get("medications") or []:
            st.write(f"• {m}")

    with col_right:
        vitals = extraction.get("vitals") or {}
        if vitals:
            st.markdown("**Vitals**")
            st.table({"Measurement": list(vitals.keys()), "Value": list(vitals.values())})

        labs = extraction.get("lab_results") or {}
        if labs:
            st.markdown("**Lab Results**")
            st.table({"Test": list(labs.keys()), "Value": list(labs.values())})

# ── TAB 2: Agent Reasoning Chain ──────────────────────────────────────────────
with tab2:
    st.subheader("Agent Reasoning Chain")
    st.caption("Each card shows what one agent found. Data flows left → right.")

    col_a, col_arr1, col_b, col_arr2, col_c = st.columns([3, 0.4, 3, 0.4, 3])

    with col_a:
        with st.container(border=True):
            st.markdown("**Agent 1: Extraction**")
            pi = extraction.get("patient_info") or {}
            st.write(f"Age: {pi.get('age', 'N/A')}  Gender: {pi.get('gender', 'N/A')}")
            st.write(f"Chief complaint: {pi.get('chief_complaint', 'N/A')}")
            st.write(f"Diagnoses found: {len(extraction.get('diagnoses') or [])}")
            st.write(f"Symptoms: {len(extraction.get('symptoms') or [])}")
            st.write(f"Medications: {len(extraction.get('medications') or [])}")
            st.write(f"Vitals recorded: {len(extraction.get('vitals') or {})}")
            st.write(f"Labs recorded: {len(extraction.get('lab_results') or {})}")

    with col_arr1:
        st.markdown(
            "<div style='text-align:center;font-size:2em;padding-top:60px'>→</div>",
            unsafe_allow_html=True,
        )

    with col_b:
        with st.container(border=True):
            st.markdown("**Agent 2: Pattern Recognition**")
            st.markdown(f"Risk: **{risk_level}**")
            st.write(f"Urgency: {pattern_risk.get('recommended_urgency', 'N/A')}")
            patterns = pattern_risk.get("clinical_patterns") or []
            if patterns:
                st.markdown("Key patterns:")
                for p in patterns[:4]:
                    st.write(f"• {p}")
            gaps = pattern_risk.get("documentation_gaps") or []
            st.write(f"Documentation gaps: {len(gaps)}")

    with col_arr2:
        st.markdown(
            "<div style='text-align:center;font-size:2em;padding-top:60px'>→</div>",
            unsafe_allow_html=True,
        )

    with col_c:
        with st.container(border=True):
            st.markdown("**Agent 3: Synthesis**")
            st.write(synthesis.get("clinical_summary", ""))
            st.write(f"ICD-10 codes: {len(synthesis.get('icd10_codes') or [])}")
            st.write(f"CPT codes: {len(synthesis.get('cpt_codes') or [])}")
            flags = synthesis.get("payment_flags") or []
            st.write(f"Payment flags: {len(flags)}")

# ── TAB 3: Billing & Coding ───────────────────────────────────────────────────
with tab3:
    st.subheader("Billing & Coding")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**ICD-10 Codes**")
        icd = synthesis.get("icd10_codes") or []
        if icd:
            st.table({"Code": [c["code"] for c in icd], "Description": [c["description"] for c in icd]})
        else:
            st.write("None identified.")

    with col2:
        st.markdown("**CPT Codes**")
        cpt = synthesis.get("cpt_codes") or []
        if cpt:
            st.table({"Code": [c["code"] for c in cpt], "Description": [c["description"] for c in cpt]})
        else:
            st.write("None identified.")

    flags = synthesis.get("payment_flags") or []
    if flags:
        st.markdown("**Payment Flags**")
        for flag in flags:
            st.warning(flag)

    st.markdown("**Recommended Actions**")
    for i, action in enumerate(synthesis.get("recommended_actions") or [], 1):
        st.write(f"{i}. {action}")

    st.markdown("**Explainability**")
    st.info(synthesis.get("explainability", "No explanation available."))

# ── TAB 4: Documentation Gaps ────────────────────────────────────────────────
with tab4:
    st.subheader("Documentation Gaps")
    st.markdown(f"**Recommended Urgency:** `{pattern_risk.get('recommended_urgency', 'N/A')}`")

    gaps = pattern_risk.get("documentation_gaps") or []
    if not gaps:
        st.success("No documentation gaps identified.")
    else:
        st.markdown(f"**{len(gaps)} gap(s) identified:**")
        is_high_risk = risk_level in ("CRITICAL", "HIGH")
        for i, gap in enumerate(gaps):
            if is_high_risk or i < 2:
                st.error(f"Critical gap: {gap}")
            else:
                st.warning(f"Warning: {gap}")
```

- [ ] **Step 2: Manual verification — run with cached results**

Prerequisite: `results/all_results.json` must exist (run `python main.py` first with a real API key).

Run: `streamlit run dashboard.py`
Expected:
- Browser opens at `http://localhost:8501`
- Title "Multi-Agent Clinical Record Intelligence" visible
- Patient dropdown shows P001, P002, P003
- All 4 tabs render without errors
- Risk badge for P001 shows red CRITICAL or orange HIGH
- Tab 4 for P003 shows multiple documentation gaps

- [ ] **Step 3: Manual verification — run without cache**

Delete `results/all_results.json` then reload the page.
Expected: Sidebar shows "Analyze All Patients" button. Clicking it shows spinner, runs pipeline, then renders results.

- [ ] **Step 4: Commit**

```bash
git add dashboard.py
git commit -m "feat: add Streamlit dashboard with 4-tab layout"
```

---

## Task 8: README.md

**Files:**
- Create: `README.md`

- [ ] **Step 1: Write README.md**

Create `README.md`:
```markdown
# Multi-Agent Clinical Record Intelligence
## Cotiviti Intern Assessment POC

### What This Does
Three GPT-4o agents analyze clinical notes in sequence: the first extracts structured data
(vitals, diagnoses, medications), the second identifies clinical patterns and assigns a risk
level, and the third generates ICD-10/CPT codes with plain-English explainability. Results
are rendered in a Streamlit dashboard with tabs for patient overview, agent reasoning chain,
billing/coding, and documentation gaps.

### Architecture

```
clinical_notes.py  →  main.py (orchestrator)
                           │
                     ┌─────▼──────┐
                     │  Agent 1   │  extraction_agent.run(note_text)
                     │ Extraction │  → {patient_info, diagnoses, vitals, ...}
                     └─────┬──────┘
                           │
                     ┌─────▼──────────┐
                     │    Agent 2     │  pattern_risk_agent.run(extraction)
                     │ Pattern + Risk │  → {risk_level, patterns, gaps, ...}
                     └─────┬──────────┘
                           │
                     ┌─────▼──────┐
                     │  Agent 3   │  synthesis_agent.run(extraction, pattern_risk)
                     │ Synthesis  │  → {icd10_codes, cpt_codes, explainability, ...}
                     └─────┬──────┘
                           │
                   results/all_results.json
                           │
                     dashboard.py (Streamlit)
```

### How to Run

1. `pip install -r requirements.txt`
2. `export OPENAI_API_KEY=your_key_here`
3. `python main.py`
4. `streamlit run dashboard.py`

### Tech Stack
- Python 3.10+, OpenAI GPT-4o, Streamlit, python-dotenv

### Agent Pipeline

| Agent | Input | Output |
|-------|-------|--------|
| Agent 1: Extraction | Raw clinical note text | Structured dict: diagnoses, vitals, labs, meds |
| Agent 2: Pattern & Risk | Extraction dict | Risk level (CRITICAL/HIGH/MEDIUM/LOW), patterns, gaps |
| Agent 3: Synthesis | Extraction + Pattern dicts | ICD-10/CPT codes, explainability, payment flags |

### Running Tests

```bash
pytest tests/ -v
```

### Disclaimer
Not intended for clinical use. Educational and demonstration purposes only.
```

- [ ] **Step 2: Commit**

```bash
git add README.md
git commit -m "docs: add README with setup instructions and architecture diagram"
```

---

## Self-Review

**Spec coverage:**
- [x] 3 hardcoded clinical notes (P001/P002/P003) — Task 2
- [x] extraction_agent with exact system/user prompts — Task 3
- [x] pattern_risk_agent with exact system/user prompts — Task 4
- [x] synthesis_agent with exact system/user prompts — Task 5
- [x] main.py orchestrator with progress prints + results/ output — Task 6
- [x] all_results.json final shape — Task 6
- [x] Streamlit dashboard 4 tabs — Task 7
- [x] Sidebar dropdown + Analyze button (cache-first) — Task 7
- [x] Risk badge colors — Task 7
- [x] Documentation gap color coding — Task 7
- [x] requirements.txt (openai, streamlit, python-dotenv) — Task 1
- [x] README with ASCII art, how-to-run, disclaimer — Task 8

**Placeholder scan:** No TBDs, TODOs, or vague steps found.

**Type consistency:**
- `extraction_agent.run(note_text: str) -> dict` — used consistently in Task 6 as `extraction_agent.run(note["note"])`
- `pattern_risk_agent.run(extraction: dict) -> dict` — used consistently in Task 6 as `pattern_risk_agent.run(extraction)`
- `synthesis_agent.run(extraction: dict, pattern_risk: dict) -> dict` — used consistently in Task 6 as `synthesis_agent.run(extraction, pattern_risk)`
- `CLINICAL_NOTES` list with `id`, `title`, `note` keys — consistent across Tasks 2 and 6
