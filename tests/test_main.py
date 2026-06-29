import json
from pathlib import Path
from unittest.mock import patch

import pytest

import main as main_module


FAKE_EXTRACTION = {"patient_info": {"age": 58}, "diagnoses": ["hypertension"]}
FAKE_PATTERN_RISK = {"risk_level": "CRITICAL", "clinical_patterns": []}
FAKE_SYNTHESIS = {"clinical_summary": "Test summary", "icd10_codes": []}


@pytest.fixture
def results_dir(tmp_path, monkeypatch):
    """Run main() with results/ redirected to a tmp directory."""
    monkeypatch.chdir(tmp_path)
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
