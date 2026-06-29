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
