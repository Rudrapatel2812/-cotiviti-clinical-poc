# Multi-Agent Clinical Record Intelligence — Design Spec
**Date:** 2026-06-28  
**Project:** Cotiviti Intern Assessment POC  
**Status:** Approved

---

## What We Are Building

A Python POC that runs 3 GPT-4o agents sequentially over hardcoded clinical notes, produces structured JSON output (extraction → pattern/risk → synthesis), and renders everything in a Streamlit dashboard.

No database. No auth. No external data source. The raw clinical notes are static Python dicts.

---

## File Structure

```
cotiviti-poc/
├── agents/
│   ├── extraction_agent.py       # Agent 1: structured info extraction
│   ├── pattern_risk_agent.py     # Agent 2: pattern recognition + risk classification
│   └── synthesis_agent.py        # Agent 3: ICD-10/CPT coding + explainability
├── data/
│   └── clinical_notes.py         # 3 hardcoded sample notes as Python dicts
├── results/                       # Created at runtime by main.py
│   ├── P001.json
│   ├── P002.json
│   ├── P003.json
│   └── all_results.json
├── docs/
│   └── superpowers/specs/         # This file
├── main.py                        # Orchestrator — runs full pipeline
├── dashboard.py                   # Streamlit UI (read-only, loads JSON)
├── requirements.txt
└── README.md
```

---

## Data

`data/clinical_notes.py` exports `CLINICAL_NOTES: list[dict]`, three hardcoded notes:

| ID   | Title                        | Key clinical detail                                      |
|------|------------------------------|----------------------------------------------------------|
| P001 | Cardiac Emergency            | 58M, chest pain, elevated troponin, HTN, T2DM, smoker   |
| P002 | Obstetric Emergency          | 34F, 28wk pregnant, BP 168/110, proteinuria, thrombocytopenia |
| P003 | Incomplete Documentation     | Fever + cough, no vitals, no labs, no history (intentional gaps) |

Each dict: `{"id": "P001", "title": "...", "note": "full text"}`.

---

## Agent Architecture

**Pattern:** Plain functions, no classes, no frameworks. Each agent file exports a single `run()` function.

**OpenAI config for all agents:**
- Model: `gpt-4o`
- `response_format={"type": "json_object"}` — guarantees parseable JSON, prevents markdown code fence wrapping
- API key from `OPENAI_API_KEY` environment variable, validated at startup in `main.py`

### Agent 1 — Extraction (`extraction_agent.run(note_text: str) -> dict`)

Extracts structured fields from raw note text:
- `patient_info`: age, gender, chief_complaint
- `diagnoses`: list
- `medications`: list
- `vitals`: dict of key→value
- `lab_results`: dict of key→value
- `symptoms`: list
- Missing fields → `null`

### Agent 2 — Pattern & Risk (`pattern_risk_agent.run(extraction: dict) -> dict`)

Takes Agent 1 output, identifies clinical patterns and assigns risk:
- `clinical_patterns`: list
- `risk_level`: one of `CRITICAL / HIGH / MEDIUM / LOW`
- `risk_reasons`: list
- `documentation_gaps`: list of missing info items
- `recommended_urgency`: `immediate / urgent / routine`

### Agent 3 — Synthesis (`synthesis_agent.run(extraction: dict, pattern_risk: dict) -> dict`)

Takes both prior outputs, generates:
- `clinical_summary`: 2–3 sentence plain English
- `icd10_codes`: list of `{code, description}`
- `cpt_codes`: list of `{code, description}`
- `recommended_actions`: list
- `explainability`: plain English WHY this risk level, citing specific data points
- `payment_flags`: list of billing concerns (empty list if none)

---

## Orchestrator (`main.py`)

1. Validate `OPENAI_API_KEY` present — raise clear error if missing
2. Create `results/` directory if absent
3. For each note in `CLINICAL_NOTES`:
   - Print progress markers
   - Run Agent 1 → Agent 2 → Agent 3 sequentially
   - Merge into final dict: `{patient_id, title, original_note, extraction, pattern_risk, synthesis}`
   - Write `results/{patient_id}.json`
4. Write `results/all_results.json` (list of all 3 final dicts)
5. Print "All patients processed"

---

## Dashboard (`dashboard.py`)

**Sidebar:**
- Patient dropdown (P001, P002, P003)
- "Analyze Patient" button
  - If `all_results.json` exists → load it (no API calls)
  - If missing → run `main.py` pipeline with `st.spinner`, then load

**Main area — 4 tabs:**

| Tab | Content |
|-----|---------|
| Patient Overview | Original note, extracted info in two columns (diagnoses/symptoms/meds left; vitals/labs right), risk badge |
| Agent Reasoning Chain | 3 side-by-side cards (one per agent) showing key findings, with arrow between them |
| Billing & Coding | ICD-10 table, CPT table, payment flags as warnings, recommended actions list |
| Documentation Gaps | Color-coded gap list (red=critical, yellow=warning), urgency level |

**Risk badge colors:**
- `CRITICAL` → red (`#ff4b4b`)
- `HIGH` → orange (`#ffa500`)
- `MEDIUM` → yellow (`#ffd700`)
- `LOW` → green (`#00cc44`)

---

## Error Handling

- Missing `OPENAI_API_KEY`: clear error at startup, not mid-run
- JSON parse failure: exception propagates naming which agent + patient failed
- Missing `results/` dir: created automatically
- Dashboard with no cache and pipeline fails: `st.error()` with message

---

## Dependencies

```
openai
streamlit
python-dotenv
```

No other libraries. Standard library only for JSON, os, pathlib.

---

## Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| Plain functions over classes | "Minimal and clean" per spec rules; easier for assessment reviewer to follow |
| `response_format=json_object` | Prevents most common demo failure (markdown in JSON response) |
| Cache-first dashboard | Avoids 9 API calls on every page refresh; user can still force re-run |
| Hardcoded notes | POC scope; no infra needed to run |
| `results/` as output dir | Separates runtime artifacts from source; clean git ignore |
