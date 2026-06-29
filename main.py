# ─────────────────────────────────────────────────────────────────────────────
# main.py  —  Pipeline Orchestrator
#
# PURPOSE:
#   Entry point for the analysis pipeline. Iterates over all 3 clinical notes
#   and runs each through the 3-agent chain sequentially:
#
#     extraction_agent  →  pattern_risk_agent  →  synthesis_agent
#
# OUTPUT FILES (written to results/):
#   results/P001.json          — full result dict for patient P001
#   results/P002.json          — full result dict for patient P002
#   results/P003.json          — full result dict for patient P003
#   results/all_results.json   — list of all 3 result dicts (loaded by dashboard)
#
# Each result dict shape:
#   {
#     "patient_id":    "P001",
#     "title":         "Cardiac Emergency",
#     "original_note": "58 year old male...",
#     "extraction":    { ...Agent 1 output... },
#     "pattern_risk":  { ...Agent 2 output... },
#     "synthesis":     { ...Agent 3 output... }
#   }
#
# HOW TO RUN:
#   export OPENAI_API_KEY=sk-...
#   python main.py
#
# NOTE: This file is intentionally importable (logic wrapped in main()).
#   This lets tests mock the agents and call main() without subprocess overhead.
# ─────────────────────────────────────────────────────────────────────────────

import json
import os
import sys
from pathlib import Path

from data.clinical_notes import CLINICAL_NOTES
from agents import extraction_agent, pattern_risk_agent, synthesis_agent


def main() -> None:
    # Fail fast if the API key is missing — better than a cryptic OpenAI error
    # partway through processing the first patient
    if not os.environ.get("OPENAI_API_KEY"):
        sys.exit("Error: OPENAI_API_KEY environment variable is not set.")

    # Create results/ directory if it doesn't exist yet
    Path("results").mkdir(exist_ok=True)

    all_results = []

    for note in CLINICAL_NOTES:
        patient_id = note["id"]
        print(f"\nAnalyzing {patient_id}: {note['title']}...")

        # ── Agent 1: Extract structured fields from the raw note text ──────
        extraction = extraction_agent.run(note["note"])
        print("  Extraction complete")

        # ── Agent 2: Identify patterns and classify risk ────────────────────
        # Receives Agent 1 output — no raw note text needed at this stage
        pattern_risk = pattern_risk_agent.run(extraction)
        print("  Pattern analysis complete")

        # ── Agent 3: Generate codes, summary, and explainability ───────────
        # Receives both Agent 1 and Agent 2 outputs to produce ICD-10/CPT codes
        synthesis = synthesis_agent.run(extraction, pattern_risk)
        print("  Synthesis complete")

        # Combine all agent outputs into a single result dict for this patient
        result = {
            "patient_id": patient_id,
            "title": note["title"],
            "original_note": note["note"],
            "extraction": extraction,
            "pattern_risk": pattern_risk,
            "synthesis": synthesis,
        }

        # Write individual patient file for debugging and partial recovery
        with open(f"results/{patient_id}.json", "w") as f:
            json.dump(result, f, indent=2)

        all_results.append(result)

    # Write the combined file that dashboard.py reads on startup
    with open("results/all_results.json", "w") as f:
        json.dump(all_results, f, indent=2)

    print("\nAll patients processed")


# Guard allows this file to be imported by tests without executing the pipeline
if __name__ == "__main__":
    main()
