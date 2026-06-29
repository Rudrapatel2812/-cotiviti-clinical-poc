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
