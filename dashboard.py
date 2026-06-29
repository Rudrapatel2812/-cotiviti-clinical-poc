# ─────────────────────────────────────────────────────────────────────────────
# dashboard.py  —  Streamlit Dashboard
#
# PURPOSE:
#   Read-only visualization layer. Loads results/all_results.json (written by
#   main.py) and renders them across 4 tabs. Can also trigger the pipeline
#   via a sidebar button if no cached results exist yet.
#
# HOW TO RUN:
#   streamlit run dashboard.py
#
# TABS:
#   1. Patient Overview     — risk badge, extracted vitals/labs, diagnoses
#   2. Agent Reasoning Chain — 3 side-by-side cards showing each agent's output
#   3. Billing & Coding     — ICD-10 table, CPT table, explainability, actions
#   4. Documentation Gaps   — color-coded list of missing clinical info
#
# CACHE BEHAVIOUR:
#   - On load: reads all_results.json if it exists (no API calls)
#   - Button click with cache present: re-runs pipeline, overwrites cache
#   - Button click with no cache: runs pipeline for the first time
# ─────────────────────────────────────────────────────────────────────────────

import json
import subprocess
import sys
from pathlib import Path

import streamlit as st

# Path to the JSON file written by main.py
RESULTS_FILE = Path("results/all_results.json")

# Risk badge colours — map GPT-4o risk_level strings to hex colours
RISK_COLORS = {
    "CRITICAL": "#ff4b4b",  # red
    "HIGH": "#ffa500",      # orange
    "MEDIUM": "#ffd700",    # yellow
    "LOW": "#00cc44",       # green
}


def load_results() -> list | None:
    """Read cached results from disk. Returns None if file doesn't exist yet."""
    if RESULTS_FILE.exists():
        with open(RESULTS_FILE) as f:
            return json.load(f)
    return None


def run_pipeline() -> list | None:
    """Invoke main.py as a subprocess, show a spinner, then reload results."""
    with st.spinner("Running analysis pipeline... (~60-90 seconds)"):
        # Run in a subprocess so Streamlit's event loop isn't blocked
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
    """Return an inline HTML <span> badge coloured by risk level."""
    color = RISK_COLORS.get(risk_level, "#888888")  # grey fallback for unknown levels
    return (
        f'<span style="background:{color};color:white;padding:6px 16px;'
        f'border-radius:4px;font-weight:bold;font-size:1.1em;">'
        f"{risk_level}</span>"
    )


# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="Clinical Record Intelligence", layout="wide")
st.title("Multi-Agent Clinical Record Intelligence")
st.caption("Cotiviti POC — Agentic AI for TPO")

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Controls")

    # Try to load cached results first — avoids triggering API calls on every page load
    results = load_results()

    if results:
        # Build dropdown options from the cached patient list
        patient_ids = [r["patient_id"] for r in results]
        labels = {r["patient_id"]: f"{r['patient_id']} — {r['title']}" for r in results}
    else:
        # No cache yet — show placeholder IDs so the dropdown still renders
        patient_ids = ["P001", "P002", "P003"]
        labels = {p: p for p in patient_ids}

    selected_id = st.selectbox(
        "Select Patient",
        patient_ids,
        format_func=lambda x: labels.get(x, x),
    )

    # Button label changes to signal that results already exist
    btn_label = "Re-analyze All Patients" if results else "Analyze All Patients"
    if st.button(btn_label, use_container_width=True):
        results = run_pipeline()

    # Stop rendering the main area until results are available
    if results is None:
        st.info("Click 'Analyze All Patients' to run the pipeline.")
        st.stop()

# ── Resolve selected patient data ─────────────────────────────────────────────
# Pull the selected patient's result dict out of the loaded list
patient_data = next((r for r in results if r["patient_id"] == selected_id), None)
if patient_data is None:
    st.error(f"No data found for {selected_id}. Try re-analyzing.")
    st.stop()

# Unpack the three agent output dicts — use `or {}` so downstream code never
# needs to guard against None (GPT-4o output is always a dict, but being safe)
extraction = patient_data.get("extraction") or {}
pattern_risk = patient_data.get("pattern_risk") or {}
synthesis = patient_data.get("synthesis") or {}
risk_level = pattern_risk.get("risk_level", "UNKNOWN")

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs(
    ["Patient Overview", "Agent Reasoning Chain", "Billing & Coding", "Documentation Gaps"]
)

# ── TAB 1: Patient Overview ───────────────────────────────────────────────────
with tab1:
    st.subheader(patient_data["title"])

    # Colour-coded risk badge rendered via unsafe_allow_html (Streamlit has no
    # native coloured badge widget — this is the standard workaround)
    st.markdown(f"**Risk Level:** {risk_badge(risk_level)}", unsafe_allow_html=True)
    st.markdown("---")

    # Show the original note text in a collapsible expander to save vertical space
    with st.expander("Original Clinical Note"):
        st.text(patient_data.get("original_note", ""))

    # Two-column layout: clinical lists on the left, numeric tables on the right
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
        # Vitals and labs come back as dicts from Agent 1; convert to table format
        vitals = extraction.get("vitals") or {}
        if vitals:
            st.markdown("**Vitals**")
            st.table({"Measurement": list(vitals.keys()), "Value": list(vitals.values())})

        labs = extraction.get("lab_results") or {}
        if labs:
            st.markdown("**Lab Results**")
            st.table({"Test": list(labs.keys()), "Value": list(labs.values())})

# ── TAB 2: Agent Reasoning Chain ─────────────────────────────────────────────
with tab2:
    st.subheader("Agent Reasoning Chain")
    st.caption("Each card shows what one agent found. Data flows left → right.")

    # 5 columns: [Agent1 card] [arrow] [Agent2 card] [arrow] [Agent3 card]
    # Narrow columns (0.4) are purely decorative arrows between the cards
    col_a, col_arr1, col_b, col_arr2, col_c = st.columns([3, 0.4, 3, 0.4, 3])

    with col_a:
        with st.container(border=True):
            st.markdown("**Agent 1: Extraction**")
            pi = extraction.get("patient_info") or {}
            st.write(f"Age: {pi.get('age', 'N/A')}  |  Gender: {pi.get('gender', 'N/A')}")
            st.write(f"Chief complaint: {pi.get('chief_complaint', 'N/A')}")
            # Show counts so reviewer can quickly see how much was extracted
            st.write(f"Diagnoses found: {len(extraction.get('diagnoses') or [])}")
            st.write(f"Symptoms: {len(extraction.get('symptoms') or [])}")
            st.write(f"Medications: {len(extraction.get('medications') or [])}")
            st.write(f"Vitals recorded: {len(extraction.get('vitals') or {})}")
            st.write(f"Labs recorded: {len(extraction.get('lab_results') or {})}")

    with col_arr1:
        # Right-arrow rendered as HTML — Streamlit has no built-in arrow widget
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
                for p in patterns[:4]:  # cap at 4 to keep card compact
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

    # ICD-10 and CPT tables side by side
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

    # Payment flags surface upcoding or billing risks — shown as warnings
    flags = synthesis.get("payment_flags") or []
    if flags:
        st.markdown("**Payment Flags**")
        for flag in flags:
            st.warning(flag)

    st.markdown("**Recommended Actions**")
    for i, action in enumerate(synthesis.get("recommended_actions") or [], 1):
        st.write(f"{i}. {action}")

    # Explainability: plain English WHY the risk level was assigned,
    # referencing specific data points (e.g. "troponin 0.8 exceeds threshold")
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

        # Color logic: CRITICAL/HIGH cases show all gaps as red errors.
        # MEDIUM/LOW cases show the first 2 as red, remainder as yellow warnings.
        # This is especially visible for P003 (incomplete note) which has 6+ gaps.
        is_high_risk = risk_level in ("CRITICAL", "HIGH")
        for i, gap in enumerate(gaps):
            if is_high_risk or i < 2:
                st.error(f"Critical gap: {gap}")
            else:
                st.warning(f"Warning: {gap}")
