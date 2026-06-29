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
    with st.spinner("Running analysis pipeline... (~60-90 seconds)"):
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
        f"{risk_level}</span>"
    )


# ── Page config ───────────────────────────────────────────────────────────────
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

# ── TAB 2: Agent Reasoning Chain ─────────────────────────────────────────────
with tab2:
    st.subheader("Agent Reasoning Chain")
    st.caption("Each card shows what one agent found. Data flows left → right.")

    col_a, col_arr1, col_b, col_arr2, col_c = st.columns([3, 0.4, 3, 0.4, 3])

    with col_a:
        with st.container(border=True):
            st.markdown("**Agent 1: Extraction**")
            pi = extraction.get("patient_info") or {}
            st.write(f"Age: {pi.get('age', 'N/A')}  |  Gender: {pi.get('gender', 'N/A')}")
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
