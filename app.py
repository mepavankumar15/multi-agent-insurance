"""
Streamlit UI for the Multi-Agent Insurance Claims Processing System.
Provides an interactive interface to submit claims and visualise the
multi-agent pipeline results.
"""

import json
import streamlit as st

from claims_agents import (
    MOCK_POLICIES,
    SAMPLE_CLAIMS,
    process_claim,
    build_graph,
)

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Insurance Claims Processor",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS
# ---------------------------------------------------------------------------

st.markdown("""
<style>
    /* Main container */
    .main .block-container {
        padding-top: 2rem;
        max-width: 1200px;
    }

    /* Header gradient */
    .header-gradient {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        padding: 2rem;
        border-radius: 16px;
        margin-bottom: 2rem;
        text-align: center;
        color: white;
    }
    .header-gradient h1 {
        color: white !important;
        font-size: 2rem;
        margin-bottom: 0.3rem;
    }
    .header-gradient p {
        color: #a8b2d1;
        font-size: 1rem;
    }

    /* Agent result cards */
    .agent-card {
        background: #f8f9fc;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 1.2rem;
        margin-bottom: 1rem;
    }
    .agent-card h4 {
        color: #1a1a2e;
        margin-bottom: 0.5rem;
    }

    /* Status badges */
    .badge-approved {
        background: #d4edda;
        color: #155724;
        padding: 4px 12px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    .badge-escalated {
        background: #fff3cd;
        color: #856404;
        padding: 4px 12px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    .badge-denied {
        background: #f8d7da;
        color: #721c24;
        padding: 4px 12px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.85rem;
    }

    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background: #000000;
        color: #ffffff;
    }
    
    section[data-testid="stSidebar"] h1, 
    section[data-testid="stSidebar"] h2, 
    section[data-testid="stSidebar"] h3, 
    section[data-testid="stSidebar"] p, 
    section[data-testid="stSidebar"] div,
    section[data-testid="stSidebar"] span {
        color: #ffffff !important;
    }

    /* Metric cards */
    .metric-row {
        display: flex;
        gap: 1rem;
        margin-bottom: 1rem;
    }
    .metric-card {
        flex: 1;
        background: black;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
    }
    .metric-card .label {
        font-size: 0.8rem;
        color: #666;
        text-transform: uppercase;
    }
    .metric-card .value {
        font-size: 1.5rem;
        font-weight: 700;
        color: #1a1a2e;
    }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------

st.markdown("""
<div class="header-gradient">
    <h1>Multi-Agent Insurance Claims Processor</h1>
    <p>Powered by LangGraph &bull; 6 Specialized AI Agents &bull; Grok (xAI)</p>
</div>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.markdown("## Sample Claims")
    st.caption("Click to load a sample claim into the input area.")

    sample_labels = {
        "approved": "Auto-Approve (Low Risk)",
        "fraud_escalated": "Fraud Escalation (Lapsed + Urgent)",
        "over_limit_escalated": "Over-Limit Escalation",
    }

    for key, label in sample_labels.items():
        if st.button(f"📋 {label}", key=f"sample_{key}", use_container_width=True):
            st.session_state["claim_input"] = SAMPLE_CLAIMS[key]

    st.markdown("---")
    st.markdown("## Mock Policy Database")

    for pol_id, pol_data in MOCK_POLICIES.items():
        status_icon = "✅" if pol_data["status"] == "active" else "❌"
        with st.expander(f"{status_icon} {pol_id}"):
            st.markdown(f"**Holder:** {pol_data['holder']}")
            st.markdown(f"**Status:** {pol_data['status']}")
            st.markdown(f"**Coverage:** {pol_data['coverage_type']}")
            st.markdown(f"**Limit:** ${pol_data['coverage_limit']:,}")
            st.markdown(f"**Deductible:** ${pol_data['deductible']:,}")

    st.markdown("---")
    st.markdown("## Pipeline Graph")
    try:
        graph = build_graph().compile()
        mermaid_str = graph.get_graph().draw_mermaid()
        st.code(mermaid_str, language="mermaid")
    except Exception as e:
        st.caption(f"Could not render graph: {e}")


# ---------------------------------------------------------------------------
# Main area  --  claim input
# ---------------------------------------------------------------------------

default_text = st.session_state.get("claim_input", "")

claim_text = st.text_area(
    "Enter raw claim text:",
    value=default_text,
    height=180,
    placeholder=(
        "Claimant: Jane Doe. Policy: POL-10234. Auto claim. "
        "Incident date 2026-08-01. Amount: $4,500. "
        "My car was rear-ended at a stoplight..."
    ),
)

process_btn = st.button("🚀 Process Claim", type="primary", use_container_width=True)


# ---------------------------------------------------------------------------
# Processing & output
# ---------------------------------------------------------------------------

if process_btn and claim_text.strip():
    with st.spinner("Processing claim through 6-agent pipeline..."):
        result = process_claim(claim_text.strip())

    st.success("Pipeline completed successfully!")

    # ---- Metrics row ----
    col1, col2, col3, col4 = st.columns(4)

    decision_status = result.get("decision_status", "unknown")
    fraud_score = result.get("fraud_score", 0)

    with col1:
        st.metric("Claimant", result.get("claimant_name", "N/A"))
    with col2:
        st.metric("Claim Amount", f"${result.get('claim_amount', 0):,.2f}")
    with col3:
        st.metric("Fraud Score", f"{fraud_score}/100")
    with col4:
        st.metric("Decision", decision_status.upper())

    # ---- Tabbed output ----
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📥 Intake",
        "📋 Policy Check",
        "🔍 Fraud Detection",
        "⚖️ Decision",
        "✉️ Letter",
        "📜 Trace Log",
    ])

    with tab1:
        st.subheader("Intake Agent - Extracted Fields")
        intake_data = {
            "claimant_name": result.get("claimant_name"),
            "policy_number": result.get("policy_number"),
            "claim_type": result.get("claim_type"),
            "incident_date": result.get("incident_date"),
            "claim_amount": result.get("claim_amount"),
            "description": result.get("description"),
        }
        st.json(intake_data)

    with tab2:
        st.subheader("Policy Validation Agent")
        policy_data = {
            "policy_found": result.get("policy_found"),
            "policy_status": result.get("policy_status"),
            "coverage_valid": result.get("coverage_valid"),
            "coverage_limit": result.get("coverage_limit"),
            "notes": result.get("policy_notes"),
        }
        st.json(policy_data)

        if result.get("coverage_valid"):
            st.success("Policy is valid and coverage matches the claim type.")
        else:
            st.error(result.get("policy_notes", "Policy validation failed."))

    with tab3:
        st.subheader("Fraud Detection Agent")

        # Progress bar for fraud score
        st.markdown(f"**Fraud Risk Score: {fraud_score}/100**")
        bar_color = "normal"
        if fraud_score >= 60:
            bar_color = "normal"  # red handled by st.error below
        st.progress(min(fraud_score / 100, 1.0))

        if fraud_score >= 60:
            st.error(f"HIGH RISK - Score: {fraud_score}/100")
        elif fraud_score >= 30:
            st.warning(f"MODERATE RISK - Score: {fraud_score}/100")
        else:
            st.success(f"LOW RISK - Score: {fraud_score}/100")

        # Red flags
        red_flags = result.get("red_flags", [])
        if red_flags:
            st.markdown("**Red Flags:**")
            for flag in red_flags:
                st.markdown(f"- 🚩 {flag}")
        else:
            st.info("No red flags detected.")

        st.markdown(f"**Reasoning:** {result.get('fraud_reasoning', 'N/A')}")

    with tab4:
        st.subheader("Decision")

        if decision_status == "approved":
            st.markdown(f'<span class="badge-approved">APPROVED</span>',
                        unsafe_allow_html=True)
        elif decision_status == "escalated":
            st.markdown(f'<span class="badge-escalated">ESCALATED</span>',
                        unsafe_allow_html=True)
        else:
            st.markdown(f'<span class="badge-denied">DENIED</span>',
                        unsafe_allow_html=True)

        st.markdown(f"\n**Reason:** {result.get('decision_reason', 'N/A')}")

        route = result.get("route", "unknown")
        if route == "auto_approve":
            st.info("Routed to: DecisionAgent (auto-approve path)")
        else:
            st.warning("Routed to: SeniorReviewAgent (escalation path)")

    with tab5:
        st.subheader("Communication - Customer Letter")
        letter = result.get("communication_letter", "No letter generated.")
        st.text_area("Generated Letter", value=letter, height=350, disabled=True)

    with tab6:
        st.subheader("Full Trace Log")
        trace = result.get("trace", [])
        for i, entry in enumerate(trace, 1):
            # Colour-code by agent
            if "IntakeAgent" in entry:
                icon = "📥"
            elif "PolicyValidation" in entry:
                icon = "📋"
            elif "FraudDetection" in entry:
                icon = "🔍"
            elif "DecisionAgent" in entry:
                icon = "⚖️"
            elif "SeniorReview" in entry:
                icon = "🔴"
            elif "Communication" in entry:
                icon = "✉️"
            else:
                icon = "🔹"
            st.text(f"{icon} {i:02d}. {entry}")

elif process_btn:
    st.warning("Please enter claim text before processing.")
