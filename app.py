"""
Streamlit UI for the Multi-Agent Insurance Claims Processing System.
Provides an interactive interface to submit claims and visualise the
multi-agent pipeline results.
"""

import json
import streamlit as st
import tempfile
import os
from datetime import date, datetime

# Lazy imports to make the app load faster
def get_claims_agents():
    from claims_agents import (
        MOCK_POLICIES,
        SAMPLE_CLAIMS,
        process_claim,
        build_graph,
        compile_graph,
        set_chroma_collection,
    )
    return MOCK_POLICIES, SAMPLE_CLAIMS, process_claim, build_graph, compile_graph, set_chroma_collection

def get_ingest():
    from ingest import ingest_policy_pdf
    return ingest_policy_pdf

# Load the modules now (only once)
MOCK_POLICIES, SAMPLE_CLAIMS, process_claim, build_graph, compile_graph, set_chroma_collection = get_claims_agents()
ingest_policy_pdf = get_ingest()


# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Insurance Claims Processor",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed",   # Changed from expanded
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

# Sidebar removed (Policy Configuration was causing receipt upload issues)

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

input_tab1, input_tab2, input_tab3, input_tab4 = st.tabs([
    "📝 Manual Entry", 
    "📄 Raw Text", 
    "🧾 Upload Receipt",
    "⚙️ Policy Setup"
])

with input_tab4:
    st.markdown("### Policy Configuration")
    st.info("Upload your insurance policy PDF here for exclusion checking. This tab is independent from receipt processing.")

    policy_file = st.file_uploader("Upload policy document", type=["pdf"], key="policy_uploader")

    if policy_file is not None:
        if st.button("📥 Process Policy Document", type="primary"):
            with st.spinner("Processing policy..."):
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                    tmp.write(policy_file.getvalue())
                    tmp_path = tmp.name

                try:
                    collection, summary = ingest_policy_pdf(tmp_path)
                    st.session_state["chroma_collection"] = collection
                    st.session_state["ingest_summary"] = summary
                    st.success(f"✅ Policy processed! {summary.get('exclusion', 0)} exclusions indexed.")
                except Exception as e:
                    st.error(f"Failed to process policy: {e}")
                finally:
                    if os.path.exists(tmp_path):
                        os.remove(tmp_path)

with input_tab1:
    st.markdown("### Claim Details")
    col_a, col_b = st.columns(2)
    with col_a:
        claimant_name = st.text_input("Claimant Name", value="Jane Doe")
        membership_number = st.text_input("Membership / Policy Number", value="POL-10234")
        benefit_type = st.selectbox("Benefit Type", ["Hospital & Surgical", "SMM", "Clinical", "Maternity", "Dental", "Network Dental"])
        bhn_card = st.radio("BHN Card", ["Using BHN Card", "Not using BHN Card"])
    with col_b:
        treatment_date = st.date_input("Date of Treatment / Discharge", value=date(2026, 8, 1))
        coverage_start_date = st.date_input("Coverage / Membership Start Date", value=date(2025, 1, 1))
        claim_amount = st.number_input("Claim Amount (HKD)", min_value=0.0, step=100.0, value=4500.0)
        
    diagnosis = st.text_area("Diagnosis / Description", height=100)
    st.markdown("### Document Checklist")
    col_c, col_d = st.columns(2)
    with col_c:
        orig_receipt = st.checkbox("Original Receipt")
        referral = st.checkbox("Referral Letter")
    with col_d:
        pre_auth = st.checkbox("Pre-authorisation Confirmation")
        discharge_summary = st.checkbox("Discharge Summary")
        
    process_manual_btn = st.button("🚀 Process Manual Claim", type="primary", use_container_width=True)

with input_tab2:
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
    process_raw_btn = st.button("🚀 Process Raw Text Claim", type="primary", use_container_width=True)

with input_tab3:
    st.markdown("### Upload Claim Receipt")
    receipt_file = st.file_uploader("Upload claim receipt (scanned or digital)", type=["pdf", "png", "jpg", "jpeg"])
    
    # Status indicator right after upload
    if receipt_file is not None:
        if st.session_state.get("receipt_filename") == receipt_file.name:
            extracted = st.session_state.get("receipt_extracted", {})
            if extracted and any(v for v in extracted.values() if v not in [None, "", "Unknown"]):
                st.success("✅ Receipt processed successfully")
            elif extracted == {}:
                st.error("❌ Receipt processing failed")
            else:
                st.info("⏳ Receipt is being processed...")
        else:
            st.info("⏳ Receipt uploaded - extracting now...")

    
    process_receipt_btn = False
    
    if receipt_file is not None:
        if "receipt_extracted" not in st.session_state or st.session_state.get("receipt_filename") != receipt_file.name:
            with st.spinner("🔄 Extracting data from receipt... Please wait"):
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf" if receipt_file.name.lower().endswith(".pdf") else ".png") as tmp:
                    tmp.write(receipt_file.getvalue())
                    tmp_path = tmp.name
                
                try:
                    from receipt_vision import is_scanned_pdf, pdf_to_images, extract_receipt_via_vision
                    try:
                        from PIL import Image
                    except ImportError:
                        Image = None
                        
                    if receipt_file.name.lower().endswith(".pdf"):
                        images = pdf_to_images(tmp_path)
                        extracted = extract_receipt_via_vision(images)
                    else:
                        img = Image.open(tmp_path)
                        extracted = extract_receipt_via_vision([img])
                        
                    st.session_state["receipt_extracted"] = extracted
                    st.session_state["receipt_filename"] = receipt_file.name
                    st.session_state["receipt_tmp_path"] = tmp_path
                    
                    # Success indicator
                    if extracted and any(v for v in extracted.values() if v not in [None, "", "Unknown"]):
                        st.success("✅ Receipt extraction successful! Review the fields below.")
                    else:
                        st.warning("⚠️ Extraction completed but some fields may be missing or unclear.")
                except Exception as e:
                    import traceback
                    err = traceback.format_exc()
                    st.error(f"Error extracting receipt: {e}")
                    st.code(err, language="text")
                    st.session_state["receipt_extracted"] = {}
        
        extracted = st.session_state.get("receipt_extracted", {})
        
        st.markdown("### Verify Extracted Data")
        st.info("Review and correct the extracted fields below before processing.")
        
        # Parse date
        try:
            from datetime import date
            extracted_date_str = extracted.get("treatment_date")
            if extracted_date_str:
                ext_date_val = date.fromisoformat(extracted_date_str)
            else:
                ext_date_val = date.today()
        except:
            ext_date_val = date.today()
            
        ext_claimant = st.text_input("Claimant Name (Extracted)", value=extracted.get("patient_name") or "")
        ext_provider = st.text_input("Provider", value=extracted.get("provider_name") or "")
        ext_date = st.date_input("Treatment Date", value=ext_date_val)
        
        try:
            amt = float(extracted.get("amount_charged") or 0.0)
        except:
            amt = 0.0
            
        ext_amount = st.number_input("Amount Charged", min_value=0.0, step=10.0, value=amt)
        ext_desc = st.text_area("Diagnosis/Description", value=extracted.get("diagnosis_or_treatment_description") or "")
        
        st.markdown("### Missing Form Fields")
        ext_policy = st.text_input("Policy Number (Not on receipt)", value="POL-10234")
        ext_benefit = st.selectbox("Benefit Type (Not on receipt)", ["Hospital & Surgical", "SMM", "Clinical", "Maternity", "Dental", "Network Dental"])
        
        process_receipt_btn = st.button("🚀 Process Receipt Claim", type="primary", use_container_width=True)

# ---------------------------------------------------------------------------
# Processing & output
# ---------------------------------------------------------------------------

claim_to_process = None
initial_data = {}

if process_manual_btn:
    # 90-day check
    deadline_exceeded = (date.today() - treatment_date).days > 90
    
    claim_to_process = (
        f"Claimant: {claimant_name}. Policy: {membership_number}. "
        f"Benefit: {benefit_type}. Date: {treatment_date.isoformat()}. Amount: ${claim_amount}. "
        f"Diagnosis: {diagnosis}"
    )
    
    initial_data = {
        "treatment_date": treatment_date.isoformat(),
        "coverage_start_date": coverage_start_date.isoformat(),
        "benefit_type": benefit_type,
        "submission_deadline_exceeded": deadline_exceeded,
        "structured_claim": {
            "claimant_name": claimant_name,
            "policy_number": membership_number,
            "claim_amount": claim_amount,
            "description": diagnosis,
            "incident_date": treatment_date.isoformat(),
            "claim_type": "health"
        }
    }
    
elif process_raw_btn and claim_text.strip():
    claim_to_process = claim_text.strip()
    
elif process_receipt_btn:
    deadline_exceeded = (date.today() - ext_date).days > 90
    
    claim_to_process = (
        f"Claimant: {ext_claimant}. Policy: {ext_policy}. Provider: {ext_provider}. "
        f"Benefit: {ext_benefit}. Date: {ext_date.isoformat()}. Amount: ${ext_amount}. "
        f"Diagnosis: {ext_desc}"
    )
    
    initial_data = {
        "file_path": st.session_state.get("receipt_tmp_path"),
        "treatment_date": ext_date.isoformat(),
        "coverage_start_date": date(2025, 1, 1).isoformat(), # mock
        "benefit_type": ext_benefit,
        "submission_deadline_exceeded": deadline_exceeded,
        "structured_claim": {
            "claimant_name": ext_claimant,
            "policy_number": ext_policy,
            "claim_amount": ext_amount,
            "description": ext_desc,
            "incident_date": ext_date.isoformat(),
            "claim_type": "health"
        }
    }

if claim_to_process:
    if initial_data.get("submission_deadline_exceeded"):
        st.warning("⚠️ Submission deadline (90 days) has been exceeded.")
        
    result = None
    app_graph = compile_graph()
    
    initial_state = {
        "raw_text": claim_to_process,
        "trace": [f"[System] Claim received at {datetime.now().isoformat()}"],
    }
    if initial_data:
        initial_state.update(initial_data)

    result = dict(initial_state)
    progress_bar = st.progress(0, text="🚀 Starting claim processing pipeline...")
    step_count = 0
    total_steps = 6

    with st.status("Evaluating Claim Pipeline...", expanded=True) as status:
        for output in app_graph.stream(initial_state):
            for node_name, node_state in output.items():
                st.write(f"⚙️ Completed **{node_name}** agent")
                step_count += 1
                prog = min(step_count / total_steps, 1.0)
                progress_bar.progress(prog, text=f"Agent completed: {node_name}...")
                result.update(node_state)
        
        status.update(label="Evaluation Complete!", state="complete", expanded=False)
    
    progress_bar.progress(1.0, text="Done!")

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
    tab1, tab2, tab_excl, tab3, tab4, tab5, tab6 = st.tabs([
        "📥 Intake",
        "📋 Policy Check",
        "🛑 Exclusion Match",
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

    with tab_excl:
        st.subheader("Exclusion Match Agent")
        exc_data = result.get("exclusion_check", {})
        
        excluded = exc_data.get("excluded", False)
        if excluded:
            st.error("Claim EXCLUDED by policy.")
        else:
            st.success("Claim NOT excluded by policy.")
            
        st.markdown(f"**Method used:** {exc_data.get('method', 'unknown')}")
        st.markdown(f"**Reasoning:** {exc_data.get('reasoning', 'N/A')}")
        
        if exc_data.get("matched_clause_text"):
            with st.expander(f"Matched Clause {exc_data.get('matched_clause_number', '')}"):
                st.info(exc_data.get("matched_clause_text"))

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

        # Clear reason display
        reason = result.get("decision_reason", "No reason provided")
        st.markdown("### Why this decision?")
        st.info(f"**{reason}**")

        route = result.get("route", "unknown")
        if route == "auto_approve":
            st.success("Routed to: DecisionAgent (auto-approve path)")
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

    # ---- Download section (always visible after result) ----
    st.markdown("---")
    st.subheader("📥 Download Result")

    claimant_safe = str(result.get("claimant_name", "unknown")).replace(" ", "_")
    col_dl1, col_dl2 = st.columns(2)

    with col_dl1:
        # Generate comprehensive PDF report
        try:
            from reportlab.lib.pagesizes import letter
            from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
            from reportlab.lib.units import inch
            from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
            from reportlab.lib import colors
            from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
            from io import BytesIO

            buffer = BytesIO()
            doc = SimpleDocTemplate(
                buffer,
                pagesize=letter,
                rightMargin=0.75*inch,
                leftMargin=0.75*inch,
                topMargin=0.75*inch,
                bottomMargin=0.75*inch
            )

            styles = getSampleStyleSheet()

            # Custom styles
            title_style = ParagraphStyle(
                'CustomTitle',
                parent=styles['Heading1'],
                fontSize=18,
                alignment=TA_CENTER,
                spaceAfter=20,
                textColor=colors.HexColor('#1a1a2e')
            )
            heading_style = ParagraphStyle(
                'CustomHeading',
                parent=styles['Heading2'],
                fontSize=13,
                spaceBefore=15,
                spaceAfter=8,
                textColor=colors.HexColor('#16213e')
            )
            normal_style = ParagraphStyle(
                'CustomNormal',
                parent=styles['Normal'],
                fontSize=10,
                leading=14,
                alignment=TA_JUSTIFY
            )
            bold_style = ParagraphStyle(
                'BoldNormal',
                parent=styles['Normal'],
                fontSize=10,
                leading=14,
                fontName='Helvetica-Bold'
            )

            story = []

            # Title
            story.append(Paragraph("Insurance Claim Evaluation Report", title_style))
            story.append(Paragraph(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", normal_style))
            story.append(Spacer(1, 20))

            # Claimant Information
            story.append(Paragraph("1. Claimant Information", heading_style))
            claimant_data = [
                ["Claimant Name", str(result.get("claimant_name", "N/A"))],
                ["Policy Number", str(result.get("policy_number", "N/A"))],
                ["Benefit Type", str(result.get("benefit_type", result.get("claim_type", "N/A")))],
                ["Treatment / Incident Date", str(result.get("treatment_date") or result.get("incident_date", "N/A"))],
                ["Coverage Start Date", str(result.get("coverage_start_date", "N/A"))],
                ["Claim Amount (HKD)", f"${result.get('claim_amount', 0):,.2f}"],
                ["Description", str(result.get("description", "N/A"))[:500]],
            ]
            claimant_table = Table(claimant_data, colWidths=[2.2*inch, 4.5*inch])
            claimant_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#f0f4f8')),
                ('TEXTCOLOR', (0, 0), (-1, -1), colors.HexColor('#1a1a2e')),
                ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, -1), 9),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('VALIGN', (0, 0), (-1, -1), 'TOP'),
                ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#cbd5e1')),
                ('LEFTPADDING', (0, 0), (-1, -1), 8),
                ('RIGHTPADDING', (0, 0), (-1, -1), 8),
                ('TOPPADDING', (0, 0), (-1, -1), 6),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
            ]))
            story.append(claimant_table)
            story.append(Spacer(1, 15))

            # Policy Validation
            story.append(Paragraph("2. Policy Validation", heading_style))
            policy_data = [
                ["Policy Found", str(result.get("policy_found", "N/A"))],
                ["Policy Status", str(result.get("policy_status", "N/A"))],
                ["Coverage Valid", str(result.get("coverage_valid", "N/A"))],
                ["Coverage Limit (HKD)", f"${result.get('coverage_limit', 0):,.2f}"],
                ["Policy Notes", str(result.get("policy_notes", "N/A"))],
            ]
            policy_table = Table(policy_data, colWidths=[2.2*inch, 4.5*inch])
            policy_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#f0f4f8')),
                ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, -1), 9),
                ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#cbd5e1')),
                ('LEFTPADDING', (0, 0), (-1, -1), 8),
                ('TOPPADDING', (0, 0), (-1, -1), 6),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
            ]))
            story.append(policy_table)
            story.append(Spacer(1, 15))

            # Exclusion Check
            story.append(Paragraph("3. Exclusion Check", heading_style))
            exc = result.get("exclusion_check", {})
            exc_data = [
                ["Excluded", str(exc.get("excluded", "N/A"))],
                ["Method", str(exc.get("method", "N/A"))],
                ["Reasoning", str(exc.get("reasoning", "N/A"))[:400]],
            ]
            if exc.get("matched_clause_text"):
                exc_data.append(["Matched Clause", str(exc.get("matched_clause_text", ""))[:300]])
            exc_table = Table(exc_data, colWidths=[2.2*inch, 4.5*inch])
            exc_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#f0f4f8')),
                ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, -1), 9),
                ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#cbd5e1')),
                ('LEFTPADDING', (0, 0), (-1, -1), 8),
                ('TOPPADDING', (0, 0), (-1, -1), 6),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
            ]))
            story.append(exc_table)
            story.append(Spacer(1, 15))

            # Fraud Detection
            story.append(Paragraph("4. Fraud Detection", heading_style))
            fraud_data = [
                ["Fraud Score", f"{result.get('fraud_score', 0)}/100"],
                ["Red Flags", ", ".join(result.get("red_flags", [])) or "None"],
                ["Reasoning", str(result.get("fraud_reasoning", "N/A"))[:500]],
            ]
            fraud_table = Table(fraud_data, colWidths=[2.2*inch, 4.5*inch])
            fraud_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#f0f4f8')),
                ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, -1), 9),
                ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#cbd5e1')),
                ('LEFTPADDING', (0, 0), (-1, -1), 8),
                ('TOPPADDING', (0, 0), (-1, -1), 6),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
            ]))
            story.append(fraud_table)
            story.append(Spacer(1, 15))

            # Final Decision
            story.append(Paragraph("5. Final Decision & Reasoning", heading_style))
            decision_data = [
                ["Decision Status", str(result.get("decision_status", "N/A")).upper()],
                ["Route", str(result.get("route", "N/A"))],
                ["Reason", str(result.get("decision_reason", "N/A"))],
            ]
            decision_table = Table(decision_data, colWidths=[2.2*inch, 4.5*inch])
            decision_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#f0f4f8')),
                ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, -1), 9),
                ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#cbd5e1')),
                ('LEFTPADDING', (0, 0), (-1, -1), 8),
                ('TOPPADDING', (0, 0), (-1, -1), 6),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
            ]))
            story.append(decision_table)
            story.append(Spacer(1, 15))

            # Communication Letter
            letter = result.get("communication_letter", "")
            if letter:
                story.append(Paragraph("6. Customer Communication Letter", heading_style))
                # Escape special characters for reportlab
                letter_clean = letter.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
                story.append(Paragraph(letter_clean.replace("\n", "<br/>"), normal_style))

            # Build PDF
            doc.build(story)
            pdf_bytes = buffer.getvalue()
            buffer.close()

            st.download_button(
                label="⬇️ Download Full Report (PDF)",
                data=pdf_bytes,
                file_name=f"claim_report_{claimant_safe}.pdf",
                mime="application/pdf",
                use_container_width=True,
                help="Complete evaluation report with all policy, user and decision details"
            )
        except Exception as e:
            st.error(f"PDF generation failed: {e}")
            st.caption("Please ensure 'reportlab' is installed: pip install reportlab")

    with col_dl2:
        summary = (
            "Insurance Claim Processing Result\n"
            "=====================================\n"
            f"Claimant: {result.get('claimant_name', 'N/A')}\n"
            f"Policy Number: {result.get('policy_number', 'N/A')}\n"
            f"Claim Amount: ${result.get('claim_amount', 0):,.2f}\n"
            f"Decision: {result.get('decision_status', 'unknown').upper()}\n"
            f"Fraud Score: {result.get('fraud_score', 0)}/100\n"
            f"Reason: {result.get('decision_reason', 'N/A')}\n"
            f"\nGenerated on: {datetime.now().isoformat()}\n"
        )
        st.download_button(
            label="⬇️ Download Summary (TXT)",
            data=summary,
            file_name=f"claim_summary_{claimant_safe}.txt",
            mime="text/plain",
            use_container_width=True,
            help="Human-readable one-page summary"
        )

elif process_raw_btn:
    st.warning("Please enter claim text before processing.")
