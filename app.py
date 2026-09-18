import os
import tempfile
from datetime import date
from pathlib import Path

import streamlit as st

st.set_page_config(
    page_title="Insurance Claims Processor",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

for key, default in [
    ("result", None),
    ("extracted", None),
    ("receipt_bytes", None),
    ("receipt_name", None),
    ("uploader_key", 0),
]:
    if key not in st.session_state:
        st.session_state[key] = default


def clear_receipt():
    st.session_state.receipt_bytes = None
    st.session_state.receipt_name = None
    st.session_state.extracted = None
    st.session_state.result = None
    st.session_state.uploader_key = int(st.session_state.get("uploader_key", 0)) + 1


show_admin = st.sidebar.checkbox("Show Admin View (Customers)", value=False)

st.title("🛡️ Insurance Claims Processor")

tab1, tab2, tab3 = st.tabs(["📝 Manual Entry", "📄 Raw Text", "🧾 Upload Receipt"])


@st.cache_resource
def get_handbook(_version: int = 2):
    from claims_agents import set_chroma_collection
    from ingest import ingest_policy_pdf

    kb_path = Path(__file__).parent / "knowledge_base" / "policy_handbook.pdf"
    if not kb_path.exists():
        return False
    collection, _ = ingest_policy_pdf(str(kb_path))
    set_chroma_collection(collection)
    return True


def process_now(text, data):
    from claims_agents import process_claim

    try:
        get_handbook(2)
    except Exception:
        pass
    return process_claim(text, data)


def extract_receipt(file_bytes: bytes, filename: str) -> dict:
    from PIL import Image
    from receipt_vision import extract_fields_from_text, extract_text_from_pdf

    is_pdf = filename.lower().endswith(".pdf")
    suffix = ".pdf" if is_pdf else ".png"
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(file_bytes)
            tmp_path = tmp.name

        if is_pdf:
            pdf_text = extract_text_from_pdf(tmp_path)
            return extract_fields_from_text(pdf_text)
        img = Image.open(tmp_path)
        from receipt_vision import extract_receipt_via_vision

        return extract_receipt_via_vision([img]) or {}
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)


def preview_block(ext: dict):
    st.subheader("Extracted receipt data")
    e1, e2, e3, e4 = st.columns(4)
    with e1:
        st.metric("Name", ext.get("patient_name") or "—")
    with e2:
        st.metric("Policy", ext.get("policy_number") or "—")
    with e3:
        st.metric("Date", ext.get("treatment_date") or "—")
    with e4:
        amt = ext.get("amount_charged")
        st.metric("Amount", f"${amt:,.2f}" if isinstance(amt, (int, float)) else (str(amt) if amt else "—"))
    st.write("**Diagnosis:**", ext.get("diagnosis") or ext.get("diagnosis_or_treatment_description") or "—")
    if ext.get("method"):
        st.caption(f"Source: {ext.get('method')}")
    raw = ext.get("raw_text") or ext.get("raw_ocr_text")
    if raw:
        with st.expander("Raw extracted text"):
            st.text(raw)


with tab1:
    st.subheader("Manual Entry")
    col1, col2 = st.columns(2)
    with col1:
        claimant_name = st.text_input("Claimant Name", "Jane Doe")
        policy_number = st.text_input("Policy Number", "POL-10234")
        benefit_type = st.selectbox(
            "Benefit Type",
            ["Hospital & Surgical", "Clinical", "Maternity", "Dental", "Network Dental"],
        )
    with col2:
        treatment_date = st.date_input("Treatment Date", value=date(2026, 8, 1))
        claim_amount = st.number_input("Claim Amount (HKD)", min_value=0.0, value=4500.0)
    diagnosis = st.text_area("Diagnosis", height=100)
    if st.button("Process Manual Claim", type="primary"):
        with st.spinner("Running claim pipeline..."):
            st.session_state.result = process_now(
                f"Claimant: {claimant_name}. Policy: {policy_number}. Amount: ${claim_amount}. Diagnosis: {diagnosis}",
                {
                    "structured_claim": {
                        "claimant_name": claimant_name,
                        "policy_number": policy_number,
                        "claim_amount": claim_amount,
                        "description": diagnosis,
                        "claim_type": "auto",
                        "incident_date": treatment_date.isoformat(),
                    }
                },
            )

with tab2:
    st.subheader("Raw Text")
    raw = st.text_area("Paste claim text", height=150)
    if st.button("Process Raw Text", type="primary"):
        if raw.strip():
            with st.spinner("Running claim pipeline..."):
                st.session_state.result = process_now(raw.strip(), {})

with tab3:
    st.subheader("Upload Receipt")
    # Do not render the uploader while a file is stored, or Streamlit puts it back after Remove.
    if st.session_state.receipt_bytes is None:
        uploaded_file = st.file_uploader(
            "Choose a receipt file",
            type=["pdf", "png", "jpg", "jpeg"],
            key=f"receipt_uploader_{st.session_state.uploader_key}",
        )
        if uploaded_file is not None:
            st.session_state.receipt_bytes = uploaded_file.getvalue()
            st.session_state.receipt_name = uploaded_file.name
            st.rerun()
    else:
        st.write(
            f"**Current file:** {st.session_state.receipt_name} "
            f"({len(st.session_state.receipt_bytes)} bytes)"
        )
        name = st.session_state.receipt_name or ""
        if not name.lower().endswith(".pdf"):
            st.image(st.session_state.receipt_bytes, caption="Uploaded Receipt", use_container_width=True)
        st.button("Remove file", on_click=clear_receipt, type="secondary")

    can_extract = st.session_state.receipt_bytes is not None
    extract = st.button("Extract data from receipt", type="primary", disabled=not can_extract)
    if extract and can_extract:
        with st.spinner("Extracting receipt (PDF text, no Grok call)..."):
            try:
                extracted = extract_receipt(st.session_state.receipt_bytes, st.session_state.receipt_name)
                st.session_state.extracted = {
                    "patient_name": extracted.get("patient_name"),
                    "policy_number": extracted.get("policy_number"),
                    "treatment_date": extracted.get("treatment_date"),
                    "amount_charged": extracted.get("amount_charged"),
                    "diagnosis": extracted.get("diagnosis_or_treatment_description"),
                    "method": extracted.get("method"),
                    "raw_text": extracted.get("raw_ocr_text"),
                }
                st.session_state.result = None
                st.success("Extraction finished.")
            except Exception as e:
                st.error(f"Extraction failed: {e}")

st.markdown("---")

if st.session_state.extracted:
    preview_block(st.session_state.extracted)
    if st.button("Run claim pipeline", type="primary"):
        ext = st.session_state.extracted
        with st.spinner("Running claim pipeline (first handbook load can take a minute)..."):
            st.session_state.result = process_now(
                ext.get("raw_text") or "Receipt claim",
                {
                    "structured_claim": {
                        "claimant_name": ext.get("patient_name"),
                        "policy_number": ext.get("policy_number"),
                        "claim_amount": ext.get("amount_charged"),
                        "description": ext.get("diagnosis"),
                        "incident_date": ext.get("treatment_date"),
                        "claim_type": "health",
                    }
                },
            )
        st.success("Pipeline finished.")

def build_result_text(res: dict) -> str:
    ver = res.get("verification") or {}
    exc = res.get("exclusion_check") or {}
    amt = res.get("claim_amount") or 0
    cust = ver.get("matched_customer") or {}
    lines = [
        "Insurance claim result",
        "======================",
        f"Decision: {str(res.get('decision_status', 'unknown')).upper()}",
        f"Reason: {res.get('decision_reason') or ver.get('detail') or 'N/A'}",
        "",
        f"Claimant: {res.get('claimant_name') or 'N/A'}",
        f"Policy: {res.get('policy_number') or 'N/A'}",
        f"Amount: ${amt:,.2f}",
        f"Description: {res.get('description') or 'N/A'}",
        "",
        f"Verification passed: {ver.get('passed')}",
        f"Customer id: {ver.get('customer_id') or cust.get('customer_id') or 'N/A'}",
        f"Remaining fund (at match): {cust.get('remaining_fund_balance', 'N/A')}",
        "",
        f"Policy found: {res.get('policy_found')}",
        f"Coverage valid: {res.get('coverage_valid')}",
        f"Policy notes: {res.get('policy_notes') or 'N/A'}",
        "",
        f"Exclusion applied: {exc.get('excluded')}",
        f"Exclusion method: {exc.get('method') or 'N/A'}",
        f"Exclusion reasoning: {exc.get('reasoning') or 'N/A'}",
        "",
        "Letter:",
        res.get("communication_letter") or "(none)",
    ]
    return "\n".join(str(x) for x in lines)


def build_result_pdf(res: dict) -> bytes:
    from io import BytesIO
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.lib.units import mm
    from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer

    buf = BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, leftMargin=18 * mm, rightMargin=18 * mm)
    styles = getSampleStyleSheet()
    story = [Paragraph("Insurance claim result", styles["Title"]), Spacer(1, 8)]
    for line in build_result_text(res).split("\n"):
        if not line.strip():
            story.append(Spacer(1, 6))
        else:
            safe = (
                line.replace("&", "&amp;")
                .replace("<", "&lt;")
                .replace(">", "&gt;")
            )
            story.append(Paragraph(safe.replace("\n", "<br/>"), styles["Normal"]))
    doc.build(story)
    return buf.getvalue()


if st.session_state.result:
    res = st.session_state.result
    ver = res.get("verification", {}) or {}
    exc = res.get("exclusion_check") or {}
    st.subheader("Claim result")
    st.button(
        "New receipt (remove file and start over)",
        on_click=clear_receipt,
        type="primary",
        key="new_receipt_after_eval",
    )
    if ver.get("passed") is False:
        st.error("Rejected at Verification")
        st.warning(ver.get("detail", ""))
    else:
        if ver:
            st.success("Verification Passed")
        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("Claimant", res.get("claimant_name", "N/A"))
        with c2:
            amt = res.get("claim_amount") or 0
            st.metric("Amount", f"${amt:,.2f}")
        with c3:
            st.metric("Decision", str(res.get("decision_status", "unknown")).upper())

        st.write("**Reason:**", res.get("decision_reason") or ver.get("detail") or "—")
        st.write("**Policy notes:**", res.get("policy_notes") or "—")
        if exc:
            st.write(
                "**Exclusion:**",
                "Yes" if exc.get("excluded") else "No",
                "—",
                exc.get("reasoning") or "",
            )
        if res.get("communication_letter"):
            with st.expander("Customer letter"):
                st.text(res.get("communication_letter"))

    txt = build_result_text(res)
    d1, d2 = st.columns(2)
    with d1:
        st.download_button(
            "Download result (TXT)",
            data=txt,
            file_name="claim_result.txt",
            mime="text/plain",
            key="dl_txt",
        )
    with d2:
        try:
            st.download_button(
                "Download result (PDF)",
                data=build_result_pdf(res),
                file_name="claim_result.pdf",
                mime="application/pdf",
                key="dl_pdf",
            )
        except Exception as e:
            st.caption(f"PDF download unavailable: {e}")

if show_admin:
    st.markdown("---")
    st.subheader("Customers")
    try:
        from customer_db import list_customers

        st.dataframe(list_customers(), use_container_width=True)
    except Exception as e:
        st.error(f"Could not load customers: {e}")
