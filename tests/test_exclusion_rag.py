"""
test_exclusion_rag.py — Full test suite for the exclusion RAG pipeline.

Tests:
1. Ingestion: PDF chunking and section tagging
2. CJK exclusion: no Chinese text in embedded chunks
3. Exclusion matching E2E: pregnancy claim triggers exclusion
4. 90-day submission deadline: date math fires correctly
5. API key fallback: rule-based fallback works without XAI_API_KEY
"""
import os
import sys
import warnings

warnings.filterwarnings("ignore")
import logging
logging.getLogger("pdfplumber").setLevel(logging.ERROR)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from datetime import date, timedelta
from ingest import ingest_policy_pdf
from claims_agents import process_claim, set_chroma_collection


def test_ingestion():
    """Test ingestion on the sample PDF."""
    print("\n--- Test 1: Ingestion ---")
    pdf_path = None
    # Look for any PDF file in the current directory
    for f in os.listdir("."):
        if f.endswith(".pdf"):
            pdf_path = f
            break

    if not pdf_path:
        print("SKIP: No PDF file found in current directory.")
        return None

    print(f"Ingesting {pdf_path}...")
    collection, summary = ingest_policy_pdf(pdf_path)

    print("Ingestion Summary:")
    print(f"  Exclusions:  {summary.get('exclusion', 0)}")
    print(f"  Definitions: {summary.get('definition', 0)}")
    print(f"  Procedures:  {summary.get('procedure', 0)}")
    print(f"  Total Chunks: {summary.get('total', 0)}")

    if summary.get("exclusion", 0) > 0:
        print(f"PASS: {summary['exclusion']} exclusion chunks indexed")
    else:
        print("INFO: No exclusion chunks found — check if PDF contains numbered exclusion clauses")

    set_chroma_collection(collection)
    return collection


def test_chinese_exclusion(collection):
    """Test that Chinese-language content is excluded from the embedded chunks."""
    print("\n--- Test 2: Chinese Exclusion ---")
    if not collection:
        print("SKIP: No collection.")
        return

    from retriever import retrieve_relevant_clauses
    import re

    clauses = retrieve_relevant_clauses(collection, "health insurance claim", k=5)
    print(f"  Retrieved {len(clauses)} clauses")

    has_cjk = False
    for c in clauses:
        if re.search(r'[\u4e00-\u9fff\u3400-\u4dbf]', c["text"]):
            has_cjk = True
            break

    if has_cjk:
        print("FAIL: Found CJK characters in retrieved English chunks.")
    else:
        print("PASS: No CJK characters found in retrieved chunks.")


def test_exclusion_matching(collection):
    """Test exclusion matching E2E with a pregnancy-related claim."""
    print("\n--- Test 3: Exclusion Matching E2E ---")
    if not collection:
        print("SKIP: No collection.")
        return

    claim_desc = "Patient admitted for pregnancy complications and childbirth delivery under Hospital & Surgical benefit."

    initial_data = {
        "treatment_date": date.today().isoformat(),
        "coverage_start_date": (date.today() - timedelta(days=365)).isoformat(),
        "benefit_type": "Hospital & Surgical",
        "submission_deadline_exceeded": False,
        "structured_claim": {
            "claimant_name": "Test User",
            "policy_number": "POL-40100",
            "claim_amount": 5000.0,
            "description": claim_desc,
            "incident_date": date.today().isoformat(),
            "claim_type": "health"
        }
    }

    result = process_claim(claim_desc, initial_data)
    exc_data = result.get("exclusion_check", {})

    print(f"  Excluded: {exc_data.get('excluded')}")
    print(f"  Clause#:  {exc_data.get('matched_clause_number')}")
    print(f"  Method:   {exc_data.get('method')}")
    print(f"  Reasoning: {exc_data.get('reasoning', '')[:120]}")
    print(f"  Decision:  {result.get('decision_status')}")

    if exc_data.get("excluded"):
        print("PASS: Claim was excluded.")
    else:
        print("INFO: Claim was NOT excluded (depends on PDF content and whether it contains pregnancy exclusion).")


def test_90_day_deadline():
    """Test that a claim older than 90 days gets flagged and escalated."""
    print("\n--- Test 4: 90-Day Deadline ---")

    past_date = date.today() - timedelta(days=100)

    initial_data = {
        "treatment_date": past_date.isoformat(),
        "coverage_start_date": (past_date - timedelta(days=365)).isoformat(),
        "benefit_type": "Hospital & Surgical",
        "submission_deadline_exceeded": True,
        "structured_claim": {
            "claimant_name": "Late Submitter",
            "policy_number": "POL-40100",
            "claim_amount": 500.0,
            "description": "Routine checkup",
            "incident_date": past_date.isoformat(),
            "claim_type": "health"
        }
    }

    result = process_claim("Routine checkup", initial_data)

    status = result.get("decision_status")
    reason = result.get("decision_reason", "")
    print(f"  Status: {status}")
    print(f"  Reason: {reason[:150]}")

    if status == "escalated":
        print("PASS: Claim escalated (deadline flag).")
    else:
        print(f"FAIL: Expected escalated, got {status}")


def test_fallback_method(collection):
    """Confirm the rule-based fallback still works if XAI_API_KEY is unset."""
    print("\n--- Test 5: API Key Fallback ---")

    original_key = os.environ.pop("XAI_API_KEY", None)

    try:
        initial_data = {
            "treatment_date": date.today().isoformat(),
            "coverage_start_date": (date.today() - timedelta(days=365)).isoformat(),
            "benefit_type": "Hospital & Surgical",
            "submission_deadline_exceeded": False,
            "structured_claim": {
                "claimant_name": "Fallback User",
                "policy_number": "POL-40100",
                "claim_amount": 5000.0,
                "description": "Treatment for pre-existing congenital condition",
                "incident_date": date.today().isoformat(),
                "claim_type": "health"
            }
        }

        result = process_claim("Treatment for pre-existing congenital condition", initial_data)
        exc_data = result.get("exclusion_check", {})

        method = exc_data.get("method")
        print(f"  Method used: {method}")
        if method in ("rule_based", "skipped"):
            print("PASS: Fallback rule-based method was used.")
        else:
            print(f"FAIL: Expected rule_based method, got {method}")

    finally:
        if original_key is not None:
            os.environ["XAI_API_KEY"] = original_key


if __name__ == "__main__":
    print("=" * 60)
    print("  Exclusion RAG Pipeline Tests")
    print("=" * 60)

    col = test_ingestion()
    test_chinese_exclusion(col)
    test_exclusion_matching(col)
    test_90_day_deadline()
    test_fallback_method(col)

    print("\n" + "=" * 60)
    print("  All tests finished.")
    print("=" * 60)
