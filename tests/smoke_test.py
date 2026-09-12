"""
smoke_test.py — Validate the full exclusion RAG pipeline end-to-end.

Outputs results to stdout.  Suppresses pdfplumber warnings on stderr.
"""
import sys, os, warnings
warnings.filterwarnings("ignore")

# Suppress pdfplumber FontBBox warnings
import logging
logging.getLogger("pdfplumber").setLevel(logging.ERROR)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from datetime import date, timedelta


def test_ingestion():
    print("\n[1] Ingestion test")
    from ingest import ingest_policy_pdf

    pdf_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "insurance_claims_pipeline_executed.pdf")
    if not os.path.exists(pdf_path):
        print("  SKIP: PDF not found at", pdf_path)
        return None

    collection, summary = ingest_policy_pdf(pdf_path)
    print(f"  Exclusions:  {summary.get('exclusion', 0)}")
    print(f"  Definitions: {summary.get('definition', 0)}")
    print(f"  Procedures:  {summary.get('procedure', 0)}")
    print(f"  Total:       {summary.get('total', 0)}")

    n_exc = summary.get("exclusion", 0)
    if n_exc > 0:
        print(f"  PASS: {n_exc} exclusion chunks indexed")
    else:
        print("  FAIL: No exclusion chunks found")

    return collection


def test_cjk_exclusion(collection):
    print("\n[2] CJK character exclusion test")
    if collection is None:
        print("  SKIP: No collection"); return
    import re
    from retriever import retrieve_relevant_clauses
    clauses = retrieve_relevant_clauses(collection, "insurance claim exclusion", k=5)
    print(f"  Retrieved {len(clauses)} clauses")
    cjk = any(re.search(r'[\u4e00-\u9fff\u3400-\u4dbf]', c["text"]) for c in clauses)
    print("  FAIL: CJK found" if cjk else "  PASS: No CJK in results")


def test_exclusion_match(collection):
    print("\n[3] Exclusion matching (pregnancy, non-Maternity)")
    if collection is None:
        print("  SKIP: No collection"); return

    from claims_agents import process_claim, set_chroma_collection
    set_chroma_collection(collection)

    desc = "Patient admitted for pregnancy complications and delivery. Hospital & Surgical benefit."
    initial = {
        "treatment_date": date.today().isoformat(),
        "coverage_start_date": (date.today() - timedelta(days=365)).isoformat(),
        "benefit_type": "Hospital & Surgical",
        "submission_deadline_exceeded": False,
        "structured_claim": {
            "claimant_name": "Test User",
            "policy_number": "POL-40100",
            "claim_amount": 5000.0,
            "description": desc,
            "incident_date": date.today().isoformat(),
            "claim_type": "health",
        },
    }

    result = process_claim(desc, initial)
    exc = result.get("exclusion_check", {})
    print(f"  excluded:  {exc.get('excluded')}")
    print(f"  clause#:   {exc.get('matched_clause_number')}")
    print(f"  method:    {exc.get('method')}")
    print(f"  reasoning: {exc.get('reasoning', '')[:120]}")
    print(f"  decision:  {result.get('decision_status')}")
    if exc.get("excluded"):
        print("  PASS: Claim was excluded")
    else:
        print("  INFO: Claim not excluded (may depend on PDF content)")


def test_90_day_deadline():
    print("\n[4] 90-day submission deadline")
    from claims_agents import process_claim

    past = (date.today() - timedelta(days=100)).isoformat()
    initial = {
        "treatment_date": past,
        "coverage_start_date": (date.today() - timedelta(days=365)).isoformat(),
        "benefit_type": "Clinical",
        "submission_deadline_exceeded": True,
        "structured_claim": {
            "claimant_name": "Old Claim",
            "policy_number": "POL-40100",
            "claim_amount": 500.0,
            "description": "Routine checkup",
            "incident_date": past,
            "claim_type": "health",
        },
    }

    result = process_claim("Routine checkup", initial)
    status = result.get("decision_status")
    reason = result.get("decision_reason", "")
    print(f"  decision: {status}")
    print(f"  reason:   {reason[:150]}")
    if status == "escalated":
        print("  PASS: Escalated (deadline flag)")
    else:
        print(f"  FAIL: Expected escalated, got {status}")


def test_fallback_no_api_key(collection):
    print("\n[5] Fallback test (no XAI_API_KEY)")
    from claims_agents import process_claim, set_chroma_collection
    if collection:
        set_chroma_collection(collection)

    orig = os.environ.pop("XAI_API_KEY", None)
    try:
        initial = {
            "treatment_date": date.today().isoformat(),
            "coverage_start_date": (date.today() - timedelta(days=365)).isoformat(),
            "benefit_type": "Hospital & Surgical",
            "submission_deadline_exceeded": False,
            "structured_claim": {
                "claimant_name": "Fallback User",
                "policy_number": "POL-40100",
                "claim_amount": 3000.0,
                "description": "Treatment for pre-existing condition",
                "incident_date": date.today().isoformat(),
                "claim_type": "health",
            },
        }
        result = process_claim("Treatment for pre-existing condition", initial)
        exc = result.get("exclusion_check", {})
        method = exc.get("method")
        print(f"  method: {method}")
        if method in ("rule_based", "skipped"):
            print("  PASS: Rule-based fallback (or skipped) without API key")
        else:
            print(f"  FAIL: Unexpected method '{method}'")
    finally:
        if orig:
            os.environ["XAI_API_KEY"] = orig


if __name__ == "__main__":
    print("=" * 60)
    print("  Exclusion RAG Pipeline — Smoke Test")
    print("=" * 60)

    col = test_ingestion()
    test_cjk_exclusion(col)
    test_exclusion_match(col)
    test_90_day_deadline()
    test_fallback_no_api_key(col)

    print("\n" + "=" * 60)
    print("  All tests finished.")
    print("=" * 60)
