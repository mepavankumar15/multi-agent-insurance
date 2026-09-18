"""
test_vision_extraction.py
-------------------------
Sanity-check script for the generic vision extraction capability in receipt_vision.py
using SROIE-style sample receipts (images + ground-truth JSON).

This is intentionally kept separate from the main claims_agents test suite.
It reports per-field accuracy and flags problematic images for manual review.
It does NOT fail the overall test run on imperfect extraction.
"""

import os
import json
import re
from difflib import SequenceMatcher
from PIL import Image
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pipeline.receipt_vision import pdf_to_images, extract_generic_receipt

RECEIPTS_DIR = "test_data/sample_receipts"


def normalize_date(d):
    if not d:
        return None
    # Try to extract YYYY-MM-DD or similar
    m = re.search(r"(\d{4})[-/](\d{1,2})[-/](\d{1,2})", str(d))
    if m:
        return f"{m.group(1)}-{int(m.group(2)):02d}-{int(m.group(3)):02d}"
    return str(d).strip()


def normalize_total(t):
    if t is None:
        return None
    s = re.sub(r"[^\d.]", "", str(t))
    try:
        return float(s)
    except:
        return None


def fuzzy_match(a, b, threshold=0.6):
    if not a or not b:
        return False
    return SequenceMatcher(None, str(a).lower(), str(b).lower()).ratio() > threshold


def run_test():
    if not os.path.isdir(RECEIPTS_DIR):
        print(f"ERROR: {RECEIPTS_DIR} does not exist. Please place SROIE sample images + .json files there.")
        return

    files = [f for f in os.listdir(RECEIPTS_DIR) if f.lower().endswith((".png", ".jpg", ".jpeg"))]
    if not files:
        print("No image files found in test_data/sample_receipts/")
        return

    results = {"company": 0, "address": 0, "date": 0, "total": 0}
    total_images = len(files)
    failures = []

    print(f"\n=== Vision Extraction Sanity Check on {total_images} receipts ===\n")

    for fname in sorted(files):
        base = os.path.splitext(fname)[0]
        img_path = os.path.join(RECEIPTS_DIR, fname)
        gt_path = os.path.join(RECEIPTS_DIR, base + ".json")

        if not os.path.exists(gt_path):
            print(f"SKIP {fname}: no matching ground-truth JSON")
            continue

        with open(gt_path, "r", encoding="utf-8") as f:
            gt = json.load(f)

        # Load image (support both direct images and single-page PDFs if needed)
        try:
            if fname.lower().endswith(".pdf"):
                images = pdf_to_images(img_path)
            else:
                images = [Image.open(img_path)]
        except Exception as e:
            print(f"ERROR loading {fname}: {e}")
            continue

        extracted = extract_generic_receipt(images)

        # Compare
        company_ok = fuzzy_match(extracted.get("company"), gt.get("company"))
        address_ok = fuzzy_match(extracted.get("address"), gt.get("address"))
        date_ok = normalize_date(extracted.get("date")) == normalize_date(gt.get("date"))
        total_ok = normalize_total(extracted.get("total")) == normalize_total(gt.get("total"))

        if company_ok: results["company"] += 1
        if address_ok: results["address"] += 1
        if date_ok:    results["date"] += 1
        if total_ok:   results["total"] += 1

        status = "PASS" if all([company_ok, address_ok, date_ok, total_ok]) else "PARTIAL"
        print(f"{status:8} | {fname}")
        print(f"         GT: company={gt.get('company')}, date={gt.get('date')}, total={gt.get('total')}")
        print(f"         EX: company={extracted.get('company')}, date={extracted.get('date')}, total={extracted.get('total')}")
        print(f"         method={extracted.get('method', 'unknown')}\n")

        if not all([company_ok, address_ok, date_ok, total_ok]):
            reason = []
            if not company_ok: reason.append("company mismatch")
            if not address_ok: reason.append("address mismatch")
            if not date_ok:    reason.append("date mismatch")
            if not total_ok:   reason.append("total mismatch")
            failures.append((fname, ", ".join(reason), extracted.get("method")))

    # Summary
    print("\n=== Overall Field Accuracy ===")
    for field, correct in results.items():
        pct = (correct / total_images) * 100 if total_images else 0
        print(f"{field:8}: {correct}/{total_images} correct ({pct:.1f}%)")

    if failures:
        print("\n=== Images needing manual review ===")
        for fname, reason, method in failures:
            print(f"  {fname}  — {reason}  (method: {method})")
    else:
        print("\nAll images passed perfectly.")


if __name__ == "__main__":
    run_test()
