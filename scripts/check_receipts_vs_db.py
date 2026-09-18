"""Compare every extractable receipt field against customer_db."""
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pdfplumber
from pipeline.customer_db import list_customers, get_customer_by_policy_number
from pipeline.paths import RECEIPTS_DIR as RECEIPT_DIR


def parse_pdf(path: Path) -> dict:
    text = ""
    with pdfplumber.open(str(path)) as pdf:
        for page in pdf.pages:
            text += (page.extract_text() or "") + "\n"

    name = None
    m = re.search(r"^Name:\s*(.+)$", text, re.M)
    if m:
        name = m.group(1).strip()

    policy = None
    m = re.search(r"Policy No:\s*(POL-?\d+)", text, re.I)
    if m:
        policy = m.group(1).upper()
    elif re.search(r"(POL-?\d+)", text, re.I):
        policy = re.search(r"(POL-?\d+)", text, re.I).group(1).upper()

    date_val = None
    m = re.search(r"^Date:\s*(.+)$", text, re.M)
    if m:
        date_val = m.group(1).strip()

    provider = None
    m = re.search(r"^Provider:\s*(.+)$", text, re.M)
    if m:
        provider = m.group(1).strip()

    diagnosis = None
    m = re.search(r"Description / Diagnosis\s*\n(.+)", text)
    if m:
        diagnosis = m.group(1).strip()

    amount = None
    m = re.search(r"Total Amount Due \(HKD\):\s*\$?\s*([\d,]+\.\d{2})", text)
    if not m:
        m = re.search(r"\$\s*([\d,]+\.\d{2})", text)
    if m:
        amount = float(m.group(1).replace(",", ""))

    header = None
    m = re.search(r"MEDICAL RECEIPT.*", text)
    if m:
        header = m.group(0).strip()

    return {
        "name": name,
        "policy": policy,
        "date": date_val,
        "provider": provider,
        "diagnosis": diagnosis,
        "amount": amount,
        "header": header,
        "raw": text,
    }


def scenario_from_filename(stem: str) -> str:
    if "unknown" in stem:
        return "unknown_customer"
    if "over_limit" in stem:
        return "over_limit"
    if "exclusion" in stem:
        return "exclusion_match"
    if "clean" in stem:
        return "clean"
    return "other"


def main():
    customers = {c["customer_id"]: c for c in list_customers()}
    print(f"DB customers: {len(customers)}")
    print("DB columns:", sorted(next(iter(customers.values())).keys()) if customers else [])
    print("=" * 90)

    files = sorted(RECEIPT_DIR.glob("*.pdf"))
    mismatches = 0
    constraint_fails = 0

    print("\n=== FIELD-BY-FIELD (name + policy vs DB) ===\n")
    for f in files:
        parsed = parse_pdf(f)
        cid = f.stem.split("_")[0]
        scenario = scenario_from_filename(f.stem)

        if scenario == "unknown_customer":
            db = get_customer_by_policy_number(parsed["policy"] or "")
            ok = db is None
            if not ok:
                mismatches += 1
            print(
                f"{f.name}\n"
                f"  PDF  name={parsed['name']!r} pol={parsed['policy']} amount={parsed['amount']} "
                f"date={parsed['date']!r} dx={parsed['diagnosis']!r}\n"
                f"  DB   {'FOUND (should NOT exist)' if db else 'not in DB (correct)'}\n"
                f"  {'FAIL' if not ok else 'OK'}\n"
            )
            continue

        cust = customers.get(cid)
        if not cust:
            mismatches += 1
            print(f"{f.name}\n  FAIL: {cid} missing from DB\n")
            continue

        issues = []
        if parsed["name"] != cust["name"]:
            issues.append(f"name PDF={parsed['name']!r} DB={cust['name']!r}")
        if parsed["policy"] != cust["policy_number"]:
            issues.append(f"policy PDF={parsed['policy']!r} DB={cust['policy_number']!r}")

        bal = float(cust["remaining_fund_balance"])
        amt = parsed["amount"]
        if amt is None:
            issues.append("amount missing from PDF")
            constraint_fails += 1
        elif scenario == "clean" and amt > bal:
            issues.append(f"clean amount {amt} > remaining {bal}")
            constraint_fails += 1
        elif scenario == "over_limit" and amt <= bal:
            issues.append(f"over_limit amount {amt} <= remaining {bal}")
            constraint_fails += 1
        elif scenario == "exclusion_match" and amt > bal:
            issues.append(f"exclusion amount {amt} > remaining {bal} (should still be in-budget)")
            constraint_fails += 1

        status = "MATCH" if not issues else "MISMATCH"
        if issues:
            mismatches += 1

        print(f"{f.name}  [{scenario}]  {status}")
        print(f"  PDF  name={parsed['name']!r}")
        print(f"       policy={parsed['policy']}")
        print(f"       amount={parsed['amount']}")
        print(f"       date={parsed['date']!r}")
        print(f"       provider={parsed['provider']!r}")
        print(f"       diagnosis={parsed['diagnosis']!r}")
        print(f"  DB   id={cust['customer_id']}")
        print(f"       name={cust['name']!r}")
        print(f"       policy={cust['policy_number']}")
        print(f"       remaining={cust['remaining_fund_balance']}")
        print(f"       total_limit={cust['total_fund_limit']}")
        print(f"       benefit={cust['benefit_type']!r}")
        print(f"       coverage_start={cust['coverage_start_date']}")
        if issues:
            print(f"  ISSUES: {issues}")
        print()

    print("=" * 90)
    print("\n=== DB FIELDS NOT PRINTED ON RECEIPTS ===")
    print("  coverage_start_date  — not on receipt (membership date)")
    print("  benefit_type         — not on receipt")
    print("  total_fund_limit      — not on receipt (only remaining is used for over_limit math)")
    print("  customer_id          — only in filename, not inside PDF body")
    print("\n=== RECEIPT FIELDS NOT IN DB ===")
    print("  Date (treatment)     — random last 30 days")
    print("  Provider             — Faker hospital/clinic name")
    print("  Diagnosis            — scenario text")
    print("  Amount               — random; constrained vs remaining_fund_balance")
    print("-" * 90)
    print(f"Identity mismatches (name/policy/unknown): {mismatches}")
    print(f"Amount-constraint failures: {constraint_fails}")
    print(f"Files checked: {len(files)}")


if __name__ == "__main__":
    main()
