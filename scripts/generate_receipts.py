"""
receipt_generator.py — Generates synthetic medical receipts for testing.
Receipts reference real customers from customer_db for consistency.
"""

import os
from datetime import date, timedelta
import random
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.pdfgen import canvas
from reportlab.lib import colors
from faker import Faker
from typing import Dict
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pipeline import customer_db
from pipeline.paths import RECEIPTS_DIR

OUTPUT_DIR = str(RECEIPTS_DIR)
os.makedirs(OUTPUT_DIR, exist_ok=True)

fake = Faker()

def _create_receipt_pdf(customer: Dict, scenario: str, amount: float, description: str, filename: str) -> str:
    """Internal function to render a receipt PDF using reportlab."""
    filepath = os.path.join(OUTPUT_DIR, filename)
    c = canvas.Canvas(filepath, pagesize=A4)
    width, height = A4

    # Header
    c.setFont("Helvetica-Bold", 16)
    c.drawCentredString(width / 2, height - 30*mm, "MEDICAL RECEIPT / TAX INVOICE")
    c.setFont("Helvetica", 9)
    c.drawCentredString(width / 2, height - 38*mm, "City General Hospital & Specialist Centre")

    # Patient Info
    y = height - 55*mm
    c.setFont("Helvetica-Bold", 10)
    c.drawString(20*mm, y, "Patient Details")
    c.setFont("Helvetica", 9)
    y -= 6*mm
    c.drawString(20*mm, y, f"Name: {customer['name']}")
    y -= 5*mm
    c.drawString(20*mm, y, f"Policy No: {customer['policy_number']}")
    y -= 5*mm
    c.drawString(20*mm, y, f"Date: {(date.today() - timedelta(days=random.randint(1, 30))).strftime('%d/%m/%Y')}")
    y -= 5*mm
    c.drawString(20*mm, y, f"Provider: {fake.company()} Medical Clinic")

    # Description / Diagnosis
    y -= 15*mm
    c.setFont("Helvetica-Bold", 10)
    c.drawString(20*mm, y, "Description / Diagnosis")
    c.setFont("Helvetica", 9)
    y -= 6*mm
    # Wrap text
    text_obj = c.beginText(20*mm, y)
    text_obj.textLine(description)
    c.drawText(text_obj)

    # Amount
    y = 80*mm
    c.setFont("Helvetica-Bold", 12)
    c.drawString(20*mm, y, "Total Amount Due (HKD):")
    c.setFont("Helvetica-Bold", 14)
    c.setFillColor(colors.red)
    c.drawRightString(width - 20*mm, y, f"${amount:,.2f}")
    c.setFillColor(colors.black)

    # Footer
    c.setFont("Helvetica", 8)
    c.drawCentredString(width / 2, 20*mm, "This is a computer generated receipt. Thank you for your visit.")

    c.save()
    return filepath

def generate_receipt(customer: Dict, scenario: str) -> str:
    """
    Generates a receipt PDF based on the scenario.
    'clean': Normal amount within balance.
    'over_limit': Amount exceeds remaining balance.
    'exclusion_match': Description contains excluded terms (e.g., 'checkup', 'pregnancy').
    'unknown_customer': Generates a receipt with a name not in the DB (handled by caller).
    """
    if scenario == "clean":
        amount = round(random.uniform(500, min(5000, customer['remaining_fund_balance'] * 0.8)), 2)
        description = random.choice(["Consultation and medication", "Minor surgical procedure", "Diagnostic imaging (MRI)"])
    elif scenario == "over_limit":
        amount = round(customer['remaining_fund_balance'] * 1.5, 2)
        description = "Emergency hospitalization and treatment"
    elif scenario == "exclusion_match":
        amount = round(random.uniform(300, 1500), 2)
        description = random.choice(["Routine annual health checkup", "Pregnancy consultation and scan", "Cosmetic dermatology consultation"])
    else:
        amount = 1000.0
        description = "General consultation"

    filename = f"{customer['customer_id']}_{scenario}.pdf"
    return _create_receipt_pdf(customer, scenario, amount, description, filename)

def generate_all_test_receipts():
    """Generates the full test suite of receipts."""
    print("[receipt_generator] Generating test receipts...")
    # Ensure DB is populated
    if not os.path.exists(customer_db.DB_PATH):
        customer_db.generate_sample_customers(25)
    
    customers = customer_db.list_customers()
    
    # 1. Clean receipts for all 25
    for cust in customers:
        generate_receipt(cust, "clean")
    
    # 2. Over limit (5 random)
    for cust in random.sample(customers, 5):
        generate_receipt(cust, "over_limit")

    # 3. Exclusion match (5 random)
    for cust in random.sample(customers, 5):
        generate_receipt(cust, "exclusion_match")

    # 4. Unknown customer (3 fake entries)
    for i in range(3):
        fake_cust = {
            "name": fake.name(),
            "policy_number": f"POL-{random.randint(10000, 99999)}",
            "customer_id": f"UNK-{i}",
            "remaining_fund_balance": 0 # Not relevant
        }
        filename = f"UNK_{i}_unknown_customer.pdf"
        _create_receipt_pdf(fake_cust, "unknown", 1500.0, "General consultation", filename)
        
    print(f"[receipt_generator] Test receipts generated in {OUTPUT_DIR}")

if __name__ == "__main__":
    generate_all_test_receipts()