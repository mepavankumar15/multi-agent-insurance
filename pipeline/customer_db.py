"""
customer_db.py — SQLite customer database for fund verification and deduction.
"""

import sqlite3
import os
from datetime import date, timedelta
from typing import Optional, Dict, List
import random

from .paths import CUSTOMERS_DB

DB_PATH = str(CUSTOMERS_DB)

BENEFIT_TYPES = ["Hospital & Surgical", "Clinical", "Maternity", "Dental", "Network Dental"]

def init_db():
    """Create the customers table if it doesn't exist."""
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS customers (
            customer_id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            policy_number TEXT UNIQUE NOT NULL,
            total_fund_limit REAL NOT NULL,
            remaining_fund_balance REAL NOT NULL,
            coverage_start_date TEXT NOT NULL,
            benefit_type TEXT NOT NULL
        )
    """)
    conn.commit()
    conn.close()
    print(f"[customer_db] Database initialized at {DB_PATH}")

def generate_sample_customers(n: int = 25):
    """Populate the database with n fake customers using faker."""
    try:
        from faker import Faker
    except ImportError:
        print("ERROR: faker library not found. Please run: pip install faker")
        return

    init_db()
    fake = Faker()
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # Clear existing data for reproducibility during setup
    cursor.execute("DELETE FROM customers")

    for i in range(1, n + 1):
        customer_id = f"CUST-{i:03d}"
        name = fake.name()
        policy_number = f"POL-{random.randint(10000, 99999)}"
        total_limit = round(random.uniform(20000, 150000), 2)
        remaining_balance = total_limit  # Start full
        # Coverage start date: 1 to 6 years ago
        years_ago = random.randint(1, 6)
        start_date = (date.today() - timedelta(days=years_ago * 365)).isoformat()
        benefit = random.choice(BENEFIT_TYPES)

        cursor.execute("""
            INSERT INTO customers (customer_id, name, policy_number, total_fund_limit, 
                                   remaining_fund_balance, coverage_start_date, benefit_type)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (customer_id, name, policy_number, total_limit, remaining_balance, start_date, benefit))

    conn.commit()
    conn.close()
    print(f"[customer_db] Generated {n} sample customers.")

def _row_to_dict(cursor, row) -> Optional[Dict]:
    if not row:
        return None
    cols = [col[0] for col in cursor.description]
    return dict(zip(cols, row))


def get_customer_by_policy_number(policy_number: str) -> Optional[Dict]:
    if not policy_number:
        return None
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM customers WHERE policy_number = ?", (policy_number,))
    row = cursor.fetchone()
    data = _row_to_dict(cursor, row)
    conn.close()
    return data

def get_customer_by_name(name: str) -> Optional[Dict]:
    """
    Fuzzy match customer name using rapidfuzz.
    """
    try:
        from rapidfuzz import process, fuzz
    except ImportError:
        print("WARNING: rapidfuzz not installed. Falling back to exact match.")
        return get_customer_by_policy_number(name) # Fallback logic placeholder

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT name, customer_id FROM customers")
    all_names = cursor.fetchall()
    conn.close()

    if not all_names:
        return None

    # Extract just the names for matching
    name_list = [r[0] for r in all_names]
    match = process.extractOne(name, name_list, scorer=fuzz.WRatio, score_cutoff=85)

    if match:
        matched_name = match[0]
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM customers WHERE name = ?", (matched_name,))
        row = cursor.fetchone()
        data = _row_to_dict(cursor, row)
        conn.close()
        return data
    return None

def deduct_fund_balance(customer_id: str, amount: float):
    """Deduct amount from balance. Raises error if insufficient funds."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT remaining_fund_balance FROM customers WHERE customer_id = ?", (customer_id,))
    row = cursor.fetchone()
    if not row:
        conn.close()
        raise ValueError(f"Customer {customer_id} not found.")
    
    current_balance = row[0]
    if current_balance < amount:
        conn.close()
        raise ValueError(f"Insufficient funds for {customer_id}. Balance: {current_balance}, Amount: {amount}")
    
    new_balance = current_balance - amount
    cursor.execute("UPDATE customers SET remaining_fund_balance = ? WHERE customer_id = ?", (new_balance, customer_id))
    conn.commit()
    conn.close()
    print(f"[customer_db] Deducted {amount} from {customer_id}. New balance: {new_balance}")

def list_customers() -> List[Dict]:
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM customers ORDER BY customer_id")
    rows = cursor.fetchall()
    cols = [col[0] for col in cursor.description]
    conn.close()
    return [dict(zip(cols, row)) for row in rows]

if __name__ == "__main__":
    generate_sample_customers(25)