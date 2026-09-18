"""
One-time setup script to create and populate the customer database.

Running this will (re)create data/customers.db with 25 fake customers.
"""

import os
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from pipeline.customer_db import init_db, generate_sample_customers
from pipeline.paths import CUSTOMERS_DB as DB_PATH


def main():
    print("=== Customer Database Setup ===\n")

    if DB_PATH.exists():
        print(f"WARNING: Database already exists at {DB_PATH}")
        choice = input("Do you want to DELETE and recreate it? (y/n): ").strip().lower()
        if choice != "y":
            print("Aborted.")
            return
        else:
            os.remove(DB_PATH)
            print("Existing database deleted.")

    print("\nInitializing database and generating 25 sample customers...")
    init_db()
    generate_sample_customers(25)

    print(f"\n✅ Customer database created successfully at: {DB_PATH}")
    print("You can now use it in the application.")


if __name__ == "__main__":
    main()