"""Project paths. Resolved from this file so it works in Streamlit, scripts, and tests."""

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
KNOWLEDGE_PDF = DATA_DIR / "knowledge" / "policy_handbook.pdf"
CUSTOMERS_DB = DATA_DIR / "customers.db"
RECEIPTS_DIR = DATA_DIR / "receipts"
