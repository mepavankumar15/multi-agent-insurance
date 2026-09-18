# Multi-Agent Insurance Claims Processor

Streamlit app for medical claim intake, member verification, handbook exclusion checks, and a written decision.

**Deploy:** Streamlit Community Cloud → this repo, file **`app.py`** (still at the project root).  
**Run locally:** `python run_app.py` → http://localhost:8501

---

## What it does

1. Read a claim from a form, pasted text, or a receipt (PDF / image).
2. Look up the member in SQLite (`data/customers.db`).
3. Check remaining fund balance.
4. Match the claim against **General exclusions** in `data/knowledge/policy_handbook.pdf`.
5. Return **approved**, **denied** (exclusion), **rejected** (unknown member / over fund), or **escalated**.

Grok (`XAI_API_KEY`) is optional. Without it, rule-based fallbacks still run.

---

## Layout

Kept simple for GitHub and for Streamlit Cloud (entry file unchanged):

```text
app.py                 ← Streamlit entry (Cloud + local)
run_app.py             ← local launcher
requirements.txt
.streamlit/config.toml

pipeline/              Python package (agents, ingest, RAG, vision, customers)
data/
  customers.db
  knowledge/policy_handbook.pdf
  receipts/            sample PDFs
scripts/               setup DB, generate receipts, compare receipts to DB
tests/
docs/notebooks/        optional notebooks
```

Nothing else is required at the root besides `app.py` and `requirements.txt`.

---

## Local setup

```bash
python -m pip install -r requirements.txt
python run_app.py
```

If you need a fresh sample database:

```bash
python scripts/setup_customer_db.py
```

Optional `.env` (not committed):

```text
XAI_API_KEY=...
```

On Streamlit Cloud, set the same key under **App settings → Secrets**.

---

## Sample receipts (`data/receipts/`)

| Pattern | Typical outcome |
|---------|-----------------|
| `CUST-*_clean.pdf` | Verify, usually approve |
| `CUST-*_over_limit.pdf` | Reject at verification (insufficient fund) |
| `CUST-*_exclusion_match.pdf` | Deny (handbook exclusion, e.g. cosmetic / checkup) |
| `UNK_*_unknown_customer.pdf` | Reject at verification (not in DB) |

---

## Scripts

```bash
python scripts/setup_customer_db.py
python scripts/generate_receipts.py
python scripts/check_receipts_vs_db.py
```
