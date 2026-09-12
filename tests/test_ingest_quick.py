import warnings; warnings.filterwarnings('ignore')
import logging; logging.getLogger('pdfplumber').setLevel(logging.ERROR)

from ingest import ingest_policy_pdf
collection, summary = ingest_policy_pdf('sample_policy.pdf')
print('Ingestion Summary:')
for k, v in summary.items():
    print(f'  {k}: {v}')

print()
from retriever import retrieve_relevant_clauses
clauses = retrieve_relevant_clauses(collection, 'pregnancy childbirth delivery', section_type='exclusion', k=3)
print(f'Retrieved {len(clauses)} exclusion clauses for pregnancy query:')
for c in clauses:
    dist = c["distance"]
    text = c["text"][:120]
    print(f'  dist={dist:.3f} | {text}...')

print()
clauses2 = retrieve_relevant_clauses(collection, 'pre-existing condition', section_type='exclusion', k=3)
print(f'Retrieved {len(clauses2)} exclusion clauses for pre-existing query:')
for c in clauses2:
    dist = c["distance"]
    text = c["text"][:120]
    print(f'  dist={dist:.3f} | {text}...')
