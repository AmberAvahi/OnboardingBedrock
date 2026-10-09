import os
import random
from concurrent.futures import ThreadPoolExecutor

from fpdf import FPDF

from common import MODELS, invoke_model

TYPES = {"nda": "Non-Disclosure Agreement", "employment": "Employment Agreement",
         "lease": "Commercial Lease Agreement", "services": "Master Services Agreement",
         "license": "Software License Agreement"}
COMPANIES = ["Acme", "Globex", "Initech", "Umbrella", "Hooli", "Stark", "Wayne", "Wonka", "Cyberdyne", "Tyrell",
             "Soylent", "Vandelay", "Oscorp", "Aperture", "Gringotts", "Dunder", "Waystar", "Pendant", "Bluth", "Prestige"]
STATES = ["Delaware", "California", "New York", "Texas", "Florida", "Illinois"]
random.seed(0)
DOCS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "legal_docs")
os.makedirs(DOCS, exist_ok=True)


def make(job):
    key, n, a, b, state, years, notice, amount = job
    text, _ = invoke_model(
        MODELS["Haiku 4.5"],
        f"Write a realistic {TYPES[key]} (about 600 words) between {a} Inc. and {b} Corp. Required facts: governing law "
        f"{state}; term {years} years; termination notice {notice} days; main fee or payment ${amount:,}. Structure it as "
        "numbered sections, each starting on its own line exactly like 'Section 1. Title' followed by the clauses. "
        "No markdown and no preamble; the first line is the document title.", 1500)
    text = text.translate(str.maketrans("“”‘’–—", "\"\"''--")).encode("latin-1", "replace").decode("latin-1")
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", size=10)
    pdf.multi_cell(0, 5, text)
    pdf.output(f"{DOCS}/{key}_{n:03d}_{a.lower()}_{b.lower()}.pdf")


jobs = [(key, n, *random.sample(COMPANIES, 2), random.choice(STATES), random.choice([1, 2, 3, 5]),
         random.choice([15, 30, 45, 60, 90]), random.randrange(5, 200) * 1000)
        for key in TYPES for n in range(1, 23)]
with ThreadPoolExecutor(8) as pool:
    list(pool.map(make, jobs))
print(f"{len(os.listdir(DOCS))} PDFs in {DOCS}")
