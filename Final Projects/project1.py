import glob
import json
import os
import re
import warnings
from concurrent.futures import ThreadPoolExecutor

import gradio as gr
import numpy as np
from pypdf import PdfReader

from common import MODELS, embed, invoke_model

FAST, SMART = MODELS["Haiku 4.5"], MODELS["Sonnet 4.5"]
warnings.filterwarnings("ignore", category=RuntimeWarning)  # numpy 2.0 on macOS emits spurious matmul warnings


def load_chunks(path):
    name, label, chunks = os.path.basename(path), "Title", []
    for page, pdf_page in enumerate(PdfReader(path).pages, 1):
        for piece in re.split(r"(?m)^(?=Section \d+)", pdf_page.extract_text()):
            if piece.strip():
                match = re.match(r"Section (\d+)", piece)
                label = f"Section {match.group(1)}" if match else label
                chunks.append({"doc": name, "type": name.split("_")[0], "page": page, "section": label, "text": piece.strip()})
    return chunks


with ThreadPoolExecutor(16) as pool:
    chunks = [c for doc in pool.map(load_chunks, glob.glob(os.path.join(os.path.dirname(os.path.abspath(__file__)), "legal_docs", "*.pdf"))) for c in doc]
    vectors = np.array(list(pool.map(embed, [f"{c['doc'][:-4].replace('_', ' ')} {c['section']}\n{c['text']}" for c in chunks])))
# One index (row ids) per document type
indexes = {t: np.array([i for i, c in enumerate(chunks) if c["type"] == t]) for t in {c["type"] for c in chunks}}
print(f"{len({c['doc'] for c in chunks})} documents, {len(chunks)} chunks, indexes: {sorted(indexes)}")


def route(question):
    reply, _ = invoke_model(
        FAST, f"Document types: {sorted(indexes)}. Which are relevant to the question? Reply only with a JSON list "
        f"of types, or all of them if unsure.\nQuestion: {question}", 60)
    return [t for t in re.findall(r"\w+", reply) if t in indexes] or list(indexes)


def retrieve(question, k=6):
    ids = np.concatenate([indexes[t] for t in route(question)])
    return [chunks[i] for i in ids[np.argsort(vectors[ids] @ embed(question))[::-1][:k]]]


def standalone(message, history):
    if not history:
        return message
    turns = "\n".join(f"User: {u}\nAssistant: {a}" for u, a in history[-3:])
    question, _ = invoke_model(
        FAST, f"{turns}\n\nRewrite this follow-up as a standalone question. Reply only with the question: {message}", 100)
    return question.strip()


def summarize(filename):
    text = "\n".join(c["text"] for c in chunks if c["doc"] == filename)
    if not text:
        return f"No document named {filename}. Use a file name from legal_docs/."
    return invoke_model(SMART, "Write an executive summary of this legal document in at most 8 bullets: parties, term, "
                        f"key obligations, fees, termination and notable risks.\n\n{text}", 600)[0]


def answer(message, history):
    if message.startswith("/summary "):
        return summarize(message.split(maxsplit=1)[1].strip())
    question = standalone(message, history)
    context = "\n\n".join(f"[{c['doc']}, p.{c['page']}, {c['section']}]\n{c['text']}" for c in retrieve(question))
    return invoke_model(
        SMART, "Answer using only these excerpts. Cite every claim as [file, p.N, Section N] exactly as labeled. "
        f"If the excerpts do not contain the answer, say so.\n\n{context}\n\nQuestion: {question}", 700)[0]


gr.ChatInterface(answer, title="Legal Assistant").launch(show_error=True)
