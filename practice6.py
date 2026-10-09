import glob
import json
import re

import gradio as gr
import numpy as np

from common import MODELS, bedrock_runtime, invoke_model

MODEL = MODELS["Haiku 4.5"]


def embed(text):
    response = bedrock_runtime.invoke_model(
        modelId="amazon.titan-embed-text-v2:0", body=json.dumps({"inputText": text}))
    vector = np.array(json.loads(response["body"].read())["embedding"])
    return vector / np.linalg.norm(vector)


def chunk(text, size=600):
    chunks, current = [], ""
    for paragraph in text.split("\n\n"):
        if current and len(current) + len(paragraph) > size:
            chunks.append(current.strip())
            current = ""
        current += paragraph + "\n\n"
    return chunks + [current.strip()]


chunks = [c for path in glob.glob("docs/*.txt") for c in chunk(open(path).read())]
vectors = np.array([embed(c) for c in chunks])


def answer(question):
    top = np.argsort(vectors @ embed(question))[::-1][:8]
    passages = "\n".join(f"[{i}] {chunks[i]}" for i in top)
    ranking, _ = invoke_model(
        MODEL, f"Question: {question}\n\nPassages:\n{passages}\n\n"
        "Reply only with the numbers of the 3 most relevant passages, best first.", 20)
    best = [int(n) for n in re.findall(r"\d+", ranking) if int(n) in top][:3] or list(top[:3])
    context = "\n\n".join(chunks[i] for i in best)
    reply, _ = invoke_model(MODEL, f"Answer using only this context:\n{context}\n\nQuestion: {question}")
    return reply, context


gr.Interface(answer, gr.Textbox(label="Question"), [gr.Textbox(label="Answer"), gr.Textbox(label="Context used")], allow_flagging="never").launch()
