import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor

from bert_score import score as bert_score
from rouge_score import rouge_scorer
from sacrebleu import sentence_bleu

from common import MODELS, invoke_model

DOMAINS = ["science", "history", "geography", "technology", "literature"]
JUDGE = "us.anthropic.claude-sonnet-4-6"  # not one of the evaluated models, so no self-judging
PRICES = {"Haiku 4.5": (1, 5), "Sonnet 4.5": (3, 15), "Opus 4.6": (5, 25)}  # approx. USD per 1M tokens (in, out)
BENCHMARK = "benchmark.json"


def generate(domain):
    text, _ = invoke_model(
        JUDGE, f"Write 10 factual questions about {domain}, each with a one-sentence reference answer. "
        "Reply only with a JSON list of objects with keys 'question' and 'answer'.", 3000)
    return [{**q, "domain": domain} for q in json.loads(re.search(r"\[.*\]", text, re.S).group())]


def evaluate(task):
    model, q = task
    try:
        start = time.perf_counter()
        answer, usage = invoke_model(MODELS[model], q["question"] + "\nAnswer in one sentence.", 150)
        latency = time.perf_counter() - start
        verdict, _ = invoke_model(
            JUDGE, f"Question: {q['question']}\nReference answer: {q['answer']}\nCandidate answer: {answer}\n"
            "Rate the candidate's correctness from 1 (wrong) to 5 (fully correct). Reply with only the digit.", 5)
        price_in, price_out = PRICES[model]
        return {"model": model, "domain": q["domain"], "answer": answer, "reference": q["answer"],
                "latency": latency, "judge": int(re.search(r"[1-5]", verdict).group()),
                "cost": (usage["input_tokens"] * price_in + usage["output_tokens"] * price_out) / 1e6}
    except Exception as e:
        print(f"skipped {model}: {e}")


# 1. Benchmark: 5 domains x 10 questions, generated once and cached
if os.path.exists(BENCHMARK):
    bench = json.load(open(BENCHMARK))
else:
    with ThreadPoolExecutor(5) as pool:
        bench = [q for qs in pool.map(generate, DOMAINS) for q in qs]
    json.dump(bench, open(BENCHMARK, "w"), indent=1)
print(f"{len(bench)} benchmark questions")

# 2. Answers + LLM judge + latency + cost
with ThreadPoolExecutor(8) as pool:
    results = [r for r in pool.map(evaluate, [(m, q) for m in MODELS for q in bench]) if r]

# 3. Automatic metrics against the reference answers
rouge = rouge_scorer.RougeScorer(["rougeL"])
bert_f1 = bert_score([r["answer"] for r in results], [r["reference"] for r in results],
                     model_type="distilbert-base-uncased")[2]
for r, f1 in zip(results, bert_f1):
    r["bleu"] = sentence_bleu(r["answer"], [r["reference"]]).score
    r["rouge"] = rouge.score(r["reference"], r["answer"])["rougeL"].fmeasure
    r["bert"] = f1.item()


def avg(key, **where):
    rows = [r for r in results if all(r[k] == v for k, v in where.items())]
    return sum(r[key] for r in rows) / len(rows) if rows else 0


METRICS = [("LLM judge (1-5)", "judge", "{:.2f}"), ("ROUGE-L F1", "rouge", "{:.3f}"), ("BLEU (0-100)", "bleu", "{:.1f}"),
           ("BERTScore F1", "bert", "{:.3f}"), ("Avg latency (s)", "latency", "{:.2f}"), ("Total cost (USD)", "cost", "${:.4f}")]
summary = {key: {m: sum(r[key] for r in results if r["model"] == m) if key == "cost" else avg(key, model=m)
                 for m in MODELS} for _, key, _ in METRICS}


# 4. HTML report: inline SVG bars (no external dependencies), one color per model across all charts
def chart(title, key, fmt):
    values, top = summary[key], max(summary[key].values()) or 1
    bars = "".join(
        f'<text x="0" y="{i * 34 + 16}">{m}</text>'
        f'<rect x="90" y="{i * 34}" width="{300 * v / top:.1f}" height="22" rx="4" fill="var(--s{i + 1})">'
        f'<title>{m}: {fmt.format(v)}</title></rect>'
        f'<text x="{96 + 300 * v / top:.1f}" y="{i * 34 + 16}">{fmt.format(v)}</text>'
        for i, (m, v) in enumerate(values.items()))
    return f'<section><h3>{title}</h3><svg viewBox="0 0 480 {len(values) * 34}">{bars}</svg></section>'


def table(headers, rows):
    head = "".join(f"<th>{h}</th>" for h in headers)
    body = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in row) + "</tr>" for row in rows)
    return f"<table><tr>{head}</tr>{body}</table>"


legend = "".join(f'<span><i style="background:var(--s{i + 1})"></i>{m}</span>' for i, m in enumerate(MODELS))
page = f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Bedrock Model Evaluation</title><style>
:root{{--bg:#f9f9f7;--card:#fcfcfb;--ink:#0b0b0b;--ink2:#52514e;--line:#e1e0d9;--s1:#2a78d6;--s2:#eb6834;--s3:#1baf7a}}
@media (prefers-color-scheme:dark){{:root{{--bg:#0d0d0d;--card:#1a1a19;--ink:#fff;--ink2:#c3c2b7;--line:#2c2c2a;--s1:#3987e5;--s2:#d95926;--s3:#199e70}}}}
body{{margin:0;padding:24px 16px;background:var(--bg);color:var(--ink);font:15px system-ui,-apple-system,"Segoe UI",sans-serif}}
main{{max-width:1000px;margin:auto}} p,small{{color:var(--ink2)}}
.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:16px}}
section{{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:16px;overflow-x:auto}}
h1,h3{{margin:0 0 8px}} svg text{{fill:var(--ink);font-size:13px}} .legend span{{margin-right:16px}}
.legend i{{display:inline-block;width:12px;height:12px;border-radius:3px;margin-right:6px}}
table{{border-collapse:collapse;width:100%}} th,td{{text-align:left;padding:6px 10px;border-bottom:1px solid var(--line)}}
</style></head><body><main>
<h1>Bedrock model evaluation</h1>
<p>{len(bench)} questions across {len(DOMAINS)} domains, {len(results)} graded answers. Judge: Sonnet 4.6.</p>
<p class="legend">{legend}</p>
<div class="grid">{"".join(chart(t, k, f) for t, k, f in METRICS)}</div>
<h3 style="margin-top:24px">Summary table</h3>
<section>{table(["Model"] + [t for t, _, _ in METRICS], [[m] + [f.format(summary[k][m]) for _, k, f in METRICS] for m in MODELS])}</section>
<h3 style="margin-top:24px">LLM judge score by domain (1-5)</h3>
<section>{table(["Domain"] + list(MODELS), [[d] + [f"{avg('judge', model=m, domain=d):.2f}" for m in MODELS] for d in DOMAINS])}</section>
<p><small>Costs use approximate Anthropic list prices per 1M tokens and exclude judge and benchmark-generation calls; Bedrock pricing may differ.
Latency is measured with 8 concurrent requests.</small></p></main></body></html>"""
open("report.html", "w").write(page)

for title, key, fmt in METRICS:
    print(f"{title:<20}" + "".join(f"{m}: {fmt.format(summary[key][m]):<10}" for m in MODELS))
print("\nReport saved to report.html")
