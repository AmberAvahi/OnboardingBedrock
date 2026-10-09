import json
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor

from common import MODELS, invoke_model

HAIKU, SONNET = MODELS["Haiku 4.5"], MODELS["Sonnet 4.5"]
PRICES = {HAIKU: (1, 5), SONNET: (3, 15)}  # approx. USD per 1M tokens (in, out)
AGENTS = {
    "analysis": "You are a rigorous analyst. Reason step by step, weigh trade-offs and give concrete conclusions.",
    "writing": "You are an expert writer. Produce clear, polished prose for the requested audience.",
    "code": "You are a senior engineer. Return correct, minimal code followed by a one-line explanation.",
}
PLANNER = ("Split the user query into the fewest subtasks needed. Reply only with a JSON list of objects: "
           '{"id": <int>, "agent": "analysis|writing|code", "title": "<short section heading>", "task": "<self-contained instruction>", '
           '"complexity": "simple|hard", "depends_on": [<ids of subtasks whose output this one needs>]}')
QUERY = ("Compare RAG and fine-tuning for a legal document assistant, write a one-paragraph recommendation for "
         "executives, and write a Python function that splits a contract into its numbered sections.")
stats = []


def call(model, prompt, agent, system=None, max_tokens=800):
    start = time.time()
    text, usage = invoke_model(model, prompt, max_tokens, **({"system": system} if system else {}))
    price_in, price_out = PRICES[model]
    stats.append({"agent": agent, "model": model, "seconds": time.time() - start,
                  "cost": (usage["input_tokens"] * price_in + usage["output_tokens"] * price_out) / 1e6})
    return text


def run(task):
    context = "".join(f"Output of subtask {d}:\n{results[d]}\n\n" for d in task["depends_on"] if d in results)
    model = SONNET if task["complexity"] == "hard" else HAIKU  # routing: only hard subtasks pay for the big model
    return call(model, f"{context}Task: {task['task']}", task["agent"], AGENTS[task["agent"]])


query = " ".join(sys.argv[1:]) or QUERY
start = time.time()
plan = json.loads(re.search(r"\[.*\]", call(HAIKU, query, "coordinator", PLANNER, 600), re.S).group())
print("PLAN")
for t in plan:
    print(f"  {t['id']}. [{t['agent']}, {t['complexity']}, after {t['depends_on']}] {t['task'][:90]}")

results, pending = {}, {t["id"]: t for t in plan}
with ThreadPoolExecutor(4) as pool:
    while pending:  # run every subtask whose dependencies are done, in parallel
        ready = [t for t in pending.values() if all(d in results for d in t["depends_on"])] or list(pending.values())
        for task, output in zip(ready, pool.map(run, ready)):
            results[task["id"]] = output
            del pending[task["id"]]

sections = [f"## {t['title']}\n{results[t['id']]}" for t in plan]
overview = call(HAIKU, f"Original query: {query}\n\nSpecialist outputs:\n\n" + "\n\n".join(sections) + "\n\nWrite a 2-3 sentence overview "
                "that ties these outputs together as the answer to the query. Do not repeat their content or mention agents.",
                "aggregator", max_tokens=300)
final = "\n\n".join([overview, *sections])  # specialist outputs are appended verbatim, not regenerated
multi_seconds = time.time() - start
print(f"\nFINAL ANSWER\n{final}")

start = time.time()
baseline = call(SONNET, query, "baseline", max_tokens=1500)
baseline_seconds = time.time() - start

print("\nCOST AND LATENCY BY AGENT")
for s in stats:
    print(f"  {s['agent']:<12}{s['model'].split('.')[-1][:24]:<26}{s['seconds']:5.1f}s  ${s['cost']:.4f}")
multi_cost = sum(s["cost"] for s in stats if s["agent"] != "baseline")
print(f"\nMulti-agent: {multi_seconds:.1f}s, ${multi_cost:.4f}, {len(final)} chars")
print(f"Single Sonnet call: {baseline_seconds:.1f}s, ${stats[-1]['cost']:.4f}, {len(baseline)} chars")
