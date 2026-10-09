import json
import os
import random
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from math import comb

from common import MODELS, bedrock_runtime, invoke_model

GENERATOR = "us.anthropic.claude-sonnet-4-6"
BASE_MODEL = MODELS["Haiku 4.5"]
FINETUNED_MODEL = os.environ.get("FINETUNED_MODEL_ARN")  # set after a real fine-tuning job
CATEGORIES = {"billing", "technical", "account", "shipping", "other"}
PRIORITIES = {"low", "medium", "high"}
SYSTEM = ('Classify the support ticket. Reply only with JSON: {"category": billing|technical|account|shipping|other, '
          '"priority": low|medium|high, "summary": "<one sentence>"}')
INDUSTRIES = ["e-commerce", "SaaS software", "telecom", "banking app", "travel booking", "food delivery", "streaming service"]
EPOCHS = 2
PRICE_PER_1M_TRAINED_TOKENS = float(sys.argv[1]) if len(sys.argv) > 1 else None  # USD; copy it from the Bedrock pricing page


# 1. Dataset: 140 labeled support tickets -> 120 train / 20 held-out test
def generate(industry):
    text, _ = invoke_model(
        GENERATOR, f"Write 20 varied, realistic customer support tickets for a {industry} company (1-3 sentences, mixed tones). "
        f"Categories: {sorted(CATEGORIES)}. Priorities: {sorted(PRIORITIES)}. Reply only with a JSON list of objects "
        "with keys ticket, category, priority, summary (one sentence).", 4000)
    return json.loads(re.search(r"\[.*\]", text, re.S).group())


def to_example(item):
    answer = json.dumps({k: item[k] for k in ("category", "priority", "summary")})
    return {"schemaVersion": "bedrock-conversation-2024", "system": [{"text": SYSTEM}], "messages": [
        {"role": "user", "content": [{"text": item["ticket"]}]},
        {"role": "assistant", "content": [{"text": answer}]}]}


def write_jsonl(path, items):
    with open(path, "w") as f:
        f.writelines(json.dumps(to_example(i)) + "\n" for i in items)


if not os.path.exists("train.jsonl"):
    with ThreadPoolExecutor(len(INDUSTRIES)) as pool:
        items = list({i["ticket"]: i for batch in pool.map(generate, INDUSTRIES) for i in batch}.values())
    random.Random(0).shuffle(items)
    write_jsonl("train.jsonl", items[:-20])
    write_jsonl("test.jsonl", items[-20:])


# 2. Format validation (same rules Bedrock applies to the conversation format, plus our label schema)
def validate(path):
    errors, seen = [], set()
    for n, line in enumerate(open(path), 1):
        try:
            example = json.loads(line)
            messages = example["messages"]
            assert example["schemaVersion"] == "bedrock-conversation-2024", "bad schemaVersion"
            assert [m["role"] for m in messages] == ["user", "assistant"], "roles must be user, assistant"
            assert all(m["content"][0]["text"].strip() for m in messages), "empty text"
            label = json.loads(messages[1]["content"][0]["text"])
            assert label["category"] in CATEGORIES and label["priority"] in PRIORITIES, "invalid label"
            ticket = messages[0]["content"][0]["text"]
            assert ticket not in seen, "duplicate ticket"
            seen.add(ticket)
        except Exception as e:
            errors.append(f"line {n}: {type(e).__name__}: {e}")
    return errors


for path in ("train.jsonl", "test.jsonl"):
    errors = validate(path)
    print(f"{path}: {sum(1 for _ in open(path))} examples, {len(errors)} errors")
    print(*errors[:5], sep="\n")

# 3. Estimated training cost (~4 characters per token)
tokens = sum(len(line) // 4 for line in open("train.jsonl")) * EPOCHS
print(f"\nTraining tokens: ~{tokens:,} ({EPOCHS} epochs)")
if PRICE_PER_1M_TRAINED_TOKENS is None:
    print("Estimated cost: pass the price as an argument, e.g. python3 practice9.py <USD per 1M trained tokens>")
else:
    print(f"Estimated cost: ${tokens / 1e6 * PRICE_PER_1M_TRAINED_TOKENS:.2f}")

# 4. A/B experiment on the held-out test set.
#    Hypothesis: B beats A on exact-label accuracy. Same inputs, paired comparison, sign test on disagreements.
#    A = base model, zero-shot. B = fine-tuned model if FINETUNED_MODEL_ARN is set, else base model with 5 few-shot examples.
train = [json.loads(line) for line in open("train.jsonl")]
test = [json.loads(line) for line in open("test.jsonl")]
shots = [m for example in train[:5] for m in example["messages"]]


def ask(model_id, example, shots=()):
    ticket = example["messages"][0]
    reply = bedrock_runtime.converse(
        modelId=model_id, system=[{"text": SYSTEM}], messages=[*shots, ticket],
        inferenceConfig={"maxTokens": 200, "temperature": 0})
    try:
        return json.loads(re.search(r"\{.*\}", reply["output"]["message"]["content"][0]["text"], re.S).group())
    except Exception:
        return None


def correct(prediction, example):
    label = json.loads(example["messages"][1]["content"][0]["text"])
    return bool(prediction) and all(prediction.get(k) == label[k] for k in ("category", "priority"))


arm_b = (FINETUNED_MODEL, ()) if FINETUNED_MODEL else (BASE_MODEL, shots)
with ThreadPoolExecutor(8) as pool:
    a = list(pool.map(lambda e: correct(ask(BASE_MODEL, e), e), test))
    b = list(pool.map(lambda e: correct(ask(arm_b[0], e, arm_b[1]), e), test))

b_wins, a_wins = sum(y and not x for x, y in zip(a, b)), sum(x and not y for x, y in zip(a, b))
n = a_wins + b_wins
p_value = min(1.0, 2 * sum(comb(n, i) for i in range(min(a_wins, b_wins) + 1)) / 2 ** n) if n else 1.0
print(f"\nA (base, zero-shot):           {sum(a)}/{len(test)} correct")
print(f"B ({'fine-tuned' if FINETUNED_MODEL else 'base + 5 few-shot'}): {sum(b)}/{len(test)} correct")
print(f"B-only wins: {b_wins}, A-only wins: {a_wins}, sign-test p-value: {p_value:.3f}")
print(f"With {len(test)} test examples only large differences are detectable; use 100+ for a reliable verdict.")
