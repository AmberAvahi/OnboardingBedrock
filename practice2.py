import sys
import time
from common import MODELS, invoke_model

PROMPTS = [
    "Explain what AWS Bedrock is in 3 sentences.",
    "Write a Python function that checks if a string is a palindrome.",
    "A train leaves at 3pm going 60 mph. Another leaves at 4pm going 80 mph on the same track. When does the second catch up? Show your reasoning.",
    "Write a short, funny product description for a self-watering coffee mug.",
]

summary = []
for prompt in sys.argv[1:] or PROMPTS:
    print(f"\n{'#' * 70}\nPROMPT: {prompt}\n{'#' * 70}")
    results = []
    for name, model_id in MODELS.items():
        start = time.time()
        try:
            text, usage = invoke_model(model_id, prompt)
        except Exception as e:
            print(f"\n=== {name} === ERROR: {e}")
            continue
        elapsed = time.time() - start
        results.append((name, elapsed, usage['output_tokens']))
        print(f"\n=== {name} ({elapsed:.2f}s, {usage['output_tokens']} output tokens) ===")
        print(text)
    summary.append((prompt, results))

print(f"\n\n{'=' * 70}\nTIMING SUMMARY\n{'=' * 70}")
for prompt, results in summary:
    print(f"\n{prompt[:60]}...")
    for name, elapsed, tokens in results:
        print(f"  {name:<12} {elapsed:6.2f}s  {tokens:5d} tokens")
