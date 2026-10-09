import time
from common import MODELS, invoke_model


def timed(prompt, max_tokens=1000):
    start = time.time()
    result, _ = invoke_model(MODELS["Sonnet 4.5"], prompt, max_tokens)
    elapsed = time.time() - start
    print(f"\n=== {prompt!r} (max_tokens={max_tokens}) ===")
    print(result)
    print(f"--- {elapsed:.2f}s, {len(result)} characters ---")


# Tasks 1 and 4: run the example and measure response time
timed("Explain what AWS Bedrock is in 3 paragraphs")

# Task 2: three different questions
timed("What is the difference between invoke_model and converse in Bedrock?")
timed("Give me 3 use cases for generative AI in customer support.")
timed("Explain what an inference profile is in Bedrock in one paragraph.")

# Task 3: max_tokens = 500
timed("Explain what AWS Bedrock is in 3 paragraphs", max_tokens=500)
