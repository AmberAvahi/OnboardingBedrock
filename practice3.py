from common import MODELS, invoke_model

PROMPT = "Write a one-sentence slogan for a coffee shop."
MODEL = MODELS["Haiku 4.5"]

CONFIGS = (
    [{"temperature": t} for t in [0, 0.3, 0.7, 1.0, 1.5]]
    + [{"top_p": p} for p in [0.1, 0.9]]
    + [{"top_k": k} for k in [1, 250]]
)

for params in CONFIGS:
    print(f"\n=== {params} ===")
    for i in range(5):
        try:
            print(f"{i + 1}. {invoke_model(MODEL, PROMPT, 100, **params)[0].strip()}")
        except Exception as e:
            print(f"ERROR: {e}")
            break
