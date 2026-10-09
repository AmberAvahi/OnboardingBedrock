import json

from common import MODELS, invoke_model

MODEL = MODELS["Haiku 4.5"]
TOKEN_LIMIT = 200
FILE = "conversation.json"


def summarize(history):
    old, recent = history[:-2], history[-2:]
    summary, _ = invoke_model(MODEL, old + [{"role": "user", "content": "Summarize our conversation so far in under 150 words."}])
    print("[history summarized]")
    return [
        {"role": "user", "content": f"Summary of our earlier conversation: {summary}"},
        {"role": "assistant", "content": "Understood."},
    ] + recent


history = []
print("Commands: /save, /load, /quit")
while (text := input("you> ").strip()) != "/quit":
    if text == "/save":
        json.dump(history, open(FILE, "w"))
    elif text == "/load":
        history = json.load(open(FILE))
    else:
        history.append({"role": "user", "content": text})
        reply, usage = invoke_model(MODEL, history, 500)
        history.append({"role": "assistant", "content": reply})
        print("bot>", reply)
        if usage["input_tokens"] > TOKEN_LIMIT and len(history) >= 4:
            history = summarize(history)
