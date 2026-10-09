import json

from common import MODELS, bedrock_runtime

MAX_TOKENS = 500


def stream_to_file(prompt, path="stream_output.txt", model=MODELS["Haiku 4.5"]):
    body = json.dumps({
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": MAX_TOKENS,
        "messages": [{"role": "user", "content": prompt}],
    })
    tokens = 0
    try:
        response = bedrock_runtime.invoke_model_with_response_stream(modelId=model, body=body)
        with open(path, "w") as f:
            for event in response["body"]:
                chunk = json.loads(event["chunk"]["bytes"])
                if chunk["type"] == "content_block_delta":
                    text = chunk["delta"]["text"]
                    f.write(text)
                    f.flush()
                    print(text, end="", flush=True)
                    tokens += 1
                elif chunk["type"] == "message_delta":
                    tokens = chunk["usage"]["output_tokens"]
    except Exception as e:
        print(f"\nStream failed after {tokens} tokens (partial output kept in {path}): {e}")
        return
    filled = 30 * min(tokens, MAX_TOKENS) // MAX_TOKENS
    print(f"\n\n[{'#' * filled}{'.' * (30 - filled)}] {tokens}/{MAX_TOKENS} tokens, saved to {path}")


stream_to_file("Explain what AWS Bedrock is in 3 paragraphs")
