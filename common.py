import json

import boto3

bedrock_runtime = boto3.client('bedrock-runtime', region_name='us-east-2')

MODELS = {
    "Haiku 4.5": "us.anthropic.claude-haiku-4-5-20251001-v1:0",
    "Sonnet 4.5": "us.anthropic.claude-sonnet-4-5-20250929-v1:0",
    "Opus 4.6": "us.anthropic.claude-opus-4-6-v1",
}


def invoke_model(model_id, prompt, max_tokens=1000, **params):
    body = json.dumps({
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": max_tokens,
        "messages": [{"role": "user", "content": prompt}] if isinstance(prompt, str) else prompt,
        **params,
    })
    response = bedrock_runtime.invoke_model(modelId=model_id, body=body)
    response_body = json.loads(response['body'].read())
    return response_body['content'][0]['text'], response_body['usage']
