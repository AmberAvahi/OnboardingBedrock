# OnboardingBedrock

AWS Bedrock practice workbook: basic and advanced practices plus three final projects.

- `practice1.py` - `practice10.5.py`: practices 1-10 (invocation, model comparison, parameters, streaming, conversations, RAG, tool use, evaluation, fine-tuning data, guardrails). `common.py` holds the shared Bedrock client.
- `Final Projects/`: legal assistant with RAG (`project1.py`), multi-agent system (`project2.py`) and evaluation platform (`project3.py`).

Authentication uses a short-term Bedrock API key exported as `AWS_BEARER_TOKEN_BEDROCK` (never committed). Run scripts from their own folder.
