import json
import re
import sqlite3
import sys
import urllib.parse
import urllib.request

from common import bedrock_runtime

MODEL = "us.anthropic.claude-sonnet-4-6"

db = sqlite3.connect(":memory:")
db.executescript("""
    CREATE TABLE employees (name TEXT, department TEXT, salary INTEGER);
    INSERT INTO employees VALUES ('Ana', 'Engineering', 120000), ('Luis', 'Engineering', 100000),
                                 ('Marta', 'Sales', 80000), ('Pedro', 'Sales', 90000);
""")


def calculator(expression):
    if not re.fullmatch(r"[\d+\-*/().% ]+", expression):
        return "Error: only numbers and + - * / ( ) % are allowed"
    return eval(expression, {"__builtins__": {}})


def web_search(query):
    url = "https://en.wikipedia.org/w/api.php?" + urllib.parse.urlencode(
        {"action": "query", "list": "search", "srsearch": query, "srlimit": 3, "format": "json"})
    request = urllib.request.Request(url, headers={"User-Agent": "bedrock-practice/1.0"})
    results = json.load(urllib.request.urlopen(request))["query"]["search"]
    return "\n".join(f"{r['title']}: {re.sub('<[^>]+>', '', r['snippet'])}" for r in results)


def query_db(sql):
    if not sql.strip().lower().startswith("select"):
        return "Error: only SELECT queries are allowed"
    return db.execute(sql).fetchall()


def spec(fn, description, param):
    return {"toolSpec": {"name": fn.__name__, "description": description, "inputSchema": {"json": {
        "type": "object", "properties": {param: {"type": "string"}}, "required": [param]}}}}


TOOLS = {f.__name__: f for f in (calculator, web_search, query_db)}
TOOL_CONFIG = {"tools": [
    spec(calculator, "Evaluate an arithmetic expression.", "expression"),
    spec(web_search, "Search Wikipedia for general knowledge.", "query"),
    spec(query_db, "Run a SELECT on the employees table (columns: name, department, salary).", "sql"),
]}

question = " ".join(sys.argv[1:]) or (
    "What is the average salary of the Engineering department in the database, "
    "what is that divided by 12, and what does Wikipedia say about AWS Bedrock?")
messages = [{"role": "user", "content": [{"text": question}]}]

for step in range(1, 11):
    response = bedrock_runtime.converse(modelId=MODEL, messages=messages, toolConfig=TOOL_CONFIG)
    message = response["output"]["message"]
    messages.append(message)
    if response["stopReason"] != "tool_use":
        print(f"\nFINAL ANSWER:\n{message['content'][0]['text']}")
        break
    results = []
    for block in message["content"]:
        if "text" in block:
            print(f"[step {step}] model says: {block['text']}")
        if "toolUse" in block:
            call = block["toolUse"]
            print(f"[step {step}] calls {call['name']}({call['input']})")
            try:
                output = str(TOOLS[call["name"]](**call["input"]))
            except Exception as e:
                output = f"Error: {e}"
            print(f"           -> {output[:200]}")
            results.append({"toolResult": {"toolUseId": call["toolUseId"], "content": [{"text": output}]}})
    messages.append({"role": "user", "content": results})
