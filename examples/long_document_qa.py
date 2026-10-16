"""Ask Gargi a question about a long text file (up to ~16K tokens is reliable), via Ollama.

  python long_document_qa.py notes.txt "What date was the meeting moved to?"
"""
import argparse, json, urllib.request

ap = argparse.ArgumentParser()
ap.add_argument("file"); ap.add_argument("question")
ap.add_argument("--model", default="hf.co/udgamlabs/Gargi-GGUF:Q8_0"); ap.add_argument("--host", default="http://localhost:11434")
a = ap.parse_args()
doc = open(a.file, encoding="utf-8").read()
if len(doc) > 60000:
    print(f"Note: the file is long ({len(doc):,} characters); Gargi is reliable up to roughly 16K tokens (~60K characters).")
prompt = f"Read the document below, then answer the question.\n\n<document>\n{doc}\n</document>\n\nQuestion: {a.question}"
body = {"model": a.model, "stream": False, "think": False, "options": {"temperature": 0, "num_ctx": 16384},
        "messages": [{"role": "user", "content": prompt}]}
r = json.loads(urllib.request.urlopen(urllib.request.Request(f"{a.host}/api/chat", data=json.dumps(body).encode()), timeout=600).read())
print(r["message"]["content"].strip())
