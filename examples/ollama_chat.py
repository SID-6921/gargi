"""Chat with Gargi through a local Ollama server (thinking on by default).

  ollama pull hf.co/udgamlabs/Gargi-GGUF:Q8_0
  python ollama_chat.py "Why is the sky blue?" --no-think
"""
import argparse, json, urllib.request

ap = argparse.ArgumentParser()
ap.add_argument("question"); ap.add_argument("--no-think", action="store_true")
ap.add_argument("--model", default="hf.co/udgamlabs/Gargi-GGUF:Q8_0"); ap.add_argument("--host", default="http://localhost:11434")
a = ap.parse_args()
body = {"model": a.model, "stream": False, "think": not a.no_think, "options": {"temperature": 0, "num_ctx": 16384},
        "messages": [{"role": "user", "content": a.question}]}
r = json.loads(urllib.request.urlopen(urllib.request.Request(f"{a.host}/api/chat", data=json.dumps(body).encode()), timeout=300).read())
if r["message"].get("thinking"):
    print("Thinking:", r["message"]["thinking"].strip(), "\n")
print(r["message"]["content"].strip())
