"""Chat with Gargi using Hugging Face transformers (CPU, CUDA or Apple MPS).

  pip install "transformers>=5.18" torch
  python transformers_chat.py "Sara has 3 boxes of 6 pencils and gives away 4. How many are left?" --think
"""
import argparse, re
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ap = argparse.ArgumentParser()
ap.add_argument("question"); ap.add_argument("--think", action="store_true", help="let Gargi reason before answering")
ap.add_argument("--model", default="udgamlabs/Gargi"); ap.add_argument("--max-new-tokens", type=int, default=512)
a = ap.parse_args()
device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
tok = AutoTokenizer.from_pretrained(a.model)
model = AutoModelForCausalLM.from_pretrained(a.model, dtype="auto").to(device)
q = a.question
if a.think and re.search(r"\d", q) and "Final answer" not in q:   # the instruction Gargi's maths training used
    q += "\n\nSolve it step by step, then give the final answer on the last line as: Final answer: <number>"
inputs = tok.apply_chat_template([{"role": "user", "content": q}], add_generation_prompt=True, enable_thinking=a.think,
                                 return_tensors="pt", return_dict=True).to(device)
out = model.generate(**inputs, max_new_tokens=a.max_new_tokens, do_sample=False)
text = tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=False)
m = re.search(r"<\|channel>thought\n?(.*?)<channel\|>", text, re.S)
if m:
    print("Thinking:", m.group(1).strip(), "\n"); text = text[m.end():]
print(re.sub(r"<[^>]{1,40}\|>|<\|[^>]{1,40}>|<eos>", "", text).strip())
