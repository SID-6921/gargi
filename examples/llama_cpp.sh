#!/usr/bin/env bash
# Gargi with llama.cpp (Q8_0 recommended; keep the context at or below ~16K tokens).
llama-cli -hf udgamlabs/Gargi-GGUF:Q8_0 -c 16384
# or an OpenAI-compatible server:
# llama-server -hf udgamlabs/Gargi-GGUF:Q8_0 -c 16384 --jinja
