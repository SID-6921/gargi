#!/usr/bin/env bash
# Gargi on Apple silicon with MLX.
pip install -U mlx-lm
mlx_lm.chat --model udgamlabs/Gargi-MLX            # interactive chat (use udgamlabs/Gargi-MLX-4bit for the 4-bit build)
