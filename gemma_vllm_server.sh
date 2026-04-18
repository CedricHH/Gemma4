#!/bin/bash
# gemma_vllm_server.sh
# Initialisiert den vLLM Server für das Gemma 4 E2B-it Modell mit Audio-Unterstützung

# Beenden des Skripts bei nicht behandelten Fehlern
set -e

echo " Aktualisiere Abhängigkeiten für Audio-Multimodalität..."
# Installation der vLLM-Audio-Variante sowie der OpenAI Client Library
pip install "vllm[audio]" openai sounddevice scipy numpy

echo " Starte vLLM Inferenz-Engine für Google Gemma 4 E2B-it..."
# Parameter-Erklärung:
# --model: Referenziert das Modell aus der Hugging Face Registry.
# --dtype: BFloat16 maximiert die Leistung auf Ampere/Ada GPU-Architekturen.
# --trust-remote-code: Zwingend erforderlich für die Custom-GELU und Routing Logik.
# --limit-mm-per-prompt: Weist die PagedAttention an, Speicherblöcke für Audio zu reservieren.
# --max-model-len: Begrenzung auf 8192 Token verhindert OOM-Abstürze bei Desktop-GPUs.
# --port: Standard OpenAI Port 8000.

python -m vllm.entrypoints.openai.api_server \
    --model google/gemma-4-E2B-it \
    --dtype bfloat16 \
    --trust-remote-code \
    --limit-mm-per-prompt "audio=1,image=0,video=0" \
    --max-model-len 8192 \
    --port 8000
