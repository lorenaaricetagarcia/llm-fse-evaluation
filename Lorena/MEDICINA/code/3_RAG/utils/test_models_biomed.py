#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test de modelos biomédicos y Ollama – Lorena Ariceta García
-------------------------------------------------------------
✅ Comprueba si los modelos (Hugging Face o Ollama) están instalados y generan texto
✅ Soporta: meditron, bio_mistral, medalpaca, biogpt, pubmedgpt, llama3, mistral, gemma
✅ Informa del tipo de modelo (HF/Ollama), dispositivo usado (GPU/CPU) y primera respuesta
"""

import torch
import requests
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# Modelos que quieres probar
MODELS = {
    "meditron": "epfl-llm/meditron-7b",
    "bio_mistral": "BioMistral/BioMistral-7B",
    "medalpaca": "medalpaca/medalpaca-13b",
    "biogpt": "microsoft/BioGPT-Large",
    "pubmedgpt": "stanford-crfm/pubmedgpt",
    # modelos Ollama locales
    "llama3": None,
    "mistral": None,
    "gemma": None,
}

# Prompt de prueba (en inglés y español)
PROMPT_EN = "Explain briefly what the heart does in the human body."
PROMPT_ES = "Explica brevemente qué función cumple el corazón en el cuerpo humano."


# === Función de prueba para Hugging Face ===
def test_hf_model(repo_name, prompt):
    try:
        print(f"🔹 Cargando modelo HF: {repo_name}")
        tok = AutoTokenizer.from_pretrained(repo_name)
        mod = AutoModelForCausalLM.from_pretrained(repo_name, device_map="auto")
        pipe = pipeline("text-generation", model=mod, tokenizer=tok)
        out = pipe(prompt, max_new_tokens=50, do_sample=False)[0]["generated_text"]
        device = "GPU" if torch.cuda.is_available() else "CPU"
        print(f"✅ {repo_name} cargado correctamente en {device}")
        print(f"🧩 Respuesta parcial: {out[:200].strip()}\n")
        return True
    except Exception as e:
        print(f"❌ Error cargando {repo_name}: {e}\n")
        return False


# === Función de prueba para Ollama local ===
def test_ollama(model_name, prompt):
    try:
        print(f"🔹 Probando modelo Ollama local: {model_name}")
        payload = {"model": model_name, "prompt": prompt, "stream": False}
        r = requests.post("http://localhost:11434/api/generate", json=payload, timeout=120)
        if r.status_code == 200:
            text = r.json().get("response", "").strip()
            print(f"✅ {model_name} (Ollama) respondió correctamente.")
            print(f"🧩 Respuesta parcial: {text[:200]}\n")
            return True
        else:
            print(f"⚠️ Ollama devolvió código {r.status_code}")
            return False
    except Exception as e:
        print(f"❌ Error probando {model_name} en Ollama: {e}\n")
        return False


# === Loop principal ===
if __name__ == "__main__":
    print("\n🧬 TEST DE MODELOS BIOMÉDICOS / OLLAMA\n" + "=" * 60)

    for name, repo in MODELS.items():
        print(f"\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        print(f"🧠 Modelo: {name}")
        print(f"📦 Fuente: {'HuggingFace' if repo else 'Ollama local'}")

        if repo:
            test_hf_model(repo, PROMPT_EN)
        else:
            test_ollama(name, PROMPT_ES)

    print("\n🏁 Test completado.\n")
