#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RAG with Wikipedia – Final (multilingual, dynamic model)
Author: Lorena Ariceta García

✔ Centralized prompt system (ES/EN)
✔ Dynamic model selection via RAG_MODEL (Ollama or HF)
✔ Context retrieved via Wikipedia REST API
✔ Accuracy metrics + leaderboard
"""

# =============================================================================
# Imports
# =============================================================================
import os, sys, re, json, time, requests, pandas as pd, argparse
from datetime import datetime
from transformers import pipeline

# =============================================================================
# Paths and config
# =============================================================================
BASE_DIR = "/home/xs1/Desktop/Lorena/MEDICINA"
if f"{BASE_DIR}/code/3_RAG" not in sys.path:
    sys.path.append(f"{BASE_DIR}/code/3_RAG")
sys.path.append(f"{BASE_DIR}/code/3_RAG")
from prompt_config import PROMPTS
from utils.load_model import load_hf_model

# =============================================================================
# Arguments / environment
# =============================================================================
parser = argparse.ArgumentParser(description="RAG Wikipedia Final – multilingual")
parser.add_argument("--lang", choices=["es", "en"], default="es")
args = parser.parse_args()

LANG = os.environ.get("RAG_LANG", args.lang)
MODEL_NAME = os.environ.get("RAG_MODEL")
assert MODEL_NAME, "❌ Environment variable RAG_MODEL must be defined."

PROMPT_RAG = PROMPTS[LANG]

print(f"\n🧠 Model selected: {MODEL_NAME}")
print(f"🌐 Language: {LANG}\n")

EXAMS_DIR = f"{BASE_DIR}/results/1_data_preparation/6_json_final/prueba"
OUTPUT_DIR = f"{BASE_DIR}/results/2_models/2_rag/final/wikipedia_final_{LANG}"
os.makedirs(OUTPUT_DIR, exist_ok=True)
LOG_FILE = os.path.join(OUTPUT_DIR, f"rag_wikipedia_final_{MODEL_NAME}_{LANG}_log.txt")

# =============================================================================
# Logger
# =============================================================================
class DualLogger:
    def __init__(self, path):
        self.terminal = sys.stdout
        self.log = open(path, "w", encoding="utf-8")
    def write(self, msg):
        self.terminal.write(msg); self.log.write(msg)
    def flush(self):
        self.terminal.flush(); self.log.flush()

sys.stdout = DualLogger(LOG_FILE)

# =============================================================================
# Functions
# =============================================================================
def generate_answer(prompt: str, model_name: str) -> str:
    """Generate an answer using Ollama (local) or Hugging Face model."""
    try:
        # Local Ollama
        if model_name.startswith("med") or model_name.lower() in ["llama3", "mistral", "gemma"]:
            payload = {"model": model_name, "prompt": prompt, "stream": False}
            resp = requests.post("http://localhost:11434/api/generate", json=payload, timeout=180)
            return resp.json().get("response", "").strip()
        # Hugging Face fallback
        pipe = load_hf_model(model_name)
        out = pipe(prompt, max_new_tokens=256, do_sample=False)[0]["generated_text"]
        return out.strip()
    except Exception as e:
        print(f"❌ Error generating response with {model_name}: {e}")
        return "SIN_RESPUESTA"

def get_wikipedia_context(query, lang="es", max_sent=3):
    """Retrieve short summary from Wikipedia (REST API)."""
    base = "es" if lang == "es" else "en"
    url = f"https://{base}.wikipedia.org/api/rest_v1/page/summary/{query.replace(' ', '_')}"
    try:
        r = requests.get(url, timeout=10)
        if r.status_code == 200:
            text = r.json().get("extract", "")
            return " ".join(text.split(". ")[:max_sent])
        return f"No relevant Wikipedia ({lang}) context found."
    except Exception:
        return f"Error retrieving Wikipedia context ({lang})."

# =============================================================================
# Execution
# =============================================================================
print(f"⏳ Starting RAG-Wikipedia Final [{LANG}] – {datetime.now():%Y-%m-%d %H:%M:%S}")

metrics_all = []
results_by_title = {}

exam_files = [f for f in os.listdir(EXAMS_DIR) if f.endswith(".json")]

for exam_file in exam_files:
    titulacion = exam_file.split("_")[0]
    path = os.path.join(EXAMS_DIR, exam_file)
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    preguntas = [
        q for q in data.get("preguntas", [])
        if not q.get("tipo") or q.get("tipo").lower() in ["texto", "text", "teorica"]
    ]
    print(f"\n📘 Processing {titulacion} → {len(preguntas)} questions")

    results_by_title[titulacion] = []
    for i, q in enumerate(preguntas, 1):
        enunciado = q["enunciado"]
        correcta = q["respuesta_correcta"]
        opciones = "\n".join(f"{idx+1}. {opt}" for idx, opt in enumerate(q["opciones"]))

        context = get_wikipedia_context(enunciado, LANG)
        prompt = (
            PROMPT_RAG +
            f"\n📚 CONTEXTO (Wikipedia):\n{context}\n\n"
            f"❓ PREGUNTA:\n{enunciado}\n\n"
            f"🔢 OPCIONES:\n{opciones}\n"
        )

        text = generate_answer(prompt, MODEL_NAME)
        match = re.search(r"\b([1-4])\b", text)
        pred = int(match.group(1)) if match else None

        results_by_title[titulacion].append({
            "numero": q["numero"],
            "enunciado": enunciado,
            "opciones": q["opciones"],
            "contexto": context,
            MODEL_NAME: pred,
            f"{MODEL_NAME}_text": text,
            "respuesta_correcta": correcta
        })

        if i % 50 == 0:
            print(f" 💾 Progress: {i}/{len(preguntas)}")

    # Metrics
    total = len(results_by_title[titulacion])
    aciertos = sum(1 for r in results_by_title[titulacion] if r.get(MODEL_NAME) == r["respuesta_correcta"])
    sin_resp = sum(1 for r in results_by_title[titulacion] if r.get(MODEL_NAME) is None)
    errores = total - aciertos - sin_resp
    respondidas = total - sin_resp
    acc = (aciertos / respondidas * 100) if respondidas else 0

    metrics_all.append({
        "Modelo": MODEL_NAME,
        "Titulación": titulacion,
        "Idioma": LANG,
        "Total preguntas": total,
        "Respondidas": respondidas,
        "Aciertos": aciertos,
        "Errores": errores,
        "Sin respuesta": sin_resp,
        "Accuracy (%)": round(acc, 2)
    })
    print(f"✅ {MODEL_NAME.upper()} ({titulacion}) → {acc:.2f}%")

# =============================================================================
# Save results
# =============================================================================
for titulacion, preguntas in results_by_title.items():
    json_out = os.path.join(OUTPUT_DIR, f"{titulacion}_{MODEL_NAME}_rag_wikipedia_final_{LANG}.json")
    with open(json_out, "w", encoding="utf-8") as f_out:
        json.dump({"preguntas": preguntas}, f_out, ensure_ascii=False, indent=2)
    print(f"💾 Saved: {json_out}")

df = pd.DataFrame(metrics_all)
csv_path = os.path.join(OUTPUT_DIR, f"rag_wikipedia_final_{LANG}_metrics.csv")
xlsx_path = os.path.join(OUTPUT_DIR, f"rag_wikipedia_final_{LANG}_metrics.xlsx")
df.to_csv(csv_path, index=False, encoding="utf-8-sig")
df.to_excel(xlsx_path, index=False)

print(f"\n📊 Metrics exported to: {xlsx_path}")

leaderboard = df.groupby("Modelo")[["Accuracy (%)"]].mean().round(2)
leaderboard_path = os.path.join(OUTPUT_DIR, f"rag_wikipedia_final_leaderboard_{MODEL_NAME}_{LANG}.xlsx")
leaderboard.to_excel(leaderboard_path)
print(f"\n🏅 Leaderboard saved to: {leaderboard_path}")
print(f"\n🧾 Log saved: {LOG_FILE}")
print("\n🏁 RAG-Wikipedia Final completed successfully.")
