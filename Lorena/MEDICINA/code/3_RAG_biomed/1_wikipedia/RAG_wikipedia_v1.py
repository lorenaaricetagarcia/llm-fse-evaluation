#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RAG with Wikipedia – v1 (multilingual, dynamic model)
Author: Lorena Ariceta García
TFM – Data Science & Bioinformatics for Precision Medicine
"""

import os, sys, re, json, time, requests, pandas as pd, argparse
import wikipediaapi
from datetime import datetime
from keybert import KeyBERT
from sentence_transformers import SentenceTransformer
from transformers import pipeline
from utils.load_model import load_hf_model

# Prompts
sys.path.append("/home/xs1/Desktop/Lorena/MEDICINA/code/3_RAG")
from prompt_config import PROMPTS

# ===============================================================
# CONFIG
# ===============================================================
parser = argparse.ArgumentParser(description="RAG Wikipedia v1 – multilingual")
parser.add_argument("--lang", choices=["es", "en"], default="es")
args = parser.parse_args()

LANG = os.getenv("RAG_LANG", args.lang)
MODEL_NAME = os.getenv("RAG_MODEL")
assert MODEL_NAME, "❌ Environment variable RAG_MODEL not defined."

BASE_DIR = "/home/xs1/Desktop/Lorena/MEDICINA"
if f"{BASE_DIR}/code/3_RAG" not in sys.path:
    sys.path.append(f"{BASE_DIR}/code/3_RAG")
PROMPT_RAG = PROMPTS[LANG]

EXAMS_DIR = f"{BASE_DIR}/results/1_data_preparation/6_json_final/prueba"
OUTPUT_DIR = f"{BASE_DIR}/results/2_models/2_rag/1_wikipedia/v1_{LANG}"
os.makedirs(OUTPUT_DIR, exist_ok=True)
LOG_FILE = os.path.join(OUTPUT_DIR, f"rag_wikipedia_v1_{MODEL_NAME}_{LANG}_log.txt")

# ===============================================================
# LOGGER
# ===============================================================
class DualLogger:
    def __init__(self, path):
        self.terminal = sys.stdout
        self.log = open(path, "w", encoding="utf-8")
    def write(self, msg): self.terminal.write(msg); self.log.write(msg)
    def flush(self): self.terminal.flush(); self.log.flush()

sys.stdout = DualLogger(LOG_FILE)

print(f"🧠 MODEL: {MODEL_NAME} | 🌐 LANG: {LANG} | ⏳ {datetime.now():%Y-%m-%d %H:%M:%S}")

# ===============================================================
# INIT
# ===============================================================
sentence_model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
kw_model = KeyBERT(model=sentence_model)
wiki = wikipediaapi.Wikipedia(LANG)

translator = None
if LANG == "en":
    translator = pipeline("translation", model="Helsinki-NLP/opus-mt-es-en")

# ===============================================================
# FUNCTIONS
# ===============================================================
def generate_answer(prompt):
    try:
        if MODEL_NAME.startswith("med") or MODEL_NAME.lower() in ["llama3", "mistral", "gemma"]:
            payload = {"model": MODEL_NAME, "prompt": prompt, "stream": False}
            r = requests.post("http://localhost:11434/api/generate", json=payload, timeout=180)
            return r.json().get("response", "").strip()
        else:
            pipe = load_hf_model(MODEL_NAME)
            return pipe(prompt, max_new_tokens=256, do_sample=False)[0]["generated_text"].strip()
    except Exception as e:
        return f"❌ Error: {e}"

def extract_keyword(text):
    try:
        kw = kw_model.extract_keywords(text, top_n=1)
        return kw[0][0] if kw else ""
    except: return ""

def translate_kw(kw):
    if translator and LANG == "en":
        try:
            return translator(kw)[0]["translation_text"]
        except: return kw
    return kw

# ===============================================================
# MAIN
# ===============================================================
metrics = []
for file in os.listdir(EXAMS_DIR):
    if not file.endswith(".json"): continue
    titulacion = file.split("_")[0]
    with open(os.path.join(EXAMS_DIR, file), "r", encoding="utf-8") as f: data = json.load(f)
    preguntas = [
        q for q in data.get("preguntas", [])
        if not q.get("tipo") or q.get("tipo").lower() in ["texto", "text", "teorica"]
    ]

    print(f"\n📘 {titulacion}: {len(preguntas)} preguntas")

    results = []
    for q in preguntas:
        kw = extract_keyword(q["enunciado"])
        kw_tr = translate_kw(kw)
        page = wiki.page(kw_tr)
        context = page.summary[:1500] if page.exists() else "Sin contexto relevante."
        opts = "\n".join(f"{i+1}. {o}" for i, o in enumerate(q["opciones"]))
        prompt = f"{PROMPT_RAG}\n📚 CONTEXTO:\n{context}\n❓ {q['enunciado']}\n{opts}"
        resp = generate_answer(prompt)
        match = re.search(r"\b([1-4])\b", resp)
        pred = int(match.group(1)) if match else None
        results.append({
            "numero": q["numero"], "respuesta_correcta": q["respuesta_correcta"],
            MODEL_NAME: pred, f"{MODEL_NAME}_texto": resp
        })

    total = len(results)
    aciertos = sum(1 for r in results if r.get(MODEL_NAME)==r["respuesta_correcta"])
    sin_resp = sum(1 for r in results if r.get(MODEL_NAME) is None)
    acc = round(aciertos/(total-sin_resp)*100,2) if total-sin_resp>0 else 0
    metrics.append({"Modelo": MODEL_NAME,"Titulación": titulacion,"Accuracy (%)": acc})

    json_out = os.path.join(OUTPUT_DIR, f"{titulacion}_{MODEL_NAME}_rag_wikipedia_v1_{LANG}.json")
    with open(json_out,"w",encoding="utf-8") as f: json.dump({"preguntas":results},f,ensure_ascii=False,indent=2)
    print(f"💾 {json_out}")

df=pd.DataFrame(metrics)
df.to_excel(os.path.join(OUTPUT_DIR,f"rag_wikipedia_v1_{MODEL_NAME}_{LANG}_metrics.xlsx"),index=False)
print("\n🏁 DONE.")
