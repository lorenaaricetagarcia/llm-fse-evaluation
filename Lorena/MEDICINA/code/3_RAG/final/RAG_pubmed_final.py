#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RAG with PubMed – FINAL (multilingual, dynamic model)
Author: Lorena Ariceta García
"""

import os, sys, json, re, requests, pandas as pd, xml.etree.ElementTree as ET
from datetime import datetime
from transformers import pipeline
from utils.load_model import load_hf_model

BASE_DIR = "/home/xs1/Desktop/Lorena/MEDICINA"
if f"{BASE_DIR}/code/3_RAG" not in sys.path:
    sys.path.append(f"{BASE_DIR}/code/3_RAG")
sys.path.append(f"{BASE_DIR}/code/3_RAG")
from prompt_config import PROMPTS

LANG = os.getenv("RAG_LANG","es")
MODEL_NAME = os.getenv("RAG_MODEL")
assert MODEL_NAME, "❌ RAG_MODEL not defined"

PROMPT_RAG = PROMPTS[LANG]
EXAMS_DIR = f"{BASE_DIR}/results/1_data_preparation/6_json_final/prueba"
print("🔍 DEBUG → EXAMS_DIR existe:", os.path.exists(EXAMS_DIR))

OUTPUT_DIR = f"{BASE_DIR}/results/2_models/2_rag/final/pubmed_final_{LANG}"
os.makedirs(OUTPUT_DIR, exist_ok=True)
LOG_FILE = os.path.join(OUTPUT_DIR, f"rag_pubmed_final_{MODEL_NAME}_{LANG}_log.txt")
sys.stdout=open(LOG_FILE,"w",encoding="utf-8")

print(f"🧠 MODEL: {MODEL_NAME} | 🌐 LANG: {LANG} | ⏳ {datetime.now():%Y-%m-%d %H:%M:%S}")

# --- Translator optional ---
try:
    translator = pipeline("translation", model="Helsinki-NLP/opus-mt-es-en")
    print("✅ Translator loaded (ES→EN)")
except Exception as e:
    translator = None
    print(f"⚠️ Translator unavailable ({e})")

def translate_query(text):
    if LANG=="es" and translator:
        try: return translator(text[:500])[0]["translation_text"]
        except: return text
    return text

def pubmed_search(term, retmax=3):
    try:
        r=requests.get(f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?db=pubmed&term={term}&retmax={retmax}",timeout=15)
        tree=ET.fromstring(r.text)
        return [i.text for i in tree.findall(".//Id")]
    except: return []

def pubmed_fetch(ids):
    if not ids: return ""
    try:
        url=f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=pubmed&id={','.join(ids)}&retmode=text&rettype=abstract"
        return requests.get(url,timeout=15).text.strip()
    except: return ""

def retrieve_context(query):
    q=translate_query(query)
    ids=pubmed_search("+".join(q.split()))
    txt=pubmed_fetch(ids)
    return txt[:2000] if txt else "No relevant PubMed context found.", ids

def generate_answer(prompt):
    try:
        if MODEL_NAME.startswith("med") or MODEL_NAME.lower() in ["llama3","mistral","gemma"]:
            r=requests.post("http://localhost:11434/api/generate",json={"model":MODEL_NAME,"prompt":prompt,"stream":False},timeout=180)
            return r.json().get("response","").strip()
        else:
            pipe=load_hf_model(MODEL_NAME)
            return pipe(prompt,max_new_tokens=256,do_sample=False)[0]["generated_text"].strip()
    except Exception as e:
        return f"❌ {e}"

metrics=[]
for f in os.listdir(EXAMS_DIR):
    print(f"🔍 DEBUG → Archivos detectados en {EXAMS_DIR}:", os.listdir(EXAMS_DIR))

    if not f.endswith(".json"):continue
    tit=f.split("_")[0]
    with open(os.path.join(EXAMS_DIR,f),"r",encoding="utf-8") as g:data=json.load(g)
    preguntas = [
        q for q in data.get("preguntas", [])
        if not q.get("tipo") or q.get("tipo").lower() in ["texto", "text", "teorica"]
    ]
    results=[]
    for q in preguntas:
        ctx,ids=retrieve_context(q["enunciado"])
        opts="\n".join(f"{i+1}. {o}" for i,o in enumerate(q["opciones"]))
        prompt=f"{PROMPT_RAG}\n📚 CONTEXTO (PubMed):\n{ctx}\n❓ {q['enunciado']}\n{opts}"
        out=generate_answer(prompt)
        m=re.search(r"\b([1-4])\b",out)
        pred=int(m.group(1)) if m else None
        results.append({"numero":q["numero"],"pmids":ids,"respuesta_correcta":q["respuesta_correcta"],MODEL_NAME:pred})
    acc=sum(1 for r in results if r[MODEL_NAME]==r["respuesta_correcta"])/max(1,len(results))*100
    metrics.append({"Modelo":MODEL_NAME,"Titulación":tit,"Accuracy (%)":round(acc,2)})
    json.dump({"preguntas":results},open(os.path.join(OUTPUT_DIR,f"{tit}_{MODEL_NAME}_rag_pubmed_final_{LANG}.json"),"w",encoding="utf-8"),ensure_ascii=False,indent=2)

df = pd.DataFrame(metrics)
xlsx_path = os.path.join(OUTPUT_DIR, f"rag_pubmed_final_{LANG}_metrics.xlsx")
df.to_excel(xlsx_path, index=False)
print(f"\n📊 Metrics exported to: {xlsx_path}")
print("🏁 Completed.")
