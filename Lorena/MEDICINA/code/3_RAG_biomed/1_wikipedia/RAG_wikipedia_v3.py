#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RAG with Wikipedia – v3 (dynamic model, KeyBERT + spaCy)
Author: Lorena Ariceta García
"""

import os, sys, re, json, time, requests, argparse, pandas as pd, spacy, wikipediaapi
from datetime import datetime
from keybert import KeyBERT
from sentence_transformers import SentenceTransformer
from transformers import pipeline
from utils.load_model import load_hf_model

sys.path.append("/home/xs1/Desktop/Lorena/MEDICINA/code/3_RAG")
from prompt_config import PROMPTS

parser = argparse.ArgumentParser(description="RAG Wikipedia v3")
parser.add_argument("--lang", choices=["es","en"], default="es")
args=parser.parse_args()

LANG=os.getenv("RAG_LANG",args.lang)
MODEL_NAME=os.getenv("RAG_MODEL")
assert MODEL_NAME,"❌ RAG_MODEL not defined"

BASE_DIR="/home/xs1/Desktop/Lorena/MEDICINA"
if f"{BASE_DIR}/code/3_RAG" not in sys.path:
    sys.path.append(f"{BASE_DIR}/code/3_RAG")
PROMPT_RAG=PROMPTS[LANG]
EXAMS_DIR=f"{BASE_DIR}/results/1_data_preparation/6_json_final/prueba"
OUTPUT_DIR=f"{BASE_DIR}/results/2_models/2_rag/1_wikipedia/v3_{LANG}"
os.makedirs(OUTPUT_DIR,exist_ok=True)
LOG_FILE=os.path.join(OUTPUT_DIR,f"rag_wikipedia_v3_{MODEL_NAME}_{LANG}_log.txt")

sys.stdout=open(LOG_FILE,"w",encoding="utf-8")
print(f"🧠 MODEL: {MODEL_NAME} | 🌐 LANG: {LANG}")

kw_model=KeyBERT(model=SentenceTransformer("all-MiniLM-L6-v2"))
nlp=spacy.load("es_core_news_sm" if LANG=="es" else "en_core_web_sm")
wiki=wikipediaapi.Wikipedia(LANG)
translator=pipeline("translation",model="Helsinki-NLP/opus-mt-es-en") if LANG=="en" else None

def generate_answer(prompt):
    try:
        if MODEL_NAME.startswith("med") or MODEL_NAME.lower() in ["llama3","mistral","gemma"]:
            r=requests.post("http://localhost:11434/api/generate",json={"model":MODEL_NAME,"prompt":prompt},timeout=180)
            return r.json().get("response","").strip()
        else:
            pipe=load_hf_model(MODEL_NAME)
            return pipe(prompt,max_new_tokens=256,do_sample=False)[0]["generated_text"].strip()
    except Exception as e:
        return f"❌ {e}"

def get_keywords(text):
    kw=[k[0] for k in kw_model.extract_keywords(text,top_n=2)]
    kw_spacy=[t.text for t in nlp(text) if t.pos_ in ["NOUN","PROPN"]]
    return list(dict.fromkeys(kw+kw_spacy))

def get_context(keywords):
    contexts=[]
    for kw in keywords:
        if translator and LANG=="en":
            kw=translator(kw)[0]["translation_text"]
        page=wiki.page(kw)
        if page.exists(): contexts.append(page.summary[:1500])
    return "\n".join(contexts) if contexts else "Sin contexto relevante."

metrics=[]
for f in os.listdir(EXAMS_DIR):
    if not f.endswith(".json"):continue
    tit=f.split("_")[0]
    with open(os.path.join(EXAMS_DIR,f),"r",encoding="utf-8") as g:data=json.load(g)
    preguntas = [
        q for q in data.get("preguntas", [])
        if not q.get("tipo") or q.get("tipo").lower() in ["texto", "text", "teorica"]
    ]
    results=[]
    for q in preguntas:
        ctx=get_context(get_keywords(q["enunciado"]))
        opts="\n".join(f"{i+1}. {o}" for i,o in enumerate(q["opciones"]))
        prompt=f"{PROMPT_RAG}\n📚 CONTEXTO:\n{ctx}\n❓ {q['enunciado']}\n{opts}"
        out=generate_answer(prompt)
        match=re.search(r"\b([1-4])\b",out)
        pred=int(match.group(1)) if match else None
        results.append({"numero":q["numero"],"respuesta_correcta":q["respuesta_correcta"],MODEL_NAME:pred})
    acc=sum(1 for r in results if r[MODEL_NAME]==r["respuesta_correcta"])/max(1,len(results))*100
    metrics.append({"Modelo":MODEL_NAME,"Titulación":tit,"Accuracy (%)":round(acc,2)})
    json.dump({"preguntas":results},open(os.path.join(OUTPUT_DIR,f"{tit}_{MODEL_NAME}_rag_wikipedia_v3_{LANG}.json"),"w",encoding="utf-8"),ensure_ascii=False,indent=2)

pd.DataFrame(metrics).to_excel(os.path.join(OUTPUT_DIR,f"rag_wikipedia_v3_{MODEL_NAME}_{LANG}_metrics.xlsx"),index=False)
print("\n🏁 Completed.")
