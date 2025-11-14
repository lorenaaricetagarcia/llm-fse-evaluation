#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RAG with PubMed – v2 (dynamic multilingual)
Author: Lorena Ariceta García
"""

import os, sys, json, re, requests, xml.etree.ElementTree as ET, pandas as pd, spacy, argparse, torch
from keybert import KeyBERT
from sentence_transformers import SentenceTransformer
from datetime import datetime
from transformers import pipeline
from utils.load_model import load_hf_model

BASE_DIR = "/home/xs1/Desktop/Lorena/MEDICINA"
if f"{BASE_DIR}/code/3_RAG" not in sys.path:
    sys.path.append(f"{BASE_DIR}/code/3_RAG")
sys.path.append(f"{BASE_DIR}/code/3_RAG")
from prompt_config import PROMPTS

# ------------------------- Args / Env -------------------------
parser = argparse.ArgumentParser(description="RAG PubMed v2 – multilingual")
parser.add_argument("--lang", choices=["es","en"], default="es")
args=parser.parse_args()
LANG=os.getenv("RAG_LANG", args.lang)
MODEL_NAME=os.getenv("RAG_MODEL")
assert MODEL_NAME, "❌ RAG_MODEL not defined"
PROMPT_RAG=PROMPTS[LANG]

EXAMS_DIR=f"{BASE_DIR}/results/1_data_preparation/6_json_final/prueba"
OUTPUT_DIR=f"{BASE_DIR}/results/2_models/2_rag/2_pubmed/v2_{LANG}"
os.makedirs(OUTPUT_DIR,exist_ok=True)
LOG_FILE=os.path.join(OUTPUT_DIR,f"rag_pubmed_v2_{MODEL_NAME}_{LANG}_log.txt")

# ------------------------- Logger -------------------------
sys.stdout=open(LOG_FILE,"w",encoding="utf-8")
print(f"🧠 MODEL: {MODEL_NAME} | 🌐 LANG: {LANG} | ⏳ {datetime.now():%Y-%m-%d %H:%M:%S}")

# ------------------------- Init -------------------------
device = 0 if torch.cuda.is_available() else -1
print("✅ GPU detected" if device==0 else "⚙️ Running on CPU")

sentence_model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
kw_model = KeyBERT(model=sentence_model)
try: nlp = spacy.load("en_core_web_sm")
except: nlp = spacy.load("es_core_news_sm")

try:
    translator = pipeline("translation", model="Helsinki-NLP/opus-mt-es-en", device=device)
    print("✅ Translator loaded (ES→EN)")
except Exception as e:
    translator = None
    print(f"⚠️ Translator unavailable ({e})")

# ------------------------- PubMed utils -------------------------
def get_keywords(text):
    try: kws=[k[0] for k in kw_model.extract_keywords(text,top_n=3)]
    except: kws=[]
    if not kws:
        kws=[t.text for t in nlp(text) if t.pos_ in ["NOUN","PROPN"]][:3]
    return kws or ["medicine","treatment","diagnosis"]

def translate_keywords(kws):
    if LANG=="en" or not translator: return kws
    try:
        phrase=", ".join(kws)
        out=translator(phrase)[0]["translation_text"]
        return [p.strip().lower() for p in re.split(r"[,;/]",out) if p.strip()]
    except: return kws

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

def retrieve_pubmed_context(query):
    kws=get_keywords(query)
    kws_en=translate_keywords(kws)
    ids=pubmed_search("+".join(kws_en))
    txt=pubmed_fetch(ids)
    return (txt[:2000] if txt else "No PubMed context found."), ids

def generate_answer(prompt):
    try:
        if MODEL_NAME.startswith("med") or MODEL_NAME.lower() in ["llama3","mistral","gemma"]:
            r=requests.post("http://localhost:11434/api/generate",json={"model":MODEL_NAME,"prompt":prompt,"stream":False},timeout=180)
            return r.json().get("response","").strip()
        else:
            pipe=load_hf_model(MODEL_NAME)
            return pipe(prompt,max_new_tokens=256,do_sample=False)[0]["generated_text"].strip()
    except Exception as e: return f"❌ {e}"

# ------------------------- Main Loop -------------------------
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
        ctx, ids = retrieve_pubmed_context(q["enunciado"])
        opts="\n".join(f"{i+1}. {o}" for i,o in enumerate(q["opciones"]))
        prompt=f"{PROMPT_RAG}\n📚 CONTEXT (PubMed):\n{ctx}\n❓ {q['enunciado']}\n{opts}"
        out=generate_answer(prompt)
        m=re.search(r"\b([1-4])\b",out)
        pred=int(m.group(1)) if m else None
        results.append({"numero":q["numero"],"pmids":ids,"respuesta_correcta":q["respuesta_correcta"],MODEL_NAME:pred})
    acc=sum(1 for r in results if r[MODEL_NAME]==r["respuesta_correcta"])/max(1,len(results))*100
    metrics.append({"Modelo":MODEL_NAME,"Titulación":tit,"Accuracy (%)":round(acc,2)})
    json.dump({"preguntas":results},open(os.path.join(OUTPUT_DIR,f"{tit}_{MODEL_NAME}_rag_pubmed_v2_{LANG}.json"),"w",encoding="utf-8"),ensure_ascii=False,indent=2)

pd.DataFrame(metrics).to_excel(os.path.join(OUTPUT_DIR,f"rag_pubmed_v2_{MODEL_NAME}_{LANG}_metrics.xlsx"),index=False)
print("🏁 Completed.")
