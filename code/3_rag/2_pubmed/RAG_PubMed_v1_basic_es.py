#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
PubMed-RAG Pipeline (v1) — Single specialization runner (for RAG_main_selective)

Reads env vars:
  RAG_LANG           -> "es" or "en"
  RAG_MODEL          -> e.g., "llama3"
  RAG_SPECIALIZATION -> e.g., "MEDICINA"
  RAG_INPUT_JSON     -> full path to specialization JSON
  RAG_PROMPT         -> base prompt text coming from prompt_config.py

For each question (tipo == "texto"):
  1) Extract keywords (KeyBERT -> fallback spaCy)
  2) Retrieve PubMed abstracts in real-time (E-utilities)
  3) Build final prompt = BASE_PROMPT + context + question + options
  4) Ask Ollama
  5) Parse predicted option 1-4
  6) Save per-run JSON + *_metrics.xlsx (so main can find it)
"""

import os
import json
import re
import time
from datetime import datetime
from typing import Optional, List, Tuple

import requests
import pandas as pd
import spacy
import xml.etree.ElementTree as ET
from keybert import KeyBERT
from sentence_transformers import SentenceTransformer


# ================================================================
# 0) ENV from RAG_main_selective
# ================================================================
RAG_LANG = os.environ.get("RAG_LANG", "es").strip().lower()
RAG_MODEL = os.environ.get("RAG_MODEL", "llama3").strip()
RAG_SPECIALIZATION = os.environ.get("RAG_SPECIALIZATION", "UNKNOWN").strip()
RAG_INPUT_JSON = os.environ.get("RAG_INPUT_JSON", "").strip()
RAG_PROMPT = os.environ.get("RAG_PROMPT", "").strip()

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434/api/generate")

if not RAG_INPUT_JSON or not os.path.exists(RAG_INPUT_JSON):
    raise FileNotFoundError(f"RAG_INPUT_JSON not found: {RAG_INPUT_JSON}")

# ================================================================
# 1) Config
# ================================================================
BASE_DIR = "/home/xs1/Desktop/Lorena"
OUTPUT_DIR = f"{BASE_DIR}/results/3_rag/2_pubmed"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# CPU keywording
SENTENCE_MODEL = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
KW_MODEL = KeyBERT(model=SENTENCE_MODEL)

# spaCy (ojo: usa es_core_news_sm incluso si RAG_LANG=en, porque tus preguntas suelen estar en ES;
# si algún día metes preguntas en EN, puedes cambiarlo a en_core_web_sm cuando RAG_LANG == "en")
NLP = spacy.load("es_core_news_sm")


# ================================================================
# 2) Helpers
# ================================================================
def get_keywords_keybert(text: str, top_n: int = 3) -> List[str]:
    try:
        kws = KW_MODEL.extract_keywords(text, top_n=top_n)
        return [k[0] for k in kws if k and k[0]]
    except Exception:
        return []

def get_keywords_spacy(text: str, top_n: int = 6) -> List[str]:
    doc = NLP(text)
    kws = [t.text.lower() for t in doc if t.pos_ in ("NOUN", "PROPN")]
    # dedupe manteniendo orden
    seen = set()
    out = []
    for k in kws:
        if k not in seen:
            seen.add(k)
            out.append(k)
        if len(out) >= top_n:
            break
    return out

def build_prompt(context: str, statement: str, options_text: str) -> str:
    """
    Usa el prompt base que viene del main (RAG_PROMPT) y le añade contexto+pregunta+opciones.
    Si no llega RAG_PROMPT, usa uno mínimo.
    """
    if RAG_PROMPT:
        base = RAG_PROMPT.strip()
    else:
        base = (
            "You are a medical professional answering a MIR-style clinical question.\n"
            "Use the provided CONTEXT when helpful.\n"
            "Answer strictly: 'The correct answer is number X.'\n"
            "Then add a short justification."
        )

    return f"""{base}

CONTEXT (PubMed):
{context}

QUESTION:
{statement}

OPTIONS:
{options_text}
"""

def parse_prediction(raw_text: str) -> Optional[int]:
    m = re.search(r"\b([1-4])\b", raw_text)
    return int(m.group(1)) if m else None

def _pubmed_esearch(term: str, retmax: int = 3) -> List[str]:
    url = (
        "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
        f"?db=pubmed&term={term}&retmax={retmax}"
    )
    r = requests.get(url, timeout=20)
    r.raise_for_status()
    root = ET.fromstring(r.text)
    return [elem.text for elem in root.findall(".//Id") if elem.text]

def _pubmed_efetch(ids: List[str]) -> str:
    if not ids:
        return ""
    id_list = ",".join(ids)
    url = (
        "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
        f"?db=pubmed&id={id_list}&retmode=text&rettype=abstract"
    )
    r = requests.get(url, timeout=20)
    r.raise_for_status()
    return (r.text or "").strip()

def retrieve_pubmed_context(statement: str, top_k: int = 3) -> Tuple[str, List[str]]:
    """
    Returns (context, keywords_used)
    """
    keywords = get_keywords_keybert(statement, top_n=3)
    if not keywords:
        keywords = get_keywords_spacy(statement)[:3]

    if not keywords:
        return "No relevant keywords found.", []

    # term: mejor con espacios como '+'; si quieres más robusto usa urllib.parse.quote_plus
    term = "+".join(keywords)

    try:
        ids = _pubmed_esearch(term, retmax=top_k)
        if not ids:
            return f"No PubMed results found for: {term}", keywords

        abstracts = _pubmed_efetch(ids)
        if not abstracts:
            return f"PubMed fetch returned empty text for: {term}", keywords

        # limita contexto
        context = "\n\n".join(abstracts.split("\n\n")[:3])[:2000]

        # respeta NCBI rate-limit un poco
        time.sleep(0.34)

        return context, keywords

    except Exception as exc:
        return f"Error retrieving PubMed context: {exc}", keywords


# ================================================================
# 3) Load data
# ================================================================
with open(RAG_INPUT_JSON, "r", encoding="utf-8") as f:
    data = json.load(f)

questions_all = data.get("preguntas", [])
questions = [q for q in questions_all if q.get("tipo") == "texto"]
total = len(questions)

print(f"\n✅ PubMed_v1 START", flush=True)
print(f"   🏷️  Spec: {RAG_SPECIALIZATION}", flush=True)
print(f"   🌐 Lang: {RAG_LANG}", flush=True)
print(f"   🧠 Model: {RAG_MODEL}", flush=True)
print(f"   📄 JSON: {RAG_INPUT_JSON}", flush=True)
print(f"   🧾 Text questions: {total}", flush=True)
print("-" * 70, flush=True)

# ================================================================
# 4) Run loop
# ================================================================
model_results = []
correct = wrong = no_answer = 0
t_start = time.time()

for idx, q in enumerate(questions, start=1):
    statement = (q.get("enunciado") or "").strip()
    options = q.get("opciones") or []
    gt = q.get("respuesta_correcta", None)

    # Context PubMed
    context, kws_used = retrieve_pubmed_context(statement, top_k=3)

    options_text = "\n".join(f"{i+1}. {opt}" for i, opt in enumerate(options))
    prompt = build_prompt(context, statement, options_text)

    # Call Ollama
    try:
        payload = {"model": RAG_MODEL, "prompt": prompt, "stream": False}
        r = requests.post(OLLAMA_URL, json=payload, timeout=180)
        r.raise_for_status()
        raw_text = (r.json().get("response", "") or "").strip()
    except Exception as exc:
        raw_text = f"ERROR: {exc}"

    pred = parse_prediction(raw_text)

    # Metrics counters
    if pred is None:
        no_answer += 1
        status = "pred=None"
    else:
        if gt is None:
            status = f"pred={pred} | gt=None"
        elif pred == gt:
            correct += 1
            status = f"✅ pred={pred} | gt={gt}"
        else:
            wrong += 1
            status = f"❌ pred={pred} | gt={gt}"

    short = raw_text.replace("\n", " ")
    short = (short[:90] + "...") if len(short) > 90 else short
    kw_txt = ", ".join(kws_used) if kws_used else "no-kws"

    print(
        f"{RAG_SPECIALIZATION} | {RAG_LANG} | {RAG_MODEL} | Q {idx}/{total} | {status} | kws: {kw_txt} | {short}",
        flush=True
    )

    model_results.append(
        {
            "numero": q.get("numero"),
            "enunciado": statement,
            "opciones": options,
            "kws_used": kws_used,
            "context": context,
            "pred": pred,
            "gt": gt,
            "raw": raw_text,
        }
    )

# ================================================================
# 5) Save outputs
# ================================================================
answered = total - no_answer
acc = (correct / answered * 100.0) if answered > 0 else 0.0

stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
json_out = os.path.join(
    OUTPUT_DIR,
    f"{RAG_SPECIALIZATION}_{RAG_MODEL}_pubmed_v1_{RAG_LANG}_{stamp}.json"
)
xlsx_out = os.path.join(
    OUTPUT_DIR,
    f"{RAG_SPECIALIZATION}_{RAG_MODEL}_pubmed_v1_{RAG_LANG}_{stamp}_metrics.xlsx"
)

with open(json_out, "w", encoding="utf-8") as f:
    json.dump({"preguntas": model_results}, f, ensure_ascii=False, indent=2)

df_metrics = pd.DataFrame([{
    "Specialization": RAG_SPECIALIZATION,
    "Lang": RAG_LANG,
    "Model": RAG_MODEL,
    "Total questions": total,
    "Answered": answered,
    "Correct": correct,
    "Errors": wrong,
    "No answer": no_answer,
    "Accuracy (%)": round(acc, 2),
    "Seconds": round(time.time() - t_start, 2),
    "JSON": os.path.basename(json_out),
}])
df_metrics.to_excel(xlsx_out, index=False)

print("-" * 70, flush=True)
print(f"✅ Saved JSON   : {json_out}", flush=True)
print(f"✅ Saved METRICS: {xlsx_out}", flush=True)
print(f"🏁 DONE | Accuracy: {acc:.2f}% | Answered: {answered}/{total}", flush=True)
