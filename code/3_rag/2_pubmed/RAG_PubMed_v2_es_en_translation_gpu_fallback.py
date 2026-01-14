#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
PubMed-RAG Pipeline (v2) — Single specialization runner (for main selector)

Reads env vars:
  RAG_LANG           -> "es" or "en"
  RAG_MODEL          -> e.g., "llama3"
  RAG_SPECIALIZATION -> e.g., "MEDICINA"
  RAG_INPUT_JSON     -> full path to the specialization JSON
  RAG_PROMPT         -> base prompt template text coming from prompt_config.py

Features (v2):
  - Keywording: KeyBERT + spaCy
  - Ensure min 3 keywords (fallback terms)
  - Optional ES->EN translation using transformers pipeline (if available)
  - Tiered PubMed search (3 -> 2 -> 1 keywords), tries EN then ES
  - Retrieves abstracts via NCBI E-utilities (ESearch + EFetch)
  - Calls Ollama
  - Prints progress: "SPEC | lang | model | Q i/total | pred/gt | query | pmids"

Outputs:
  - JSON:  <SPEC>_<MODEL>_pubmed_v2_<LANG>_<timestamp>.json
  - METRICS Excel: <SPEC>_<MODEL>_pubmed_v2_<LANG>_<timestamp>_metrics.xlsx
    (important: *_metrics.xlsx so main can find it)
"""

import json
import os
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Tuple, Dict, Any
import xml.etree.ElementTree as ET

import requests
import pandas as pd
import spacy
from keybert import KeyBERT
from sentence_transformers import SentenceTransformer

# -----------------------------
# ENV from main selector
# -----------------------------
RAG_LANG = os.environ.get("RAG_LANG", "es").strip().lower()
RAG_MODEL = os.environ.get("RAG_MODEL", "llama3").strip()
RAG_SPECIALIZATION = os.environ.get("RAG_SPECIALIZATION", "UNKNOWN").strip()
RAG_INPUT_JSON = os.environ.get("RAG_INPUT_JSON", "").strip()
RAG_PROMPT = os.environ.get("RAG_PROMPT", "").strip()

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434/api/generate")

if not RAG_INPUT_JSON or not os.path.exists(RAG_INPUT_JSON):
    raise FileNotFoundError(f"RAG_INPUT_JSON not found: {RAG_INPUT_JSON}")

# -----------------------------
# Config / models
# -----------------------------
sentence_model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
kw_model = KeyBERT(model=sentence_model)

# spaCy
# (si quieres EN, podrías condicionar el modelo, pero tu dataset parece ES)
nlp = spacy.load("es_core_news_sm")

# Translator optional (ES->EN). If not available, we keep ES keywords.
translator = None
try:
    from transformers import pipeline  # type: ignore
    import torch  # type: ignore

    if torch.cuda.is_available():
        try:
            _ = torch.cuda.get_device_name(0)
            device = 0
        except Exception:
            device = -1
    else:
        device = -1

    translator = pipeline("translation", model="Helsinki-NLP/opus-mt-es-en", device=device)
    print("✅ Translator loaded: Helsinki-NLP/opus-mt-es-en", flush=True)
except Exception as e:
    print(f"⚠️ Translator unavailable, continuing without translation: {e}", flush=True)
    translator = None

BASE_DIR = Path(os.getenv("FSE_BASE_DIR", Path(__file__).resolve().parents[3]))
OUTPUT_DIR = Path(
    os.getenv(
        "FSE_OUTPUT_DIR",
        BASE_DIR / "results/3_rag/2_pubmed",
    )
)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PUBMED_SLEEP_SECONDS = 0.4
PUBMED_TOP_K = 3
CONTEXT_CHAR_BUDGET = 2000

# -----------------------------
# Helpers
# -----------------------------
def get_keywords_keybert(text: str, top_n: int = 3) -> List[str]:
    try:
        kws = kw_model.extract_keywords(text, top_n=top_n)
        return [k[0] for k in kws if k and k[0]]
    except Exception:
        return []

def get_keywords_spacy(text: str, top_n: int = 5) -> List[str]:
    doc = nlp(text)
    kws = [t.text.lower().strip() for t in doc if t.pos_ in ("NOUN", "PROPN")]
    # unique preserve order
    seen = set()
    out = []
    for kw in kws:
        if kw and kw not in seen:
            seen.add(kw)
            out.append(kw)
    return out[:top_n]

def ensure_three_keywords(keywords: List[str]) -> List[str]:
    keywords = [k for k in keywords if isinstance(k, str) and k.strip()]
    fallback = ["medicine", "diagnosis", "treatment"]
    for term in fallback:
        if len(keywords) >= 3:
            break
        if term not in keywords:
            keywords.append(term)
    return keywords[:3]

def translate_keywords_es_en(keywords_es: List[str]) -> List[str]:
    if translator is None:
        return keywords_es
    try:
        phrase = ", ".join(keywords_es)
        translated = translator(phrase)[0]["translation_text"]
        parts = [p.strip().lower() for p in re.split(r"[,;/]", translated) if p.strip()]
        # sanitize
        parts = [re.sub(r"[^a-z0-9 \-()]", "", p) for p in parts]
        parts = [p.strip() for p in parts if p.strip()]
        if len(parts) >= 1:
            return ensure_three_keywords(parts)
        return keywords_es
    except Exception:
        return keywords_es

def pubmed_esearch(term: str, retmax: int) -> List[str]:
    url = (
        "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
        f"?db=pubmed&term={term}&retmax={retmax}"
    )
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    tree = ET.fromstring(r.text)
    return [elem.text for elem in tree.findall(".//Id") if elem.text]

def pubmed_efetch(ids: List[str]) -> str:
    id_list = ",".join(ids)
    url = (
        "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
        f"?db=pubmed&id={id_list}&retmode=text&rettype=abstract"
    )
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    return r.text.strip()

def tiered_pubmed_search(keywords: List[str], top_k: int, label: str) -> Tuple[List[str], List[str], str]:
    """
    Try with 3 -> 2 -> 1 keywords
    """
    for n in (3, 2, 1):
        subset = keywords[:n]
        query = "+".join([k.replace(" ", "+") for k in subset if k])
        if not query:
            continue
        print(f"    🔎 PubMed query {label} ({n} kw): {query}", flush=True)
        try:
            ids = pubmed_esearch(query, top_k)
        except Exception as e:
            print(f"    ⚠️ ESearch error: {e}", flush=True)
            ids = []
        if ids:
            print(f"    📚 PMIDs: {', '.join(ids)}", flush=True)
            return ids, subset, query
        print("    ⚠️ No results, trying fewer keywords...", flush=True)
    return [], [], ""

def retrieve_pubmed_context(statement: str, top_k: int = PUBMED_TOP_K) -> Tuple[str, List[str], List[str], List[str], str]:
    """
    Returns:
      context, pmids, keywords_es, keywords_en, query_used
    """
    kb = get_keywords_keybert(statement, top_n=3)
    sp = get_keywords_spacy(statement, top_n=5)

    merged = []
    seen = set()
    for kw in kb + sp:
        kw2 = kw.lower().strip()
        if kw2 and kw2 not in seen:
            seen.add(kw2)
            merged.append(kw2)

    keywords_es = ensure_three_keywords(merged[:3])
    keywords_en = translate_keywords_es_en(keywords_es)

    print(f"    🔍 Keywords (ES): {', '.join(keywords_es)}", flush=True)
    print(f"    🌍 Keywords (EN): {', '.join(keywords_en)}", flush=True)

    # Try EN first, then ES
    ids, _, query_used = tiered_pubmed_search(keywords_en, top_k, "[EN]")
    if not ids:
        ids, _, query_used = tiered_pubmed_search(keywords_es, top_k, "[ES]")

    if not ids:
        return "No relevant PubMed results found.", [], keywords_es, keywords_en, query_used

    try:
        raw = pubmed_efetch(ids)
    except Exception as e:
        return f"EFetch error: {e}", ids, keywords_es, keywords_en, query_used

    # Take first "blocks" and truncate
    context = "\n\n".join(raw.split("\n\n")[:3])[:CONTEXT_CHAR_BUDGET]
    time.sleep(PUBMED_SLEEP_SECONDS)
    return context, ids, keywords_es, keywords_en, query_used

def build_prompt(context: str, statement: str, options_text: str) -> str:
    base = RAG_PROMPT.strip() if RAG_PROMPT else (
        "You are a medical professional answering a MIR-style clinical question.\n"
        "Use the provided CONTEXT if relevant.\n"
        "Answer strictly in the format: 'The correct answer is number X.'\n"
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

# -----------------------------
# Load JSON
# -----------------------------
with open(RAG_INPUT_JSON, "r", encoding="utf-8") as f:
    data = json.load(f)

questions_all = data.get("preguntas", [])
questions = [q for q in questions_all if q.get("tipo") == "texto"]
total = len(questions)

print(f"\n✅ PubMed_v2 START", flush=True)
print(f"   🏷️  Spec: {RAG_SPECIALIZATION}", flush=True)
print(f"   🌐 Lang: {RAG_LANG}", flush=True)
print(f"   🧠 Model: {RAG_MODEL}", flush=True)
print(f"   📄 JSON: {RAG_INPUT_JSON}", flush=True)
print(f"   🧾 Text questions: {total}", flush=True)
print("-" * 70, flush=True)

# -----------------------------
# Main loop
# -----------------------------
t_start = time.time()
rows: List[Dict[str, Any]] = []

correct = wrong = no_answer = 0

for idx, q in enumerate(questions, start=1):
    statement = q.get("enunciado", "")
    options = q.get("opciones", [])
    gt = q.get("respuesta_correcta", None)

    print(f"\n{RAG_SPECIALIZATION} | {RAG_LANG} | {RAG_MODEL} | Q {idx}/{total}", flush=True)

    context, pmids, kw_es, kw_en, query_used = retrieve_pubmed_context(statement)

    options_text = "\n".join(f"{i+1}. {opt}" for i, opt in enumerate(options))
    prompt = build_prompt(context, statement, options_text)

    # Ollama
    try:
        payload = {"model": RAG_MODEL, "prompt": prompt, "stream": False}
        r = requests.post(OLLAMA_URL, json=payload, timeout=180)
        r.raise_for_status()
        raw_text = (r.json().get("response", "") or "").strip()
    except Exception as exc:
        raw_text = f"ERROR: {exc}"

    pred = parse_prediction(raw_text)

    # Metrics
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

    print(
        f"{RAG_SPECIALIZATION} | {RAG_LANG} | {RAG_MODEL} | Q {idx}/{total} | {status} | "
        f"pmids={len(pmids)} | {query_used} | {short}",
        flush=True
    )

    rows.append({
        "numero": q.get("numero"),
        "enunciado": statement,
        "opciones": options,
        "gt": gt,
        "pred": pred,
        "raw": raw_text,
        "keywords_es": kw_es,
        "keywords_en": kw_en,
        "pubmed_query": query_used,
        "pmids": pmids,
        "context": context,
    })

# -----------------------------
# Save outputs
# -----------------------------
answered = total - no_answer
acc = (correct / answered * 100.0) if answered > 0 else 0.0

stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
json_out = (
    OUTPUT_DIR
    / f"{RAG_SPECIALIZATION}_{RAG_MODEL}_pubmed_v2_{RAG_LANG}_{stamp}.json"
)
xlsx_out = (
    OUTPUT_DIR
    / f"{RAG_SPECIALIZATION}_{RAG_MODEL}_pubmed_v2_{RAG_LANG}_{stamp}_metrics.xlsx"
)

with open(json_out, "w", encoding="utf-8") as f:
    json.dump({"preguntas": rows}, f, ensure_ascii=False, indent=2)

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
    "JSON": json_out.name,
}])
df_metrics.to_excel(xlsx_out, index=False)

print("-" * 70, flush=True)
print(f"✅ Saved JSON   : {json_out}", flush=True)
print(f"✅ Saved METRICS: {xlsx_out}", flush=True)
print(f"🏁 DONE | Accuracy: {acc:.2f}% | Answered: {answered}/{total}", flush=True)
