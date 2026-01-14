#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
PubMed-RAG Pipeline (v3) — Single specialization runner
- Like v2 (tiered + optional translation)
- Adds checkpoint saves every CHECKPOINT_EVERY questions
"""

import json
import os
import re
import time
from datetime import datetime
from pathlib import Path
from typing import List, Tuple, Optional
import xml.etree.ElementTree as ET

import requests
import pandas as pd
import spacy
from keybert import KeyBERT
from sentence_transformers import SentenceTransformer

try:
    import torch
    from transformers import pipeline as hf_pipeline
except Exception:
    torch = None
    hf_pipeline = None


# =========================
# ENV
# =========================
RAG_LANG = os.environ.get("RAG_LANG", "es").strip().lower()
RAG_MODEL = os.environ.get("RAG_MODEL", "llama3").strip()
RAG_SPECIALIZATION = os.environ.get("RAG_SPECIALIZATION", "UNKNOWN").strip()
RAG_INPUT_JSON = os.environ.get("RAG_INPUT_JSON", "").strip()
RAG_PROMPT = os.environ.get("RAG_PROMPT", "").strip()
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434/api/generate").strip()

CHECKPOINT_EVERY = int(os.environ.get("RAG_CHECKPOINT_EVERY", "50"))

if not RAG_INPUT_JSON or not os.path.exists(RAG_INPUT_JSON):
    raise FileNotFoundError(f"RAG_INPUT_JSON not found: {RAG_INPUT_JSON}")

BASE_DIR = Path(os.getenv("FSE_BASE_DIR", Path(__file__).resolve().parents[3]))
OUTPUT_DIR = Path(
    os.getenv(
        "FSE_OUTPUT_DIR",
        BASE_DIR / "results/3_rag/2_pubmed",
    )
)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# =========================
# Models
# =========================
sentence_model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
kw_model = KeyBERT(model=sentence_model)
nlp = spacy.load("es_core_news_sm")

translator = None
if hf_pipeline is not None:
    device = -1
    if torch is not None and getattr(torch, "cuda", None) is not None and torch.cuda.is_available():
        try:
            torch.cuda.get_device_name(0)
            device = 0
        except Exception:
            device = -1
    try:
        translator = hf_pipeline("translation", model="Helsinki-NLP/opus-mt-es-en", device=device)
    except Exception:
        translator = None


# =========================
# Helpers
# =========================
def ensure_base_prompt() -> str:
    if RAG_PROMPT:
        return RAG_PROMPT.strip()
    return (
        "You are a medical professional answering a MIR-style multiple-choice question.\n"
        "Read the CONTEXT and then the QUESTION.\n"
        "Answer strictly in the format: 'The correct answer is number X.' (X from 1 to 4)\n"
        "Then add one short justification sentence.\n"
    )

def get_keywords_keybert(text: str, top_n: int = 3) -> List[str]:
    try:
        kws = kw_model.extract_keywords(text, top_n=top_n)
        return [k[0] for k in kws if k and k[0]]
    except Exception:
        return []

def get_keywords_spacy(text: str, top_n: int = 5) -> List[str]:
    doc = nlp(text)
    cands = [t.text.lower() for t in doc if t.pos_ in ("NOUN", "PROPN")]
    seen, out = set(), []
    for c in cands:
        if c and c not in seen:
            seen.add(c)
            out.append(c)
    return out[:top_n]

def ensure_three_keywords(kws: List[str]) -> List[str]:
    kws = [k.strip().lower() for k in kws if isinstance(k, str) and k.strip()]
    fallback = ["medicine", "diagnosis", "treatment"] if RAG_LANG == "en" else ["medicina", "diagnóstico", "tratamiento"]
    for t in fallback:
        if len(kws) >= 3:
            break
        if t not in kws:
            kws.append(t)
    return kws[:3]

def translate_keywords_es_en(keywords_es: List[str]) -> List[str]:
    if translator is None:
        return keywords_es
    try:
        phrase = ", ".join(keywords_es)
        translated = translator(phrase)[0]["translation_text"]
        parts = [p.strip().lower() for p in re.split(r"[,;/]", translated) if p.strip()]
        if len(parts) < 3:
            parts = []
            for kw in keywords_es:
                try:
                    parts.append(translator(kw)[0]["translation_text"].strip().lower())
                except Exception:
                    parts.append(kw)
        parts = [re.sub(r"[^a-z0-9 \-()]", "", p) for p in parts]
        return ensure_three_keywords(parts)
    except Exception:
        return keywords_es

def pubmed_esearch(term: str, retmax: int = 3) -> List[str]:
    url = (
        "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
        f"?db=pubmed&term={term}&retmax={retmax}"
    )
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    tree = ET.fromstring(r.text)
    return [e.text for e in tree.findall(".//Id") if e.text]

def pubmed_efetch(pmids: List[str]) -> str:
    if not pmids:
        return ""
    id_list = ",".join(pmids)
    url = (
        "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
        f"?db=pubmed&id={id_list}&retmode=text&rettype=abstract"
    )
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    return r.text.strip()

def tiered_pubmed_search(keywords: List[str], top_k: int, tag: str) -> Tuple[List[str], List[str], str]:
    for n in (3, 2, 1):
        used = keywords[:n]
        term = "+".join(used)
        pmids = pubmed_esearch(term, retmax=top_k)
        if pmids:
            return pmids, used, term
    return [], [], ""

def retrieve_pubmed_context(question: str, top_k: int = 3) -> Tuple[str, List[str], List[str], List[str], str]:
    kb = get_keywords_keybert(question, top_n=3)
    sp = get_keywords_spacy(question, top_n=5)

    merged, seen = [], set()
    for kw in kb + sp:
        kw = kw.strip().lower()
        if kw and kw not in seen:
            seen.add(kw)
            merged.append(kw)

    keywords_es = ensure_three_keywords(merged[:3])
    keywords_en = translate_keywords_es_en(keywords_es)

    pmids, _, used_term = tiered_pubmed_search(keywords_en, top_k=top_k, tag="EN")
    if not pmids:
        pmids, _, used_term = tiered_pubmed_search(keywords_es, top_k=top_k, tag="ES")

    if not pmids:
        return "No relevant PubMed results found.", [], keywords_es, keywords_en, used_term

    raw = pubmed_efetch(pmids)
    context = "\n\n".join(raw.split("\n\n")[:3])[:2000]
    time.sleep(0.4)
    return context, pmids, keywords_es, keywords_en, used_term

def build_prompt(context: str, statement: str, options_text: str) -> str:
    base = ensure_base_prompt()
    return f"""{base}

CONTEXT (PubMed):
{context}

QUESTION:
{statement}

OPTIONS:
{options_text}
"""

def call_ollama(model: str, prompt: str, timeout_s: int = 180) -> str:
    payload = {"model": model, "prompt": prompt, "stream": False}
    r = requests.post(OLLAMA_URL, json=payload, timeout=timeout_s)
    r.raise_for_status()
    return (r.json().get("response", "") or "").strip()

def parse_choice(text: str) -> Optional[int]:
    m = re.search(r"\b([1-4])\b", text)
    return int(m.group(1)) if m else None


# =========================
# Load JSON
# =========================
with open(RAG_INPUT_JSON, "r", encoding="utf-8") as f:
    data = json.load(f)

questions = [q for q in data.get("preguntas", []) if q.get("tipo") == "texto"]
total = len(questions)

print(f"\n✅ PubMed_v3 START")
print(f"   🏷️  Spec: {RAG_SPECIALIZATION}")
print(f"   🌐 Lang: {RAG_LANG}")
print(f"   🧠 Model: {RAG_MODEL}")
print(f"   📄 JSON: {RAG_INPUT_JSON}")
print(f"   🧾 Text questions: {total}")
print(f"   💾 Checkpoint every: {CHECKPOINT_EVERY}")
print("-" * 70, flush=True)


# =========================
# Loop
# =========================
model_results = []
correct = wrong = no_answer = 0
t0 = time.time()

stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
tmp_json_path = (
    OUTPUT_DIR
    / f"{RAG_SPECIALIZATION}_{RAG_MODEL}_pubmed_v3_{RAG_LANG}_{stamp}_checkpoint.json"
)

for idx, q in enumerate(questions, start=1):
    statement = q.get("enunciado", "")
    options = q.get("opciones", [])
    gt = q.get("respuesta_correcta", None)

    options_text = "\n".join(f"{i+1}. {opt}" for i, opt in enumerate(options))

    context, pmids, kws_es, kws_en, used_term = retrieve_pubmed_context(statement, top_k=3)
    prompt = build_prompt(context, statement, options_text)

    try:
        raw = call_ollama(RAG_MODEL, prompt)
    except Exception as exc:
        raw = f"ERROR: {exc}"

    pred = parse_choice(raw)

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

    short = raw.replace("\n", " ")
    short = (short[:90] + "...") if len(short) > 90 else short
    pmids_txt = ",".join(pmids) if pmids else "-"

    print(
        f"{RAG_SPECIALIZATION} | {RAG_LANG} | {RAG_MODEL} | Q {idx}/{total} | {status} | "
        f"term={used_term or '-'} | pmids={pmids_txt} | {short}",
        flush=True
    )

    model_results.append({
        "numero": q.get("numero"),
        "enunciado": statement,
        "opciones": options,
        "keywords_es": kws_es,
        "keywords_en": kws_en,
        "pubmed_term": used_term,
        "pmids": pmids,
        "context": context,
        "pred": pred,
        "gt": gt,
        "raw": raw,
    })

    if CHECKPOINT_EVERY > 0 and (idx % CHECKPOINT_EVERY == 0):
        with open(tmp_json_path, "w", encoding="utf-8") as ftmp:
            json.dump({"preguntas": model_results}, ftmp, ensure_ascii=False, indent=2)
        print(f"💾 Checkpoint saved: {tmp_json_path} ({idx}/{total})", flush=True)


# =========================
# Save final
# =========================
answered = total - no_answer
acc = (correct / answered * 100.0) if answered > 0 else 0.0

json_out = (
    OUTPUT_DIR
    / f"{RAG_SPECIALIZATION}_{RAG_MODEL}_pubmed_v3_{RAG_LANG}_{stamp}.json"
)
xlsx_out = (
    OUTPUT_DIR
    / f"{RAG_SPECIALIZATION}_{RAG_MODEL}_pubmed_v3_{RAG_LANG}_{stamp}_metrics.xlsx"
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
    "Seconds": round(time.time() - t0, 2),
    "JSON": json_out.name,
}])
df_metrics.to_excel(xlsx_out, index=False)

print("-" * 70)
print(f"✅ Saved JSON   : {json_out}")
print(f"✅ Saved METRICS: {xlsx_out}")
print(f"🏁 DONE | Accuracy: {acc:.2f}% | Answered: {answered}/{total}", flush=True)
