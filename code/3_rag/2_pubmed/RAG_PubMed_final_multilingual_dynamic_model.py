#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
PubMed-RAG Pipeline (FINAL) — Single specialization runner (for main selector)

ENV (from RAG_main_selective):
  RAG_LANG           -> "es" or "en"
  RAG_MODEL          -> e.g. "llama3"
  RAG_SPECIALIZATION -> e.g. "MEDICINA"
  RAG_INPUT_JSON     -> full path to specialization JSON
  RAG_PROMPT         -> base prompt template text from prompt_config.py

For each question (tipo == "texto"):
  1) Extract keywords (KeyBERT)
  2) Retrieve PubMed abstracts via NCBI E-utilities (tiered query 3->2->1)
  3) Build final prompt = RAG_PROMPT + context + question + options
  4) Ask Ollama
  5) Parse predicted option 1-4
  6) Save JSON + *_metrics.xlsx (timestamped) so main can find metrics
"""

import json
import os
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Tuple
import xml.etree.ElementTree as ET

import requests
import pandas as pd
from keybert import KeyBERT
from sentence_transformers import SentenceTransformer


# =============================================================================
# 0) ENV from RAG_main_selective
# =============================================================================
RAG_LANG = os.environ.get("RAG_LANG", "es").strip().lower()
RAG_MODEL = os.environ.get("RAG_MODEL", "llama3").strip()
RAG_SPECIALIZATION = os.environ.get("RAG_SPECIALIZATION", "UNKNOWN").strip()
RAG_INPUT_JSON = os.environ.get("RAG_INPUT_JSON", "").strip()
RAG_PROMPT = os.environ.get("RAG_PROMPT", "").strip()

if not RAG_INPUT_JSON or not os.path.exists(RAG_INPUT_JSON):
    raise FileNotFoundError(f"RAG_INPUT_JSON not found: {RAG_INPUT_JSON}")

# =============================================================================
# 1) Config
# =============================================================================
BASE_DIR = Path(os.getenv("FSE_BASE_DIR", Path(__file__).resolve().parents[3]))
OUTPUT_DIR = Path(
    os.getenv(
        "FSE_OUTPUT_DIR",
        BASE_DIR / "results/3_rag/2_pubmed",
    )
)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Ollama endpoint
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434/api/generate").strip()

# PubMed E-utilities
ESEARCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
EFETCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"

PUBMED_TOP_K = int(os.environ.get("PUBMED_TOP_K", "3"))
PUBMED_MAX_CHARS = int(os.environ.get("PUBMED_MAX_CHARS", "2000"))
PUBMED_SLEEP = float(os.environ.get("PUBMED_SLEEP", "0.4"))

# Keyword extraction (CPU)
SENTENCE_MODEL = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
KW_MODEL = KeyBERT(model=SENTENCE_MODEL)


# =============================================================================
# 2) Helpers
# =============================================================================

# ================================================================
# 2.b) Optional ES->EN keyword translation (only used when RAG_LANG == "en")
#     - Silent fallback if transformers/torch/model not available
# ================================================================
TRANSLATOR = None
try:
    from transformers import pipeline  # type: ignore
    import torch  # type: ignore

    if torch.cuda.is_available():
        try:
            _ = torch.cuda.get_device_name(0)
            _device = 0
        except Exception:
            _device = -1
    else:
        _device = -1

    TRANSLATOR = pipeline("translation", model="Helsinki-NLP/opus-mt-es-en", device=_device)
except Exception:
    TRANSLATOR = None


def translate_keywords_es_en(keywords_es: List[str], top_n: int = 3) -> List[str]:
    """Translate ES keywords to EN (best-effort). Returns up to top_n terms."""
    if not keywords_es:
        return []
    if TRANSLATOR is None:
        return keywords_es[:top_n]

    try:
        phrase = ", ".join([k for k in keywords_es if isinstance(k, str) and k.strip()])
        if not phrase:
            return keywords_es[:top_n]

        translated = TRANSLATOR(phrase)[0].get("translation_text", "") or ""
        parts = [p.strip().lower() for p in re.split(r"[,;/]", translated) if p.strip()]
        parts = [re.sub(r"[^a-z0-9 \-()]", "", p).strip() for p in parts]
        parts = [p for p in parts if p]

        # de-dup preserve order
        seen = set()
        out = []
        for p in parts:
            if p not in seen:
                seen.add(p)
                out.append(p)
            if len(out) >= top_n:
                break

        return out if out else keywords_es[:top_n]
    except Exception:
        return keywords_es[:top_n]

def extract_keywords(text: str, top_n: int = 3) -> List[str]:
    """KeyBERT top_n keywords; fallback to empty list."""
    try:
        kws = KW_MODEL.extract_keywords(text, top_n=top_n)
        out = []
        for kw, _score in kws:
            kw = (kw or "").strip()
            if kw:
                out.append(kw)
        return out
    except Exception:
        return []


def ensure_three_keywords(kws: List[str]) -> List[str]:
    """Guarantee at least 3 query terms (generic fallback)."""
    base = []
    seen = set()
    for k in kws:
        k2 = (k or "").strip().lower()
        if k2 and k2 not in seen:
            seen.add(k2)
            base.append(k2)

    fallback = ["medicine", "diagnosis", "treatment"] if RAG_LANG == "en" else ["medicina", "diagnóstico", "tratamiento"]
    for term in fallback:
        if len(base) >= 3:
            break
        if term not in base:
            base.append(term)
    return base[:3]


def tiered_pubmed_search(keywords: List[str], top_k: int) -> Tuple[List[str], str]:
    """
    Try 3 keywords, then 2, then 1.
    Returns (pmids, used_term).
    """
    for n in (3, 2, 1):
        subset = keywords[:n]
        term = "+".join(subset)
        params = {"db": "pubmed", "term": term, "retmax": str(top_k)}
        r = requests.get(ESEARCH_URL, params=params, timeout=20)
        r.raise_for_status()

        root = ET.fromstring(r.text)
        pmids = [e.text for e in root.findall(".//Id") if e.text]
        if pmids:
            return pmids, term
    return [], ""


def pubmed_fetch_abstracts(pmids: List[str]) -> str:
    if not pmids:
        return ""
    params = {
        "db": "pubmed",
        "id": ",".join(pmids),
        "retmode": "text",
        "rettype": "abstract",
    }
    r = requests.get(EFETCH_URL, params=params, timeout=25)
    r.raise_for_status()
    return (r.text or "").strip()


def retrieve_pubmed_context(question_text: str) -> Tuple[str, List[str], str, List[str]]:
    """
    Returns: (context, pmids, used_term, keywords_used)
    """
    kws_es = extract_keywords(question_text, top_n=3)
    kws_es = ensure_three_keywords(kws_es)
    kws = kws_es
    if (RAG_LANG or '').strip().lower() == 'en':
        kws = translate_keywords_es_en(kws_es, top_n=3)

    try:
        pmids, used_term = tiered_pubmed_search(kws, top_k=PUBMED_TOP_K)
        if not pmids:
            return "No PubMed results found.", [], used_term, kws

        raw = pubmed_fetch_abstracts(pmids)
        if not raw:
            return "No PubMed abstracts retrieved.", pmids, used_term, kws

        # Compact context
        context = "\n\n".join(raw.split("\n\n")[:3])[:PUBMED_MAX_CHARS]
        time.sleep(PUBMED_SLEEP)
        return context, pmids, used_term, kws

    except Exception as exc:
        return f"PubMed retrieval error: {exc}", [], "", kws


def build_prompt(context: str, statement: str, options_text: str) -> str:
    """
    Uses RAG_PROMPT passed from main selector + appends context/question/options.
    """
    if RAG_PROMPT:
        base = RAG_PROMPT.strip()
    else:
        base = (
            "You are a medical professional answering a clinical multiple-choice question.\n"
            "Answer strictly as: 'The correct answer is number X.'\n"
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


def call_ollama(model: str, prompt: str, timeout_s: int = 180) -> str:
    payload = {"model": model, "prompt": prompt, "stream": False}
    r = requests.post(OLLAMA_URL, json=payload, timeout=timeout_s)
    r.raise_for_status()
    return (r.json().get("response", "") or "").strip()


# =============================================================================
# 3) Load data
# =============================================================================
with open(RAG_INPUT_JSON, "r", encoding="utf-8") as f:
    base_data = json.load(f)

questions_all = base_data.get("preguntas", [])
questions = [q for q in questions_all if q.get("tipo") == "texto"]
total = len(questions)

print(f"\n✅ PubMed_FINAL START")
print(f"   🏷️  Spec: {RAG_SPECIALIZATION}")
print(f"   🌐 Lang: {RAG_LANG}")
print(f"   🧠 Model: {RAG_MODEL}")
print(f"   📄 JSON: {RAG_INPUT_JSON}")
print(f"   🧾 Text questions: {total}")
print("-" * 70, flush=True)

# =============================================================================
# 4) Run loop
# =============================================================================
model_results = []
correct = wrong = no_answer = 0
t_start = time.time()

for idx, q in enumerate(questions, start=1):
    statement = q.get("enunciado", "")
    options = q.get("opciones", [])
    gt = q.get("respuesta_correcta", None)

    options_text = "\n".join(f"{i+1}. {opt}" for i, opt in enumerate(options))

    # PubMed context
    context, pmids, used_term, kws = retrieve_pubmed_context(statement)

    prompt = build_prompt(context, statement, options_text)

    # Ask model
    try:
        raw_text = call_ollama(RAG_MODEL, prompt, timeout_s=180)
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
    pmids_short = ",".join(pmids[:3]) if pmids else "-"
    kws_short = ",".join(kws) if kws else "-"

    print(
        f"{RAG_SPECIALIZATION} | {RAG_LANG} | {RAG_MODEL} | Q {idx}/{total} | {status} | "
        f"kws={kws_short} | term={used_term or '-'} | pmids={pmids_short} | {short}",
        flush=True
    )

    model_results.append(
        {
            "numero": q.get("numero"),
            "enunciado": statement,
            "opciones": options,
            "keywords": kws,
            "pubmed_term": used_term,
            "pmids": pmids,
            "context": context,
            "pred": pred,
            "gt": gt,
            "raw": raw_text,
        }
    )

# =============================================================================
# 5) Save outputs
# =============================================================================
answered = total - no_answer
acc = (correct / answered * 100.0) if answered > 0 else 0.0

stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
json_out = (
    OUTPUT_DIR
    / f"{RAG_SPECIALIZATION}_{RAG_MODEL}_pubmed_final_{RAG_LANG}_{stamp}.json"
)
xlsx_out = (
    OUTPUT_DIR
    / f"{RAG_SPECIALIZATION}_{RAG_MODEL}_pubmed_final_{RAG_LANG}_{stamp}_metrics.xlsx"
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
    "JSON": json_out.name,
}])
df_metrics.to_excel(xlsx_out, index=False)

print("-" * 70)
print(f"✅ Saved JSON   : {json_out}")
print(f"✅ Saved METRICS: {xlsx_out}")
print(f"🏁 DONE | Accuracy: {acc:.2f}% | Answered: {answered}/{total}", flush=True)
