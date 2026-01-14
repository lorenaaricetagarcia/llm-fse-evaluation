#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Wikipedia-RAG Pipeline (v3) — Single specialization runner (for main selector)

- Reads env vars:
  RAG_LANG           -> "es" or "en"
  RAG_MODEL          -> e.g., "llama3"
  RAG_SPECIALIZATION -> e.g., "MEDICINA"
  RAG_INPUT_JSON     -> full path to the specialization JSON
  RAG_PROMPT         -> base prompt template text coming from prompt_config.py

v3 behavior (close to your original v3 intent):
- Extract top-N keywords (KeyBERT)
- Also extract spaCy NOUN/PROPN keywords (for logs)
- Retrieve Wikipedia content per KeyBERT keyword with casing fallbacks
- Build context from FIRST N LINES of each retrieved page (default: 3 lines)
- Build final prompt = BASE_PROMPT (from env) + context + question + options
- Ask Ollama (local)
- Parse predicted option 1–4
- Save per-run JSON + *_metrics.xlsx (timestamped)

Console output:
- Prints progress: "SPEC | lang | model | Q i/total | status | kws | snippet"
"""

import json
import os
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple, List

import requests
import pandas as pd
import wikipediaapi
import spacy
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
BASE_DIR = Path(os.getenv("FSE_BASE_DIR", Path(__file__).resolve().parents[3]))
OUTPUT_DIR = Path(
    os.getenv(
        "FSE_OUTPUT_DIR",
        BASE_DIR / "results/3_rag/1_wikipedia",
    )
)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# KeyBERT CPU
SENTENCE_MODEL = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
KW_MODEL = KeyBERT(model=SENTENCE_MODEL)

# spaCy model (language-aware)
# - For EN you might want en_core_web_sm; but your original v3 used Spanish.
SPACY_MODEL = os.environ.get("RAG_SPACY_MODEL", "es_core_news_sm").strip()
nlp = spacy.load(SPACY_MODEL)

# Wikipedia API (user_agent required in newer versions)
WIKI = wikipediaapi.Wikipedia(
    language=RAG_LANG,  # "es" or "en"
    user_agent="LorenaTFM-RAG/1.0 (contact: lorena.ariceta@uni)"
)

# v3 parameters (can be overridden by env)
KEYBERT_TOP_N = int(os.environ.get("RAG_KEYBERT_TOP_N", "3"))
WIKI_SLEEP_SECONDS = float(os.environ.get("RAG_WIKI_SLEEP", "0.3"))
WIKI_LINES_PER_ARTICLE = int(os.environ.get("RAG_WIKI_LINES", "3"))
MAX_CONTEXT_CHARS = int(os.environ.get("RAG_MAX_CONTEXT_CHARS", "4000"))


# ================================================================
# 2) Helpers
# ================================================================
def get_keywords_keybert(text: str, top_n: int) -> List[str]:
    kws = KW_MODEL.extract_keywords(text, top_n=top_n)
    return [k[0] for k in kws] if kws else []


def get_keywords_spacy(text: str) -> List[str]:
    doc = nlp(text)
    return [t.text.lower() for t in doc if t.pos_ in ("NOUN", "PROPN")]


def search_wikipedia_with_fallback(keyword: str) -> Tuple[Optional[str], bool, Optional[str]]:
    """
    Returns: (content_text_or_none, used_fallback, suggested_title_or_none)
    """
    page = WIKI.page(keyword)
    if page.exists():
        return page.text, False, None

    variants = [keyword.lower(), keyword.capitalize(), keyword.title()]
    for suggested in variants:
        page_alt = WIKI.page(suggested)
        if page_alt.exists():
            return page_alt.text, True, suggested

    return None, False, None


def take_first_lines(text: str, n_lines: int) -> str:
    lines = [ln.strip() for ln in text.split("\n") if ln.strip()]
    return "\n".join(lines[:n_lines])


def build_prompt(context: str, statement: str, options_text: str) -> str:
    """
    Uses RAG_PROMPT from main and appends context/question/options.
    Falls back to a minimal prompt if not provided.
    """
    if RAG_PROMPT:
        base = RAG_PROMPT.strip()
    else:
        base = (
            "You are a medical professional answering a MIR-style question.\n"
            "Use the CONTEXT if helpful.\n"
            "Answer strictly: 'The correct answer is number X.' and one short justification."
        )

    return f"""{base}

CONTEXT (Wikipedia):
{context}

QUESTION:
{statement}

OPTIONS:
{options_text}
"""


def call_ollama(prompt: str, timeout_s: int = 180) -> str:
    payload = {"model": RAG_MODEL, "prompt": prompt, "stream": False}
    r = requests.post(OLLAMA_URL, json=payload, timeout=timeout_s)
    r.raise_for_status()
    return (r.json().get("response", "") or "").strip()


def parse_prediction(raw_text: str) -> Optional[int]:
    m = re.search(r"\b([1-4])\b", raw_text)
    return int(m.group(1)) if m else None


# ================================================================
# 3) Load data
# ================================================================
with open(RAG_INPUT_JSON, "r", encoding="utf-8") as f:
    base_data = json.load(f)

questions_all = base_data.get("preguntas", [])
questions = [q for q in questions_all if q.get("tipo") == "texto"]
total = len(questions)

print(f"\n✅ Wikipedia_v3 START", flush=True)
print(f"   🏷️  Spec : {RAG_SPECIALIZATION}", flush=True)
print(f"   🌐 Lang : {RAG_LANG}", flush=True)
print(f"   🧠 Model: {RAG_MODEL}", flush=True)
print(f"   📄 JSON : {RAG_INPUT_JSON}", flush=True)
print(f"   🧾 Text questions: {total}", flush=True)
print("-" * 80, flush=True)


# ================================================================
# 4) Run loop
# ================================================================
model_results = []

# v3 extra logs (saved inside JSON to keep single-run outputs)
keywords_log = []
suggestions_log = []
stats = {
    "no_keywords_keybert": 0,
    "no_keywords_spacy": 0,
    "questions_with_overlap": 0,
    "suggestions_used": 0,
}

correct = wrong = no_answer = 0
t_start = time.time()

for idx, q in enumerate(questions, start=1):
    statement = (q.get("enunciado", "") or "").strip()
    options = q.get("opciones", []) or []
    gt = q.get("respuesta_correcta", None)

    # keywords
    kb_kws = get_keywords_keybert(statement, top_n=KEYBERT_TOP_N)
    sp_kws = get_keywords_spacy(statement)

    if not kb_kws:
        stats["no_keywords_keybert"] += 1
    if not sp_kws:
        stats["no_keywords_spacy"] += 1

    overlap = len(set(map(str.lower, kb_kws)) & set(sp_kws))
    if overlap > 0:
        stats["questions_with_overlap"] += 1

    keywords_log.append({
        "q_index": idx,
        "numero": q.get("numero"),
        "keybert": kb_kws,
        "spacy": sp_kws,
        "overlap": overlap,
    })

    # Wikipedia contexts from KeyBERT kws
    contexts = []
    used_titles = []
    for kw in kb_kws:
        try:
            content, used_fallback, suggested = search_wikipedia_with_fallback(kw)
            if content:
                short = take_first_lines(content, WIKI_LINES_PER_ARTICLE)
                contexts.append(short)
                used_titles.append(suggested if used_fallback else kw)
                if used_fallback:
                    stats["suggestions_used"] += 1
                    suggestions_log.append({"q_index": idx, "original": kw, "suggested": suggested})
            time.sleep(WIKI_SLEEP_SECONDS)
        except Exception as e:
            # don’t crash a whole run for one keyword
            suggestions_log.append({"q_index": idx, "original": kw, "error": str(e)})

    if not contexts:
        print(
            f"{RAG_SPECIALIZATION} | {RAG_LANG} | {RAG_MODEL} | Q {idx}/{total} | ❌ no context | kws={kb_kws}",
            flush=True
        )
        model_results.append({
            "numero": q.get("numero"),
            "enunciado": statement,
            "opciones": options,
            "keybert_kws": kb_kws,
            "spacy_kws": sp_kws,
            "context": "",
            "pred": None,
            "gt": gt,
            "raw": "",
        })
        no_answer += 1
        continue

    context_full = "\n\n".join(contexts)
    if len(context_full) > MAX_CONTEXT_CHARS:
        context_full = context_full[:MAX_CONTEXT_CHARS]

    options_text = "\n".join(f"{i+1}. {opt}" for i, opt in enumerate(options))
    prompt = build_prompt(context_full, statement, options_text)

    # Call Ollama
    try:
        raw_text = call_ollama(prompt)
    except Exception as exc:
        raw_text = f"ERROR: {exc}"

    pred = parse_prediction(raw_text)

    # metrics counters
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

    # progress print
    short_out = raw_text.replace("\n", " ")
    short_out = (short_out[:90] + "...") if len(short_out) > 90 else short_out
    kws_show = ", ".join(used_titles[:KEYBERT_TOP_N]) if used_titles else ", ".join(kb_kws)

    print(
        f"{RAG_SPECIALIZATION} | {RAG_LANG} | {RAG_MODEL} | Q {idx}/{total} | {status} | kws={kws_show} | {short_out}",
        flush=True
    )

    model_results.append({
        "numero": q.get("numero"),
        "enunciado": statement,
        "opciones": options,
        "keybert_kws": kb_kws,
        "spacy_kws": sp_kws,
        "context": context_full,
        "pred": pred,
        "gt": gt,
        "raw": raw_text,
    })


# ================================================================
# 5) Save outputs
# ================================================================
answered = total - no_answer
acc = (correct / answered * 100.0) if answered > 0 else 0.0

stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
json_out = (
    OUTPUT_DIR
    / f"{RAG_SPECIALIZATION}_{RAG_MODEL}_wikipedia_v3_{RAG_LANG}_{stamp}.json"
)
xlsx_out = (
    OUTPUT_DIR
    / f"{RAG_SPECIALIZATION}_{RAG_MODEL}_wikipedia_v3_{RAG_LANG}_{stamp}_metrics.xlsx"
)

with open(json_out, "w", encoding="utf-8") as f:
    json.dump(
        {
            "meta": {
                "specialization": RAG_SPECIALIZATION,
                "lang": RAG_LANG,
                "model": RAG_MODEL,
                "input_json": RAG_INPUT_JSON,
                "keybert_top_n": KEYBERT_TOP_N,
                "wiki_lines_per_article": WIKI_LINES_PER_ARTICLE,
                "max_context_chars": MAX_CONTEXT_CHARS,
                "spacy_model": SPACY_MODEL,
            },
            "stats": stats,
            "keywords_log": keywords_log,
            "suggestions_log": suggestions_log,
            "preguntas": model_results,
        },
        f,
        ensure_ascii=False,
        indent=2
    )

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
    "No keywords (KeyBERT)": stats["no_keywords_keybert"],
    "No keywords (spaCy)": stats["no_keywords_spacy"],
    "Overlap questions": stats["questions_with_overlap"],
    "Wiki suggestions used": stats["suggestions_used"],
}])
df_metrics.to_excel(xlsx_out, index=False)

print("-" * 80, flush=True)
print(f"✅ Saved JSON   : {json_out}", flush=True)
print(f"✅ Saved METRICS: {xlsx_out}", flush=True)
print(f"🏁 DONE | Accuracy: {acc:.2f}% | Answered: {answered}/{total}", flush=True)
