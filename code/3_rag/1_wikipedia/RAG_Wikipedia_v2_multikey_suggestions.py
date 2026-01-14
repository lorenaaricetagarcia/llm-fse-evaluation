#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Wikipedia-RAG Pipeline (v2) — Multi-keyword retrieval + suggestions (single specialization runner)

Reads env vars from RAG_main_selective:
  RAG_LANG           -> "es" or "en"
  RAG_MODEL          -> e.g., "llama3"
  RAG_SPECIALIZATION -> e.g., "MEDICINA"
  RAG_INPUT_JSON     -> full path to specialization JSON
  RAG_PROMPT         -> base prompt template text (from prompt_config.py)
  OLLAMA_URL         -> optional (default http://localhost:11434/api/generate)

Behavior:
  - Only processes tipo == "texto"
  - Extracts top_n keywords (KeyBERT), optionally spaCy nouns/proper nouns (for logs)
  - Retrieves up to 3 Wikipedia pages (with simple title fallback variants)
  - Builds final prompt = RAG_PROMPT + CONTEXT + QUESTION + OPTIONS
  - Calls Ollama, parses option 1-4
  - Prints progress: "SPEC | lang | model | Q i/total | pred/gt | kw=[...] | snippet"
  - Saves *_metrics.xlsx with timestamp so main can find latest metrics
"""

import json
import os
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Tuple

import requests
import pandas as pd
import wikipediaapi
from keybert import KeyBERT
from sentence_transformers import SentenceTransformer

# spaCy is optional: if missing model, we continue without it
try:
    import spacy  # type: ignore
    _SPACY_AVAILABLE = True
except Exception:
    spacy = None
    _SPACY_AVAILABLE = False


# ================================================================
# 0) ENV from RAG_main_selective
# ================================================================
RAG_LANG = os.environ.get("RAG_LANG", "es").strip().lower()
RAG_MODEL = os.environ.get("RAG_MODEL", "llama3").strip()
RAG_SPECIALIZATION = os.environ.get("RAG_SPECIALIZATION", "UNKNOWN").strip()
RAG_INPUT_JSON = os.environ.get("RAG_INPUT_JSON", "").strip()
RAG_PROMPT = os.environ.get("RAG_PROMPT", "").strip()
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434/api/generate").strip()

if not RAG_INPUT_JSON or not os.path.exists(RAG_INPUT_JSON):
    raise FileNotFoundError(f"RAG_INPUT_JSON not found: {RAG_INPUT_JSON}")

BASE_DIR = Path(os.getenv("FSE_BASE_DIR", Path(__file__).resolve().parents[3]))
OUTPUT_DIR = Path(
    os.getenv(
        "FSE_OUTPUT_DIR",
        BASE_DIR / "results/3_rag/1_wikipedia",
    )
)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ================================================================
# 1) Config
# ================================================================
# KeyBERT CPU
SENTENCE_MODEL = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
KW_MODEL = KeyBERT(model=SENTENCE_MODEL)

# spaCy (optional)
NLP = None
if _SPACY_AVAILABLE:
    try:
        # for Spanish; for English you could switch to en_core_web_sm if you want
        if RAG_LANG == "es":
            NLP = spacy.load("es_core_news_sm")
        elif RAG_LANG == "en":
            # optional; only if installed
            NLP = spacy.load("en_core_web_sm")
    except Exception:
        NLP = None

# Wikipedia API (requires user_agent in recent versions)
WIKI = wikipediaapi.Wikipedia(
    language=RAG_LANG,
    user_agent="LorenaTFM-RAG/1.0 (contact: lorena.ariceta@uni)"
)


# ================================================================
# 2) Helpers
# ================================================================
def get_keywords_keybert(text: str, top_n: int = 3) -> List[str]:
    kws = KW_MODEL.extract_keywords(text, top_n=top_n)
    return [k[0] for k in kws] if kws else []


def get_keywords_spacy(text: str) -> List[str]:
    if NLP is None:
        return []
    doc = NLP(text)
    return [t.text.lower() for t in doc if t.pos_ in ("NOUN", "PROPN")]


def search_wikipedia_with_fallback(keyword: str) -> Tuple[Optional[str], bool, Optional[str]]:
    """
    Returns: (content_or_none, used_fallback, suggested_title_or_none)
    """
    page = WIKI.page(keyword)
    if page.exists():
        return page.text, False, None

    variants = []
    k = keyword.strip()
    if not k:
        return None, False, None

    variants.extend([k.lower(), k.capitalize(), k.title()])
    seen = set()
    for v in variants:
        if v in seen:
            continue
        seen.add(v)
        page_alt = WIKI.page(v)
        if page_alt.exists():
            return page_alt.text, True, v

    return None, False, None


def build_prompt(context: str, statement: str, options_text: str) -> str:
    base = RAG_PROMPT.strip() if RAG_PROMPT else (
        "Answer the question using the provided context when useful.\n"
        "Answer strictly as: 'The correct answer is number X.'\n"
        "Then add a short justification."
    )
    return f"""{base}

CONTEXT:
{context}

QUESTION:
{statement}

OPTIONS:
{options_text}
"""


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

print(f"\n✅ Wikipedia_v2 START")
print(f"   🏷️  Spec: {RAG_SPECIALIZATION}")
print(f"   🌐 Lang: {RAG_LANG}")
print(f"   🧠 Model: {RAG_MODEL}")
print(f"   📄 JSON: {RAG_INPUT_JSON}")
print(f"   🧾 Text questions: {total}")
if NLP is None:
    print(f"   ⚠️ spaCy disabled (model not available). Continuing with KeyBERT only.")
print("-" * 70, flush=True)


# ================================================================
# 4) Run loop
# ================================================================
model_results = []
keywords_log = []
suggestions_log = []

no_keywords_keybert = 0
no_keywords_spacy = 0
keyword_overlap_count = 0
suggestions_used = 0

correct = wrong = no_answer = 0
t_start = time.time()

# how many KeyBERT keywords
TOP_N = 3
# context constraints
MAX_PAGES = 3
MAX_CHARS = 2000

for idx, q in enumerate(questions, start=1):
    statement = q.get("enunciado", "")
    options = q.get("opciones", [])
    gt = q.get("respuesta_correcta", None)

    keybert_kws = get_keywords_keybert(statement, top_n=TOP_N)
    spacy_kws = get_keywords_spacy(statement)

    if not keybert_kws:
        no_keywords_keybert += 1
    if NLP is not None and not spacy_kws:
        no_keywords_spacy += 1

    overlap = 0
    if keybert_kws and spacy_kws:
        overlap = len(set(map(str.lower, keybert_kws)) & set(spacy_kws))
        if overlap > 0:
            keyword_overlap_count += 1

    keywords_log.append({
        "question": statement,
        "keybert": keybert_kws,
        "spacy": spacy_kws,
        "overlap": overlap
    })

    if not keybert_kws:
        # no keywords => cannot retrieve => count as no_answer
        print(f"{RAG_SPECIALIZATION} | {RAG_LANG} | {RAG_MODEL} | Q {idx}/{total} | ❌ no keywords",
              flush=True)
        model_results.append({
            "numero": q.get("numero"),
            "enunciado": statement,
            "opciones": options,
            "keywords": [],
            "context": "",
            "pred": None,
            "gt": gt,
            "raw": "",
        })
        no_answer += 1
        continue

    # Retrieve multiple contexts
    contexts = []
    used_keywords = []
    for kw in keybert_kws:
        try:
            content, used_suggestion, suggested_title = search_wikipedia_with_fallback(kw)
            if content:
                contexts.append(content)
                used_keywords.append(suggested_title if used_suggestion and suggested_title else kw)
                if used_suggestion:
                    suggestions_used += 1
                    suggestions_log.append({"original": kw, "suggested": suggested_title})
            time.sleep(0.25)
        except Exception as e:
            # keep going
            continue

        if len(contexts) >= MAX_PAGES:
            break

    if not contexts:
        print(f"{RAG_SPECIALIZATION} | {RAG_LANG} | {RAG_MODEL} | Q {idx}/{total} | ❌ no wiki context | kw={keybert_kws}",
              flush=True)
        model_results.append({
            "numero": q.get("numero"),
            "enunciado": statement,
            "opciones": options,
            "keywords": keybert_kws,
            "context": "",
            "pred": None,
            "gt": gt,
            "raw": "",
        })
        no_answer += 1
        continue

    context_full = ("\n\n".join(contexts))[:MAX_CHARS]
    options_text = "\n".join(f"{i+1}. {opt}" for i, opt in enumerate(options))
    prompt = build_prompt(context_full, statement, options_text)

    # Call Ollama
    try:
        payload = {"model": RAG_MODEL, "prompt": prompt, "stream": False}
        r = requests.post(OLLAMA_URL, json=payload, timeout=180)
        r.raise_for_status()
        raw_text = (r.json().get("response", "") or "").strip()
    except Exception as exc:
        raw_text = f"ERROR: {exc}"

    pred = parse_prediction(raw_text)

    # Update metrics
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
    kw_show = used_keywords if used_keywords else keybert_kws

    print(
        f"{RAG_SPECIALIZATION} | {RAG_LANG} | {RAG_MODEL} | Q {idx}/{total} | {status} | kw={kw_show} | {short}",
        flush=True
    )

    model_results.append({
        "numero": q.get("numero"),
        "enunciado": statement,
        "opciones": options,
        "keywords": keybert_kws,
        "keywords_used": kw_show,
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
    / f"{RAG_SPECIALIZATION}_{RAG_MODEL}_wikipedia_v2_{RAG_LANG}_{stamp}.json"
)
xlsx_out = (
    OUTPUT_DIR
    / f"{RAG_SPECIALIZATION}_{RAG_MODEL}_wikipedia_v2_{RAG_LANG}_{stamp}_metrics.xlsx"
)
keywords_path = (
    OUTPUT_DIR
    / f"keywords_summary_v2_{RAG_SPECIALIZATION}_{RAG_MODEL}_{RAG_LANG}_{stamp}.txt"
)
suggestions_path = (
    OUTPUT_DIR
    / (
        "wikipedia_suggestions_used_v2_"
        f"{RAG_SPECIALIZATION}_{RAG_MODEL}_{RAG_LANG}_{stamp}.txt"
    )
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
    "Suggestions used": suggestions_used,
    "KeyBERT top_n": TOP_N,
    "Context max pages": MAX_PAGES,
    "Context max chars": MAX_CHARS,
}])
df_metrics.to_excel(xlsx_out, index=False)

# logs
with open(keywords_path, "w", encoding="utf-8") as f_kw:
    for i, item in enumerate(keywords_log, 1):
        f_kw.write(f"Question {i}:\n")
        f_kw.write(f"  Statement: {item['question']}\n")
        f_kw.write(f"  KeyBERT: {item['keybert']}\n")
        f_kw.write(f"  spaCy: {item['spacy']}\n")
        f_kw.write(f"  Overlap: {item['overlap']}\n\n")
    f_kw.write("=== Statistics ===\n")
    f_kw.write(f"Questions without keywords (KeyBERT): {no_keywords_keybert}\n")
    if NLP is not None:
        f_kw.write(f"Questions without keywords (spaCy): {no_keywords_spacy}\n")
        f_kw.write(f"Questions with overlap: {keyword_overlap_count}\n")
    f_kw.write(f"Total processed (text questions): {total}\n")

with open(suggestions_path, "w", encoding="utf-8") as f_sug:
    f_sug.write("=== Wikipedia suggestion fallbacks used ===\n\n")
    for item in suggestions_log:
        f_sug.write(f"Original: {item['original']} -> Suggested: {item['suggested']}\n")
    f_sug.write(f"\nTotal suggestions used: {suggestions_used}\n")

print("-" * 70)
print(f"✅ Saved JSON   : {json_out}")
print(f"✅ Saved METRICS: {xlsx_out}")
print(f"📝 Saved KW log : {keywords_path}")
print(f"📝 Saved SUG log: {suggestions_path}")
print(f"🏁 DONE | Accuracy: {acc:.2f}% | Answered: {answered}/{total}", flush=True)
