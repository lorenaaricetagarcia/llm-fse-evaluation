#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Wikipedia-RAG Pipeline (v1) — Single specialization runner (for main selector)

- Reads env vars:
  RAG_LANG           -> "es" or "en"
  RAG_MODEL          -> e.g., "llama3"
  RAG_SPECIALIZATION -> e.g., "MEDICINA"
  RAG_INPUT_JSON     -> full path to the specialization JSON
  RAG_PROMPT         -> base prompt template text coming from prompt_config.py

- For each question (tipo == "texto"):
  1) Extract keyword (KeyBERT)
  2) Retrieve Wikipedia summary
  3) Build final prompt = BASE_PROMPT + context + question + options
  4) Ask Ollama
  5) Parse predicted option 1-4
  6) Save per-run metrics Excel named *_metrics.xlsx so your main can find it
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import re
import time
from datetime import datetime
from typing import Optional

import requests
import pandas as pd
import wikipediaapi
from keybert import KeyBERT
from sentence_transformers import SentenceTransformer


# ================================================================
# 0) ENV from RAG_main_selective
# ================================================================
RAG_LANG = os.environ.get("RAG_LANG", "es").strip().lower()
RAG_MODEL = os.environ.get("RAG_MODEL", "llama3").strip()
RAG_SPECIALIZATION = os.environ.get("RAG_SPECIALIZATION", "UNKNOWN").strip()
RAG_INPUT_JSON = os.environ.get("RAG_INPUT_JSON", "").strip()

# Prompt base (lo pasa el main)
RAG_PROMPT = os.environ.get("RAG_PROMPT", "").strip()

if not RAG_INPUT_JSON or not os.path.exists(RAG_INPUT_JSON):
    raise FileNotFoundError(f"RAG_INPUT_JSON not found: {RAG_INPUT_JSON}")

# ================================================================
# 1) Config
# ================================================================
# KeyBERT en CPU
SENTENCE_MODEL = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
KW_MODEL = KeyBERT(model=SENTENCE_MODEL)

# Wikipedia (obligatorio user_agent en versiones nuevas)
WIKI = wikipediaapi.Wikipedia(
    language=RAG_LANG,  # es/en
    user_agent="LorenaTFM-RAG/1.0 (contact: lorena.ariceta@uni)"
)

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434/api/generate")

BASE_DIR = "/home/xs1/Desktop/Lorena"
OUTPUT_DIR = f"{BASE_DIR}/results/3_rag/1_wikipedia"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ================================================================
# 2) Helpers
# ================================================================
def extract_keyword(text: str) -> Optional[str]:
    keywords = KW_MODEL.extract_keywords(text, top_n=1)
    if keywords:
        return keywords[0][0]
    return None


def build_prompt(context: str, statement: str, options_text: str) -> str:
    """
    Usa el prompt base que viene del main (RAG_PROMPT) y le añade contexto+pregunta+opciones.
    Si no llega RAG_PROMPT, usa uno mínimo.
    """
    if RAG_PROMPT:
        base = RAG_PROMPT.strip()
    else:
        base = (
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

# solo texto
questions = [q for q in questions_all if q.get("tipo") == "texto"]
total = len(questions)

print(f"\n✅ Wikipedia_v1 START")
print(f"   🏷️  Spec: {RAG_SPECIALIZATION}")
print(f"   🌐 Lang: {RAG_LANG}")
print(f"   🧠 Model: {RAG_MODEL}")
print(f"   📄 JSON: {RAG_INPUT_JSON}")
print(f"   🧾 Text questions: {total}")
print("-" * 70, flush=True)

# ================================================================
# 4) Run loop
# ================================================================
model_results = []
correct = wrong = no_answer = 0

t_start = time.time()

for idx, q in enumerate(questions, start=1):
    statement = q.get("enunciado", "")
    options = q.get("opciones", [])
    gt = q.get("respuesta_correcta", None)

    # Keyword
    keyword = extract_keyword(statement)
    if not keyword:
        print(f"{RAG_SPECIALIZATION} | {RAG_LANG} | {RAG_MODEL} | Q {idx}/{total} | ❌ no keyword",
              flush=True)
        model_results.append({"numero": q.get("numero"), "pred": None, "raw": "", "keyword": None})
        no_answer += 1
        continue

    # Wikipedia context
    page = WIKI.page(keyword)
    if not page.exists():
        print(f"{RAG_SPECIALIZATION} | {RAG_LANG} | {RAG_MODEL} | Q {idx}/{total} | ❌ no wiki page: {keyword}",
              flush=True)
        model_results.append({"numero": q.get("numero"), "pred": None, "raw": "", "keyword": keyword})
        no_answer += 1
        continue

    context = (page.summary or "")[:1500]

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

    # Update metrics counters
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

    # Progress print
    short = raw_text.replace("\n", " ")
    short = (short[:90] + "...") if len(short) > 90 else short

    print(
        f"{RAG_SPECIALIZATION} | {RAG_LANG} | {RAG_MODEL} | Q {idx}/{total} | {status} | {keyword} | {short}",
        flush=True
    )

    model_results.append(
        {
            "numero": q.get("numero"),
            "enunciado": statement,
            "opciones": options,
            "keyword": keyword,
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
    f"{RAG_SPECIALIZATION}_{RAG_MODEL}_wikipedia_v1_{RAG_LANG}_{stamp}.json"
)
xlsx_out = os.path.join(
    OUTPUT_DIR,
    f"{RAG_SPECIALIZATION}_{RAG_MODEL}_wikipedia_v1_{RAG_LANG}_{stamp}_metrics.xlsx"
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

print("-" * 70)
print(f"✅ Saved JSON   : {json_out}")
print(f"✅ Saved METRICS: {xlsx_out}")
print(f"🏁 DONE | Accuracy: {acc:.2f}% | Answered: {answered}/{total}", flush=True)
