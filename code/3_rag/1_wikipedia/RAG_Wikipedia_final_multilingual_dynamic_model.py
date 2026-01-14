#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Wikipedia-RAG Pipeline (FINAL) — Single specialization runner (for main selector)

Reads env vars:
  RAG_LANG           -> "es" or "en"
  RAG_MODEL          -> e.g., "llama3"
  RAG_SPECIALIZATION -> e.g., "MEDICINA"
  RAG_INPUT_JSON     -> full path to the specialization JSON
  RAG_PROMPT         -> base prompt text from prompt_config.py (passed by main)

Behavior:
  - For each question (tipo == "texto"):
    1) Extract top keywords (KeyBERT) (multi-keyword)
    2) Retrieve Wikipedia contexts for several keywords (REST summary endpoint)
    3) Build final prompt = BASE_PROMPT + CONTEXT + QUESTION + OPTIONS
    4) Ask Ollama
    5) Parse predicted option 1-4
    6) Print progress "SPEC | lang | model | Q i/total | pred/gt | ..."
  - Saves JSON + *_metrics.xlsx (so main selector can find it)
"""

import json
import os
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Tuple
from urllib.parse import quote

import requests
import pandas as pd
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

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434/api/generate").strip()

if not RAG_INPUT_JSON or not os.path.exists(RAG_INPUT_JSON):
    raise FileNotFoundError(f"RAG_INPUT_JSON not found: {RAG_INPUT_JSON}")

if RAG_LANG not in ("es", "en"):
    RAG_LANG = "es"


# ================================================================
# 1) Paths
# ================================================================
BASE_DIR = Path(os.getenv("FSE_BASE_DIR", Path(__file__).resolve().parents[3]))
OUTPUT_DIR = Path(
    os.getenv(
        "FSE_OUTPUT_DIR",
        BASE_DIR / "results/3_rag/1_wikipedia",
    )
)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ================================================================
# 2) Models: KeyBERT CPU
# ================================================================
SENTENCE_MODEL = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
KW_MODEL = KeyBERT(model=SENTENCE_MODEL)


# ================================================================
# 3) Helpers
# ================================================================

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

def get_keywords_keybert(text: str, top_n: int = 3) -> List[str]:
    try:
        kws = KW_MODEL.extract_keywords(text, top_n=top_n)
        return [k[0] for k in kws if k and k[0]]
    except Exception:
        return []


def wiki_summary_rest(title: str, lang: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Use Wikipedia REST summary endpoint:
      https://{lang}.wikipedia.org/api/rest_v1/page/summary/{title}
    Returns (extract, used_title)
    """
    if not title:
        return None, None

    lang_code = "es" if lang == "es" else "en"
    safe_title = quote(title.replace(" ", "_"), safe=":_()-%")
    url = f"https://{lang_code}.wikipedia.org/api/rest_v1/page/summary/{safe_title}"

    try:
        r = requests.get(url, timeout=15, headers={"User-Agent": "LorenaTFM-RAG/1.0"})
        if r.status_code != 200:
            return None, None
        data = r.json()
        extract = (data.get("extract") or "").strip()
        if not extract:
            return None, None
        return extract, title
    except Exception:
        return None, None


def retrieve_multi_context(statement: str, lang: str, top_n: int = 3, max_chars: int = 2000) -> Tuple[str, List[str]]:
    """
    Get context from up to top_n keywords, concatenated and truncated.
    Returns (context, used_keywords)
    """
    kws = get_keywords_keybert(statement, top_n=top_n)
    if (lang or '').strip().lower() == 'en':
        kws = translate_keywords_es_en(kws, top_n=top_n)

    contexts = []
    used = []

    for kw in kws:
        extract, used_title = wiki_summary_rest(kw, lang)
        if extract:
            contexts.append(f"[{kw}]\n{extract}")
            used.append(kw)
        time.sleep(0.2)

    if not contexts:
        return "", []

    full = "\n\n".join(contexts)
    return full[:max_chars], used


def build_prompt(context: str, statement: str, options_text: str) -> str:
    base = RAG_PROMPT.strip() if RAG_PROMPT else (
        "Answer using the provided context when useful.\n"
        "Answer strictly as: 'La respuesta correcta es la número X.' (X=1..4).\n"
        "Then add a short justification."
    )

    return f"""{base}

📚 CONTEXTO (Wikipedia):
{context}

❓ PREGUNTA:
{statement}

🔢 OPCIONES:
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


# ================================================================
# 4) Load JSON
# ================================================================
with open(RAG_INPUT_JSON, "r", encoding="utf-8") as f:
    data = json.load(f)

all_q = data.get("preguntas", [])
questions = [q for q in all_q if q.get("tipo") == "texto"]
total = len(questions)

print(f"\n✅ Wikipedia_final START")
print(f"   🏷️  Spec : {RAG_SPECIALIZATION}")
print(f"   🌐 Lang : {RAG_LANG}")
print(f"   🧠 Model: {RAG_MODEL}")
print(f"   📄 JSON : {RAG_INPUT_JSON}")
print(f"   🧾 Text questions: {total}")
print("-" * 90, flush=True)


# ================================================================
# 5) Run loop
# ================================================================
t0 = time.time()
model_results = []
correct = wrong = no_answer = 0

for idx, q in enumerate(questions, start=1):
    statement = (q.get("enunciado") or "").strip()
    options = q.get("opciones", []) or []
    gt = q.get("respuesta_correcta", None)

    options_text = "\n".join(f"{i+1}. {opt}" for i, opt in enumerate(options))

    context, used_kws = retrieve_multi_context(statement, RAG_LANG, top_n=3, max_chars=2000)
    if not context:
        no_answer += 1
        print(f"{RAG_SPECIALIZATION} | {RAG_LANG} | {RAG_MODEL} | Q {idx}/{total} | ❌ no context",
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
        continue

    prompt = build_prompt(context, statement, options_text)

    try:
        raw_text = call_ollama(RAG_MODEL, prompt, timeout_s=180)
    except Exception as exc:
        raw_text = f"ERROR: {exc}"

    pred = parse_prediction(raw_text)

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
        f"{RAG_SPECIALIZATION} | {RAG_LANG} | {RAG_MODEL} | Q {idx}/{total} | {status} | kws={used_kws} | {short}",
        flush=True
    )

    model_results.append({
        "numero": q.get("numero"),
        "enunciado": statement,
        "opciones": options,
        "keywords": used_kws,
        "context": context,
        "pred": pred,
        "gt": gt,
        "raw": raw_text,
    })


# ================================================================
# 6) Save outputs
# ================================================================
answered = total - no_answer
acc = (correct / answered * 100.0) if answered > 0 else 0.0
stamp = datetime.now().strftime("%Y%m%d_%H%M%S")

json_out = (
    OUTPUT_DIR
    / f"{RAG_SPECIALIZATION}_{RAG_MODEL}_wikipedia_final_{RAG_LANG}_{stamp}.json"
)
xlsx_out = (
    OUTPUT_DIR
    / f"{RAG_SPECIALIZATION}_{RAG_MODEL}_wikipedia_final_{RAG_LANG}_{stamp}_metrics.xlsx"
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

print("-" * 90)
print(f"✅ Saved JSON   : {json_out}")
print(f"✅ Saved METRICS: {xlsx_out}")
print(f"🏁 DONE | Accuracy: {acc:.2f}% | Answered: {answered}/{total}", flush=True)
