#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RAG with Wikipedia – v2 (Revised, Publication-Ready)
Author: Lorena Ariceta García  
TFM – Data Science & Bioinformatics for Precision Medicine  

Description:
  - Processes MIR-style multiple-choice questions (`tipo == "texto"`)
  - Extracts hybrid keywords (KeyBERT + spaCy)
  - Retrieves biomedical context from Wikipedia (with fallback suggestions)
  - Queries local Ollama models (llama3, mistral, gemma)
  - Annotates each question with `found_wikipedia` flag
  - Computes metrics (Global / Con Wikipedia / Sin Wikipedia)
  - Exports results to structured JSON + Excel (3 sheets)
"""

import os
import sys
import json
import re
import time
import requests
import spacy
import wikipediaapi
import pandas as pd
from typing import List, Dict, Optional, Tuple
from keybert import KeyBERT
from sentence_transformers import SentenceTransformer
from collections import Counter
from datetime import datetime

# =============================================================================
# Configuration
# =============================================================================

MODELS = ["llama3", "mistral", "gemma"]

EXAMS_DIR = "/home/xs1/Desktop/Lorena/MEDICINA/results/1_data_preparation/6_json_final/prueba"
OUTPUT_DIR = "/home/xs1/Desktop/Lorena/MEDICINA/results/2_models/2_rag/1_wikipedia/v2"
os.makedirs(OUTPUT_DIR, exist_ok=True)

LOG_FILE = os.path.join(OUTPUT_DIR, "rag_wikipedia_v2_log.txt")

# =============================================================================
# Dual Logging (console + file)
# =============================================================================

class DualLogger:
    def __init__(self, path: str):
        self.terminal = sys.stdout
        self.log = open(path, "w", encoding="utf-8")
    def write(self, msg: str):
        self.terminal.write(msg)
        self.log.write(msg)
    def flush(self):
        self.terminal.flush()
        self.log.flush()

sys.stdout = DualLogger(LOG_FILE)

# =============================================================================
# Initialization
# =============================================================================

print(f"⏳ Starting RAG-Wikipedia v2 – {datetime.now():%Y-%m-%d %H:%M:%S}\n")
sentence_model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
kw_model = KeyBERT(model=sentence_model)
nlp = spacy.load("es_core_news_sm")
wiki = wikipediaapi.Wikipedia("es")

# =============================================================================
# Helper functions
# =============================================================================

def extract_keywords_keybert(text: str, top_n: int = 3) -> List[str]:
    try:
        return [kw[0] for kw in kw_model.extract_keywords(text, top_n=top_n)]
    except Exception:
        return []

def extract_keywords_spacy(text: str) -> List[str]:
    doc = nlp(text)
    return [t.text.lower() for t in doc if t.pos_ in ["NOUN", "PROPN"]]

def search_with_suggestion(keyword: str) -> Tuple[Optional[str], bool, Optional[str]]:
    page = wiki.page(keyword)
    if page.exists():
        return page.text, False, None
    for variant in [keyword.lower(), keyword.capitalize(), keyword.title()]:
        alt = wiki.page(variant)
        if alt.exists():
            return alt.text, True, variant
    return None, False, None

# =============================================================================
# Metric Computation
# =============================================================================

def compute_metrics_fixed(results: List[Dict], model: str) -> Dict[str, float]:
    total = len(results)
    correct = sum(1 for r in results if r.get(model) == r.get("respuesta_correcta"))
    none_cnt = sum(1 for r in results if r.get(model) is None)
    errors = total - correct - none_cnt
    found = sum(1 for r in results if r.get("found_wikipedia"))
    acc_global = (correct / total * 100) if total > 0 else 0.0
    return {
        "Modelo": model,
        "Total preguntas": total,
        "Aciertos": correct,
        "Errores": errors,
        "Sin respuesta": none_cnt,
        "Con Wikipedia": found,
        "Accuracy (%)": round(acc_global, 2)
    }

def compute_submetrics(results: List[Dict], model: str, flag: bool) -> Dict[str, float]:
    subset = [r for r in results if r.get("found_wikipedia") == flag]
    total = len(subset)
    correct = sum(1 for r in subset if r.get(model) == r.get("respuesta_correcta"))
    none_cnt = sum(1 for r in subset if r.get(model) is None)
    errors = total - correct - none_cnt
    acc = (correct / total * 100) if total > 0 else 0.0
    return {
        "Modelo": model,
        "Subset": "Con Wikipedia" if flag else "Sin Wikipedia",
        "Total preguntas": total,
        "Aciertos": correct,
        "Errores": errors,
        "Sin respuesta": none_cnt,
        "Accuracy (%)": round(acc, 2)
    }

# =============================================================================
# Processing
# =============================================================================

def process_exam(path: str):
    with open(path, "r", encoding="utf-8") as f:
        exam = json.load(f)

    titulacion = exam.get("titulacion", os.path.splitext(os.path.basename(path))[0])
    preguntas = [q for q in exam.get("preguntas", []) if q.get("tipo") == "texto"]
    year = next((q.get("convocatoria") for q in preguntas if q.get("convocatoria")), "SIN_AÑO")
    print(f"📘 Processing {titulacion} ({year}) → {len(preguntas)} texto questions")

    all_results = []

    for model in MODELS:
        print(f"🔹 Model: {model}")
        results = []

        for i, q in enumerate(preguntas, 1):
            enunciado = q["enunciado"]
            correcta = q.get("respuesta_correcta")

            kw_kb = extract_keywords_keybert(enunciado)
            kw_sp = extract_keywords_spacy(enunciado)

            contexts = []
            for kw in kw_kb:
                text, _, _ = search_with_suggestion(kw)
                if text:
                    contexts.append("\n".join(text.split("\n")[:3]))
                time.sleep(0.2)

            found_wikipedia = len(contexts) > 0
            context_text = "\n\n".join(contexts) if found_wikipedia else "Sin contexto relevante."

            options = "\n".join(f"{i+1}. {opt}" for i, opt in enumerate(q["opciones"]))
            prompt = (
                "Eres un profesional médico que debe responder una pregunta tipo MIR.\n"
                "Usa el contexto si está disponible; si no, responde con tu conocimiento médico.\n\n"
                f"📚 Contexto:\n{context_text}\n\n"
                f"❓ Pregunta:\n{enunciado}\n\n"
                f"🔢 Opciones:\n{options}\n\n"
                "Responde exactamente en formato: 'La respuesta correcta es la número X.'"
            )

            try:
                resp = requests.post(
                    "http://localhost:11434/api/generate",
                    json={"model": model, "prompt": prompt, "stream": False},
                    timeout=120
                )
                text = resp.json().get("response", "").strip()
            except Exception as e:
                text = f"❌ Error: {e}"

            match = re.search(r"\b([1-4])\b", text)
            pred = int(match.group(1)) if match else None

            results.append({
                "titulacion": titulacion,
                "año": year,
                "numero": q.get("numero"),
                "respuesta_correcta": correcta,
                "found_wikipedia": found_wikipedia,
                model: pred,
                f"{model}_texto": text
            })

            if i % 50 == 0:
                print(f"💾 Progress: {i}/{len(preguntas)} questions processed")

        all_results.append((model, results))

    return all_results, titulacion, year

# =============================================================================
# Main
# =============================================================================

def main():
    metrics_global, metrics_with, metrics_without = [], [], []

    for exam_file in os.listdir(EXAMS_DIR):
        if not exam_file.endswith(".json"):
            continue
        path = os.path.join(EXAMS_DIR, exam_file)
        model_results, titulacion, year = process_exam(path)

        for model, recs in model_results:
            out_name = f"{titulacion}_{model}_rag_wikipedia_v2.json"
            out_path = os.path.join(OUTPUT_DIR, out_name)
            with open(out_path, "w", encoding="utf-8") as f_out:
                json.dump({"preguntas": recs}, f_out, ensure_ascii=False, indent=2)
            print(f"✅ Saved JSON: {out_path}")

            m_global = compute_metrics_fixed(recs, model)
            m_with = compute_submetrics(recs, model, True)
            m_without = compute_submetrics(recs, model, False)

            m_global["Titulación"] = m_with["Titulación"] = m_without["Titulación"] = titulacion
            m_global["Año"] = m_with["Año"] = m_without["Año"] = year

            metrics_global.append(m_global)
            metrics_with.append(m_with)
            metrics_without.append(m_without)

            print(f"📊 {model.upper()} → Global: {m_global['Accuracy (%)']}% | "
                  f"Con Wikipedia: {m_with['Accuracy (%)']}% | "
                  f"Sin Wikipedia: {m_without['Accuracy (%)']}%")

    # === Export Excel ===
    df_g = pd.DataFrame(metrics_global)
    df_w = pd.DataFrame(metrics_with)
    df_n = pd.DataFrame(metrics_without)

    excel_path = os.path.join(OUTPUT_DIR, "rag_wikipedia_v2_metrics.xlsx")
    with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
        df_g.to_excel(writer, sheet_name="Global", index=False)
        df_w.to_excel(writer, sheet_name="Con Wikipedia", index=False)
        df_n.to_excel(writer, sheet_name="Sin Wikipedia", index=False)

    print("\n✅ Metrics computed and exported successfully.")
    print(f"📘 Excel saved to: {excel_path}")
    print(f"🧾 Log file: {LOG_FILE}")

if __name__ == "__main__":
    main()
