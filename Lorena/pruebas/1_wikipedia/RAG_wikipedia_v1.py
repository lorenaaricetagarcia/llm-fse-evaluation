#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RAG with Wikipedia – v1 (Revised, Publication-Ready)
Author: Lorena Ariceta García  
TFM – Data Science & Bioinformatics for Precision Medicine  

Description:
  - Processes MIR-style multiple-choice questions (`tipo == "texto"`)
  - Retrieves biomedical context from Wikipedia (via KeyBERT → WikipediaAPI)
  - Queries local Ollama models (llama3, mistral, gemma)
  - Stores only numeric predictions (1–4) and Wikipedia retrieval flag
  - Computes and exports metrics (Global / With Wikipedia / Without Wikipedia)
  - Produces Excel report with 3 sheets and detailed logging
"""

import os
import sys
import json
import re
import time
import requests
import wikipediaapi
import pandas as pd
from typing import Optional, Dict, List
from keybert import KeyBERT
from sentence_transformers import SentenceTransformer
from collections import Counter

# =============================================================================
# Configuration
# =============================================================================

MODELS = ["llama3", "mistral", "gemma"]

EXAMS_DIR = "/home/xs1/Desktop/Lorena/results/1_data_preparation/6_json_final/prueba"
OUTPUT_DIR = "/home/xs1/Desktop/Lorena/MEDICINA/results/2_models/2_rag/1_wikipedia/v1"
os.makedirs(OUTPUT_DIR, exist_ok=True)

LOG_FILE = os.path.join(OUTPUT_DIR, "rag_wikipedia_v1_log.txt")

# =============================================================================
# Dual Logging (console + file)
# =============================================================================

class DualLogger:
    """Log all printed output to both terminal and file (for reproducibility)."""
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

print("⏳ Loading models and Wikipedia API...")
start_time = time.time()
sentence_model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
kw_model = KeyBERT(model=sentence_model)
wiki = wikipediaapi.Wikipedia("es")

# =============================================================================
# Utility Functions
# =============================================================================

def extract_keyword(text: str) -> Optional[str]:
    """Return the top keyword via KeyBERT, or None if no keyword extracted."""
    try:
        kws = kw_model.extract_keywords(text, top_n=1)
        return kws[0][0] if kws else None
    except Exception:
        return None

# =============================================================================
# Metric Computation
# =============================================================================

def compute_metrics_fixed(results: List[Dict], model: str) -> Dict[str, float]:
    """Compute evaluation metrics correctly aligned to 'texto' subset."""
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
    """Compute accuracy for subset with/without Wikipedia."""
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
# Core Processing
# =============================================================================

def process_exam(exam_path: str) -> List[Dict]:
    """Process a single exam and return per-model predictions."""
    with open(exam_path, "r", encoding="utf-8") as f:
        exam = json.load(f)

    titulacion = exam.get("titulacion", os.path.splitext(os.path.basename(exam_path))[0])
    preguntas = exam.get("preguntas", [])
    year = next((q.get("convocatoria") for q in preguntas if q.get("convocatoria")), "SIN_AÑO")

    texto_qs = [q for q in preguntas if q.get("tipo") == "texto"]
    total = len(texto_qs)
    print(f"\n📘 Processing {titulacion} ({year}) → {total} texto questions")

    results_all = []
    for model in MODELS:
        print(f"   🔹 Model: {model}")
        results = []

        for idx, q in enumerate(texto_qs, 1):
            keyword = extract_keyword(q["enunciado"])
            found = False
            context = ""
            if keyword:
                page = wiki.page(keyword)
                found = page.exists()
                context = page.summary.strip()[:1500] if found else ""

            options = "\n".join(f"{i+1}. {opt}" for i, opt in enumerate(q["opciones"]))
            prompt = (
                "Eres un profesional médico que debe responder una pregunta tipo MIR.\n"
                "Si el contexto de Wikipedia es relevante, úsalo.\n\n"
                f"Contexto: {context}\n"
                f"Pregunta: {q['enunciado']}\n"
                f"Opciones:\n{options}\n\n"
                "Responde exactamente: 'La respuesta correcta es la número X.'"
            )

            try:
                resp = requests.post("http://localhost:11434/api/generate",
                                     json={"model": model, "prompt": prompt, "stream": False},
                                     timeout=90)
                text = resp.json().get("response", "").strip()
            except Exception as e:
                print(f"❌ Error on question {idx}: {e}")
                text = None

            match = re.search(r"\b([1-4])\b", text or "")
            pred = int(match.group(1)) if match else None

            results.append({
                "titulacion": titulacion,
                "año": year,
                "numero": q.get("numero"),
                "respuesta_correcta": q.get("respuesta_correcta"),
                "found_wikipedia": found,
                model: pred,
                f"{model}_texto": text
            })

            if idx % 50 == 0:
                print(f"      💾 Progress: {idx}/{total} questions processed")

        results_all.append((model, results))

    return results_all, titulacion, year

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
            out_name = f"{titulacion}_{model}_rag_wikipedia_v1.json"
            out_path = os.path.join(OUTPUT_DIR, out_name)
            with open(out_path, "w", encoding="utf-8") as fw:
                json.dump({"preguntas": recs}, fw, ensure_ascii=False, indent=2)
            print(f"   ✅ Saved JSON: {out_path}")

            # Global + subsets
            met_g = compute_metrics_fixed(recs, model)
            met_w = compute_submetrics(recs, model, True)
            met_n = compute_submetrics(recs, model, False)

            met_g["Titulación"] = met_w["Titulación"] = met_n["Titulación"] = titulacion
            met_g["Año"] = met_w["Año"] = met_n["Año"] = year

            metrics_global.append(met_g)
            metrics_with.append(met_w)
            metrics_without.append(met_n)

            print(f"📊 {model.upper()} → Global Accuracy: {met_g['Accuracy (%)']}% | "
                  f"With Wikipedia: {met_w['Accuracy (%)']}% | Without: {met_n['Accuracy (%)']}%")

    # Export Excel
    df_g = pd.DataFrame(metrics_global)
    df_w = pd.DataFrame(metrics_with)
    df_n = pd.DataFrame(metrics_without)

    excel_path = os.path.join(OUTPUT_DIR, "rag_wikipedia_v1_metrics.xlsx")
    with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
        df_g.to_excel(writer, sheet_name="Global", index=False)
        df_w.to_excel(writer, sheet_name="Con Wikipedia", index=False)
        df_n.to_excel(writer, sheet_name="Sin Wikipedia", index=False)

    print("\n✅ Metrics successfully computed and exported.")
    print(f"📘 Excel saved to: {excel_path}")

    duration = (time.time() - start_time) / 60
    print(f"🕒 Execution time: {duration:.2f} minutes")
    print(f"🧾 Log file: {LOG_FILE}")

# =============================================================================
# Entry Point
# =============================================================================

if __name__ == "__main__":
    main()
