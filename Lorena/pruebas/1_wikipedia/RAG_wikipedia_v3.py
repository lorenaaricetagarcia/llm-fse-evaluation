#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RAG with Wikipedia – v3 (Revised, Publication-Ready)
Author: Lorena Ariceta García  
TFM – Data Science & Bioinformatics for Precision Medicine  

Description:
  - Processes 900 MIR-style questions (tipo == "texto")
  - Keeps all questions (with or without Wikipedia context)
  - Annotates `found_wikipedia` flag per question
  - Computes metrics for:
      • Global accuracy (all)
      • With Wikipedia context
      • Without Wikipedia context
  - Exports results to JSON + Excel (3 sheets)
"""

import os
import re
import json
import time
import requests
import spacy
import wikipediaapi
import pandas as pd
from keybert import KeyBERT
from sentence_transformers import SentenceTransformer
from collections import Counter
from datetime import datetime
import sys

# =============================================================================
# Configuration
# =============================================================================

MODELOS = ["llama3", "mistral", "gemma"]

CARPETA_EXAMENES = "/home/xs1/Desktop/Lorena/MEDICINA/results/1_data_preparation/6_json_final/prueba"
CARPETA_SALIDA = "/home/xs1/Desktop/Lorena/MEDICINA/results/2_models/2_rag/1_wikipedia/v3"
os.makedirs(CARPETA_SALIDA, exist_ok=True)

LOG_FILE = os.path.join(CARPETA_SALIDA, "rag_wikipedia_v3_log.txt")

# =============================================================================
# Dual Logger
# =============================================================================
class DualLogger:
    def __init__(self, path: str):
        self.terminal = sys.stdout
        self.log = open(path, "w", encoding="utf-8")
    def write(self, message: str):
        self.terminal.write(message)
        self.log.write(message)
    def flush(self):
        self.terminal.flush()
        self.log.flush()

sys.stdout = DualLogger(LOG_FILE)

# =============================================================================
# Initialization
# =============================================================================
print(f"⏳ Starting RAG-Wikipedia v3 – {datetime.now():%Y-%m-%d %H:%M:%S}")

sentence_model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
kw_model = KeyBERT(model=sentence_model)
nlp = spacy.load("es_core_news_sm")
wiki_api = wikipediaapi.Wikipedia(language="es")

# =============================================================================
# Helpers
# =============================================================================
def get_keywords_keybert(texto, top_n=3):
    try:
        return [kw[0] for kw in kw_model.extract_keywords(texto, top_n=top_n)]
    except Exception:
        return []

def get_keywords_spacy(texto):
    doc = nlp(texto)
    return [t.text.lower() for t in doc if t.pos_ in ["NOUN", "PROPN"]]

def buscar_con_sugerencia(keyword):
    """Busca en Wikipedia, probando variantes si no se encuentra."""
    page = wiki_api.page(keyword)
    if page.exists():
        return page.text, False, None
    for sugerida in [keyword.lower(), keyword.capitalize(), keyword.title()]:
        alt_page = wiki_api.page(sugerida)
        if alt_page.exists():
            return alt_page.text, True, sugerida
    return None, False, None

# =============================================================================
# Métricas
# =============================================================================
def compute_metrics_fixed(results, modelo):
    total = len(results)
    correct = sum(1 for r in results if r.get(modelo) == r.get("respuesta_correcta"))
    none_cnt = sum(1 for r in results if r.get(modelo) is None)
    errors = total - correct - none_cnt
    found = sum(1 for r in results if r.get("found_wikipedia"))
    acc_global = (correct / total * 100) if total > 0 else 0.0
    return {
        "Modelo": modelo,
        "Total preguntas": total,
        "Aciertos": correct,
        "Errores": errors,
        "Sin respuesta": none_cnt,
        "Con Wikipedia": found,
        "Accuracy (%)": round(acc_global, 2)
    }

def compute_submetrics(results, modelo, flag):
    subset = [r for r in results if r.get("found_wikipedia") == flag]
    total = len(subset)
    correct = sum(1 for r in subset if r.get(modelo) == r.get("respuesta_correcta"))
    none_cnt = sum(1 for r in subset if r.get(modelo) is None)
    errors = total - correct - none_cnt
    acc = (correct / total * 100) if total > 0 else 0.0
    return {
        "Modelo": modelo,
        "Subset": "Con Wikipedia" if flag else "Sin Wikipedia",
        "Total preguntas": total,
        "Aciertos": correct,
        "Errores": errors,
        "Sin respuesta": none_cnt,
        "Accuracy (%)": round(acc, 2)
    }

# =============================================================================
# Prompt Template
# =============================================================================
PROMPT_RAG = (
    "Eres un profesional médico que debe responder una pregunta tipo examen clínico (MIR).\n"
    "Lee cuidadosamente el CONTEXTO recuperado y luego la PREGUNTA.\n"
    "Si el contexto contiene información útil, utilízala; si no, aplica tu conocimiento clínico.\n"
    "Responde estrictamente en el formato: 'La respuesta correcta es la número X.'\n"
    "Después añade una breve frase justificativa.\n"
)

# =============================================================================
# Main Loop
# =============================================================================
metricas_global, metricas_conwiki, metricas_sinwiki = [], [], []
resultados_titulacion = {modelo: {} for modelo in MODELOS}
archivos_json = [f for f in os.listdir(CARPETA_EXAMENES) if f.endswith(".json")]

for archivo_json in archivos_json:
    nombre_examen = os.path.splitext(archivo_json)[0]
    titulacion = nombre_examen.split("_")[0] if "_" in nombre_examen else nombre_examen
    ruta_json = os.path.join(CARPETA_EXAMENES, archivo_json)

    with open(ruta_json, "r", encoding="utf-8") as f:
        base_data = json.load(f)

    preguntas_texto = [p for p in base_data["preguntas"] if p.get("tipo") == "texto"]
    print(f"\n📘 Procesando {titulacion} ({len(preguntas_texto)} preguntas tipo texto)")

    for modelo in MODELOS:
        print(f"\n🔹 Modelo: {modelo}")
        resultados_titulacion[modelo][titulacion] = []
        preguntas_resultado = []

        for i, pregunta in enumerate(preguntas_texto, 1):
            enunciado = pregunta["enunciado"]
            keywords_keybert = get_keywords_keybert(enunciado)
            keywords_spacy = get_keywords_spacy(enunciado)

            contextos = []
            for kw in keywords_keybert:
                try:
                    contenido, usada, sugerida = buscar_con_sugerencia(kw)
                    if contenido:
                        resumen = "\n".join(contenido.split("\n")[:3])
                        contextos.append(resumen)
                    time.sleep(0.25)
                except Exception as e:
                    print(f"⚠️ Error buscando '{kw}': {e}")
                    continue

            found_wikipedia = len(contextos) > 0
            contexto_completo = "\n\n".join(contextos) if found_wikipedia else "Sin contexto relevante encontrado."

            opciones = "\n".join(f"{idx+1}. {op}" for idx, op in enumerate(pregunta["opciones"]))
            prompt = f"{PROMPT_RAG}\nCONTEXTO:\n{contexto_completo}\n\nPREGUNTA:\n{enunciado}\n\nOPCIONES:\n{opciones}\n"

            try:
                payload = {"model": modelo, "prompt": prompt, "stream": False}
                response = requests.post("http://localhost:11434/api/generate", json=payload, timeout=180)
                texto = response.json().get("response", "").strip()
            except Exception as e:
                texto = f"❌ Error: {e}"

            match = re.search(r"\b([1-4])\b", texto)
            seleccion = int(match.group(1)) if match else None

            preguntas_resultado.append({
                "numero": pregunta.get("numero"),
                "enunciado": enunciado,
                "opciones": pregunta.get("opciones"),
                "respuesta_correcta": pregunta.get("respuesta_correcta"),
                "found_wikipedia": found_wikipedia,
                modelo: seleccion,
                f"{modelo}_texto": texto
            })

            if i % 50 == 0:
                print(f"💾 Progreso: {i}/{len(preguntas_texto)} procesadas")

        resultados_titulacion[modelo][titulacion].extend(preguntas_resultado)

        # === Métricas ===
        m_global = compute_metrics_fixed(preguntas_resultado, modelo)
        m_con = compute_submetrics(preguntas_resultado, modelo, True)
        m_sin = compute_submetrics(preguntas_resultado, modelo, False)

        m_global["Titulación"] = m_con["Titulación"] = m_sin["Titulación"] = titulacion

        metricas_global.append(m_global)
        metricas_conwiki.append(m_con)
        metricas_sinwiki.append(m_sin)

        print(f"📊 {modelo.upper()} → Global: {m_global['Accuracy (%)']}% | "
              f"Con Wikipedia: {m_con['Accuracy (%)']}% | "
              f"Sin Wikipedia: {m_sin['Accuracy (%)']}%")

# =============================================================================
# Save Outputs
# =============================================================================
for modelo in MODELOS:
    for titulacion, preguntas in resultados_titulacion[modelo].items():
        salida_json = os.path.join(CARPETA_SALIDA, f"{titulacion}_{modelo}_rag_wikipedia_v3.json")
        with open(salida_json, "w", encoding="utf-8") as f_out:
            json.dump({"preguntas": preguntas}, f_out, ensure_ascii=False, indent=2)
        print(f"💾 Guardado JSON: {salida_json}")

# === Export Excel ===
df_g = pd.DataFrame(metricas_global)
df_w = pd.DataFrame(metricas_conwiki)
df_n = pd.DataFrame(metricas_sinwiki)
excel_path = os.path.join(CARPETA_SALIDA, "rag_wikipedia_v3_metrics.xlsx")
with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
    df_g.to_excel(writer, sheet_name="Global", index=False)
    df_w.to_excel(writer, sheet_name="Con Wikipedia", index=False)
    df_n.to_excel(writer, sheet_name="Sin Wikipedia", index=False)

print(f"\n✅ Métricas exportadas correctamente en: {excel_path}")
print(f"🧾 Log completo: {LOG_FILE}")
print("\n🏁 Pipeline RAG-Wikipedia v3 completado correctamente.")
