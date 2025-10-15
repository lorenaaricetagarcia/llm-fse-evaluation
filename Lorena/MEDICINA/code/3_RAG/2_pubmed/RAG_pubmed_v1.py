#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RAG with PubMed – v1 (Revised, Publication-Ready)
Author: Lorena Ariceta García  
TFM – Data Science & Bioinformatics for Precision Medicine  

Description:
  - Retrieves biomedical context from PubMed abstracts via E-utilities
  - Annotates each question with `found_pubmed`
  - Evaluates local LLMs (llama3, mistral, gemma)
  - Computes metrics (Global / With PubMed / Without PubMed)
  - Exports JSONs + Excel (3 sheets) with detailed reproducible logs
"""

import os
import json
import requests
import re
import time
import spacy
import xml.etree.ElementTree as ET
import pandas as pd
from keybert import KeyBERT
from sentence_transformers import SentenceTransformer
from datetime import datetime
import sys

# =============================================================================
# ⚙️ CONFIGURACIÓN
# =============================================================================

MODELOS = ["llama3", "mistral", "gemma"]

CARPETA_EXAMENES = "/home/xs1/Desktop/Lorena/MEDICINA/results/1_data_preparation/6_json_final/prueba"
CARPETA_SALIDA = "/home/xs1/Desktop/Lorena/MEDICINA/results/2_models/2_rag/2_pubmed/v1"
os.makedirs(CARPETA_SALIDA, exist_ok=True)

LOG_FILE = os.path.join(CARPETA_SALIDA, "rag_pubmed_v1_log.txt")

# =============================================================================
# 🧾 LOG DUAL (pantalla + archivo)
# =============================================================================

class DualLogger:
    def __init__(self, path):
        self.terminal = sys.stdout
        self.log = open(path, "w", encoding="utf-8")
    def write(self, msg):
        self.terminal.write(msg)
        self.log.write(msg)
    def flush(self):
        self.terminal.flush()
        self.log.flush()

sys.stdout = DualLogger(LOG_FILE)

# =============================================================================
# 🔍 INICIALIZACIÓN
# =============================================================================

print(f"⏳ Starting RAG-PubMed v1 – {datetime.now():%Y-%m-%d %H:%M:%S}")
sentence_model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
kw_model = KeyBERT(model=sentence_model)
nlp = spacy.load("es_core_news_sm")

# =============================================================================
# 🔬 FUNCIONES DE KEYWORDS Y BÚSQUEDA PUBMED
# =============================================================================

def get_keywords_keybert(texto, top_n=3):
    try:
        return [kw[0] for kw in kw_model.extract_keywords(texto, top_n=top_n)]
    except Exception:
        return []

def get_keywords_spacy(texto):
    doc = nlp(texto)
    return [t.text.lower() for t in doc if t.pos_ in ["NOUN", "PROPN"]]

def retrieve_pubmed_context(query, top_k=3):
    """Busca abstracts relevantes en PubMed usando la API oficial (E-utilities)."""
    try:
        keywords = get_keywords_keybert(query, top_n=3)
        if not keywords:
            keywords = get_keywords_spacy(query)[:3]
        if not keywords:
            return "No keywords", False

        search_term = "+".join(keywords)
        url_search = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?db=pubmed&term={search_term}&retmax={top_k}"
        search_resp = requests.get(url_search, timeout=30)
        tree = ET.fromstring(search_resp.text)
        ids = [id_elem.text for id_elem in tree.findall(".//Id")]

        if not ids:
            return "No PubMed results", False

        id_list = ",".join(ids)
        url_fetch = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=pubmed&id={id_list}&retmode=text&rettype=abstract"
        fetch_resp = requests.get(url_fetch, timeout=30)
        abstracts = fetch_resp.text.strip()
        context = "\n\n".join(abstracts.split("\n\n")[:3])[:2000]

        if not context:
            return "No abstract text", False

        time.sleep(0.3)
        return context, True
    except Exception as e:
        return f"⚠️ Error PubMed: {e}", False

# =============================================================================
# 📊 MÉTRICAS
# =============================================================================

def compute_metrics_fixed(results, modelo):
    total = len(results)
    correct = sum(1 for r in results if r.get(modelo) == r.get("respuesta_correcta"))
    none_cnt = sum(1 for r in results if r.get(modelo) is None)
    errors = total - correct - none_cnt
    found = sum(1 for r in results if r.get("found_pubmed"))
    acc_global = (correct / total * 100) if total > 0 else 0.0
    return {
        "Modelo": modelo,
        "Total preguntas": total,
        "Aciertos": correct,
        "Errores": errors,
        "Sin respuesta": none_cnt,
        "Con PubMed": found,
        "Accuracy (%)": round(acc_global, 2)
    }

def compute_submetrics(results, modelo, flag):
    subset = [r for r in results if r.get("found_pubmed") == flag]
    total = len(subset)
    correct = sum(1 for r in subset if r.get(modelo) == r.get("respuesta_correcta"))
    none_cnt = sum(1 for r in subset if r.get(modelo) is None)
    errors = total - correct - none_cnt
    acc = (correct / total * 100) if total > 0 else 0.0
    return {
        "Modelo": modelo,
        "Subset": "Con PubMed" if flag else "Sin PubMed",
        "Total preguntas": total,
        "Aciertos": correct,
        "Errores": errors,
        "Sin respuesta": none_cnt,
        "Accuracy (%)": round(acc, 2)
    }

# =============================================================================
# 🚀 LOOP PRINCIPAL
# =============================================================================

archivos_json = [f for f in os.listdir(CARPETA_EXAMENES) if f.endswith(".json")]
metricas_global, metricas_conpub, metricas_sinpub = [], [], []
resultados_titulacion = {modelo: {} for modelo in MODELOS}

for archivo_json in archivos_json:
    nombre_examen = os.path.splitext(archivo_json)[0]
    titulacion = nombre_examen.split("_")[0]
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
            contexto, found_pubmed = retrieve_pubmed_context(enunciado)

            opciones = "\n".join(f"{idx+1}. {op}" for idx, op in enumerate(pregunta["opciones"]))
            prompt = (
                "Eres un profesional médico que debe responder una pregunta tipo MIR.\n"
                "Usa los abstracts de PubMed como evidencia científica.\n\n"
                f"📚 CONTEXTO:\n{contexto}\n\n"
                f"❓ PREGUNTA:\n{enunciado}\n\n"
                f"🔢 OPCIONES:\n{opciones}\n\n"
                "Responde exactamente: 'La respuesta correcta es la número X.'"
            )

            try:
                resp = requests.post("http://localhost:11434/api/generate",
                                     json={"model": modelo, "prompt": prompt, "stream": False},
                                     timeout=180)
                text = resp.json().get("response", "").strip()
            except Exception as e:
                text = f"❌ Error: {e}"

            match = re.search(r"\b([1-4])\b", text)
            pred = int(match.group(1)) if match else None

            preguntas_resultado.append({
                "numero": pregunta.get("numero"),
                "enunciado": enunciado,
                "opciones": pregunta.get("opciones"),
                "respuesta_correcta": pregunta.get("respuesta_correcta"),
                "found_pubmed": found_pubmed,
                modelo: pred,
                f"{modelo}_texto": text
            })

            if i % 50 == 0:
                print(f"💾 Progreso: {i}/{len(preguntas_texto)} preguntas procesadas")

        # === Métricas ===
        m_global = compute_metrics_fixed(preguntas_resultado, modelo)
        m_con = compute_submetrics(preguntas_resultado, modelo, True)
        m_sin = compute_submetrics(preguntas_resultado, modelo, False)

        m_global["Titulación"] = m_con["Titulación"] = m_sin["Titulación"] = titulacion

        metricas_global.append(m_global)
        metricas_conpub.append(m_con)
        metricas_sinpub.append(m_sin)

        print(f"📊 {modelo.upper()} → Global: {m_global['Accuracy (%)']}% | "
              f"Con PubMed: {m_con['Accuracy (%)']}% | "
              f"Sin PubMed: {m_sin['Accuracy (%)']}%")

        resultados_titulacion[modelo][titulacion] = preguntas_resultado

# =============================================================================
# 💾 GUARDAR RESULTADOS
# =============================================================================

for modelo in MODELOS:
    for titulacion, preguntas in resultados_titulacion[modelo].items():
        salida_json = os.path.join(CARPETA_SALIDA, f"{titulacion}_{modelo}_rag_pubmed_v1.json")
        with open(salida_json, "w", encoding="utf-8") as f_out:
            json.dump({"preguntas": preguntas}, f_out, ensure_ascii=False, indent=2)
        print(f"💾 Guardado JSON: {salida_json}")

# === Exportar métricas ===
df_g = pd.DataFrame(metricas_global)
df_w = pd.DataFrame(metricas_conpub)
df_n = pd.DataFrame(metricas_sinpub)
excel_path = os.path.join(CARPETA_SALIDA, "rag_pubmed_v1_metrics.xlsx")

with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
    df_g.to_excel(writer, sheet_name="Global", index=False)
    df_w.to_excel(writer, sheet_name="Con PubMed", index=False)
    df_n.to_excel(writer, sheet_name="Sin PubMed", index=False)

print(f"\n✅ Métricas exportadas correctamente en: {excel_path}")
print(f"🧾 Log completo: {LOG_FILE}")
print("\n🏁 Pipeline RAG-PubMed v1 completado correctamente.")
