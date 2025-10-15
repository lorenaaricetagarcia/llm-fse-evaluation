#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RAG with PubMed – v2 (Revised, Publication-Ready)
Author: Lorena Ariceta García
TFM – Data Science & Bioinformatics for Precision Medicine

✔ Guarda todas las preguntas (aunque no haya contexto)
✔ Añade 'found_pubmed': True/False
✔ Calcula métricas globales + con/sin PubMed
✔ Exporta JSONs + Excel (3 hojas) + log reproducible
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
from transformers import pipeline
import torch
from datetime import datetime
import sys

# =============================================================================
# CONFIGURACIÓN
# =============================================================================
MODELOS = ["llama3", "mistral", "gemma"]

CARPETA_EXAMENES = "/home/xs1/Desktop/Lorena/MEDICINA/results/1_data_preparation/6_json_final/prueba"
CARPETA_SALIDA = "/home/xs1/Desktop/Lorena/MEDICINA/results/2_models/2_rag/2_pubmed/v2"
os.makedirs(CARPETA_SALIDA, exist_ok=True)

LOG_FILE = os.path.join(CARPETA_SALIDA, "rag_pubmed_v2_log.txt")

# =============================================================================
# LOG DUAL (pantalla + archivo)
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
# INICIALIZACIÓN
# =============================================================================
print(f"⏳ Starting RAG-PubMed v2 – {datetime.now():%Y-%m-%d %H:%M:%S}")

sentence_model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
kw_model = KeyBERT(model=sentence_model)
nlp = spacy.load("es_core_news_sm")

if torch.cuda.is_available():
    try:
        print(f"✅ GPU detectada: {torch.cuda.get_device_name(0)}")
        device = 0
    except Exception:
        device = -1
        print("⚠️ GPU no compatible, usando CPU.")
else:
    device = -1
    print("⚙️ No se detecta GPU → usando CPU.")

try:
    translator = pipeline("translation", model="Helsinki-NLP/opus-mt-es-en", device=device)
    print("✅ Traductor ES→EN cargado.")
except Exception as e:
    print(f"⚠️ No se pudo cargar el traductor: {e}")
    translator = None

# =============================================================================
# FUNCIONES AUXILIARES
# =============================================================================
def get_keywords_keybert(texto, top_n=3):
    try:
        return [kw[0] for kw in kw_model.extract_keywords(texto, top_n=top_n)]
    except Exception:
        return []

def get_keywords_spacy(texto, top_n=5):
    doc = nlp(texto)
    kws = [t.text.lower() for t in doc if t.pos_ in ["NOUN", "PROPN"]]
    seen, out = set(), []
    for k in kws:
        if k not in seen:
            seen.add(k)
            out.append(k)
    return out[:top_n]

def ensure_3_keywords(base_kw):
    kws = [k for k in base_kw if k and isinstance(k, str)]
    extras = ["medicina", "diagnóstico", "tratamiento"]
    for e in extras:
        if len(kws) >= 3: break
        if e not in kws: kws.append(e)
    return kws[:3]

def translate_keywords_es_en(kws_es):
    if translator is None:
        return kws_es
    try:
        phrase = ", ".join(kws_es)
        translated = translator(phrase)[0]["translation_text"]
        parts = [p.strip().lower() for p in re.split(r"[,;/]", translated) if p.strip()]
        return ensure_3_keywords(parts or kws_es)
    except Exception:
        return kws_es

def pubmed_esearch(term, retmax):
    url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?db=pubmed&term={term}&retmax={retmax}"
    r = requests.get(url, timeout=30)
    tree = ET.fromstring(r.text)
    return [id_elem.text for id_elem in tree.findall(".//Id")]

def pubmed_efetch(ids):
    id_list = ",".join(ids)
    url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=pubmed&id={id_list}&retmode=text&rettype=abstract"
    r = requests.get(url, timeout=30)
    return r.text.strip()

def retrieve_pubmed_context(query, top_k=3):
    try:
        kws_es = ensure_3_keywords(get_keywords_keybert(query) + get_keywords_spacy(query))
        kws_en = translate_keywords_es_en(kws_es)
        term = "+".join(kws_en)
        ids = pubmed_esearch(term, top_k)
        if not ids:
            return "Sin contexto relevante.", [], kws_es, kws_en, term, False
        raw_text = pubmed_efetch(ids)
        context = "\n\n".join(raw_text.split("\n\n")[:3])[:2000]
        time.sleep(0.3)
        return context, ids, kws_es, kws_en, term, True
    except Exception as e:
        return f"⚠️ Error PubMed: {e}", [], [], [], "", False

# =============================================================================
# MÉTRICAS
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
# LOOP PRINCIPAL
# =============================================================================
metricas_global, metricas_conpub, metricas_sinpub = [], [], []
resultados_titulacion = {m: {} for m in MODELOS}
archivos_json = [f for f in os.listdir(CARPETA_EXAMENES) if f.endswith(".json")]

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

        for i, p in enumerate(preguntas_texto, 1):
            enunciado = p["enunciado"]
            contexto, pmids, kws_es, kws_en, term, found_pubmed = retrieve_pubmed_context(enunciado)

            opciones = "\n".join(f"{idx+1}. {op}" for idx, op in enumerate(p["opciones"]))
            prompt = (
                "Eres un profesional médico que responde preguntas tipo MIR.\n"
                "Utiliza PubMed si hay información relevante; si no, responde con tu conocimiento clínico.\n"
                "Formato: 'La respuesta correcta es la número X.'\n\n"
                f"📘 CONTEXTO (PubMed):\n{contexto}\n\n"
                f"❓ PREGUNTA:\n{enunciado}\n\n"
                f"🔢 OPCIONES:\n{opciones}\n"
            )

            try:
                resp = requests.post(
                    "http://localhost:11434/api/generate",
                    json={"model": modelo, "prompt": prompt, "stream": False},
                    timeout=180
                )
                texto = resp.json().get("response", "").strip()
            except Exception as e:
                texto = f"❌ Error: {e}"

            match = re.search(r"\b([1-4])\b", texto)
            seleccion = int(match.group(1)) if match else None

            preguntas_resultado.append({
                "numero": p.get("numero"),
                "enunciado": enunciado,
                "opciones": p.get("opciones"),
                "keywords_es": kws_es,
                "keywords_en": kws_en,
                "pubmed_term": term,
                "pmids_usados": pmids,
                "found_pubmed": found_pubmed,
                modelo: seleccion,
                f"{modelo}_texto": texto,
                "respuesta_correcta": p.get("respuesta_correcta")
            })

            if i % 50 == 0:
                print(f"💾 Progreso: {i}/{len(preguntas_texto)} procesadas")

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
# GUARDADO
# =============================================================================
for modelo in MODELOS:
    for titulacion, preguntas in resultados_titulacion[modelo].items():
        path = os.path.join(CARPETA_SALIDA, f"{titulacion}_{modelo}_rag_pubmed_v2.json")
        with open(path, "w", encoding="utf-8") as f_out:
            json.dump({"preguntas": preguntas}, f_out, ensure_ascii=False, indent=2)
        print(f"💾 Guardado JSON: {path}")

# === Export Excel ===
df_g = pd.DataFrame(metricas_global)
df_w = pd.DataFrame(metricas_conpub)
df_n = pd.DataFrame(metricas_sinpub)
excel_path = os.path.join(CARPETA_SALIDA, "rag_pubmed_v2_metrics.xlsx")

with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
    df_g.to_excel(writer, sheet_name="Global", index=False)
    df_w.to_excel(writer, sheet_name="Con PubMed", index=False)
    df_n.to_excel(writer, sheet_name="Sin PubMed", index=False)

print(f"\n✅ Métricas exportadas correctamente en: {excel_path}")
print(f"🧾 Log completo: {LOG_FILE}")
print("\n🏁 Pipeline RAG-PubMed v2 completado correctamente.")
