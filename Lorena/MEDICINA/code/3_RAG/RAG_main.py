#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RAG Integrator – Wikipedia + PubMed (v1–v3)
Author: Lorena Ariceta García
TFM – Data Science & Bioinformatics for Precision Medicine

Descripción:
  - Ejecuta automáticamente los 6 pipelines RAG (Wikipedia v1–v3, PubMed v1–v3)
  - Fusiona métricas en un Excel resumen global con promedios por modelo
  - Guarda logs de ejecución y comprobación de errores
"""

import os
import subprocess
import pandas as pd
from datetime import datetime

# =============================================================================
# ⚙️ CONFIGURACIÓN DE RUTAS
# =============================================================================
BASE_DIR = "/home/xs1/Desktop/Lorena/MEDICINA/results"
SCRIPTS = {
    "Wikipedia_v1": f"{BASE_DIR}/2_models/2_rag/1_wikipedia/v1/rag_wikipedia_v1.py",
    "Wikipedia_v2": f"{BASE_DIR}/2_models/2_rag/1_wikipedia/v2/rag_wikipedia_v2.py",
    "Wikipedia_v3": f"{BASE_DIR}/2_models/2_rag/1_wikipedia/v3/rag_wikipedia_v3.py",
    "PubMed_v1": f"{BASE_DIR}/2_models/2_rag/2_pubmed/v1/rag_pubmed_v1.py",
    "PubMed_v2": f"{BASE_DIR}/2_models/2_rag/2_pubmed/v2/rag_pubmed_v2.py",
    "PubMed_v3": f"{BASE_DIR}/2_models/2_rag/2_pubmed/v3/rag_pubmed_v3.py",
}

METRICS_PATHS = {
    "Wikipedia_v1": f"{BASE_DIR}/2_models/2_rag/1_wikipedia/v1/rag_wikipedia_v1_metrics.xlsx",
    "Wikipedia_v2": f"{BASE_DIR}/2_models/2_rag/1_wikipedia/v2/rag_wikipedia_v2_metrics.xlsx",
    "Wikipedia_v3": f"{BASE_DIR}/2_models/2_rag/1_wikipedia/v3/rag_wikipedia_v3_metrics.xlsx",
    "PubMed_v1": f"{BASE_DIR}/2_models/2_rag/2_pubmed/v1/rag_pubmed_v1_metrics.xlsx",
    "PubMed_v2": f"{BASE_DIR}/2_models/2_rag/2_pubmed/v2/rag_pubmed_v2_metrics.xlsx",
    "PubMed_v3": f"{BASE_DIR}/2_models/2_rag/2_pubmed/v3/rag_pubmed_v3_metrics.xlsx",
}

SUMMARY_DIR = f"{BASE_DIR}/summary"
os.makedirs(SUMMARY_DIR, exist_ok=True)

GLOBAL_LOG = os.path.join(SUMMARY_DIR, "integrator_log.txt")
SUMMARY_EXCEL = os.path.join(SUMMARY_DIR, "rag_overall_summary.xlsx")

# =============================================================================
# 🧾 LOG INICIAL
# =============================================================================
with open(GLOBAL_LOG, "w", encoding="utf-8") as log:
    log.write(f"📘 RAG Integrator – Inicio: {datetime.now():%Y-%m-%d %H:%M:%S}\n")
    log.write("=" * 80 + "\n\n")

# =============================================================================
# 🚀 EJECUCIÓN SECUENCIAL DE LOS 6 SCRIPTS
# =============================================================================
for name, script_path in SCRIPTS.items():
    print(f"\n🔹 Ejecutando {name}...")
    with open(GLOBAL_LOG, "a", encoding="utf-8") as log:
        log.write(f"\n▶️ Ejecutando {name} – {datetime.now():%H:%M:%S}\n")
        if not os.path.exists(script_path):
            log.write(f"❌ No se encontró el script en {script_path}\n")
            continue

        try:
            subprocess.run(["python3", script_path], check=True)
            log.write(f"✅ {name} completado correctamente.\n")
        except subprocess.CalledProcessError as e:
            log.write(f"⚠️ Error ejecutando {name}: {e}\n")
        except Exception as e:
            log.write(f"❌ Error inesperado en {name}: {e}\n")

# =============================================================================
# 📊 FUSIÓN DE MÉTRICAS
# =============================================================================
print("\n📊 Fusionando métricas de los 6 pipelines...")

summary_frames = {}
for name, path in METRICS_PATHS.items():
    if os.path.exists(path):
        try:
            df = pd.read_excel(path, sheet_name=0)
            df["Fuente"] = name
            summary_frames[name] = df
        except Exception as e:
            print(f"⚠️ No se pudo leer {name}: {e}")
    else:
        print(f"⚠️ Archivo de métricas no encontrado: {path}")

if summary_frames:
    all_df = pd.concat(summary_frames.values(), ignore_index=True)
    resumen_global = (
        all_df.groupby(["Modelo", "Fuente"])["Accuracy (%)"]
        .mean()
        .reset_index()
        .pivot(index="Modelo", columns="Fuente", values="Accuracy (%)")
    )
    resumen_global["Media general"] = resumen_global.mean(axis=1).round(2)

    with pd.ExcelWriter(SUMMARY_EXCEL, engine="openpyxl") as writer:
        for name, df in summary_frames.items():
            df.to_excel(writer, sheet_name=name, index=False)
        resumen_global.to_excel(writer, sheet_name="Resumen global")

    print(f"✅ Resumen final exportado a:\n   {SUMMARY_EXCEL}")
    with open(GLOBAL_LOG, "a", encoding="utf-8") as log:
        log.write(f"\n📊 Fusión completada correctamente: {SUMMARY_EXCEL}\n")
else:
    print("⚠️ No se encontró ningún archivo de métricas para fusionar.")
    with open(GLOBAL_LOG, "a", encoding="utf-8") as log:
        log.write("⚠️ No se encontraron métricas para combinar.\n")

# =============================================================================
# ✅ FINALIZACIÓN
# =============================================================================
print("\n🏁 Integración completada con éxito.")
with open(GLOBAL_LOG, "a", encoding="utf-8") as log:
    log.write(f"\n🏁 Integración finalizada: {datetime.now():%Y-%m-%d %H:%M:%S}\n")
    log.write("=" * 80 + "\n")
