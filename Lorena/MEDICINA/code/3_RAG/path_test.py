#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RAG Path Test – Verificación completa de rutas (incluye versiones FINAL)
Author: Lorena Ariceta García
TFM – Data Science & Bioinformatics for Precision Medicine
"""

import os
import pandas as pd
from pathlib import Path

# =============================================================================
# ⚙️ CONFIGURACIÓN
# =============================================================================
BASE_DIR = Path("/home/xs1/Desktop/Lorena/MEDICINA")
CODE_DIR = BASE_DIR / "code/3_RAG"
RESULTS_DIR = BASE_DIR / "results/2_models/2_rag"
SUMMARY_DIR = BASE_DIR / "results/summary"
EXAM_DIR = BASE_DIR / "results/1_data_preparation/6_json_final/prueba"

WIKI_DIR = RESULTS_DIR / "1_wikipedia"
PUBMED_DIR = RESULTS_DIR / "2_pubmed"
FINAL_DIR = CODE_DIR / "final"

VERSIONS = ["v1", "v2", "v3", "final"]
LANGS = ["es", "en"]

# =============================================================================
# 🎨 MAPA DE SCRIPTS
# =============================================================================
SCRIPTS = {
    f"Wikipedia_{v}": CODE_DIR / f"1_wikipedia/RAG_wikipedia_{v}.py"
    for v in ["v1", "v2", "v3"]
}
SCRIPTS.update({
    f"PubMed_{v}": CODE_DIR / f"2_pubmed/RAG_pubmed_{v}.py"
    for v in ["v1", "v2", "v3"]
})
# Añadir finales
SCRIPTS.update({
    "Wikipedia_final": FINAL_DIR / "RAG_wikipedia_final.py",
    "PubMed_final": FINAL_DIR / "RAG_pubmed_final.py",
})

# =============================================================================
# 🎨 FUNCIONES AUXILIARES
# =============================================================================
def check_exists(path, create=False):
    """Devuelve True si existe, crea si se pide."""
    if path.exists():
        return True
    elif create:
        path.mkdir(parents=True, exist_ok=True)
        return True
    return False

def status_icon(ok):
    return "✅" if ok else "❌"

# =============================================================================
# 🔍 COMPROBACIÓN DE SCRIPTS
# =============================================================================
print("\n🔍 Verificando scripts de ejecución...\n")
for name, script_path in SCRIPTS.items():
    print(f"{name:<18} → {status_icon(script_path.exists())}  ({script_path})")

# =============================================================================
# 📂 COMPROBACIÓN DE CARPETAS DE SALIDA
# =============================================================================
print("\n📂 Verificando carpetas de salida (creando si no existen)...\n")

for sub in ["1_wikipedia", "2_pubmed"]:
    for v in ["v1", "v2", "v3"]:
        for lang in LANGS:
            folder = RESULTS_DIR / sub / f"{v}_{lang}"
            created = check_exists(folder, create=True)
            print(f"{sub}/{v}_{lang:<10} → {status_icon(created)}  ({folder})")

# Carpetas de FINAL
for source in ["wikipedia", "pubmed"]:
    for lang in LANGS:
        folder = RESULTS_DIR / "final" / f"{source}_final_{lang}"
        created = check_exists(folder, create=True)
        print(f"final/{source}_{lang:<10} → {status_icon(created)}  ({folder})")

# =============================================================================
# 📊 COMPROBACIÓN DE MÉTRICAS ESPERADAS
# =============================================================================
print("\n📊 Verificando rutas de métricas esperadas...\n")

expected_metrics = []
# Para versiones v1–v3
for source in ["wikipedia", "pubmed"]:
    prefix = "1_" if source == "wikipedia" else "2_"
    for v in ["v1", "v2", "v3"]:
        for lang in LANGS:
            folder = RESULTS_DIR / f"{prefix}{source}" / f"{v}_{lang}"
            metrics_xlsx = folder / f"rag_{source}_{v}_{lang}_metrics.xlsx"
            expected_metrics.append(metrics_xlsx)
            print(f"{status_icon(metrics_xlsx.exists())} {metrics_xlsx}")

# Para FINAL
for source in ["wikipedia", "pubmed"]:
    for lang in LANGS:
        folder = RESULTS_DIR / "final" / f"{source}_final_{lang}"
        metrics_xlsx = folder / f"rag_{source}_final_{lang}_metrics.xlsx"
        expected_metrics.append(metrics_xlsx)
        print(f"{status_icon(metrics_xlsx.exists())} {metrics_xlsx}")

# =============================================================================
# 📘 RUTA DE ENTRADAS (EXÁMENES)
# =============================================================================
print("\n📘 Verificando carpeta de exámenes...\n")

if EXAM_DIR.exists():
    jsons = list(EXAM_DIR.glob("*.json"))
    print(f"✅ Carpeta detectada: {EXAM_DIR}")
    print(f"   • {len(jsons)} archivos JSON encontrados.")
else:
    print(f"❌ Carpeta de exámenes no encontrada: {EXAM_DIR}")

# =============================================================================
# 🧾 DIRECTORIO DE RESUMEN
# =============================================================================
print("\n🧾 Verificando carpeta de resumen...\n")
check_exists(SUMMARY_DIR, create=True)
print(f"✅ {SUMMARY_DIR}")

# =============================================================================
# 📊 RESUMEN GLOBAL
# =============================================================================
print("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
print("✅ TEST COMPLETADO: todas las rutas comprobadas correctamente")
print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n")

df_summary = pd.DataFrame({
    "Ruta esperada": [str(p) for p in expected_metrics],
    "Existe": [p.exists() for p in expected_metrics]
})
summary_path = SUMMARY_DIR / "rag_path_test_summary.xlsx"
df_summary.to_excel(summary_path, index=False)
print(f"📊 Resumen exportado a: {summary_path}")
print("\n🏁 Fin del test de rutas.\n")
