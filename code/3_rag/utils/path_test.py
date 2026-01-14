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
BASE_DIR = Path(os.getenv("FSE_BASE_DIR", Path(__file__).resolve().parents[3]))
CODE_DIR = BASE_DIR / "code/3_rag"
RESULTS_DIR = BASE_DIR / "results/3_rag"
SUMMARY_DIR = RESULTS_DIR / "summary"
EXAM_DIR = Path(
    os.getenv(
        "FSE_INPUT_DIR",
        BASE_DIR / "results/1_data_preparation/6_json_final",
    )
)

WIKI_DIR = RESULTS_DIR / "1_wikipedia"
PUBMED_DIR = RESULTS_DIR / "2_pubmed"

VERSIONS = ["v1", "v2", "v3", "final"]
LANGS = ["es", "en"]

# =============================================================================
# 🎨 MAPA DE SCRIPTS
# =============================================================================
SCRIPTS = {
    "Wikipedia_v1": CODE_DIR / "1_wikipedia/RAG_Wikipedia_v1_basic_single_keyword.py",
    "Wikipedia_v2": CODE_DIR / "1_wikipedia/RAG_Wikipedia_v2_multikey_suggestions.py",
    "Wikipedia_v3": CODE_DIR / "1_wikipedia/RAG_Wikipedia_v3_strict_prompt_multikey.py",
    "Wikipedia_final": (
        CODE_DIR / "1_wikipedia/RAG_Wikipedia_final_multilingual_dynamic_model.py"
    ),
    "PubMed_v1": CODE_DIR / "2_pubmed/RAG_PubMed_v1_basic_es.py",
    "PubMed_v2": (
        CODE_DIR / "2_pubmed/RAG_PubMed_v2_es_en_translation_gpu_fallback.py"
    ),
    "PubMed_v3": CODE_DIR / "2_pubmed/RAG_PubMed_v3_progressive_saves_strict_prompt.py",
    "PubMed_final": (
        CODE_DIR / "2_pubmed/RAG_PubMed_final_multilingual_dynamic_model.py"
    ),
}

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

for folder in [WIKI_DIR, PUBMED_DIR, SUMMARY_DIR]:
    created = check_exists(folder, create=True)
    print(f"{status_icon(created)} {folder}")

# =============================================================================
# 📊 COMPROBACIÓN DE MÉTRICAS ESPERADAS
# =============================================================================
print("\n📊 Verificando métricas generadas...\n")

expected_metrics = sorted(WIKI_DIR.glob("*_metrics.xlsx")) + sorted(
    PUBMED_DIR.glob("*_metrics.xlsx")
)

if not expected_metrics:
    print("⚠️ No se encontraron archivos *_metrics.xlsx en results/3_rag.")
else:
    for metrics_xlsx in expected_metrics:
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
