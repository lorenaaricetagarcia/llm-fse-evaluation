#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RAG Integrator v3 (prompt-checked, updated for FINAL)
Autor: Lorena Ariceta García
TFM – Data Science & Bioinformatics for Precision Medicine

✅ Ejecuta Wikipedia (v1–v3 + final) y PubMed (v1–v3 + final)
✅ Muestra en pantalla el prompt usado en cada ejecución
✅ Inyecta variable RAG_LANG en cada script
✅ Lee métricas .xlsx directamente y fusiona todo
"""

import os, subprocess, pandas as pd, time, re
from datetime import datetime

# === CONFIGURACIÓN ===
BASE_DIR = "/home/xs1/Desktop/Lorena/MEDICINA"
RESULTS_DIR = f"{BASE_DIR}/results/2_models/2_rag"
SUMMARY_DIR = f"{BASE_DIR}/results/summary"
PROMPT_FILE = f"{BASE_DIR}/code/3_RAG/prompt_config.py"
os.makedirs(SUMMARY_DIR, exist_ok=True)

# === Cargar prompts desde prompt_config.py ===
PROMPTS = {}
with open(PROMPT_FILE, "r", encoding="utf-8") as f:
    code = f.read()
    PROMPTS["es"] = re.search(r'"es":\s*\(\s*"(.*?)"\s*\)', code, re.S).group(1).replace("\\n", "\n")
    PROMPTS["en"] = re.search(r'"en":\s*\(\s*"(.*?)"\s*\)', code, re.S).group(1).replace("\\n", "\n")

# === MATRIZ DE PIPELINES ===
RUNS = [
    # === Wikipedia ===
    ("Wikipedia_v1_es", f"{BASE_DIR}/code/3_RAG/1_wikipedia/RAG_wikipedia_v1.py", "es"),
    ("Wikipedia_v2_es", f"{BASE_DIR}/code/3_RAG/1_wikipedia/RAG_wikipedia_v2.py", "es"),
    ("Wikipedia_v3_es", f"{BASE_DIR}/code/3_RAG/1_wikipedia/RAG_wikipedia_v3.py", "es"),
    ("Wikipedia_v1_en", f"{BASE_DIR}/code/3_RAG/1_wikipedia/RAG_wikipedia_v1.py", "en"),
    ("Wikipedia_v2_en", f"{BASE_DIR}/code/3_RAG/1_wikipedia/RAG_wikipedia_v2.py", "en"),
    ("Wikipedia_v3_en", f"{BASE_DIR}/code/3_RAG/1_wikipedia/RAG_wikipedia_v3.py", "en"),

    # === PubMed ===
    ("PubMed_v1_es", f"{BASE_DIR}/code/3_RAG/2_pubmed/RAG_pubmed_v1.py", "es"),
    ("PubMed_v1_en", f"{BASE_DIR}/code/3_RAG/2_pubmed/RAG_pubmed_v1.py", "en"),
    ("PubMed_v2_es", f"{BASE_DIR}/code/3_RAG/2_pubmed/RAG_pubmed_v2.py", "es"),
    ("PubMed_v2_en", f"{BASE_DIR}/code/3_RAG/2_pubmed/RAG_pubmed_v2.py", "en"),
    ("PubMed_v3_es", f"{BASE_DIR}/code/3_RAG/2_pubmed/RAG_pubmed_v3.py", "es"),
    ("PubMed_v3_en", f"{BASE_DIR}/code/3_RAG/2_pubmed/RAG_pubmed_v3.py", "en"),

    # === FINAL versions ===
    ("Wikipedia_final_es", f"{BASE_DIR}/code/3_RAG/final/RAG_wikipedia_final.py", "es"),
    ("Wikipedia_final_en", f"{BASE_DIR}/code/3_RAG/final/RAG_wikipedia_final.py", "en"),
    ("PubMed_final_es", f"{BASE_DIR}/code/3_RAG/final/RAG_pubmed_final.py", "es"),
    ("PubMed_final_en", f"{BASE_DIR}/code/3_RAG/final/RAG_pubmed_final.py", "en"),
]

# === FUNCIONES AUXILIARES ===
def version_of(name):
    if "_v" in name:
        return re.search(r"_v(\d)_", name).group(1)
    elif "_final" in name:
        return "final"
    else:
        return "?"

def source_of(name):
    return "Wikipedia" if name.startswith("Wikipedia") else "PubMed"

def format_time(s):
    h, m, ss = int(s // 3600), int((s % 3600) // 60), int(s % 60)
    return f"{h:02d}:{m:02d}:{ss:02d}"

# === INICIO ===
print("\n🚀 Starting full RAG evaluation (with prompt check)...\n")
t0 = time.time()
script_times, frames = [], []

for name, script, lang in RUNS:
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print(f"🔹 Running {name} ...")
    print(f"🌐 Language: {lang}")
    print("🧠 Prompt being used:")
    print("-" * 50)
    print(PROMPTS[lang])
    print("-" * 50)

    env = os.environ.copy()
    env["RAG_LANG"] = lang

    start = time.time()
    try:
        proc = subprocess.run(["python3", script], env=env, capture_output=True, text=True)
        elapsed = time.time() - start
        print("\n".join(proc.stdout.splitlines()[-10:]))  # último bloque de salida
        print(f"✅ Completed {name} in {format_time(elapsed)}")
        script_times.append((name, elapsed))

        # === Buscar métricas ===
        src = "1_wikipedia" if "Wikipedia" in name else "2_pubmed"
        version = f"v{version_of(name)}" if "v" in name else "final"
        metrics_path = (
            f"{RESULTS_DIR}/{src}/{version}_{lang}/rag_{source_of(name).lower()}_{version}_{lang}_metrics.xlsx"
            if "v" in name
            else f"{RESULTS_DIR}/final/{source_of(name).lower()}_final_{lang}/rag_{source_of(name).lower()}_final_{lang}_metrics.xlsx"
        )

        if os.path.exists(metrics_path):
            df = pd.read_excel(metrics_path, sheet_name=0)
            df["Source"] = source_of(name)
            df["Version"] = version
            df["Lang"] = lang
            frames.append(df)
            print(f"📊 Loaded metrics: {os.path.basename(metrics_path)}")
        else:
            print(f"⚠️ Metrics not found for {name}")

    except Exception as e:
        print(f"❌ Error executing {name}: {e}")

# === CONSOLIDAR ===
if frames:
    merged = pd.concat(frames, ignore_index=True)
    cols = ["Source", "Modelo", "Version", "Lang", "Total preguntas", "Aciertos", "Errores", "Sin respuesta", "Accuracy (%)"]
    merged = merged[[c for c in cols if c in merged.columns]]
    merged_path = f"{SUMMARY_DIR}/rag_all_langs_summary.xlsx"
    merged.to_excel(merged_path, index=False)
    print(f"\n✅ Global summary saved to: {merged_path}")
else:
    print("\n⚠️ No metrics loaded – please verify run outputs.")

# === TIEMPOS ===
print("\n🕒 EXECUTION TIME SUMMARY")
print("──────────────────────────────")
for name, secs in script_times:
    print(f"  {name:<25} → {format_time(secs)}")
print("──────────────────────────────")
print(f"  TOTAL TIME                     → {format_time(time.time()-t0)}")
print("🏁 Done.\n")
