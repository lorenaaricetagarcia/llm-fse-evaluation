#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RAG Integrator – Model-Selectable Version
Author: Lorena Ariceta García
TFM – Data Science & Bioinformatics for Precision Medicine

✅ Permite elegir modelos (generales o biomédicos)
✅ Ejecuta pipelines seleccionados (Wikipedia / PubMed / Final)
✅ Carga prompts y métricas automáticamente
✅ Genera resumen global consolidado
"""

import os, subprocess, pandas as pd, time, re
from datetime import datetime

# === CONFIGURACIÓN ===
BASE_DIR = "/home/xs1/Desktop/Lorena/MEDICINA"
RESULTS_DIR = f"{BASE_DIR}/results/2_models/2_rag"
SUMMARY_DIR = f"{BASE_DIR}/results/summary"
PROMPT_FILE = f"{BASE_DIR}/code/3_RAG/prompt_config.py"
os.makedirs(SUMMARY_DIR, exist_ok=True)

# === MATRIZ DE PIPELINES DISPONIBLES ===
RUNS = [
    # Wikipedia
    ("Wikipedia_v1", f"{BASE_DIR}/code/3_RAG/1_wikipedia/RAG_wikipedia_v1.py"),
    ("Wikipedia_v2", f"{BASE_DIR}/code/3_RAG/1_wikipedia/RAG_wikipedia_v2.py"),
    ("Wikipedia_v3", f"{BASE_DIR}/code/3_RAG/1_wikipedia/RAG_wikipedia_v3.py"),
    ("Wikipedia_final", f"{BASE_DIR}/code/3_RAG/final/RAG_wikipedia_final.py"),

    # PubMed
    ("PubMed_v1", f"{BASE_DIR}/code/3_RAG/2_pubmed/RAG_pubmed_v1.py"),
    ("PubMed_v2", f"{BASE_DIR}/code/3_RAG/2_pubmed/RAG_pubmed_v2.py"),
    ("PubMed_v3", f"{BASE_DIR}/code/3_RAG/2_pubmed/RAG_pubmed_v3.py"),
    ("PubMed_final", f"{BASE_DIR}/code/3_RAG/final/RAG_pubmed_final.py"),
]

# === MODELOS DISPONIBLES ===
GENERAL_MODELS = ["llama3", "mistral", "gemma"]
BIOMED_MODELS = ["medllama2"]

# === CARGAR PROMPTS ===
PROMPTS = {}
with open(PROMPT_FILE, "r", encoding="utf-8") as f:
    code = f.read()
    PROMPTS["es"] = re.search(r'"es":\s*\(\s*"(.*?)"\s*\)', code, re.S).group(1).replace("\\n", "\n")
    PROMPTS["en"] = re.search(r'"en":\s*\(\s*"(.*?)"\s*\)', code, re.S).group(1).replace("\\n", "\n")

# === FUNCIONES AUXILIARES ===
def version_of(name):
    if "_v" in name:
        return re.search(r"_v(\d)", name).group(1)
    elif "_final" in name:
        return "final"
    else:
        return "?"

def source_of(name):
    return "Wikipedia" if "Wikipedia" in name else "PubMed"

def format_time(s):
    h, m, ss = int(s // 3600), int((s % 3600) // 60), int(s % 60)
    return f"{h:02d}:{m:02d}:{ss:02d}"

# === MENÚ DE SELECCIÓN ===
print("\n🧩 MODELOS DISPONIBLES\n")
for i, m in enumerate(GENERAL_MODELS + BIOMED_MODELS, 1):
    print(f"  {i:>2}. {m}")
model_choice = input("\n👉 Elige modelos (ej. 1,3,5) o 'all' para todos: ").strip().lower()

if model_choice == "all":
    selected_models = GENERAL_MODELS + BIOMED_MODELS
else:
    try:
        indices = [int(x.strip()) for x in model_choice.split(",") if x.strip().isdigit()]
        all_models = GENERAL_MODELS + BIOMED_MODELS
        selected_models = [all_models[i-1] for i in indices if 1 <= i <= len(all_models)]
    except Exception:
        print("❌ Selección inválida. Saliendo.")
        exit(1)

if not selected_models:
    print("❌ No hay modelos válidos seleccionados. Saliendo.")
    exit(0)

print("\n📘 PIPELINES DISPONIBLES\n")
for i, (name, _) in enumerate(RUNS, 1):
    print(f"  {i:>2}. {name}")
pipe_choice = input("\n👉 Elige pipelines (ej. 1,4,8) o 'all': ").strip().lower()

if pipe_choice == "all":
    selected_runs = RUNS
else:
    try:
        indices = [int(x.strip()) for x in pipe_choice.split(",") if x.strip().isdigit()]
        selected_runs = [RUNS[i-1] for i in indices if 1 <= i <= len(RUNS)]
    except Exception:
        print("❌ Selección inválida. Saliendo.")
        exit(1)

langs = ["es", "en"]
print("\n🌐 Idiomas disponibles: [es, en, ambos]")
lang_choice = input("👉 Elige idioma (es/en/all): ").strip().lower()
if lang_choice == "all":
    selected_langs = langs
elif lang_choice in langs:
    selected_langs = [lang_choice]
else:
    print("❌ Idioma no válido. Saliendo.")
    exit(0)

# === EJECUCIÓN ===
print(f"\n🚀 Ejecutando {len(selected_runs)} pipelines con modelos {selected_models} y lenguajes {selected_langs}\n")
t0 = time.time()
script_times, frames = [], []

for model in selected_models:
    for name, script in selected_runs:
        for lang in selected_langs:
            print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
            print(f"🔹 {name} | 🧠 Modelo: {model} | 🌐 Idioma: {lang}")
            print("🧩 Prompt utilizado:")
            print("-" * 50)
            print(PROMPTS[lang])
            print("-" * 50)

            env = os.environ.copy()
            env["RAG_LANG"] = lang
            env["RAG_MODEL"] = model

            start = time.time()
            try:
                proc = subprocess.run(
                    ["python3", script],
                    cwd=os.path.dirname(script),   # ejecuta dentro del directorio del script
                    env=env,
                    capture_output=True,
                    text=True
                )
                elapsed = time.time() - start
                print("\n".join(proc.stdout.splitlines()[-10:]))
                print(f"✅ {name} completado en {format_time(elapsed)}")
                script_times.append((f"{model}-{name}-{lang}", elapsed))

                src = "1_wikipedia" if "Wikipedia" in name else "2_pubmed"
                version = f"v{version_of(name)}" if "v" in name else "final"
                metrics_path = (
                    f"{RESULTS_DIR}/{src}/{version}_{lang}/rag_{source_of(name).lower()}_{version}_{lang}_metrics.xlsx"
                )

                if os.path.exists(metrics_path):
                    df = pd.read_excel(metrics_path)
                    df["Source"] = source_of(name)
                    df["Version"] = version
                    df["Lang"] = lang
                    df["Model"] = model
                    frames.append(df)
                    print(f"📊 Métricas cargadas: {os.path.basename(metrics_path)}")
                else:
                    print(f"⚠️ Métricas no encontradas para {name}")

            except Exception as e:
                print(f"❌ Error ejecutando {name}: {e}")

# === CONSOLIDAR ===
if frames:
    merged = pd.concat(frames, ignore_index=True)
    cols = ["Source", "Model", "Version", "Lang", "Total preguntas", "Aciertos", "Errores", "Sin respuesta", "Accuracy (%)"]
    merged = merged[[c for c in cols if c in merged.columns]]
    merged_path = f"{SUMMARY_DIR}/rag_selected_summary_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx"
    merged.to_excel(merged_path, index=False)
    print(f"\n✅ Resumen global guardado en: {merged_path}")
else:
    print("\n⚠️ No se cargaron métricas – verifica salidas.")

# === TIEMPOS ===
print("\n🕒 RESUMEN DE TIEMPOS")
print("──────────────────────────────")
for name, secs in script_times:
    print(f"  {name:<35} → {format_time(secs)}")
print("──────────────────────────────")
print(f"  TIEMPO TOTAL                     → {format_time(time.time()-t0)}")
print("🏁 Finalizado.\n")
