#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import subprocess
import time
import re
import glob
from datetime import datetime
import sys

import pandas as pd


# ---------------------------------------------------------------------
# 1. Configuration
# ---------------------------------------------------------------------
BASE_DIR = "/home/xs1/Desktop/Lorena"

RESULTS_DIR = f"{BASE_DIR}/results/3_rag"
SUMMARY_DIR = f"{RESULTS_DIR}/summary"
PROMPT_FILE = f"{BASE_DIR}/code/3_RAG/prompt_config.py"
JSON_FINAL_DIR = f"{BASE_DIR}/results/1_data_preparation/6_json_final"

os.makedirs(SUMMARY_DIR, exist_ok=True)


# ---------------------------------------------------------------------
# 2. Available pipelines (Wikipedia / PubMed)
# ---------------------------------------------------------------------
RUNS = [
    # Wikipedia
    ("Wikipedia_v1", f"{BASE_DIR}/code/3_RAG/1_wikipedia/RAG_Wikipedia_v1_basic_single_keyword.py"),
    ("Wikipedia_v2", f"{BASE_DIR}/code/3_RAG/1_wikipedia/RAG_Wikipedia_v2_multikey_suggestions.py"),
    ("Wikipedia_v3", f"{BASE_DIR}/code/3_RAG/1_wikipedia/RAG_Wikipedia_v3_strict_prompt_multikey.py"),
    ("Wikipedia_final", f"{BASE_DIR}/code/3_RAG/1_wikipedia/RAG_Wikipedia_final_multilingual_dynamic_model.py"),

    # PubMed
    ("PubMed_v1", f"{BASE_DIR}/code/3_RAG/2_pubmed/RAG_PubMed_v1_basic_es.py"),
    ("PubMed_v2", f"{BASE_DIR}/code/3_RAG/2_pubmed/RAG_PubMed_v2_es_en_translation_gpu_fallback.py"),
    ("PubMed_v3", f"{BASE_DIR}/code/3_RAG/2_pubmed/RAG_PubMed_v3_progressive_saves_strict_prompt.py"),
    ("PubMed_final", f"{BASE_DIR}/code/3_RAG/2_pubmed/RAG_PubMed_final_multilingual_dynamic_model.py"),
]


# ---------------------------------------------------------------------
# 3. Available models
# ---------------------------------------------------------------------
GENERAL_MODELS = ["llama3", "mistral", "gemma"]
BIOMED_MODELS = ["medllama2"]


# ---------------------------------------------------------------------
# 4. Load prompts from prompt_config.py
# ---------------------------------------------------------------------
PROMPTS = {}

with open(PROMPT_FILE, "r", encoding="utf-8") as f:
    code = f.read()

PROMPTS["es"] = (
    re.search(r'"es":\s*\(\s*"(.*?)"\s*\)', code, re.S)
    .group(1)
    .replace("\\n", "\n")
)
PROMPTS["en"] = (
    re.search(r'"en":\s*\(\s*"(.*?)"\s*\)', code, re.S)
    .group(1)
    .replace("\\n", "\n")
)


# ---------------------------------------------------------------------
# 5. Helper functions
# ---------------------------------------------------------------------
def version_of(name: str) -> str:
    if "_v" in name:
        m = re.search(r"_v(\d)", name)
        return m.group(1) if m else "?"
    if "_final" in name:
        return "final"
    return "?"


def source_of(name: str) -> str:
    return "Wikipedia" if "Wikipedia" in name else "PubMed"


def format_time(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def list_specialization_files(folder: str):
    items = []
    if not os.path.isdir(folder):
        raise FileNotFoundError(f"JSON_FINAL_DIR not found: {folder}")

    for entry in sorted(os.listdir(folder)):
        if entry.startswith("."):
            continue
        full = os.path.join(folder, entry)
        if os.path.isdir(full):
            continue
        if entry.lower().endswith(".json") or "." not in entry:
            spec_name = os.path.splitext(entry)[0]
            items.append((spec_name, full))

    return items


def find_latest_metrics_file(after_ts: float):
    pattern = os.path.join(RESULTS_DIR, "**", "*_metrics.xlsx")
    candidates = glob.glob(pattern, recursive=True)

    recent = []
    for p in candidates:
        try:
            mtime = os.path.getmtime(p)
            if mtime >= after_ts:
                recent.append((mtime, p))
        except OSError:
            pass

    if not recent:
        return None

    recent.sort(key=lambda x: x[0], reverse=True)
    return recent[0][1]


def run_script_streaming(script_path: str, cwd: str, env: dict, header_prefix: str) -> int:
    """
    Run a python script and stream stdout/stderr live to the terminal.
    Returns process return code.
    """
    # -u => unbuffered para que los prints del pipeline salgan al momento
    cmd = [sys.executable, "-u", script_path]

    proc = subprocess.Popen(
        cmd,
        cwd=cwd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,   # mezclamos stderr con stdout para verlo todo en orden
        text=True,
        bufsize=1,
        universal_newlines=True,
    )

    assert proc.stdout is not None
    for line in proc.stdout:
        # Prefijo para que siempre sepas "dónde estás"
        # (si el pipeline ya imprime su propio progreso, lo verás aquí)
        print(f"{header_prefix}{line}", end="")

    proc.wait()
    return proc.returncode


# ---------------------------------------------------------------------
# 6. Interactive selection menu
# ---------------------------------------------------------------------
print("\n🧩 AVAILABLE MODELS\n")
for i, m in enumerate(GENERAL_MODELS + BIOMED_MODELS, 1):
    print(f"  {i:>2}. {m}")

model_choice = input("\n👉 Select models (e.g., 1,3,5) or 'all' for all models: ").strip().lower()

if model_choice == "all":
    selected_models = GENERAL_MODELS + BIOMED_MODELS
else:
    try:
        indices = [int(x.strip()) for x in model_choice.split(",") if x.strip().isdigit()]
        all_models = GENERAL_MODELS + BIOMED_MODELS
        selected_models = [all_models[i - 1] for i in indices if 1 <= i <= len(all_models)]
    except Exception:
        print("❌ Invalid selection. Exiting.")
        raise SystemExit(1)

if not selected_models:
    print("❌ No valid models selected. Exiting.")
    raise SystemExit(0)

print("\n📘 AVAILABLE PIPELINES\n")
for i, (name, _) in enumerate(RUNS, 1):
    print(f"  {i:>2}. {name}")

pipe_choice = input("\n👉 Select pipelines (e.g., 1,4,8) or 'all': ").strip().lower()

if pipe_choice == "all":
    selected_runs = RUNS
else:
    try:
        indices = [int(x.strip()) for x in pipe_choice.split(",") if x.strip().isdigit()]
        selected_runs = [RUNS[i - 1] for i in indices if 1 <= i <= len(RUNS)]
    except Exception:
        print("❌ Invalid selection. Exiting.")
        raise SystemExit(1)

langs = ["es", "en"]
print("\n🌐 Available languages: [es, en, all]")
lang_choice = input("👉 Select language (es/en/all): ").strip().lower()

if lang_choice == "all":
    selected_langs = langs
elif lang_choice in langs:
    selected_langs = [lang_choice]
else:
    print("❌ Invalid language. Exiting.")
    raise SystemExit(0)


# ---------------------------------------------------------------------
# 7. Specializations discovery
# ---------------------------------------------------------------------
specializations = list_specialization_files(JSON_FINAL_DIR)

if not specializations:
    print(f"\n❌ No specialization files found in: {JSON_FINAL_DIR}")
    raise SystemExit(1)

print("\n📚 SPECIALIZATIONS DETECTED")
for spec_name, spec_path in specializations:
    print(f"  - {spec_name}  ({os.path.basename(spec_path)})")

print(
    f"\n🚀 Running: {len(specializations)} specializations × {len(selected_models)} models × "
    f"{len(selected_runs)} pipelines × {len(selected_langs)} languages\n"
)


# ---------------------------------------------------------------------
# 8. Execution loop (per specialization)
# ---------------------------------------------------------------------
t0 = time.time()
script_times = []
global_frames = []

for spec_name, spec_json_path in specializations:
    print("\n" + "=" * 78)
    print(f"🏷️  SPECIALIZATION: {spec_name}")
    print("=" * 78)

    per_spec_frames = []

    spec_summary_dir = os.path.join(SUMMARY_DIR, spec_name)
    os.makedirs(spec_summary_dir, exist_ok=True)

    for model in selected_models:
        for pipeline_name, script in selected_runs:
            for lang in selected_langs:
                print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
                print(f"🔹 {pipeline_name} | 🧠 Model: {model} | 🌐 Language: {lang} | 🏷️ Spec: {spec_name}")

                # (Si no quieres que te escupa el prompt entero, comenta estas 3 líneas)
                print("🧩 Prompt used:")
                print("-" * 50)
                print(PROMPTS[lang])
                print("-" * 50)

                env = os.environ.copy()
                env["RAG_LANG"] = lang
                env["RAG_MODEL"] = model
                env["RAG_SPECIALIZATION"] = spec_name
                env["RAG_INPUT_JSON"] = spec_json_path

                # 🔥 CLAVE: pasar el prompt al pipeline
                env["RAG_PROMPT"] = PROMPTS[lang]

                # 🔥 CLAVE: forzar unbuffered por env también
                env["PYTHONUNBUFFERED"] = "1"

                start = time.time()

                header_prefix = f"[{spec_name} | {pipeline_name} | {model} | {lang}] "

                try:
                    rc = run_script_streaming(
                        script_path=script,
                        cwd=os.path.dirname(script),
                        env=env,
                        header_prefix=header_prefix
                    )

                    elapsed = time.time() - start
                    script_times.append((f"{spec_name} | {model}-{pipeline_name}-{lang}", elapsed))

                    if rc != 0:
                        print(f"\n❌ Pipeline exited with code {rc} ({header_prefix.strip()})")
                    else:
                        print(f"\n✅ Completed in {format_time(elapsed)} ({header_prefix.strip()})")

                    metrics_path = find_latest_metrics_file(after_ts=start)

                    if metrics_path and os.path.exists(metrics_path):
                        df = pd.read_excel(metrics_path)

                        df["Specialization"] = spec_name
                        df["Source"] = source_of(pipeline_name)
                        df["Version"] = f"v{version_of(pipeline_name)}" if "_v" in pipeline_name else "final"
                        df["Lang"] = lang
                        df["Model"] = model
                        df["MetricsFile"] = os.path.basename(metrics_path)

                        per_spec_frames.append(df)
                        global_frames.append(df)

                        print(f"📊 Metrics loaded: {metrics_path}")
                    else:
                        print("⚠️ No recent metrics file found for this run.")

                except Exception as e:
                    print(f"❌ Error while executing: {e}")

    # -----------------------------------------------------------------
    # 8.1 Save per-specialization summary
    # -----------------------------------------------------------------
    if per_spec_frames:
        merged_spec = pd.concat(per_spec_frames, ignore_index=True)

        preferred_cols = [
            "Specialization",
            "Source",
            "Model",
            "Version",
            "Lang",
            "Total questions",
            "Correct",
            "Errors",
            "No answer",
            "Accuracy (%)",
            "MetricsFile",
        ]
        merged_spec = merged_spec[[c for c in preferred_cols if c in merged_spec.columns]]

        out_path = os.path.join(
            spec_summary_dir,
            f"rag_summary_{spec_name}_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx"
        )
        merged_spec.to_excel(out_path, index=False)
        print(f"\n✅ Per-specialization summary saved to: {out_path}")
    else:
        print("\n⚠️ No metrics loaded for this specialization.")


# ---------------------------------------------------------------------
# 9. Optional global summary
# ---------------------------------------------------------------------
if global_frames:
    merged_global = pd.concat(global_frames, ignore_index=True)

    preferred_cols = [
        "Specialization",
        "Source",
        "Model",
        "Version",
        "Lang",
        "Total questions",
        "Correct",
        "Errors",
        "No answer",
        "Accuracy (%)",
        "MetricsFile",
    ]
    merged_global = merged_global[[c for c in preferred_cols if c in merged_global.columns]]

    global_out = os.path.join(
        SUMMARY_DIR,
        f"rag_selected_summary_GLOBAL_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx"
    )
    merged_global.to_excel(global_out, index=False)
    print(f"\n✅ Global summary saved to: {global_out}")
else:
    print("\n⚠️ No global metrics were loaded – please check pipeline outputs.")


# ---------------------------------------------------------------------
# 10. Execution time summary
# ---------------------------------------------------------------------
print("\n🕒 EXECUTION TIME SUMMARY")
print("──────────────────────────────────────────────────────────────")
for run_name, secs in script_times:
    print(f"  {run_name:<60} → {format_time(secs)}")
print("──────────────────────────────────────────────────────────────")
print(f"  TOTAL TIME → {format_time(time.time() - t0)}")
print("🏁 Finished.\n")
