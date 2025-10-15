#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compute Prompt-Series Metrics (no-prompt / prompt-es / prompt-en / deepseek)
Author: Lorena Ariceta García
TFM – Data Science & Bioinformatics for Precision Medicine

Descripción:
  - Calcula métricas de accuracy para las 4 variantes de prompts
  - Compara predicciones con las respuestas correctas originales (por posición)
  - Excluye preguntas no textuales
  - Exporta CSV y Excel profesional con resumen global
"""

import os
import json
import pandas as pd
from datetime import datetime

# =============================================================================
# 📁 CONFIGURACIÓN DE RUTAS
# =============================================================================
BASE_DIR = "/home/xs1/Desktop/Lorena/results/2_models/1_prompt"
CORRECTAS_DIR = "/home/xs1/Desktop/Lorena/results/1_data_preparation/6_json_final"
OUTPUT_DIR = os.path.join(BASE_DIR, "metrics")
os.makedirs(OUTPUT_DIR, exist_ok=True)

LOG_FILE = os.path.join(OUTPUT_DIR, "metrics_prompt_series_log.txt")
CSV_PATH = os.path.join(OUTPUT_DIR, "prompt_series_metrics.csv")
XLSX_PATH = os.path.join(OUTPUT_DIR, "prompt_series_metrics.xlsx")

PROMPT_VARIANTS = [
    "1_no_prompt",
    "2_prompt_es",
    "3_prompt_deepseek_phi3",
    "4_prompt_en",
]

# =============================================================================
# 🧾 FUNCIÓN AUXILIAR
# =============================================================================
def compute_metrics_for_folder(prompt_variant_path, prompt_name):
    metrics = []
    print(f"\n🔹 Procesando: {prompt_name}")
    for model_name in os.listdir(prompt_variant_path):
        model_dir = os.path.join(prompt_variant_path, model_name)
        if not os.path.isdir(model_dir):
            continue

        print(f"   ⚙️ Modelo: {model_name}")
        for file in os.listdir(model_dir):
            if not file.endswith(".json"):
                continue
            titulacion = file.split("_")[0]
            correctas_path = os.path.join(CORRECTAS_DIR, f"{titulacion}.json")
            if not os.path.exists(correctas_path):
                print(f"   ⚠️ No se encuentra el archivo de respuestas correctas para {titulacion}")
                continue

            # Leer datos
            with open(os.path.join(model_dir, file), "r", encoding="utf-8") as f:
                pred_data = json.load(f)
            with open(correctas_path, "r", encoding="utf-8") as f:
                corr_data = json.load(f)

            preguntas_pred = [p for p in pred_data.get("preguntas", []) if p.get("tipo", "texto") == "texto"]
            preguntas_corr = [p for p in corr_data.get("preguntas", []) if p.get("tipo", "texto") == "texto"]

            total = min(len(preguntas_pred), len(preguntas_corr))
            aciertos = errores = sin_resp = 0

            for i in range(total):
                pred = preguntas_pred[i].get(model_name)
                correcta = preguntas_corr[i].get("respuesta_correcta")
                if pred is None:
                    sin_resp += 1
                elif pred == correcta:
                    aciertos += 1
                else:
                    errores += 1

                if (i + 1) % 50 == 0:
                    print(f"      💾 {i+1}/{total} preguntas procesadas...")

            respondidas = total - sin_resp
            acc = (aciertos / respondidas * 100) if respondidas > 0 else 0

            metrics.append({
                "Prompt": prompt_name,
                "Modelo": model_name,
                "Titulación": titulacion,
                "Total preguntas": total,
                "Respondidas": respondidas,
                "Aciertos": aciertos,
                "Errores": errores,
                "Sin respuesta": sin_resp,
                "Accuracy (%)": round(acc, 2)
            })

            print(f"      ✅ {titulacion}: {acc:.2f}% ({aciertos}/{respondidas})")

    return metrics


# =============================================================================
# 🚀 EJECUCIÓN PRINCIPAL
# =============================================================================
def main():
    all_metrics = []
    with open(LOG_FILE, "w", encoding="utf-8") as log:
        log.write(f"📘 MÉTRICAS PROMPT SERIES – Inicio: {datetime.now():%Y-%m-%d %H:%M:%S}\n")
        log.write("=" * 80 + "\n\n")

    for variant in PROMPT_VARIANTS:
        folder_path = os.path.join(BASE_DIR, variant)
        if not os.path.exists(folder_path):
            print(f"⚠️ Carpeta no encontrada: {folder_path}")
            continue
        variant_metrics = compute_metrics_for_folder(folder_path, variant)
        all_metrics.extend(variant_metrics)

    if not all_metrics:
        print("❌ No se encontraron resultados para procesar.")
        return

    df = pd.DataFrame(all_metrics)
    df.to_csv(CSV_PATH, index=False, encoding="utf-8-sig")
    df.to_excel(XLSX_PATH, index=False)
    print(f"\n📊 Métricas guardadas en:\n   • CSV : {CSV_PATH}\n   • XLSX: {XLSX_PATH}")

    # === Resumen global (para paper/Q1)
    resumen = (
        df.groupby(["Prompt", "Modelo"])["Accuracy (%)"]
        .mean()
        .reset_index()
        .pivot(index="Modelo", columns="Prompt", values="Accuracy (%)")
        .round(2)
    )
    resumen["Media general"] = resumen.mean(axis=1).round(2)

    resumen_path = os.path.join(OUTPUT_DIR, "prompt_series_summary.xlsx")
    with pd.ExcelWriter(XLSX_PATH, engine="openpyxl", mode="a") as writer:
        resumen.to_excel(writer, sheet_name="Resumen global")

    print("\n🏆 RESUMEN GLOBAL DE ACCURACY (%)")
    print(resumen)
    print(f"\n✅ Resumen añadido al Excel: {resumen_path}")

    with open(LOG_FILE, "a", encoding="utf-8") as log:
        log.write(f"\n📊 Ejecución completada: {datetime.now():%Y-%m-%d %H:%M:%S}\n")
        log.write(f"Archivos generados:\n - {CSV_PATH}\n - {XLSX_PATH}\n")


# =============================================================================
# 🏁 ENTRY POINT
# =============================================================================
if __name__ == "__main__":
    main()
