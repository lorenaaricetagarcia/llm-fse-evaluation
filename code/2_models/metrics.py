#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script Title: Prompt-Series Metrics Computation (No Prompt / Spanish Prompt / English Prompt / DeepSeek+Phi3)
Author: Lorena Ariceta Garcia
TFM: AI-Based Diagnosis: Ally or Risk?
     An Analysis of Language Models

Description
-----------
This script computes accuracy metrics for four prompt variants used in the
model-evaluation pipeline:

    1) No Prompt
    2) Prompt (Spanish)
    3) Prompt (Spanish) — DeepSeek + Phi3 (specific model IDs)
    4) Prompt (English)

Predictions are compared against the original ground-truth answers stored in
the source JSON files. To ensure robustness against potential duplicated
question numbers, the comparison is performed by position (index alignment).

Only textual questions are considered (non-text questions are excluded).

Outputs
-------
- A CSV file with per-exam metrics
- An Excel file with per-exam metrics plus an additional sheet containing a
  global summary (mean accuracy by model and prompt variant)

Requirements
------------
- Python 3.x
- pandas
- openpyxl (required for appending sheets to Excel)

Methodological Notes
--------------------
- Accuracy is computed as: correct / answered * 100
- "Answered" excludes predictions equal to None
- Text-only filtering is applied using the field: pregunta["tipo"] == "texto"
"""

import os
import json
from datetime import datetime

import pandas as pd


# =============================================================================
# 1. Path configuration
# =============================================================================
BASE_DIR = "/home/xs1/Desktop/Lorena/results/2_models/1_prompt"
GROUND_TRUTH_DIR = "/home/xs1/Desktop/Lorena/results/1_data_preparation/6_json_final"

OUTPUT_DIR = os.path.join(BASE_DIR, "metrics")
os.makedirs(OUTPUT_DIR, exist_ok=True)

LOG_FILE = os.path.join(OUTPUT_DIR, "metrics_prompt_series_log.txt")
CSV_PATH = os.path.join(OUTPUT_DIR, "prompt_series_metrics.csv")
XLSX_PATH = os.path.join(OUTPUT_DIR, "prompt_series_metrics.xlsx")

PROMPT_VARIANTS = [
    "1_no_prompt",
    "2_prompt_es",                # Spanish prompt
    "3_prompt_deepseek_phi3",     # Spanish prompt (DeepSeek + Phi3 specific IDs)
    "4_prompt_en",                # English prompt
]


# =============================================================================
# 2. Helper function
# =============================================================================
def compute_metrics_for_folder(prompt_variant_path: str, prompt_name: str) -> list[dict]:
    """
    Compute metrics for a given prompt-variant folder.

    Folder structure expected:
        <prompt_variant_path>/
            <model_name>/
                <EXAM>_<model_name>.json

    Ground-truth files expected:
        GROUND_TRUTH_DIR/<EXAM>.json

    Returns a list of dictionaries (rows) with metrics.
    """
    metrics: list[dict] = []
    print(f"\nProcessing prompt variant: {prompt_name}")

    for model_name in os.listdir(prompt_variant_path):
        model_dir = os.path.join(prompt_variant_path, model_name)
        if not os.path.isdir(model_dir):
            continue

        print(f"  Model: {model_name}")

        for filename in os.listdir(model_dir):
            if not filename.endswith(".json"):
                continue

            exam_name = filename.split("_")[0]
            ground_truth_path = os.path.join(GROUND_TRUTH_DIR, f"{exam_name}.json")

            if not os.path.exists(ground_truth_path):
                print(f"  Warning: ground-truth file not found for exam '{exam_name}'")
                continue

            # Load prediction data
            pred_path = os.path.join(model_dir, filename)
            with open(pred_path, "r", encoding="utf-8") as f:
                pred_data = json.load(f)

            # Load ground-truth data
            with open(ground_truth_path, "r", encoding="utf-8") as f:
                gt_data = json.load(f)

            # Text-only filtering
            pred_questions = [
                q for q in pred_data.get("preguntas", [])
                if q.get("tipo", "texto") == "texto"
            ]
            gt_questions = [
                q for q in gt_data.get("preguntas", [])
                if q.get("tipo", "texto") == "texto"
            ]

            total = min(len(pred_questions), len(gt_questions))
            correct = wrong = no_response = 0

            for i in range(total):
                prediction = pred_questions[i].get(model_name)
                correct_answer = gt_questions[i].get("respuesta_correcta")

                if prediction is None:
                    no_response += 1
                elif prediction == correct_answer:
                    correct += 1
                else:
                    wrong += 1

                if (i + 1) % 50 == 0:
                    print(f"    Processed {i + 1}/{total} questions...")

            answered = total - no_response
            accuracy = (correct / answered * 100) if answered > 0 else 0.0

            metrics.append({
                "Prompt Variant": prompt_name,
                "Model": model_name,
                "Exam": exam_name,
                "Total Questions": total,
                "Answered": answered,
                "Correct": correct,
                "Wrong": wrong,
                "No Response": no_response,
                "Accuracy (%)": round(accuracy, 2),
            })

            print(f"    {exam_name}: {accuracy:.2f}% ({correct}/{answered})")

    return metrics


# =============================================================================
# 3. Main execution
# =============================================================================
def main() -> None:
    all_metrics: list[dict] = []

    # Logging header
    with open(LOG_FILE, "w", encoding="utf-8") as log:
        log.write(
            f"PROMPT-SERIES METRICS — Start: {datetime.now():%Y-%m-%d %H:%M:%S}\n"
        )
        log.write("=" * 80 + "\n\n")

    # Compute metrics per prompt variant
    for variant in PROMPT_VARIANTS:
        variant_path = os.path.join(BASE_DIR, variant)

        if not os.path.exists(variant_path):
            print(f"Warning: folder not found: {variant_path}")
            continue

        variant_metrics = compute_metrics_for_folder(variant_path, variant)
        all_metrics.extend(variant_metrics)

    if not all_metrics:
        print("No results found to process.")
        return

    # Export CSV + Excel
    df = pd.DataFrame(all_metrics)
    df.to_csv(CSV_PATH, index=False, encoding="utf-8-sig")
    df.to_excel(XLSX_PATH, index=False)

    print("\nMetrics saved:")
    print(f"  • CSV : {CSV_PATH}")
    print(f"  • XLSX: {XLSX_PATH}")

    # Global summary: mean accuracy by model and prompt
    summary = (
        df.groupby(["Prompt Variant", "Model"])["Accuracy (%)"]
        .mean()
        .reset_index()
        .pivot(index="Model", columns="Prompt Variant", values="Accuracy (%)")
        .round(2)
    )
    summary["Overall Mean"] = summary.mean(axis=1).round(2)

    # Append summary sheet to the same Excel file
    with pd.ExcelWriter(XLSX_PATH, engine="openpyxl", mode="a") as writer:
        summary.to_excel(writer, sheet_name="Global Summary")

    print("\nGLOBAL SUMMARY (mean Accuracy %)")
    print(summary)

    # Append to log
    with open(LOG_FILE, "a", encoding="utf-8") as log:
        log.write(f"\nExecution completed: {datetime.now():%Y-%m-%d %H:%M:%S}\n")
        log.write("Generated files:\n")
        log.write(f" - {CSV_PATH}\n")
        log.write(f" - {XLSX_PATH}\n")


# =============================================================================
# 4. Entry point
# =============================================================================
if __name__ == "__main__":
    main()
