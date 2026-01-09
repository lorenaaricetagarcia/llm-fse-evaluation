"""
Script Title: LLM Benchmarking on MIR Multiple-Choice Questions (No-Prompt Setting)
Author: Lorena Ariceta Garcia
TFM: AI-Based Diagnosis: Ally or Risk?
     An Analysis of Language Models

Description
-----------
This script evaluates multiple local Large Language Models (LLMs) on MIR-style
multiple-choice questions stored in JSON format. For each input exam file, the
script sends each question (stem + four options) to each model via the Ollama
HTTP API and parses the model's response to extract a selected option (1–4).

The script:
1) Generates a per-model JSON file with the model's predicted option and raw text.
2) Computes accuracy metrics (correct / incorrect / unanswered) and prints per-file results.
3) Aggregates global results across all processed files and exports:
   - CSV:  no_prompt_metrics.csv
   - Excel: no_prompt_metrics.xlsx

Requirements
------------
- Python 3.x
- requests
- pandas
- Ollama running locally (default endpoint: http://localhost:11434)

Methodological Notes
--------------------
- The first occurrence of an integer in {1,2,3,4} found in the model response is
  used as the selected option.
- Some input files may contain duplicated question numbers; in that case,
  evaluation is performed by positional index rather than question number.
- For ENFERMERÍA and MEDICINA, non-text questions can be skipped if they are
  labeled as image-type items.
"""

import csv
import json
import os
import re
import sys
from collections import Counter, OrderedDict

import pandas as pd
import requests


# ================================================================
# 1) OUTPUT CONFIGURATION
# ================================================================
OUTPUT_DIR = "/home/xs1/Desktop/Lorena/results/2_models/1_prompt/1_no_prompt"
os.makedirs(OUTPUT_DIR, exist_ok=True)


class DualOutput:
    """
    Redirect stdout to both console and a log file.
    """

    def __init__(self, path: str):
        self.terminal = sys.__stdout__
        self.log = open(path, "w", encoding="utf-8")

    def write(self, message: str):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()


# Redirect all prints to both console and a log file
sys.stdout = DualOutput(os.path.join(OUTPUT_DIR, "log_no_prompt.txt"))


# ================================================================
# 2) MODELS AND INPUT FILES
# ================================================================
MODELS = ["llama3", "mistral", "gemma", "deepseek-coder", "phi3"]

INPUT_DIR = "/home/xs1/Desktop/Lorena/results/1_data_preparation/6_json_final"
json_files = [f for f in os.listdir(INPUT_DIR) if f.endswith(".json")]

OLLAMA_ENDPOINT = "http://localhost:11434/api/generate"
REQUEST_TIMEOUT_SECONDS = 180

# Global accumulator for metrics
global_summary = {
    model: {
        "correct": 0,
        "incorrect": 0,
        "no_answer": 0,
        "total": 0,
        "error_examples": [],
    }
    for model in MODELS
}


# ================================================================
# 3) MAIN LOOP (PER FILE, PER MODEL)
# ================================================================
for json_filename in json_files:
    base_name = os.path.splitext(json_filename)[0]
    json_path = os.path.join(INPUT_DIR, json_filename)

    with open(json_path, "r", encoding="utf-8") as f:
        base_data = json.load(f)

    # Warn if duplicated question numbers exist (evaluation will be positional)
    numbers = [q.get("numero") for q in base_data.get("preguntas", [])]
    duplicate_count = sum(1 for _, c in Counter(numbers).items() if c > 1)
    if duplicate_count > 0:
        print(
            f"⚠️ {json_filename}: {duplicate_count} duplicated values detected in 'numero' "
            "(evaluation will be performed by position/index).\n"
        )

    for model in MODELS:
        print(f"\n🚀 Processing model: {model} | File: {json_filename}")

        model_output = {"preguntas": []}
        model_dir = os.path.join(OUTPUT_DIR, model)
        os.makedirs(model_dir, exist_ok=True)

        # ============================================================
        # 3.1 GENERATE + STORE MODEL RESPONSES
        # ============================================================
        for idx, question in enumerate(base_data.get("preguntas", []), start=1):
            # Optional filtering for image questions in specific files
            if json_filename in ["ENFERMERÍA.json", "MEDICINA.json"] and question.get("tipo") != "texto":
                continue

            prompt = f"{question.get('enunciado', '')}\n\n"
            for opt_i, option in enumerate(question.get("opciones", []), start=1):
                prompt += f"{opt_i}. {option}\n"

            print(f"\n📤 [{idx}] Sending question to {model}...")

            payload = {"model": model, "prompt": prompt, "stream": False}

            try:
                response = requests.post(
                    OLLAMA_ENDPOINT,
                    json=payload,
                    timeout=REQUEST_TIMEOUT_SECONDS,
                )
                response.raise_for_status()

                data_model = response.json()
                raw_text = data_model.get("response", "").strip()

                print("🧠 Model response:")
                print(raw_text)

                # Extract first standalone digit 1..4
                match = re.search(r"\b([1-4])\b", raw_text)
                selection = int(match.group(1)) if match else None

                # Copy original fields while avoiding collisions
                new_question = OrderedDict()
                for key in question:
                    if key not in (model, f"{model}_text"):
                        new_question[key] = question[key]

                new_question[model] = selection
                new_question[f"{model}_text"] = raw_text
                model_output["preguntas"].append(new_question)

            except requests.exceptions.Timeout:
                print("❌ Model timeout.")
            except Exception as exc:
                print(f"❌ Error on question {idx}: {exc}")

        # Save per-model JSON output
        output_json_path = os.path.join(model_dir, f"{base_name}_{model}.json")
        with open(output_json_path, "w", encoding="utf-8") as f_out:
            json.dump(model_output, f_out, ensure_ascii=False, indent=2)

        print(f"\n✅ Saved: {output_json_path}")

        # ============================================================
        # 3.2 METRICS (POSITIONAL COMPARISON)
        # ============================================================
        print(f"\n📊 Evaluating model {model.upper()} | Exam: {base_name}")

        predicted_questions = model_output.get("preguntas", [])
        total = len(predicted_questions)

        correct = 0
        incorrect = 0
        no_answer = 0
        error_examples = []

        for i, q_pred in enumerate(predicted_questions):
            pred = q_pred.get(model)

            # Positional ground truth (index-based)
            gt = base_data.get("preguntas", [])[i].get("respuesta_correcta") if i < len(base_data.get("preguntas", [])) else None

            if pred is None:
                no_answer += 1
                continue

            if gt is None:
                # If ground truth is missing, we ignore it for accuracy
                continue

            if pred == gt:
                correct += 1
            else:
                incorrect += 1
                error_examples.append(
                    {
                        "index": i + 1,
                        "predicted": pred,
                        "correct": gt,
                        "stem": q_pred.get("enunciado", ""),
                    }
                )

        answered = total - no_answer
        accuracy_pct = (correct / answered * 100) if answered > 0 else 0.0

        # Update global summary
        global_summary[model]["correct"] += correct
        global_summary[model]["incorrect"] += incorrect
        global_summary[model]["no_answer"] += no_answer
        global_summary[model]["total"] += total

        # Store a small number of representative errors
        global_summary[model]["error_examples"].extend(error_examples[:3])

        print("-" * 60)
        print(f"Total questions           : {total}")
        print(f"Answered by the model     : {answered}")
        print(f"Correct                  : {correct}")
        print(f"Incorrect                : {incorrect}")
        print(f"No answer (None)          : {no_answer}")
        print(f"📈 Accuracy               : {accuracy_pct:.2f}%")

        print("\n🔍 Example errors:")
        for err in error_examples[:5]:
            print(f"  ➤ Question {err['index']}: predicted {err['predicted']}, correct {err['correct']}")
            print(f"    {err['stem']}")


# ================================================================
# 4) GLOBAL SUMMARY + CSV/EXCEL EXPORT
# ================================================================
print("\n📊📊📊 GLOBAL SUMMARY BY MODEL 📊📊📊")

csv_path = os.path.join(OUTPUT_DIR, "no_prompt_metrics.csv")
excel_path = os.path.join(OUTPUT_DIR, "no_prompt_metrics.xlsx")

rows = []

for model in MODELS:
    total = global_summary[model]["total"]
    correct = global_summary[model]["correct"]
    incorrect = global_summary[model]["incorrect"]
    no_answer = global_summary[model]["no_answer"]

    answered = total - no_answer
    accuracy_pct = (correct / answered * 100) if answered > 0 else 0.0

    print(f"\n🧠 Model: {model.upper()}")
    print("-" * 50)
    print(f"Total questions           : {total}")
    print(f"Answered                  : {answered}")
    print(f"Correct                   : {correct}")
    print(f"Incorrect                 : {incorrect}")
    print(f"No answer (None)          : {no_answer}")
    print(f"📈 Accuracy               : {accuracy_pct:.2f}%")

    print("🔍 Example errors:")
    for err in global_summary[model]["error_examples"][:5]:
        print(f"  ➤ Question {err['index']}: predicted {err['predicted']}, correct {err['correct']}")
        print(f"    {err['stem']}")

    rows.append(
        {
            "Model": model,
            "Total": total,
            "Answered": answered,
            "Correct": correct,
            "Incorrect": incorrect,
            "No answer": no_answer,
            "Accuracy (%)": round(accuracy_pct, 2),
        }
    )

# Save CSV
with open(csv_path, "w", newline="", encoding="utf-8") as f_csv:
    writer = csv.DictWriter(f_csv, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)

# Save Excel
df = pd.DataFrame(rows)
df.to_excel(excel_path, index=False)

print("\n✅ Global results saved to:")
print(f"   • CSV  : {csv_path}")
print(f"   • Excel: {excel_path}")
print("\n✅ Pipeline completed successfully.")
