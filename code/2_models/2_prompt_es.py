"""
Script Title: LLM Benchmarking on MIR Multiple-Choice Questions
              (Spanish Prompt Condition)
Author: Lorena Ariceta Garcia
TFM: AI-Based Diagnosis: Ally or Risk?
     An Analysis of Language Models

Description
-----------
This script evaluates multiple Large Language Models (LLMs) on MIR-style
multiple-choice medical questions using an explicit *Spanish-language prompt*.

Compared to the no-prompt baseline, this script prepends a structured
instructional prompt written in Spanish, designed to:
- enforce a clinical-exam reasoning style,
- constrain the response format,
- and reduce ambiguity in the model outputs.

For each examination file and each model, the script:
1) Sends each question (statement + options) together with the Spanish prompt
   to the model via the Ollama HTTP API.
2) Extracts the selected answer option (1–4) from the model response.
3) Stores both the predicted option and the raw generated text.
4) Computes accuracy metrics using positional alignment.
5) Aggregates global results and exports them to CSV and Excel formats.

Requirements
------------
- Python 3.x
- requests
- pandas
- Ollama running locally (http://localhost:11434)

Methodological Notes
--------------------
- The Spanish prompt explicitly instructs the model to return a single numeric
  answer (1–4) followed by a brief justification.
- Accuracy is computed by positional index to handle duplicated question numbers.
- Image-based questions are excluded for selected specializations when required.
"""

import json
import requests
import re
import os
import sys
import csv
import pandas as pd
from collections import OrderedDict, Counter


# ================================================================
# OUTPUT CONFIGURATION
# ================================================================
OUTPUT_DIR = "/home/xs1/Desktop/Lorena/results/2_models/1_prompt/2_prompt_es"
os.makedirs(OUTPUT_DIR, exist_ok=True)


class DualOutput:
    """Redirect stdout to both console and a log file."""

    def __init__(self, path: str):
        self.terminal = sys.__stdout__
        self.log = open(path, "w", encoding="utf-8")

    def write(self, message: str):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()


# Activate output redirection
sys.stdout = DualOutput(os.path.join(OUTPUT_DIR, "log_prompt_es.txt"))


# ================================================================
# MODELS AND SPANISH PROMPT
# ================================================================
MODELS = ["llama3", "mistral", "gemma", "deepseek-coder", "phi3"]

SPANISH_PROMPT = (
    "Eres un profesional médico que debe responder una pregunta tipo examen clínico (MIR).\n"
    "Lee cuidadosamente el CONTEXTO recuperado y luego la PREGUNTA.\n"
    "Si el contexto contiene información útil y directa, utilízala para responder.\n"
    "Si el contexto no aporta la respuesta, usa tu conocimiento médico general.\n"
    "Tu respuesta debe seguir estrictamente este formato:\n"
    "'La respuesta correcta es la número X' (donde X es un número del 1 al 4).\n"
    "Después, añade una sola frase breve con la justificación principal.\n"
    "No respondas con 'No estoy seguro', no proporciones varias opciones ni copies el contexto.\n"
    "Responde siempre con una única opción numérica (1–4) y una frase concisa.\n\n"
)

INPUT_DIR = "results/1_data_preparation/6_json_final"
json_files = [f for f in os.listdir(INPUT_DIR) if f.endswith(".json")]


# ================================================================
# GLOBAL RESULTS STRUCTURE
# ================================================================
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
# MAIN LOOP (PER FILE, PER MODEL)
# ================================================================
for json_filename in json_files:
    exam_name = os.path.splitext(json_filename)[0]
    json_path = os.path.join(INPUT_DIR, json_filename)

    with open(json_path, "r", encoding="utf-8") as f:
        base_data = json.load(f)

    # Warn about duplicated question numbers
    numbers = [q.get("numero") for q in base_data["preguntas"]]
    duplicate_count = sum(1 for _, c in Counter(numbers).items() if c > 1)
    if duplicate_count > 0:
        print(
            f"⚠️ {json_filename}: {duplicate_count} duplicated question numbers detected "
            "(evaluation will be positional).\n"
        )

    for model in MODELS:
        print(f"\n🚀 Processing exam '{exam_name}' with model: {model}")
        model_data = {"preguntas": []}

        model_dir = os.path.join(OUTPUT_DIR, model)
        os.makedirs(model_dir, exist_ok=True)

        # ------------------------------------------------------------
        # Generate model responses
        # ------------------------------------------------------------
        for idx, question in enumerate(base_data["preguntas"], start=1):
            if json_filename in ["ENFERMERÍA.json", "MEDICINA.json"] and question.get("tipo") != "texto":
                continue

            prompt = SPANISH_PROMPT + question["enunciado"] + "\n\n"
            for opt_i, option in enumerate(question["opciones"], start=1):
                prompt += f"{opt_i}. {option}\n"

            print(f"\n📤 [{idx}] Sending question to {model}...")

            payload = {"model": model, "prompt": prompt, "stream": False}

            try:
                response = requests.post(
                    "http://localhost:11434/api/generate",
                    json=payload,
                    timeout=180,
                )
                data_model = response.json()
                raw_text = data_model.get("response", "").strip()

                print("🧠 Model response:")
                print(raw_text)

                match = re.search(r"\b([1-4])\b", raw_text)
                selection = int(match.group(1)) if match else None

                new_question = OrderedDict()
                for key in question:
                    if key not in (model, f"{model}_texto"):
                        new_question[key] = question[key]

                new_question[model] = selection
                new_question[f"{model}_texto"] = raw_text
                model_data["preguntas"].append(new_question)

            except requests.exceptions.Timeout:
                print("❌ Model timeout.")
            except Exception as exc:
                print(f"❌ Error on question {idx}: {exc}")

        # Save per-model JSON
        output_json = os.path.join(model_dir, f"{exam_name}_{model}.json")
        with open(output_json, "w", encoding="utf-8") as f_out:
            json.dump(model_data, f_out, ensure_ascii=False, indent=2)

        print(f"\n✅ Saved: {output_json}")

        # ------------------------------------------------------------
        # Metrics (positional comparison)
        # ------------------------------------------------------------
        questions = model_data["preguntas"]
        total = len(questions)
        correct = incorrect = no_answer = 0
        error_examples = []

        for i, q in enumerate(questions):
            pred = q.get(model)
            gt = base_data["preguntas"][i].get("respuesta_correcta") if i < len(base_data["preguntas"]) else None

            if pred is None:
                no_answer += 1
            elif gt is None:
                continue
            elif pred == gt:
                correct += 1
            else:
                incorrect += 1
                error_examples.append(
                    {
                        "index": i + 1,
                        "predicted": pred,
                        "correct": gt,
                        "stem": q["enunciado"],
                    }
                )

        answered = total - no_answer
        accuracy = (correct / answered * 100) if answered > 0 else 0

        global_summary[model]["correct"] += correct
        global_summary[model]["incorrect"] += incorrect
        global_summary[model]["no_answer"] += no_answer
        global_summary[model]["total"] += total
        global_summary[model]["error_examples"].extend(error_examples[:3])

        print("-" * 60)
        print(f"Total questions           : {total}")
        print(f"Answered                  : {answered}")
        print(f"Correct                   : {correct}")
        print(f"Incorrect                 : {incorrect}")
        print(f"No answer (None)          : {no_answer}")
        print(f"📈 Accuracy               : {accuracy:.2f}%")


# ================================================================
# GLOBAL SUMMARY EXPORT
# ================================================================
csv_path = os.path.join(OUTPUT_DIR, "prompt_es_metrics.csv")
excel_path = os.path.join(OUTPUT_DIR, "prompt_es_metrics.xlsx")

rows = []

for model in MODELS:
    total = global_summary[model]["total"]
    correct = global_summary[model]["correct"]
    no_answer = global_summary[model]["no_answer"]
    answered = total - no_answer
    accuracy = (correct / answered * 100) if answered > 0 else 0

    rows.append(
        {
            "Model": model,
            "Total": total,
            "Answered": answered,
            "Correct": correct,
            "Incorrect": global_summary[model]["incorrect"],
            "No answer": no_answer,
            "Accuracy (%)": round(accuracy, 2),
        }
    )

pd.DataFrame(rows).to_csv(csv_path, index=False, encoding="utf-8-sig")
pd.DataFrame(rows).to_excel(excel_path, index=False)

print("\n✅ Global results saved:")
print(f"   • CSV  : {csv_path}")
print(f"   • Excel: {excel_path}")
print("\n✅ Spanish-prompt evaluation completed successfully.")
