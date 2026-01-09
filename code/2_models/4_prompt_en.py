"""
Script Title: MIR Model Evaluation with English Prompt (Local LLM Inference via Ollama)
Author: Lorena Ariceta Garcia
TFM: AI-Based Diagnosis: Ally or Risk?
     An Analysis of Language Models

Description
-----------
This script evaluates multiple locally hosted large language models (LLMs) on
MIR-style multiple-choice questions. Each question is presented using a
standardized English prompt that instructs the model to output exactly one
option (1–4) using a strict response format.

The script:
1) Loads preprocessed MIR exam questions and gold answers from JSON files.
2) Sends each question to each model via the Ollama local API.
3) Extracts the selected option (1–4) from the model output.
4) Saves model predictions and raw outputs to per-model JSON files.
5) Computes accuracy metrics (by position to handle possible duplicate IDs).
6) Exports a global summary to CSV and Excel, and also saves a JSON summary.

Input
-----
- Directory containing exam JSON files:
  results/1_data_preparation/6_json_final/

Output
------
- Root output directory:
  /home/xs1/Desktop/Lorena/results/2_models/1_prompt/4_prompt_en/

Within it:
- Per-model subfolders with prediction JSON files
- Execution log:
  log_prompt_en.txt
- Global metrics:
  prompt_en_metrics.csv
  prompt_en_metrics.xlsx
  prompt_en_metrics.json

Requirements
------------
- Python 3.x
- requests
- pandas
- Ollama running locally and serving models at:
  http://localhost:11434/api/generate

Methodological Notes
--------------------
- For MEDICINA and ENFERMERÍA, only questions labeled as "texto" are evaluated,
  skipping image-based questions.
- If duplicated question identifiers are detected, evaluation is performed by
  positional alignment rather than by "numero".
"""

import json
import os
import re
import sys
import csv
import requests
import pandas as pd
from collections import OrderedDict, Counter


# ================================================================
# CONFIGURATION AND OUTPUT
# ================================================================

OUTPUT_DIR = "/home/xs1/Desktop/Lorena/results/2_models/1_prompt/4_prompt_en"
os.makedirs(OUTPUT_DIR, exist_ok=True)


class DualOutput:
    """
    Redirects stdout to both terminal and a log file.
    """
    def __init__(self, path: str):
        self.terminal = sys.__stdout__
        self.log = open(path, "w", encoding="utf-8")

    def write(self, message: str) -> None:
        self.terminal.write(message)
        self.log.write(message)

    def flush(self) -> None:
        self.terminal.flush()
        self.log.flush()


# Redirect all prints to console + file
sys.stdout = DualOutput(os.path.join(OUTPUT_DIR, "log_prompt_en.txt"))


# ================================================================
# MODELS AND ENGLISH PROMPT
# ================================================================

MODELS = [
    "llama3",
    "mistral",
    "gemma",
    "deepseek-coder",
    "deepseek-llm",
    "phi3",
    "phi3:instruct",
]

EN_PROMPT = (
    "You are a medical professional who must answer a clinical exam-style question "
    "(similar to the MIR exam).\n"
    "Carefully read the retrieved CONTEXT and then the QUESTION.\n"
    "If the context contains useful and direct information, use it to answer.\n"
    "If the context does not provide the answer, rely on your general medical knowledge.\n"
    "Your answer must strictly follow this format:\n"
    "'The correct answer is number X' (where X is a number from 1 to 4).\n"
    "Then, add one short sentence with the main justification.\n"
    "Do not answer with 'I'm not sure,' do not provide multiple options, and do not copy the context.\n"
    "Always respond with a single numeric option (1–4) and a concise justification sentence.\n\n"
)

INPUT_DIR = "results/1_data_preparation/6_json_final"
exam_files = [f for f in os.listdir(INPUT_DIR) if f.endswith(".json")]


# ================================================================
# GLOBAL SUMMARY STRUCTURE
# ================================================================

global_summary = {
    model: {
        "correct": 0,
        "wrong": 0,
        "no_response": 0,
        "total": 0,
        "error_examples": [],
    }
    for model in MODELS
}


# ================================================================
# MAIN LOOP (PER FILE, PER MODEL)
# ================================================================

for exam_file in exam_files:
    exam_name = os.path.splitext(exam_file)[0]
    exam_path = os.path.join(INPUT_DIR, exam_file)

    with open(exam_path, "r", encoding="utf-8") as f:
        base_data = json.load(f)

    # Warn if duplicated question numbers exist
    numbers = [q.get("numero") for q in base_data.get("preguntas", [])]
    dup_count = sum(1 for _, c in Counter(numbers).items() if c > 1)
    if dup_count > 0:
        print(
            f"⚠️ {exam_name}: detected {dup_count} duplicated 'numero' values "
            f"— accuracy will be computed by position.\n"
        )

    for model in MODELS:
        print(f"\n🚀 Running model: {model} on exam: {exam_name}")

        run_data = {"preguntas": []}
        model_dir = os.path.join(OUTPUT_DIR, model)
        os.makedirs(model_dir, exist_ok=True)

        # ------------------------------------------------------------
        # 1) Generate and store model outputs
        # ------------------------------------------------------------
        for idx, question in enumerate(base_data.get("preguntas", []), 1):
            # For MEDICINA and ENFERMERÍA, skip non-text questions
            if exam_file in ["ENFERMERÍA.json", "MEDICINA.json"] and question.get("tipo") != "texto":
                continue

            prompt = EN_PROMPT + question.get("enunciado", "") + "\n\n"
            for opt_i, option in enumerate(question.get("opciones", []), 1):
                prompt += f"{opt_i}. {option}\n"

            print(f"\n📤 [{idx}] Sending question to {model}...")

            payload = {"model": model, "prompt": prompt, "stream": False}

            try:
                response = requests.post(
                    "http://localhost:11434/api/generate",
                    json=payload,
                    timeout=180
                )
                response.raise_for_status()

                model_json = response.json()
                raw_text = model_json.get("response", "").strip()

                print("🧠 Model response:")
                print(raw_text)

                match = re.search(r"\b([1-4])\b", raw_text)
                selection = int(match.group(1)) if match else None

                new_question = OrderedDict()
                for key in question:
                    if key not in (model, f"{model}_text"):
                        new_question[key] = question[key]

                new_question[model] = selection
                new_question[f"{model}_text"] = raw_text
                run_data["preguntas"].append(new_question)

            except requests.exceptions.Timeout:
                print("❌ Model timeout.")
            except Exception as e:
                print(f"❌ Error on question {idx}: {e}")

        # Save per-model JSON
        out_json = os.path.join(model_dir, f"{exam_name}_{model}.json")
        with open(out_json, "w", encoding="utf-8") as f_out:
            json.dump(run_data, f_out, ensure_ascii=False, indent=2)
        print(f"\n✅ Saved: {out_json}")

        # ------------------------------------------------------------
        # 2) Metrics (computed by position)
        # ------------------------------------------------------------
        print(f"\n📊 Results for {model.upper()} - Exam: {exam_name}")
        print("-" * 60)

        questions_pred = run_data.get("preguntas", [])
        total = len(questions_pred)
        correct = wrong = no_response = 0
        error_examples = []

        for i, q_pred in enumerate(questions_pred):
            pred = q_pred.get(model)

            # Positional alignment with the original file
            gold = base_data["preguntas"][i].get("respuesta_correcta") if i < len(base_data["preguntas"]) else None

            if pred is None:
                no_response += 1
            elif gold is None:
                # If no gold label is available, skip from scoring
                continue
            elif pred == gold:
                correct += 1
            else:
                wrong += 1
                error_examples.append(
                    {
                        "index": i + 1,
                        "predicted": pred,
                        "correct": gold,
                        "statement": q_pred.get("enunciado", ""),
                    }
                )

        answered = total - no_response
        accuracy = (correct / answered * 100) if answered > 0 else 0.0

        print(f"Total questions        : {total}")
        print(f"Answered by model      : {answered}")
        print(f"Correct answers        : {correct}")
        print(f"Wrong answers          : {wrong}")
        print(f"No response (None)     : {no_response}")
        print(f"📈 Accuracy rate       : {accuracy:.2f}%")

        print("\n🔍 Example errors:")
        for err in error_examples[:5]:
            print(f"  ➤ Q{err['index']}: predicted {err['predicted']}, correct {err['correct']}")
            print(f"    {err['statement']}")

        # Aggregate to global summary
        global_summary[model]["correct"] += correct
        global_summary[model]["wrong"] += wrong
        global_summary[model]["no_response"] += no_response
        global_summary[model]["total"] += total
        global_summary[model]["error_examples"].extend(error_examples[:3])


# ================================================================
# GLOBAL SUMMARY + CSV + EXCEL + JSON EXPORT
# ================================================================

print("\n📊📊📊 GLOBAL MODEL SUMMARY 📊📊📊")

csv_path = os.path.join(OUTPUT_DIR, "prompt_en_metrics.csv")
excel_path = os.path.join(OUTPUT_DIR, "prompt_en_metrics.xlsx")
json_path = os.path.join(OUTPUT_DIR, "prompt_en_metrics.json")

rows = []

for model in MODELS:
    total = global_summary[model]["total"]
    correct = global_summary[model]["correct"]
    wrong = global_summary[model]["wrong"]
    no_response = global_summary[model]["no_response"]
    answered = total - no_response
    accuracy = (correct / answered * 100) if answered > 0 else 0.0

    print(f"\n🧠 Model: {model.upper()}")
    print("-" * 50)
    print(f"Total questions       : {total}")
    print(f"Answered              : {answered}")
    print(f"Correct               : {correct}")
    print(f"Incorrect             : {wrong}")
    print(f"Unanswered (None)     : {no_response}")
    print(f"📈 Accuracy rate       : {accuracy:.2f}%")

    rows.append(
        {
            "Model": model,
            "Total": total,
            "Answered": answered,
            "Correct": correct,
            "Wrong": wrong,
            "No response": no_response,
            "Accuracy (%)": round(accuracy, 2),
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

# Save JSON
summary_json = {
    m: {
        "total": global_summary[m]["total"],
        "correct": global_summary[m]["correct"],
        "wrong": global_summary[m]["wrong"],
        "no_response": global_summary[m]["no_response"],
    }
    for m in MODELS
}
with open(json_path, "w", encoding="utf-8") as f_out:
    json.dump(summary_json, f_out, indent=2, ensure_ascii=False)

print("\n✅ Global results saved:")
print(f"   • JSON : {json_path}")
print(f"   • CSV  : {csv_path}")
print(f"   • Excel: {excel_path}")
print("\n✅ Pipeline completed successfully.")
