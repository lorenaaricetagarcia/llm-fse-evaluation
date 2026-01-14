"""
Script Title: Batch Evaluation of LLMs on FSE/MIR Multiple-Choice Exams (Spanish Prompt)
Author: Lorena Ariceta Garcia
TFM: AI-Based Diagnosis: Ally or Risk?
     An Analysis of Language Models

Description
-----------
This script evaluates multiple local language models (via Ollama) on a set of
multiple-choice exam questions stored in JSON format, using a Spanish clinical
instruction prompt.

For each JSON exam file and each model, the script:
1) Loads the exam dataset (JSON).
2) Builds the evaluation question list (optionally filtering to text questions).
3) Sends each question to the model using the Ollama API, prepending a Spanish prompt.
4) Extracts the first standalone digit (1..4) from the model’s response.
5) Saves a per-model JSON containing predictions + raw model text.
6) Computes evaluation metrics aligned with the evaluated subset of questions.
7) Aggregates global metrics and exports results to CSV and Excel.

Output directory structure
--------------------------
<OUTPUT_DIR>/
    ├── log_prompt_es.txt
    ├── prompt_es_metrics.csv
    ├── prompt_es_metrics.xlsx
    └── <model>/
        └── <exam>_<model>.json

Requirements
------------
- Python 3.x
- requests
- pandas

Methodological Notes
--------------------
- The Ollama endpoint is used with `stream=False` for simple request/response.
- Answer selection is parsed using a regex capturing the first standalone digit 1..4.
- The evaluated question list is built once per file to ensure alignment between:
  (a) ground-truth answers and (b) model predictions.
- Some files are filtered to include only text-based questions as intended.
- A Spanish prompt enforces a constrained response format for improved parsing.
"""

import csv
import json
import os
import re
import sys
from collections import Counter, OrderedDict
from pathlib import Path

import pandas as pd
import requests


# ---------------------------------------------------------------------
# 1. Output configuration
# ---------------------------------------------------------------------
BASE_DIR = Path(os.getenv("FSE_BASE_DIR", Path(__file__).resolve().parents[2]))
OUTPUT_DIR = Path(
    os.getenv(
        "FSE_OUTPUT_DIR",
        BASE_DIR / "results/2_models/2_prompt_es",
    )
)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


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


sys.stdout = DualOutput(str(OUTPUT_DIR / "log_prompt_es.txt"))


# ---------------------------------------------------------------------
# 2. Models and Spanish prompt
# ---------------------------------------------------------------------
MODELS = [
    "llama3",
    "mistral",
    "gemma",
    "deepseek-coder",
    "deepseek-llm",
    "phi3",
    "phi3:instruct",
]

SPANISH_PROMPT = (
    "Eres un profesional médico que debe responder una pregunta tipo examen clínico (MIR).\n"
    "Lee cuidadosamente la PREGUNTA y las OPCIONES.\n"
    "Aplica tu conocimiento clínico general para seleccionar la respuesta correcta.\n"
    "Responde estrictamente en el formato: 'La respuesta correcta es la número X.'\n"
    "Después añade una breve frase justificativa.\n"
)

INPUT_DIR = Path(
    os.getenv(
        "FSE_INPUT_DIR",
        BASE_DIR / "results/1_data_preparation/6_json_final",
    )
)
json_files = [path.name for path in INPUT_DIR.glob("*.json")]

OLLAMA_ENDPOINT = "http://localhost:11434/api/generate"
REQUEST_TIMEOUT_SECONDS = 180


def build_eval_questions(json_filename: str, base_data: dict) -> list:
    """Build the list of questions that will be prompted AND evaluated."""
    questions = base_data.get("preguntas", [])
    if json_filename in ["ENFERMERÍA.json", "MEDICINA.json"]:
        questions = [q for q in questions if q.get("tipo") == "texto"]
    return questions


# ---------------------------------------------------------------------
# 3. Global results structure (RAG-like)
# ---------------------------------------------------------------------
global_summary = {
    model: {
        "total": 0,
        "correct": 0,
        "errors": 0,
        "no_answer": 0,
        "no_available": 0,
        "error_examples": [],
    }
    for model in MODELS
}


# ---------------------------------------------------------------------
# 4. Main loop (per file, per model)
# ---------------------------------------------------------------------
for json_filename in json_files:
    exam_name = os.path.splitext(json_filename)[0]
    json_path = INPUT_DIR / json_filename

    with open(json_path, "r", encoding="utf-8") as f:
        base_data = json.load(f)

    # Warn about duplicated question numbers
    numbers = [q.get("numero") for q in base_data.get("preguntas", [])]
    duplicate_count = sum(1 for _, c in Counter(numbers).items() if c > 1)
    if duplicate_count > 0:
        print(
            f"⚠️ {json_filename}: {duplicate_count} duplicated question numbers detected "
            "(evaluation will be positional).\n"
        )

    # ✅ Filter once, then use this list everywhere (prompt + GT)
    eval_questions = build_eval_questions(json_filename, base_data)

    for model in MODELS:
        print(f"\n🚀 Processing exam '{exam_name}' with model: {model}")
        model_data = {"preguntas": []}

        model_dir = OUTPUT_DIR / model
        model_dir.mkdir(parents=True, exist_ok=True)

        # -------------------------------------------------------------
        # 4.1 Generate model responses
        # -------------------------------------------------------------
        for idx, question in enumerate(eval_questions, start=1):
            prompt = SPANISH_PROMPT + question.get("enunciado", "") + "\n\n"
            for opt_i, option in enumerate(question.get("opciones", []), start=1):
                prompt += f"{opt_i}. {option}\n"

            print(f"\n📤 [{idx}] Sending question to {model}...")

            payload = {"model": model, "prompt": prompt, "stream": False}

            raw_text = ""
            selection = None

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

                match = re.search(r"\b([1-4])\b", raw_text)
                selection = int(match.group(1)) if match else None

            except requests.exceptions.Timeout:
                print("❌ Model timeout.")
            except Exception as exc:
                print(f"❌ Error on question {idx}: {exc}")

            new_question = OrderedDict()
            for key in question:
                if key not in (model, f"{model}_texto"):
                    new_question[key] = question[key]

            new_question[model] = selection
            new_question[f"{model}_texto"] = raw_text
            model_data["preguntas"].append(new_question)

        # Save per-model JSON
        output_json = model_dir / f"{exam_name}_{model}.json"
        with open(output_json, "w", encoding="utf-8") as f_out:
            json.dump(model_data, f_out, ensure_ascii=False, indent=2)

        print(f"\n✅ Saved: {output_json}")

        # -------------------------------------------------------------
        # 4.2 Metrics (RAG-aligned)
        # -------------------------------------------------------------
        questions = model_data.get("preguntas", [])
        total = len(questions)

        correct = 0
        errors = 0
        no_answer = 0
        no_available = 0
        error_examples = []

        for i, q in enumerate(questions):
            pred = q.get(model)
            gt = eval_questions[i].get("respuesta_correcta")

            if gt is None:
                no_available += 1

            if pred is None:
                no_answer += 1
                continue

            # Only evaluable if GT exists
            if gt is None:
                continue

            if pred == gt:
                correct += 1
            else:
                errors += 1
                error_examples.append(
                    {
                        "index": i + 1,
                        "predicted": pred,
                        "correct": gt,
                        "stem": q.get("enunciado", ""),
                    }
                )

        answered_evaluable = correct + errors
        accuracy = (
            (correct / answered_evaluable * 100) if answered_evaluable > 0 else 0.0
        )

        global_summary[model]["total"] += total
        global_summary[model]["correct"] += correct
        global_summary[model]["errors"] += errors
        global_summary[model]["no_answer"] += no_answer
        global_summary[model]["no_available"] += no_available
        global_summary[model]["error_examples"].extend(error_examples[:3])

        print("-" * 60)
        print(f"Total questions                : {total}")
        print(f"Correct                        : {correct}")
        print(f"Errors                         : {errors}")
        print(f"No answer (pred=None)          : {no_answer}")
        print(f"No available answer (gt=None)  : {no_available}")
        print(f"Answered (evaluable)           : {answered_evaluable}")
        print(f"📈 Accuracy (evaluable)         : {accuracy:.2f}%")


# ---------------------------------------------------------------------
# 5. Global summary export (RAG-like)
# ---------------------------------------------------------------------
csv_path = OUTPUT_DIR / "prompt_es_metrics.csv"
excel_path = OUTPUT_DIR / "prompt_es_metrics.xlsx"

rows = []
for model in MODELS:
    total = global_summary[model]["total"]
    correct = global_summary[model]["correct"]
    errors = global_summary[model]["errors"]
    no_answer = global_summary[model]["no_answer"]
    no_available = global_summary[model]["no_available"]

    answered_evaluable = correct + errors
    accuracy = (correct / answered_evaluable * 100) if answered_evaluable > 0 else 0.0
    pct_no_answer = (no_answer / total * 100) if total > 0 else 0.0
    pct_no_available = (no_available / total * 100) if total > 0 else 0.0

    rows.append(
        {
            "Model": model,
            "Total questions": total,
            "Correct": correct,
            "Errors": errors,
            "Accuracy (%)": round(accuracy, 2),
            "No answer": no_answer,
            "% no answer": round(pct_no_answer, 2),
            "No available answer": no_available,
            "% no available": round(pct_no_available, 2),
        }
    )

pd.DataFrame(rows).to_csv(csv_path, index=False, encoding="utf-8-sig")
pd.DataFrame(rows).to_excel(excel_path, index=False)

print("\n✅ Global results saved:")
print(f"   • CSV  : {csv_path}")
print(f"   • Excel: {excel_path}")
print("\n✅ Spanish-prompt evaluation completed successfully.")
