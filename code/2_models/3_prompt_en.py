"""
Script Title: Batch Evaluation of LLMs on FSE/MIR Multiple-Choice Exams (English Prompt)
Author: Lorena Ariceta Garcia
TFM: AI-Based Diagnosis: Ally or Risk?
     An Analysis of Language Models

Description
-----------
This script evaluates multiple local language models (via Ollama) on a set of
multiple-choice exam questions stored in JSON format, using an English clinical
instruction prompt (prompt-only, no retrieved context).

For each JSON exam file and each model, the script:
1) Loads the exam dataset (JSON).
2) Builds the evaluation question list (optionally filtering to text questions).
3) Sends each question to the model using the Ollama API, prepending an English prompt.
4) Extracts the first standalone digit (1..4) from the model’s response.
5) Saves a per-model JSON containing predictions + raw model text.
6) Computes evaluation metrics aligned with the evaluated subset of questions.
7) Aggregates global metrics and exports results to CSV, Excel, and JSON.

Output directory structure
--------------------------
<OUTPUT_DIR>/
    ├── log_prompt_en.txt
    ├── prompt_en_metrics.csv
    ├── prompt_en_metrics.xlsx
    ├── prompt_en_metrics.json
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
- The English prompt enforces a constrained response format for improved parsing.
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
# 1. Configuration and output
# ---------------------------------------------------------------------
BASE_DIR = Path(os.getenv("FSE_BASE_DIR", Path(__file__).resolve().parents[2]))
OUTPUT_DIR = Path(
    os.getenv(
        "FSE_OUTPUT_DIR",
        BASE_DIR / "results/2_models/1_prompt/3_prompt_en",
    )
)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


class DualOutput:
    """Redirects stdout to both terminal and a log file."""

    def __init__(self, path: str):
        self.terminal = sys.__stdout__
        self.log = open(path, "w", encoding="utf-8")

    def write(self, message: str) -> None:
        self.terminal.write(message)
        self.log.write(message)

    def flush(self) -> None:
        self.terminal.flush()
        self.log.flush()


sys.stdout = DualOutput(str(OUTPUT_DIR / "log_prompt_en.txt"))


# ---------------------------------------------------------------------
# 2. Models and English prompt (prompt-only)
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

# ✅ Prompt-only template: no mention of retrieved context
EN_PROMPT = (
    "You are a medical professional answering a clinical exam-style question (similar to the Spanish MIR).\n"
    "Carefully read the QUESTION and the ANSWER OPTIONS.\n"
    "Apply your general medical knowledge to select the correct answer.\n"
    "Answer strictly in the format: 'The correct answer is number X.'\n"
    "Then add a short explanatory sentence.\n"
)

INPUT_DIR = Path(
    os.getenv(
        "FSE_INPUT_DIR",
        BASE_DIR / "results/1_data_preparation/6_json_final",
    )
)
exam_files = [path.name for path in INPUT_DIR.glob("*.json")]

OLLAMA_ENDPOINT = "http://localhost:11434/api/generate"
REQUEST_TIMEOUT_SECONDS = 180


def build_eval_questions(exam_file: str, base_data: dict) -> list:
    """Build the list of questions that will be prompted AND evaluated."""
    questions = base_data.get("preguntas", [])
    if exam_file in ["ENFERMERÍA.json", "MEDICINA.json"]:
        questions = [q for q in questions if q.get("tipo") == "texto"]
    return questions


# ---------------------------------------------------------------------
# 3. Global summary structure (RAG-like)
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
for exam_file in exam_files:
    exam_name = os.path.splitext(exam_file)[0]
    exam_path = INPUT_DIR / exam_file

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

    # ✅ Filter once; use for both prompting and GT alignment
    eval_questions = build_eval_questions(exam_file, base_data)

    for model in MODELS:
        print(f"\n🚀 Running model: {model} on exam: {exam_name}")

        run_data = {"preguntas": []}
        model_dir = OUTPUT_DIR / model
        model_dir.mkdir(parents=True, exist_ok=True)

        # -------------------------------------------------------------
        # 4.1 Generate and store model outputs
        # -------------------------------------------------------------
        for idx, question in enumerate(eval_questions, 1):
            prompt = EN_PROMPT + question.get("enunciado", "") + "\n\n"
            for opt_i, option in enumerate(question.get("opciones", []), 1):
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

                model_json = response.json()
                raw_text = model_json.get("response", "").strip()

                print("🧠 Model response:")
                print(raw_text)

                match = re.search(r"\b([1-4])\b", raw_text)
                selection = int(match.group(1)) if match else None

            except requests.exceptions.Timeout:
                print("❌ Model timeout.")
            except Exception as e:
                print(f"❌ Error on question {idx}: {e}")

            new_question = OrderedDict()
            for key in question:
                if key not in (model, f"{model}_text"):
                    new_question[key] = question[key]

            new_question[model] = selection
            new_question[f"{model}_text"] = raw_text
            run_data["preguntas"].append(new_question)

        # Save per-model JSON
        out_json = model_dir / f"{exam_name}_{model}.json"
        with open(out_json, "w", encoding="utf-8") as f_out:
            json.dump(run_data, f_out, ensure_ascii=False, indent=2)
        print(f"\n✅ Saved: {out_json}")

        # -------------------------------------------------------------
        # 4.2 Metrics (RAG-aligned)
        # -------------------------------------------------------------
        print(f"\n📊 Results for {model.upper()} - Exam: {exam_name}")
        print("-" * 60)

        questions_pred = run_data.get("preguntas", [])
        total = len(questions_pred)

        correct = 0
        errors = 0
        no_answer = 0
        no_available = 0
        error_examples = []

        for i, q_pred in enumerate(questions_pred):
            pred = q_pred.get(model)
            gold = eval_questions[i].get("respuesta_correcta")

            if gold is None:
                no_available += 1

            if pred is None:
                no_answer += 1
                continue

            # Only evaluable if gold exists
            if gold is None:
                continue

            if pred == gold:
                correct += 1
            else:
                errors += 1
                error_examples.append(
                    {
                        "index": i + 1,
                        "predicted": pred,
                        "correct": gold,
                        "statement": q_pred.get("enunciado", ""),
                    }
                )

        answered_evaluable = correct + errors
        accuracy = (
            (correct / answered_evaluable * 100) if answered_evaluable > 0 else 0.0
        )

        print(f"Total questions                : {total}")
        print(f"Correct                        : {correct}")
        print(f"Errors                         : {errors}")
        print(f"No answer (pred=None)          : {no_answer}")
        print(f"No available answer (gt=None)  : {no_available}")
        print(f"Answered (evaluable)           : {answered_evaluable}")
        print(f"📈 Accuracy (evaluable)         : {accuracy:.2f}%")

        print("\n🔍 Example errors:")
        for err in error_examples[:5]:
            print(
                f"  ➤ Q{err['index']}: predicted {err['predicted']}, correct {err['correct']}"
            )
            print(f"    {err['statement']}")

        # Aggregate to global summary
        global_summary[model]["total"] += total
        global_summary[model]["correct"] += correct
        global_summary[model]["errors"] += errors
        global_summary[model]["no_answer"] += no_answer
        global_summary[model]["no_available"] += no_available
        global_summary[model]["error_examples"].extend(error_examples[:3])


# ---------------------------------------------------------------------
# 5. Global summary + CSV + Excel + JSON export (RAG-like)
# ---------------------------------------------------------------------
print("\n📊📊📊 GLOBAL MODEL SUMMARY 📊📊📊")

csv_path = OUTPUT_DIR / "prompt_en_metrics.csv"
excel_path = OUTPUT_DIR / "prompt_en_metrics.xlsx"
json_path = OUTPUT_DIR / "prompt_en_metrics.json"

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

    print(f"\n🧠 Model: {model.upper()}")
    print("-" * 50)
    print(f"Total questions                : {total}")
    print(f"Correct                        : {correct}")
    print(f"Errors                         : {errors}")
    print(f"No answer (pred=None)          : {no_answer}")
    print(f"No available answer (gt=None)  : {no_available}")
    print(f"Answered (evaluable)           : {answered_evaluable}")
    print(f"📈 Accuracy (evaluable)         : {accuracy:.2f}%")

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
        "Total questions": global_summary[m]["total"],
        "Correct": global_summary[m]["correct"],
        "Errors": global_summary[m]["errors"],
        "Accuracy (%)": round(
            (
                global_summary[m]["correct"]
                / (global_summary[m]["correct"] + global_summary[m]["errors"])
                * 100
            )
            if (global_summary[m]["correct"] + global_summary[m]["errors"]) > 0
            else 0.0,
            2,
        ),
        "No answer": global_summary[m]["no_answer"],
        "% no answer": round(
            (global_summary[m]["no_answer"] / global_summary[m]["total"] * 100)
            if global_summary[m]["total"] > 0
            else 0.0,
            2,
        ),
        "No available answer": global_summary[m]["no_available"],
        "% no available": round(
            (global_summary[m]["no_available"] / global_summary[m]["total"] * 100)
            if global_summary[m]["total"] > 0
            else 0.0,
            2,
        ),
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
