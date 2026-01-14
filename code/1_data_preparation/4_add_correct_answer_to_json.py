"""
Script Title: Integration of FSE Questions with Official Correct Answers
Author: Lorena Ariceta Garcia
TFM: AI-Based Diagnosis: Ally or Risk?
     An Analysis of Language Models

Description
-----------
This script merges previously extracted FSE examination questions with their
corresponding official correct answers (Version 0 only).

The script:
1) Loads all JSON files containing questions grouped by specialization.
2) Loads all JSON files containing official answers, indexed by year and specialization.
3) Matches each question to its correct answer using:
   - the question number
   - the examination year inferred from the source PDF filename
4) Enriches each question with:
   - the correct answer
   - the specialization label
   - the examination year (convocatoria)
5) Writes a new JSON file per specialization containing the enriched data.

Input directories
-----------------
- results/1_data_preparation/1_json_por_titulacion
  JSON files with extracted questions.

- results/1_data_preparation/2_respuestas_json
  JSON files with official answers (Version 0 only).

Output directory
----------------
- results/1_data_preparation/3_json_con_respuesta

Requirements
------------
- Python 3.x
"""

import os
import json
import re


# ---------------------------------------------------------------------
# 1. Directory configuration
# ---------------------------------------------------------------------
QUESTIONS_DIR = "results/1_data_preparation/1_json_por_titulacion"
ANSWERS_DIR = "results/1_data_preparation/2_respuestas_json"
OUTPUT_DIR = "results/1_data_preparation/3_json_con_respuesta"

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ---------------------------------------------------------------------
# 2. Load official answers into memory
# ---------------------------------------------------------------------
answers_by_file: dict[str, dict[int, int]] = {}

for filename in os.listdir(ANSWERS_DIR):
    if filename.endswith(".json"):
        file_path = os.path.join(ANSWERS_DIR, filename)

        with open(file_path, "r", encoding="utf-8") as file:
            raw_answers = json.load(file)
            answers_by_file[filename] = {
                int(question_number): answer
                for question_number, answer in raw_answers.items()
            }


# ---------------------------------------------------------------------
# 3. Process each questions file
# ---------------------------------------------------------------------
for filename in os.listdir(QUESTIONS_DIR):
    if not filename.endswith(".json"):
        continue

    file_path = os.path.join(QUESTIONS_DIR, filename)
    with open(file_path, "r", encoding="utf-8") as file:
        data = json.load(file)

    specialization = data.get("titulacion", "").strip().upper()
    questions = data.get("preguntas", [])

    for question in questions:
        source_file = question.get("archivo_origen", "")
        question_number = question.get("numero")

        # Extract examination year from the source PDF filename
        # Example pattern: "Cuaderno_2020_MEDICINA_0_C.pdf"
        match = re.search(
            rf"Cuaderno_(\d{{4}})_{re.escape(specialization)}_\d+_C",
            source_file
        )

        if not match:
            question["respuesta_correcta"] = None
            continue

        year = match.group(1)

        # Answer files follow the format: <SPECIALIZATION>_<YEAR>.json
        answers_filename = f"{specialization}_{year}.json"
        answers = answers_by_file.get(answers_filename)

        if answers:
            question["respuesta_correcta"] = answers.get(question_number)
        else:
            question["respuesta_correcta"] = None

        # Add explicit metadata
        question["titulacion"] = specialization
        question["convocatoria"] = year

    # -----------------------------------------------------------------
    # 4. Save enriched output file
    # -----------------------------------------------------------------
    output_path = os.path.join(OUTPUT_DIR, filename)

    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)

    print(f"✅ Saved to {output_path}")
