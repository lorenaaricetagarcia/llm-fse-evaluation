"""
Script Title: Final Labeling of MIR Question Datasets by Specialization and Year
Author: Lorena Ariceta Garcia
TFM: AI-Based Diagnosis: Ally or Risk?
     An Analysis of Language Models

Description
-----------
This script performs a final labeling step in the data preparation pipeline.
It enriches each processed MIR dataset with explicit, top-level metadata
indicating:

- the medical specialization (titulación)
- the examination year (convocatoria)

The examination year is inferred from the original PDF filename associated
with the questions. Once identified, the metadata is added at the dataset
level to facilitate downstream analysis and model training.

Input directory
---------------
- results/1_data_preparation/5_json_type

Output directory
----------------
- results/1_data_preparation/6_json_final

Requirements
------------
- Python 3.x

Methodological Notes
--------------------
- The examination year is extracted using a regular expression applied to
  the original PDF filename.
- If no valid year is found, the value "UNKNOWN" is assigned.
- The labeling is performed once per dataset, as all questions in a file
  belong to the same specialization and examination year.
"""

import os
import json
import re


# ---------------------------------------------------------------------
# 1. Directory configuration
# ---------------------------------------------------------------------
INPUT_DIRECTORY = "results/1_data_preparation/5_json_type"
OUTPUT_DIRECTORY = "results/1_data_preparation/6_json_final"

os.makedirs(OUTPUT_DIRECTORY, exist_ok=True)


# ---------------------------------------------------------------------
# 2. Process and label each dataset
# ---------------------------------------------------------------------
for filename in os.listdir(INPUT_DIRECTORY):
    if not filename.endswith(".json"):
        continue

    input_path = os.path.join(INPUT_DIRECTORY, filename)

    with open(input_path, "r", encoding="utf-8") as file:
        data = json.load(file)

    specialization = data.get("titulacion", "").strip().upper()
    questions = data.get("preguntas", [])

    examination_year = "UNKNOWN"

    for question in questions:
        source_file = question.get("archivo_origen", "")
        match = re.search(
            rf"Cuaderno_(\d{{4}})_{re.escape(specialization)}_\d+_C",
            source_file
        )
        if match:
            examination_year = match.group(1)
            break

    # Add dataset-level labels
    data["label_specialization"] = specialization
    data["label_examination_year"] = examination_year

    # Save updated dataset
    output_path = os.path.join(OUTPUT_DIRECTORY, filename)
    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)

    print(f"✅ {filename} labeled as {specialization} - {examination_year}")

print("\n📦 All files processed and saved in '6_json_final'.")
