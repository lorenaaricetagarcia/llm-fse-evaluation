"""
Script Title: Classification of FSE Questions by Content Type (Text vs. Image)
Author: Lorena Ariceta Garcia
TFM: AI-Based Diagnosis: Ally or Risk?
     An Analysis of Language Models

Description
-----------
This script performs the final step of the data preparation pipeline by
classifying each FSE examination question according to its content type:
- text-based questions
- image-associated questions

The classification is based on a heuristic rule applied to the question
statement and is restricted to specific specializations in which image-based
questions are known to appear.

For each processed dataset, the script:
1) Assigns a `tipo` attribute ("texto" or "imagen") to every question.
2) Produces per-file statistics on question types.
3) Computes and displays global summary statistics across all datasets.
4) Writes the updated JSON files to a new output directory.

Input directory
---------------
- results/1_data_preparation/4_json_corrected

Output directory
----------------
- results/1_data_preparation/5_json_type

Requirements
------------
- Python 3.x

Methodological Notes
--------------------
- Only selected specializations are considered eligible for image-based
  questions.
- A keyword-based heuristic ("pregunta asociada a la imagen") is used to
  identify image-associated questions within the statement text.
- Global counters are maintained to provide an aggregate overview of the
  dataset composition.
"""

import os
import json


# ---------------------------------------------------------------------
# 1. Directory configuration
# ---------------------------------------------------------------------
INPUT_DIRECTORY = "results/1_data_preparation/4_json_corrected"
OUTPUT_DIRECTORY = "results/1_data_preparation/5_json_type"

os.makedirs(OUTPUT_DIRECTORY, exist_ok=True)


# ---------------------------------------------------------------------
# 2. Specializations eligible for image-based questions
# ---------------------------------------------------------------------
SPECIALIZATIONS_WITH_IMAGES = {"MEDICINA", "ENFERMERÍA"}


# ---------------------------------------------------------------------
# 3. Global counters
# ---------------------------------------------------------------------
total_global = 0
text_global = 0
image_global = 0


# ---------------------------------------------------------------------
# 4. Process each dataset
# ---------------------------------------------------------------------
for filename in os.listdir(INPUT_DIRECTORY):
    if not filename.endswith(".json"):
        continue

    with open(os.path.join(INPUT_DIRECTORY, filename), "r", encoding="utf-8") as file:
        data = json.load(file)

    specialization = data.get("titulacion", "").upper()

    total = 0
    text_count = 0
    image_count = 0

    for question in data.get("preguntas", []):
        total += 1
        statement = question.get("enunciado", "").lower()

        if (
            specialization in SPECIALIZATIONS_WITH_IMAGES
            and "pregunta asociada a la imagen" in statement
        ):
            question["tipo"] = "imagen"
            image_count += 1
        else:
            question["tipo"] = "texto"
            text_count += 1

    # Update global counters
    total_global += total
    text_global += text_count
    image_global += image_count

    # Save updated JSON file
    output_path = os.path.join(OUTPUT_DIRECTORY, filename)
    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)

    print(f"✅ {filename} processed:")
    print(f"   📊 Total questions: {total}")
    print(f"   🖼️ Image-based: {image_count} | 📝 Text-based: {text_count}")


# ---------------------------------------------------------------------
# 5. Global summary
# ---------------------------------------------------------------------
print("\n📦 GLOBAL SUMMARY:")
print(f"🔢 Total questions processed: {total_global}")
print(f"🖼️ Total image-based questions: {image_global}")
print(f"📝 Total text-based questions: {text_global}")

if total_global > 0:
    image_percentage = round((image_global / total_global) * 100, 2)
    text_percentage = round((text_global / total_global) * 100, 2)

    print("\n📊 Global percentages:")
    print(f"   🖼️ Image-based: {image_percentage}%")
    print(f"   📝 Text-based: {text_percentage}%")
else:
    print("⚠️ No questions were processed.")
