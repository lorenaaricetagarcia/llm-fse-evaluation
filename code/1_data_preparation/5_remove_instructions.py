"""
Script Title: Correction of Question Statements and First Answer Option
Author: Lorena Ariceta Garcia
TFM: AI-Based Diagnosis: Ally or Risk?
     An Analysis of Language Models

Description
-----------
This script applies a post-processing correction to previously enriched MIR
question datasets in order to fix formatting artifacts affecting *Question 1*
in some examinations.

In certain cases, the statement of Question 1 is partially embedded in the
first answer option due to PDF extraction inconsistencies. This script:

1) Detects Question 1 in each JSON file.
2) Separates the question statement from the first answer option when both
   are incorrectly merged.
3) Updates the question statement and the first option accordingly.
4) Writes the corrected dataset to a new output directory.

Input directory
---------------
- results/1_data_preparation/3_json_with_answers

Output directory
----------------
- results/1_data_preparation/4_json_corrected

Requirements
------------
- Python 3.x

Methodological Notes
--------------------
- A regular expression is used to identify common leading markers
  (e.g., "1." or "A.") that indicate the boundary between the statement
  and the first answer option.
- Only Question 1 is modified, as the observed extraction issue is
  systematically limited to the first question of the examination.
"""

import os
import json
import re


# ---------------------------------------------------------------------
# 1. Directory configuration
# ---------------------------------------------------------------------
INPUT_DIRECTORY = "results/1_data_preparation/3_json_with_answers"
OUTPUT_DIRECTORY = "results/1_data_preparation/4_json_corrected"

os.makedirs(OUTPUT_DIRECTORY, exist_ok=True)


# ---------------------------------------------------------------------
# 2. Helper function
# ---------------------------------------------------------------------
def split_statement_and_option(text: str) -> tuple[str, str]:
    """
    Splits a merged question statement and first answer option.

    Parameters
    ----------
    text : str
        Raw text potentially containing both the statement and the first option.

    Returns
    -------
    tuple[str, str]
        A tuple containing:
        - corrected statement
        - corrected first answer option
    """
    match = re.search(r"^(.*?)(?:\s*[1A]\.\s*)(.+)$", text.strip())

    if match:
        statement = match.group(1).strip()
        first_option = match.group(2).strip()
        return statement, first_option

    return "", text.strip()


# ---------------------------------------------------------------------
# 3. Apply correction to each file
# ---------------------------------------------------------------------
for filename in os.listdir(INPUT_DIRECTORY):
    if not filename.endswith(".json"):
        continue

    input_path = os.path.join(INPUT_DIRECTORY, filename)
    with open(input_path, "r", encoding="utf-8") as file:
        data = json.load(file)

    for question in data.get("preguntas", []):
        if question.get("numero") == 1:
            options = question.get("opciones", [])
            if options:
                new_statement, first_option = split_statement_and_option(options[0])

                if new_statement:
                    question["enunciado"] = new_statement
                    question["opciones"][0] = first_option
                else:
                    question["enunciado"] = ""

    output_path = os.path.join(OUTPUT_DIRECTORY, filename)
    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)

    print(f"✅ Corrected: {filename}")
