"""
Script Title: Extraction of MIR Questions from Text-Based PDF Booklets
Author: Lorena Ariceta Garcia
TFM: AI-Based Diagnosis: Ally or Risk?
     An Analysis of Language Models

Description
-----------
This script extracts multiple-choice questions from the MIR examination
*text-based* PDF booklets previously downloaded and organized by specialization.

For each specialization, the script:
1) Reads every PDF file located under the `text_booklet/` directory.
2) Extracts the full text from all pages using PyMuPDF (fitz).
3) Identifies questions and four answer options using a regular expression.
4) Exports one JSON file per specialization containing all extracted questions.

Expected input directory structure (from Script 1_extract_pdf_from_ministerio.py)
--------------------------------------------------
FSE_exams_v0/
    └── <specialization>/
        ├── text_booklet/
        └── image_booklet/

Output directory structure
--------------------------
results/1_data_preparation/1_json_specialization/
    └── <specialization>.json

Requirements
------------
- Python 3.x
- PyMuPDF (fitz)

Methodological Notes
--------------------
- Text is extracted page-by-page and concatenated into a single string.
- A regex-based approach is used to capture:
    - Question number
    - Question statement
    - Options 1 to 4
- Extracted content is normalized by replacing line breaks with spaces.
- The resulting JSON preserves UTF-8 encoding and uses readable indentation.
"""

import os
import re
import json
import fitz  # PyMuPDF
from typing import List, Dict


# ---------------------------------------------------------------------
# 1. PDF text extraction
# ---------------------------------------------------------------------
def extract_text_from_pdf(pdf_path: str) -> str:
    """
    Extracts the full text from all pages of a PDF file.

    Parameters
    ----------
    pdf_path : str
        Absolute or relative path to the PDF file.

    Returns
    -------
    str
        Concatenated text from all pages, or an empty string if an error occurs.
    """
    try:
        pdf_document = fitz.open(pdf_path)
        all_text = []

        for page_num in range(len(pdf_document)):
            page = pdf_document[page_num]
            page_text = page.get_text()
            all_text.append(page_text.strip())

        pdf_document.close()
        return "\n".join(all_text)

    except Exception as exception:
        print(f"⚠️ Error while reading {pdf_path}: {exception}")
        return ""


# ---------------------------------------------------------------------
# 2. Question and answer option extraction
# ---------------------------------------------------------------------
def extract_questions(text: str, source_pdf_name: str) -> list[dict]:
    """
    Extracts questions and answer options from text using a regular expression.

    Parameters
    ----------
    text : str
        Full text extracted from the PDF.
    source_pdf_name : str
        Name of the source PDF file (stored for traceability).

    Returns
    -------
    list[dict]
        A list of dictionaries, each containing:
            - question_number (int)
            - statement (str)
            - options (list[str])
            - source_file (str)
    """
    pattern = re.compile(
        r"\n?(\d+)\.\s*(.*?)\s*"
        r"1\.\s*(.*?)\s*"
        r"2\.\s*(.*?)\s*"
        r"3\.\s*(.*?)\s*"
        r"4\.\s*(.*?)(?=\n\d+\.|\Z)",
        re.DOTALL,
    )

    questions = []
    for match in pattern.finditer(text):
        question_number = int(match.group(1))
        statement = match.group(2).strip().replace("\n", " ")
        options = [
            match.group(i).strip().replace("\n", " ")
            for i in range(3, 7)
        ]

        questions.append(
            {
                "numero": question_number,
                "enunciado": statement,
                "opciones": options,
                "archivo_origen": source_pdf_name,
            }
        )

    return questions


# ---------------------------------------------------------------------
# 3. Specialization-level processing
# ---------------------------------------------------------------------
def process_specialization(specialization_path: str, output_dir: str) -> None:
    """
    Processes all text-based PDF booklets within a given specialization.

    Parameters
    ----------
    specialization_path : str
        Path to the specialization folder (e.g., FSE_exams_v0/<specialization>).
    output_dir : str
        Output directory where the JSON file will be saved.
    """
    text_booklet_path = os.path.join(specialization_path, "text_booklet")

    if not os.path.isdir(text_booklet_path):
        print(f"⚠️ Folder 'text_booklet' not found in: {specialization_path}")
        return

    all_questions = []

    for filename in os.listdir(text_booklet_path):
        if filename.lower().endswith(".pdf"):
            pdf_path = os.path.join(text_booklet_path, filename)
            print(f"📄 Processing: {pdf_path}")

            text = extract_text_from_pdf(pdf_path)
            extracted = extract_questions(text, filename)
            all_questions.extend(extracted)

    specialization_name = os.path.basename(specialization_path)

    output_payload = {
        "titulacion": specialization_name,
        "preguntas": all_questions,
    }

    os.makedirs(output_dir, exist_ok=True)

    json_path = os.path.join(output_dir, f"{specialization_name}.json")
    with open(json_path, "w", encoding="utf-8") as file:
        json.dump(output_payload, file, ensure_ascii=False, indent=2)

    print(f"✅ Saved: {json_path} ({len(all_questions)} questions)")


# ---------------------------------------------------------------------
# 4. Main execution
# ---------------------------------------------------------------------
def main() -> None:
    """
    Iterates through all specializations in the base directory and generates
    one JSON file per specialization containing extracted questions.
    """
    base_dir = "FSE_exams_v0"  # Root directory containing specializations
    output_dir = "results/1_data_preparation/1_json_specialization"

    if not os.path.isdir(base_dir):
        print(f"❌ Base directory not found: {base_dir}")
        return

    for specialization in os.listdir(base_dir):
        specialization_path = os.path.join(base_dir, specialization)

        if os.path.isdir(specialization_path):
            print(f"\n🧪 Processing specialization: {specialization}")
            process_specialization(specialization_path, output_dir)


if __name__ == "__main__":
    main()
