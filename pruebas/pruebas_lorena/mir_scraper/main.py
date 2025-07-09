import os
import re
import json
from prueba_mirial import download_all_pdfs
from extract_pdf import extract_text_from_pdf
from create_json import extract_questions_from_text

# Paso 1: Descargar todos los PDFs
download_all_pdfs()

# Paso 2: Procesar todos los PDFs
pdf_folder = "mir_pdfs"
output_json_path = "all_mir_questions.json"
preguntas_por_anyo = {}

for filename in os.listdir(pdf_folder):
    if filename.lower().endswith(".pdf"):
        pdf_path = os.path.join(pdf_folder, filename)
        print(f"📄 Procesando: {filename}")

        match_year = re.search(r"\b(19|20)\d{2}\b", filename)
        year = match_year.group() if match_year else "desconocido"

        text = extract_text_from_pdf(pdf_path)
        preguntas = extract_questions_from_text(text)
        preguntas_por_anyo[year] = preguntas

# Paso 3: Guardar en JSON
with open(output_json_path, "w", encoding="utf-8") as f:
    json.dump(preguntas_por_anyo, f, ensure_ascii=False, indent=2)

print(f"\n✅ ¡Todas las preguntas han sido guardadas en '{output_json_path}'!")
