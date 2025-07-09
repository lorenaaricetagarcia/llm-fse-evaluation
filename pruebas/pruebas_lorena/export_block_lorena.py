import fitz  # PyMuPDF

def extract_text_by_page(pdf_path):
    try:
        pdf_document = fitz.open(pdf_path)
        all_text = []

        for page_num in range(len(pdf_document)):
            page = pdf_document[page_num]
            text = page.get_text()
            all_text.append(f"\n--- Página {page_num + 1} ---\n{text.strip()}")

        pdf_document.close()
        return "\n".join(all_text)

    except Exception as e:
        return f"Error leyendo el PDF: {e}"

# Ruta del archivo PDF
pdf_file_path = "mir.pdf"

# Ejecutar extracción
extracted_text = extract_text_by_page(pdf_file_path)

# Guardar el resultado en un archivo de texto
with open("mir_extraido_completo.txt", "w", encoding="utf-8") as file:
    file.write(extracted_text)

print("Extracción completada. Revisa el archivo 'mir_extraido_completo.txt'.")
