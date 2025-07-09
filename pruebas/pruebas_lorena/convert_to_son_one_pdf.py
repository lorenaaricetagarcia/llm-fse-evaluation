import fitz  # PyMuPDF
import re
import json

def extract_text_by_page(pdf_path):
    """Extrae el texto completo de cada página del PDF."""
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

def extraer_preguntas_de_texto(texto):
    """Extrae preguntas con sus opciones del texto plano."""
    patron = re.compile(
        r"\n?(\d+)\.\s*(.*?)\s*"
        r"1\.\s*(.*?)\s*"
        r"2\.\s*(.*?)\s*"
        r"3\.\s*(.*?)\s*"
        r"4\.\s*(.*?)(?=\n\d+\.|\Z)",
        re.DOTALL
    )

    preguntas = []
    for match in patron.finditer(texto):
        numero = int(match.group(1))
        enunciado = match.group(2).strip().replace('\n', ' ')
        opciones = [match.group(i).strip().replace('\n', ' ') for i in range(3, 7)]

        preguntas.append({
            "numero": numero,
            "enunciado": enunciado,
            "opciones": opciones
        })

    return preguntas

def main():
    # Ruta del archivo PDF original
    pdf_file_path = "mir.pdf"

    # Paso 1: extraer texto del PDF
    print("📄 Extrayendo texto del PDF...")
    texto_completo = extract_text_by_page(pdf_file_path)

    # Paso 2: guardar el texto como respaldo (opcional)
    with open("mir_extraido_completo.txt", "w", encoding="utf-8") as file:
        file.write(texto_completo)

    # Paso 3: extraer preguntas del texto
    print("🔍 Analizando preguntas...")
    preguntas = extraer_preguntas_de_texto(texto_completo)

    # Paso 4: crear el archivo JSON
    estructura_json = {
        "archivo": pdf_file_path,
        "preguntas": preguntas
    }

    with open("mir_preguntas.json", "w", encoding="utf-8") as salida:
        json.dump(estructura_json, salida, ensure_ascii=False, indent=2)

    print(f"✅ Proceso completado. Se han extraído {len(preguntas)} preguntas.")
    print("📦 Archivo generado: mir_preguntas.json")

if __name__ == "__main__":
    main()
