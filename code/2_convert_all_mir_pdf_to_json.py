import fitz  # PyMuPDF
import re
import json
import os

def extract_text_from_pdf(pdf_path):
    """Extrae texto completo de todas las páginas del PDF."""
    try:
        pdf_document = fitz.open(pdf_path)
        all_text = []

        for page_num in range(len(pdf_document)):
            page = pdf_document[page_num]
            text = page.get_text()
            all_text.append(text.strip())

        pdf_document.close()
        return "\n".join(all_text)

    except Exception as e:
        print(f"⚠️ Error al leer {pdf_path}: {e}")
        return ""

def extract_questions(text, nombre_pdf):
    """Extrae preguntas y opciones del texto usando regex."""
    patron = re.compile(
        r"\n?(\d+)\.\s*(.*?)\s*"
        r"1\.\s*(.*?)\s*"
        r"2\.\s*(.*?)\s*"
        r"3\.\s*(.*?)\s*"
        r"4\.\s*(.*?)(?=\n\d+\.|\Z)",
        re.DOTALL
    )

    preguntas = []
    for match in patron.finditer(text):
        numero = int(match.group(1))
        enunciado = match.group(2).strip().replace('\n', ' ')
        opciones = [match.group(i).strip().replace('\n', ' ') for i in range(3, 7)]

        preguntas.append({
            "numero": numero,
            "enunciado": enunciado,
            "opciones": opciones,
            "archivo_origen": nombre_pdf
        })

    return preguntas

def procesar_titulacion(titulacion_path, salida_dir):
    """Procesa todos los PDFs de texto en una titulación dada."""
    cuaderno_texto_path = os.path.join(titulacion_path, "cuaderno_texto")
    if not os.path.isdir(cuaderno_texto_path):
        print(f"⚠️ No se encontró carpeta cuaderno_texto en {titulacion_path}")
        return

    preguntas_total = []
    for filename in os.listdir(cuaderno_texto_path):
        if filename.lower().endswith(".pdf"):
            pdf_path = os.path.join(cuaderno_texto_path, filename)
            print(f"📄 Procesando: {pdf_path}")
            texto = extract_text_from_pdf(pdf_path)
            preguntas = extract_questions(texto, filename)
            preguntas_total.extend(preguntas)

    # Guardar JSON por titulación
    nombre_titulacion = os.path.basename(titulacion_path)
    output = {
        "titulacion": nombre_titulacion,
        "preguntas": preguntas_total
    }

    os.makedirs(salida_dir, exist_ok=True)
    json_path = os.path.join(salida_dir, f"{nombre_titulacion}.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"✅ Guardado: {json_path} ({len(preguntas_total)} preguntas)")

def main():
    base_dir = "examenes_mir_v_0"       # Carpeta raíz donde están las titulaciones
    salida_dir = "results/1_json_por_titulacion"  # Carpeta de salida

    for titulacion in os.listdir(base_dir):
        titulacion_path = os.path.join(base_dir, titulacion)
        if os.path.isdir(titulacion_path):
            print(f"\n🧪 Procesando titulación: {titulacion}")
            procesar_titulacion(titulacion_path, salida_dir)

if __name__ == "__main__":
    main()
