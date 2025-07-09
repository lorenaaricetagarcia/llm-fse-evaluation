import os
import re
import json
import pdfplumber

def extraer_preguntas(texto, nombre_archivo):
    # Buscar bloques que empiecen por número seguido de punto o guión y espacio (1. o 1 -)
    patron = r"(?<=\n|\A)(\d{1,3})[\.\-]\s+(.*?)(?=(?:\n\d{1,3}[\.\-]\s)|\Z)"
    bloques = re.findall(patron, texto, flags=re.DOTALL)

    preguntas = []
    for numero, contenido in bloques:
        lineas = contenido.strip().split("\n")
        enunciado = []
        opciones = {}
        letra = '1'

        for linea in lineas:
            # Detectar opciones tipo 1. ..., 2. ..., A) ..., etc.
            op_match = re.match(r"^\s*(\d+|[A-Da-d])[\.\)]\s+(.*)", linea)
            if op_match:
                letra = op_match.group(1)
                opciones[letra] = op_match.group(2).strip()
            else:
                enunciado.append(linea.strip())

        pregunta = {
            "numero": numero,
            "pdf": nombre_archivo,
            "pregunta": " ".join(enunciado).strip(),
            "opciones": opciones,
            "respuesta_correcta": None
        }
        preguntas.append(pregunta)
    return preguntas

def procesar_pdfs(directorio):
    todas = []
    for archivo in os.listdir(directorio):
        if archivo.endswith(".pdf"):
            ruta = os.path.join(directorio, archivo)
            print(f"📄 Procesando: {archivo}")
            try:
                with pdfplumber.open(ruta) as pdf:
                    texto = ""
                    for pagina in pdf.pages:
                        texto += pagina.extract_text(x_tolerance=1, y_tolerance=1) + "\n"
                    preguntas = extraer_preguntas(texto, archivo)
                    todas.extend(preguntas)
            except Exception as e:
                print(f"❌ Error procesando {archivo}: {e}")
    return todas

if __name__ == "__main__":
    resultados = procesar_pdfs("mir_pdfs")
    with open("all_mir_questions.json", "w", encoding="utf-8") as f:
        json.dump({"preguntas": resultados}, f, indent=2, ensure_ascii=False)
    print("✅ ¡Preguntas guardadas correctamente en 'all_mir_questions.json'!")
