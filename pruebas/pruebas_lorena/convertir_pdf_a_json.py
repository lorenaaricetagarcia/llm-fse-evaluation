# Importación de librerías necesarias
import os           # Para manejar rutas de archivos y carpetas
import fitz         # PyMuPDF, para leer contenido de archivos PDF
import json         # Para guardar los datos estructurados en formato JSON
import re           # Para trabajar con expresiones regulares

# Ruta base donde se encuentran los exámenes organizados por titulación
base_dir = r"C:\Users\loren\OneDrive\Escritorio\TFM_UCM\examenes_mir"

# Expresiones regulares para identificar preguntas y opciones en el texto
pregunta_re = re.compile(r"^\s*(\d+)\.\s*(.+)", re.DOTALL)  # Ej. "1. ¿Qué es...?"
opcion_re = re.compile(r"^\s*(\d)\.\s*(.+)")                # Ej. "1. Opción A"

# Función para extraer preguntas y sus opciones a partir del texto completo de un examen
def parse_preguntas(texto):
    lineas = texto.splitlines()  # Divide el texto completo en líneas individuales
    preguntas = []               # Lista para almacenar las preguntas extraídas
    i = 0                        # Índice de la línea actual

    while i < len(lineas):       # Recorre todas las líneas
        linea = lineas[i].strip()                           # Elimina espacios en blanco al inicio y fin
        match = re.match(r"^(\d+)\.\s*(.*)", linea)         # Busca si es una línea que comienza con número y punto
        if match:
            num = int(match.group(1))                       # Número de pregunta
            enunciado = match.group(2)                      # Primer fragmento del enunciado

            # Continua leyendo líneas mientras no aparezca una opción ("1.")
            i += 1
            while i < len(lineas) and not re.match(r"^\s*1\.", lineas[i]):
                enunciado += " " + lineas[i].strip()        # Agrega la línea al enunciado
                i += 1

            # Extraer las 4 opciones de respuesta
            opciones = []
            for _ in range(4):                              # Se espera un máximo de 4 opciones
                if i < len(lineas) and re.match(r"^\s*\d\.", lineas[i]):
                    # Elimina el número y punto (ej. "1. ") para quedarse con el texto de la opción
                    opciones.append(re.sub(r"^\s*\d\.\s*", "", lineas[i].strip()))
                    i += 1
                else:
                    break  # Si no encuentra una opción válida, se detiene

            # Solo guarda preguntas que tengan exactamente 4 opciones
            if len(opciones) == 4:
                preguntas.append({
                    "numero": num,
                    "enunciado": enunciado.strip(),
                    "opciones": opciones
                })
        else:
            i += 1  # Si no es una pregunta, pasa a la siguiente línea

    return preguntas  # Devuelve la lista de preguntas encontradas

# Bucle principal: procesa cada carpeta de titulación
for titulacion in os.listdir(base_dir):
    texto_dir = os.path.join(base_dir, titulacion, "cuaderno_texto")  # Ruta a los PDFs de esa titulación
    if not os.path.isdir(texto_dir):  # Si no existe la carpeta "cuaderno_texto", salta a la siguiente titulación
        continue

    print(f"Procesando titulación: {titulacion}")
    data = []  # Lista para almacenar los datos de todos los PDFs de esa titulación

    # Procesa cada archivo PDF dentro de la carpeta
    for pdf_file in os.listdir(texto_dir):
        if not pdf_file.lower().endswith(".pdf"):  # Solo se procesan archivos .pdf
            continue

        pdf_path = os.path.join(texto_dir, pdf_file)  # Ruta completa al archivo PDF
        print(f"Extrayendo: {pdf_file}")

        # Abre el archivo PDF y extrae todo su texto
        with fitz.open(pdf_path) as doc:
            full_text = "\n".join(page.get_text() for page in doc)  # Concatena el texto de todas las páginas 

        # Llama a la función para extraer preguntas del texto
        preguntas = parse_preguntas(full_text)
        if preguntas:  # Si se encontraron preguntas válidas
            data.append({
                "archivo": pdf_file,
                "preguntas": preguntas
            })

    # Si se extrajeron preguntas de algún archivo, se guardan en un archivo JSON
    if data:
        json_path = os.path.join(base_dir, f"{titulacion.lower()}.json")  # Nombre del archivo JSON
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)  # Guarda con codificación y formato legible
        print(f"Guardado en {json_path}")

# Mensaje final
print("Conversión finalizada.")
