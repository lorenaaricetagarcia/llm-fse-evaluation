import os
import json
import re

# Carpetas
carpeta_preguntas = "1_json_por_titulacion"     # Donde están los JSON con las preguntas
carpeta_respuestas = "2_respuestas_json"        # Donde están los JSON con las respuestas (solo versión 0)
carpeta_salida = "3_json_con_respuesta"         # Carpeta donde se guardarán los JSON con respuestas añadidas
os.makedirs(carpeta_salida, exist_ok=True)      # Crear carpeta de salida si no existe

# === CARGAR RESPUESTAS EN MEMORIA ===
respuestas_por_archivo = {}  # Diccionario para guardar respuestas por archivo

for archivo in os.listdir(carpeta_respuestas):
    if archivo.endswith(".json"):  # Solo JSON
        ruta = os.path.join(carpeta_respuestas, archivo)
        with open(ruta, "r", encoding="utf-8") as f:
            respuestas_por_archivo[archivo] = {
                int(k): v for k, v in json.load(f).items()
            }

# === PROCESAR CADA ARCHIVO DE PREGUNTAS ===
for archivo in os.listdir(carpeta_preguntas):
    if not archivo.endswith(".json"):
        continue

    ruta = os.path.join(carpeta_preguntas, archivo)
    with open(ruta, "r", encoding="utf-8") as f:
        data = json.load(f)

    titulacion = data.get("titulacion", "").strip().upper()
    preguntas = data.get("preguntas", [])

    for pregunta in preguntas:
        archivo_origen = pregunta.get("archivo_origen", "")
        numero = pregunta.get("numero")

        # Buscar año desde el nombre del PDF: "Cuaderno_2020_MEDICINA_0_C.pdf"
        match = re.search(r"Cuaderno_(\d{4})_" + re.escape(titulacion) + r"_\d+_C", archivo_origen)
        if not match:
            pregunta["respuesta_correcta"] = None
            continue

        año = match.group(1)

        # El nuevo nombre de archivo de respuestas solo tiene año, sin _v0
        archivo_respuestas = f"{titulacion}_{año}.json"

        respuestas = respuestas_por_archivo.get(archivo_respuestas)
        if respuestas:
            pregunta["respuesta_correcta"] = respuestas.get(numero, None)
        else:
            pregunta["respuesta_correcta"] = None

        # 👉 Añadir etiquetas de titulación y año
        pregunta["titulacion"] = titulacion
        pregunta["convocatoria"] = año

    # Guardar salida
    salida = os.path.join(carpeta_salida, archivo)
    with open(salida, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"✅ Guardado en {salida}")
