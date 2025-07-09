import os
import json
import re

# Carpetas
carpeta_preguntas = "1_json_por_titulacion" # Donde están los JSON con las preguntas
carpeta_respuestas = "2_respuestas_json"    # Donde están los JSON con las respuestas
carpeta_salida = "3_json_con_respuesta"     # Carpeta donde se guardaran los JSON con respuestas añadidas
os.makedirs(carpeta_salida, exist_ok=True)  # Crear carpeta de salida si no existe

# CARGAR TODOS LOS ARCHIVOS DE RESPUESTAS EN MEMORIA
respuestas_por_archivo = {} # Diccionario para guardar respuestas por archivo
# Recorremos todos los archivos de la carpeta de respuestas
for archivo in os.listdir(carpeta_respuestas):
    if archivo.endswith(".json"):   # Solo procesamos JSON
        ruta = os.path.join(carpeta_respuestas, archivo)
        with open(ruta, "r", encoding="utf-8") as f:
            # Cargamos el JSON que es un dict {numero_pregunta: respuesta_correcta}
            respuestas_por_archivo[archivo] = {
                int(k): v for k, v in json.load(f).items()
            }
# Resultado: respuesta_por_archivo["BIOLOGÍA_2024_v2.json"] = {1: 3, 2: 1, ...}

# PROCESAR CADA ARCHIVO DE PREGUNTAS
for archivo in os.listdir(carpeta_preguntas):
    if not archivo.endswith(".json"):
        continue

    ruta = os.path.join(carpeta_preguntas, archivo) # Abre el archivo 
    with open(ruta, "r", encoding="utf-8") as f:
        data = json.load(f) # Cargamos el contenido del archivo JSON (titulación + pregutas)

    # Extraer titulación y preguntas
    titulacion = data.get("titulacion", "").strip().upper() # Ej: "BIOLOGÍA"
    preguntas = data.get("preguntas", [])

    # Para cada pregunta encontrar su respuesta correcta
    for pregunta in preguntas:
        archivo_origen = pregunta.get("archivo_origen", "") # Ej: "Cuaderno_2024_BIOLOGIA_2_C.pdf"
        numero = pregunta.get("numero")

        # Extraer año y versión del nombre del archivo origen para unificar formatos
        match = re.search(r"Cuaderno_(\d{4})_" + re.escape(titulacion) + r"_(\d+)_C", archivo_origen)
        if not match:   # Si no encuentra coincidencias
            pregunta["respuesta_correcta"] = None
            continue
        
        # Construir nombre dle archivo de respuestas asociado a esta pregunta
        año, version = match.groups()   # Extrae año y versión del match
        archivo_respuestas = f"{titulacion}_{año}_v{version}.json"  # Ej: "BIOLOGÍA_2024_V2.json"

        # Buscar esta respuesta en el diccionario cargado
        respuestas = respuestas_por_archivo.get(archivo_respuestas) # Diccionario cono {número: opción_correcta}
        if respuestas:
            pregunta["respuesta_correcta"] = respuestas.get(numero, None)   # Añade la respuesta a la pregunta
        else:
            pregunta["respuesta_correcta"] = None

    # Guardar salida
    salida = os.path.join(carpeta_salida, archivo)
    with open(salida, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"✅ Guardado en {salida}")
