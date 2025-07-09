import os
import json
import re

CARPETA_ENTRADA = "3_json_con_respuesta"
CARPETA_SALIDA = "4_json_corregido"

os.makedirs(CARPETA_SALIDA, exist_ok=True)

def separar_enunciado_y_opcion(texto):
    # Elimina marcador inicial como "1." o "A."
    match = re.search(r"^(.*?)(?:\s*[1A]\.\s*)(.+)$", texto.strip())
    if match:
        enunciado = match.group(1).strip()
        primera_opcion = match.group(2).strip()
        return enunciado, primera_opcion
    else:
        return "", texto.strip()

for nombre_archivo in os.listdir(CARPETA_ENTRADA):
    if not nombre_archivo.endswith(".json"):
        continue

    ruta_entrada = os.path.join(CARPETA_ENTRADA, nombre_archivo)
    with open(ruta_entrada, "r", encoding="utf-8") as f:
        data = json.load(f)

    for pregunta in data.get("preguntas", []):
        if pregunta.get("numero") == 1:
            opciones = pregunta.get("opciones", [])
            if opciones:
                nuevo_enunciado, primera_opcion = separar_enunciado_y_opcion(opciones[0])
                if nuevo_enunciado:
                    pregunta["enunciado"] = nuevo_enunciado
                    pregunta["opciones"][0] = primera_opcion
                else:
                    pregunta["enunciado"] = ""

    ruta_salida = os.path.join(CARPETA_SALIDA, nombre_archivo)
    with open(ruta_salida, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"✅ Corregido: {nombre_archivo}")