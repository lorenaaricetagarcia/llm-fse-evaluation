import os
import json
import re

carpeta_entrada = "5_json_type"
carpeta_salida = "6_json_final"
os.makedirs(carpeta_salida, exist_ok=True)

for nombre_archivo in os.listdir(carpeta_entrada):
    if not nombre_archivo.endswith(".json"):
        continue

    ruta = os.path.join(carpeta_entrada, nombre_archivo)

    with open(ruta, "r", encoding="utf-8") as f:
        data = json.load(f)

    titulacion = data.get("titulacion", "").strip().upper()
    preguntas = data.get("preguntas", [])

    año = "DESCONOCIDO"
    for pregunta in preguntas:
        archivo_origen = pregunta.get("archivo_origen", "")
        match = re.search(r"Cuaderno_(\d{4})_" + re.escape(titulacion) + r"_\d+_C", archivo_origen)
        if match:
            año = match.group(1)
            break  # Ya encontrado, salimos

    # Añadir etiquetas al JSON
    data["etiqueta_titulacion"] = titulacion
    data["etiqueta_convocatoria"] = año

    # Guardar en carpeta nueva
    ruta_salida = os.path.join(carpeta_salida, nombre_archivo)
    with open(ruta_salida, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"✅ {nombre_archivo} etiquetado como {titulacion} - {año}")

print("\n📦 Todos los archivos procesados y guardados en '6_json_final'.")
