import json
import requests
import re
import os
from collections import OrderedDict

# Ruta al archivo de entrada
archivo_json = "BIOLOGÍA.json"

# Crear carpeta de salida si no existe
carpeta_salida = "results/llama3"
os.makedirs(carpeta_salida, exist_ok=True)

# Cargar datos JSON
with open(archivo_json, "r", encoding="utf-8") as f:
    data = json.load(f)

for i, pregunta in enumerate(data["preguntas"], 1):
    # Crear el prompt
    prompt = f"{pregunta['enunciado']}\n\n"
    for idx, opcion in enumerate(pregunta["opciones"], 1):
        prompt += f"{idx}. {opcion}\n"

    print(f"\n📤 [{i}] Enviando pregunta a LLaMA 3...")

    payload = {
        "model": "llama3",
        "prompt": prompt,
        "stream": False
    }

    try:
        response = requests.post("http://localhost:11434/api/generate", json=payload, timeout=180)
        data_llama = response.json()
        texto_llama = data_llama.get("response", "").strip()

        print("🧠 Respuesta del modelo:")
        print(texto_llama)

        # Extraer opción detectada
        match = re.search(r'\b([1-4])\b', texto_llama)
        llama3_resp = int(match.group(1)) if match else None

        # Reordenar los campos con llama3 primero, luego llama3_texto
        nueva_pregunta = OrderedDict()
        for clave in pregunta:
            if clave not in ("llama3", "llama3_texto"):
                nueva_pregunta[clave] = pregunta[clave]
        nueva_pregunta["llama3"] = llama3_resp
        nueva_pregunta["llama3_texto"] = texto_llama

        data["preguntas"][i - 1] = nueva_pregunta

    except requests.exceptions.Timeout:
        print("❌ Timeout: el modelo tardó demasiado.")
    except Exception as e:
        print(f"❌ Error con la pregunta {i}: {e}")

# Guardar en carpeta results/llama3
nombre_archivo = os.path.basename(archivo_json).replace(".json", "_llama3.json")
ruta_salida = os.path.join(carpeta_salida, nombre_archivo)

with open(ruta_salida, "w", encoding="utf-8") as f_out:
    json.dump(data, f_out, ensure_ascii=False, indent=2)

print(f"\n✅ Anotaciones guardadas en: {ruta_salida}")