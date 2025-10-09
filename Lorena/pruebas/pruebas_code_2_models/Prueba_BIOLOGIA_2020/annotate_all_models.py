import json
import requests
import re
import os
from collections import OrderedDict

# Modelos que puedes usar con Ollama
modelos = ["llama3", "mistral", "mixtral", "gemma", "deepseek-coder", "phi3"]

# Ruta al archivo de entrada original
archivo_json = "BIOLOGÍA.json"

# Cargar archivo de preguntas
with open(archivo_json, "r", encoding="utf-8") as f:
    base_data = json.load(f)

# === EJECUCIÓN CON MODELOS ===
for modelo in modelos:
    print(f"\n🚀 Procesando con modelo: {modelo}")
    data = {"preguntas": []}
    carpeta_salida = f"results/{modelo}"
    os.makedirs(carpeta_salida, exist_ok=True)

    for i, pregunta in enumerate(base_data["preguntas"], 1):
        prompt = f"{pregunta['enunciado']}\n\n"
        for idx, opcion in enumerate(pregunta["opciones"], 1):
            prompt += f"{idx}. {opcion}\n"

        print(f"\n📤 [{i}] Enviando pregunta a {modelo}...")

        payload = {
            "model": modelo,
            "prompt": prompt,
            "stream": False
        }

        try:
            response = requests.post("http://localhost:11434/api/generate", json=payload, timeout=180)
            data_model = response.json()
            texto = data_model.get("response", "").strip()

            print("🧠 Respuesta del modelo:")
            print(texto)

            match = re.search(r'\b([1-4])\b', texto)
            seleccion = int(match.group(1)) if match else None

            nueva_pregunta = OrderedDict()
            for clave in pregunta:
                if clave not in (modelo, f"{modelo}_texto"):
                    nueva_pregunta[clave] = pregunta[clave]
            nueva_pregunta[modelo] = seleccion
            nueva_pregunta[f"{modelo}_texto"] = texto

            data["preguntas"].append(nueva_pregunta)

        except requests.exceptions.Timeout:
            print("❌ Timeout del modelo.")
        except Exception as e:
            print(f"❌ Error en pregunta {i}: {e}")

    salida = os.path.join(carpeta_salida, f"BIOLOGÍA_{modelo}.json")
    with open(salida, "w", encoding="utf-8") as f_out:
        json.dump(data, f_out, ensure_ascii=False, indent=2)

    print(f"\n✅ Guardado: {salida}")

# === ANÁLISIS DE RESULTADOS ===
for modelo in modelos:
    archivo_respuestas = f"results/{modelo}/BIOLOGÍA_{modelo}.json"
    if not os.path.exists(archivo_respuestas):
        print(f"\n⚠️ No se encontró el archivo de {modelo}")
        continue

    with open(archivo_respuestas, "r", encoding="utf-8") as f:
        data = json.load(f)

    preguntas = data.get("preguntas", [])
    aciertos = 0
    errores = 0
    sin_respuesta = 0
    errores_detalle = []

    for pregunta in preguntas:
        pred = pregunta.get(modelo)
        correcta = pregunta.get("respuesta_correcta")

        if pred is None:
            sin_respuesta += 1
        elif pred == correcta:
            aciertos += 1
        else:
            errores += 1
            errores_detalle.append({
                "número": pregunta["numero"],
                "predicha": pred,
                "correcta": correcta,
                "enunciado": pregunta["enunciado"]
            })

    total = len(preguntas)
    respondidas = total - sin_respuesta
    acierto_pct = (aciertos / respondidas * 100) if respondidas > 0 else 0

    print(f"\n📊 Resultados del modelo {modelo.upper()}")
    print("-" * 60)
    print(f"Total de preguntas        : {total}")
    print(f"Respondidas por el modelo : {respondidas}")
    print(f"Aciertos                  : {aciertos}")
    print(f"Errores                   : {errores}")
    print(f"No respondió (None)       : {sin_respuesta}")
    print(f"📈 Porcentaje de acierto  : {acierto_pct:.2f}%")

    print("\n🔍 Ejemplos de errores:")
    for err in errores_detalle[:5]:
        print(f"  ➤ Pregunta {err['número']}: predijo {err['predicha']}, correcta {err['correcta']}")
        print(f"    {err['enunciado']}")
