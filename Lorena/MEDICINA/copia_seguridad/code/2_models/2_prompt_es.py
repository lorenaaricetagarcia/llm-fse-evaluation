import json
import requests
import re
import os
import sys
from collections import OrderedDict

# 📁 Crear carpeta de resultados si no existe
os.makedirs("results/3_analysis", exist_ok=True)

# 🔄 Redirección de salida a consola + archivo
class DualOutput:
    def __init__(self, path):
        self.terminal = sys.__stdout__
        self.log = open(path, "w", encoding="utf-8")
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
    def flush(self):
        self.terminal.flush()
        self.log.flush()

# Activar redirección
sys.stdout = DualOutput("results/3_analysis/2_resumen_completo_prompt_es.txt")

# Modelos a usar con Ollama
modelos = ["llama3", "mistral", "gemma", "deepseek-coder", "phi3"]

# Prompt de sistema actualizado
PROMPT_INICIAL = (
    "Responde a la siguiente pregunta de opción múltiple seleccionando únicamente la opción correcta entre 1 y 4.\n"
    "Tu respuesta debe seguir este formato: 'La respuesta correcta es la número X.' (siendo X un número del 1 al 4), "
    "seguido de una breve explicación.\n"
    "Si no estás completamente seguro de la respuesta, responde únicamente: 'No estoy seguro.'\n\n"
)

# Carpeta de entrada con todos los exámenes
carpeta_examenes = "results/1_data_preparation/6_json_final"
archivos_json = [f for f in os.listdir(carpeta_examenes) if f.endswith(".json")]

for archivo_json in archivos_json:
    nombre_examen = os.path.splitext(archivo_json)[0]
    ruta_json = os.path.join(carpeta_examenes, archivo_json)

    with open(ruta_json, "r", encoding="utf-8") as f:
        base_data = json.load(f)

    for modelo in modelos:
        print(f"\n🚀 Procesando '{nombre_examen}' con modelo: {modelo}")
        data = {"preguntas": []}
        carpeta_salida = f"results/2_models/prompt/ES/{modelo}"
        os.makedirs(carpeta_salida, exist_ok=True)

        for i, pregunta in enumerate(base_data["preguntas"], 1):
            if archivo_json in ["ENFERMERÍA.json", "MEDICINA.json"] and pregunta.get("tipo") != "texto":
                continue

            prompt = PROMPT_INICIAL + pregunta["enunciado"] + "\n\n"
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

        salida = os.path.join(carpeta_salida, f"{nombre_examen}_{modelo}.json")
        with open(salida, "w", encoding="utf-8") as f_out:
            json.dump(data, f_out, ensure_ascii=False, indent=2)

        print(f"\n✅ Guardado: {salida}")

        # === ANÁLISIS DE RESULTADOS ===
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

        print(f"\n📊 Resultados del modelo {modelo.upper()} - Examen: {nombre_examen}")
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
