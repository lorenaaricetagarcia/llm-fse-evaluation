import json
import requests
import re
import os
import sys
from collections import OrderedDict

# Crear carpeta de salida de análisis
os.makedirs("results/3_analysis", exist_ok=True)

# Redirigir salida a archivo txt
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
sys.stdout = DualOutput("results/3_analysis/resumen_completo_prompt_EN.txt")

# Modelos a usar
modelos = ["llama3", "mistral", "gemma", "deepseek-coder", "deepseek-llm", "phi3", "phi3:instruct"]

PROMPT_INICIAL = (
    "Answer the following multiple-choice question by selecting only the correct option between 1 and 4.\n"
    "Your response must follow this format: 'The correct answer is number X.' (where X is a number from 1 to 4), "
    "followed by a short explanation.\n"
    "If you are not completely sure about the answer, respond only with: 'I'm not sure.'\n\n"
)

carpeta_entrada = "results/1_data_preparation/6_json_final"
archivos_json = [f for f in os.listdir(carpeta_entrada) if f.endswith(".json")]

resumen_global = {
    modelo: {
        "aciertos": 0,
        "errores": 0,
        "sin_respuesta": 0,
        "total": 0,
        "errores_detalle": []
    } for modelo in modelos
}

for archivo_json in archivos_json:
    exam_name = os.path.splitext(archivo_json)[0]
    ruta_json = os.path.join(carpeta_entrada, archivo_json)

    with open(ruta_json, "r", encoding="utf-8") as f:
        base_data = json.load(f)

    for modelo in modelos:
        print(f"\n🚀 Running model: {modelo} on exam: {exam_name}")
        data = {"preguntas": []}
        carpeta_salida = f"results/2_models/prompt/EN/{modelo}"
        os.makedirs(carpeta_salida, exist_ok=True)

        for i, pregunta in enumerate(base_data["preguntas"], 1):
            if archivo_json in ["ENFERMERÍA.json", "MEDICINA.json"] and pregunta.get("tipo") != "texto":
                continue

            prompt = PROMPT_INICIAL + pregunta["enunciado"] + "\n\n"
            for idx, opcion in enumerate(pregunta["opciones"], 1):
                prompt += f"{idx}. {opcion}\n"

            print(f"\n📤 [{i}] Sending question to {modelo}...")

            payload = {
                "model": modelo,
                "prompt": prompt,
                "stream": False
            }

            try:
                response = requests.post("http://localhost:11434/api/generate", json=payload, timeout=180)
                data_model = response.json()
                texto = data_model.get("response", "").strip()
                print("🧠 Model response:")
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
                print("❌ Model timeout.")
            except Exception as e:
                print(f"❌ Error on question {i}: {e}")

        output_file = os.path.join(carpeta_salida, f"{exam_name}_{modelo}.json")
        with open(output_file, "w", encoding="utf-8") as f_out:
            json.dump(data, f_out, ensure_ascii=False, indent=2)

        print(f"\n✅ Saved: {output_file}")

        preguntas = data.get("preguntas", [])
        aciertos = errores = sin_respuesta = 0
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

        print(f"\n📊 Results for {modelo.upper()} - Exam: {exam_name}")
        print("-" * 60)
        print(f"Total questions        : {total}")
        print(f"Answered by model      : {respondidas}")
        print(f"Correct answers        : {aciertos}")
        print(f"Wrong answers          : {errores}")
        print(f"No response (None)     : {sin_respuesta}")
        print(f"📈 Accuracy rate        : {acierto_pct:.2f}%")

        print("\n🔍 Example errors:")
        for err in errores_detalle[:5]:
            print(f"  ➤ Question {err['número']}: predicted {err['predicha']}, correct {err['correcta']}")
            print(f"    {err['enunciado']}")

        resumen_global[modelo]["aciertos"] += aciertos
        resumen_global[modelo]["errores"] += errores
        resumen_global[modelo]["sin_respuesta"] += sin_respuesta
        resumen_global[modelo]["total"] += total
        resumen_global[modelo]["errores_detalle"].extend(errores_detalle[:3])

# === GLOBAL SUMMARY ===
print("\n📊📊📊 GLOBAL MODEL SUMMARY 📊📊📊")
resumen_json = {}

for modelo in modelos:
    total = resumen_global[modelo]["total"]
    aciertos = resumen_global[modelo]["aciertos"]
    errores = resumen_global[modelo]["errores"]
    sin_respuesta = resumen_global[modelo]["sin_respuesta"]
    respondidas = total - sin_respuesta
    acierto_pct = (aciertos / respondidas * 100) if respondidas > 0 else 0

    print(f"\n🧠 Model: {modelo.upper()}")
    print("-" * 50)
    print(f"Total questions       : {total}")
    print(f"Answered              : {respondidas}")
    print(f"Correct               : {aciertos}")
    print(f"Incorrect             : {errores}")
    print(f"Unanswered (None)     : {sin_respuesta}")
    print(f"📈 Accuracy rate       : {acierto_pct:.2f}%")

    print("🔍 Sample errors:")
    for err in resumen_global[modelo]["errores_detalle"][:5]:
        print(f"  ➤ Q{err['número']}: predicted {err['predicha']}, correct {err['correcta']}")
        print(f"    {err['enunciado']}")

    resumen_json[modelo] = {
        "total": total,
        "respondidas": respondidas,
        "aciertos": aciertos,
        "errores": errores,
        "sin_respuesta": sin_respuesta,
        "acierto_pct": round(acierto_pct, 2)
    }

with open("results/resumen_global_modelos_EN.json", "w", encoding="utf-8") as f_out:
    json.dump(resumen_json, f_out, indent=2, ensure_ascii=False)

print("\n✅ Global summary saved in: results/resumen_global_modelos_EN.json")
