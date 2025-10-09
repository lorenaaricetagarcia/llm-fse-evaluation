import json
import requests
import re
import os
from collections import OrderedDict

# Models to use with Ollama
modelos = ["llama3", "mistral", "gemma", "deepseek-coder", "deepseek-llm", "phi3", "phi3:instruct"]

# English system prompt
PROMPT_INICIAL = (
    "You are an expert in biology. "
    "Your task is to answer the following multiple-choice question by selecting the only correct option (between 1 and 4). "
    "Your answer must follow this format: 'The correct answer is number X' (X being a number between 1 and 4), followed by a short explanation. "
    "If you are not completely sure about the answer, do not respond.\n\n"
)

# Input file path
archivo_json = "BIOLOGÍA.json"

with open(archivo_json, "r", encoding="utf-8") as f:
    base_data = json.load(f)

# === RUN MODELS ===
for modelo in modelos:
    print(f"\n🚀 Running model: {modelo}")
    data = {"preguntas": []}
    carpeta_salida = f"results/prompt/{modelo}"
    os.makedirs(carpeta_salida, exist_ok=True)

    for i, pregunta in enumerate(base_data["preguntas"], 1):
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

            # Extract the number between 1-4, or None if missing
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

    salida = os.path.join(carpeta_salida, f"BIOLOGÍA_{modelo}.json")
    with open(salida, "w", encoding="utf-8") as f_out:
        json.dump(data, f_out, ensure_ascii=False, indent=2)

    print(f"\n✅ Saved: {salida}")

# === ANALYZE RESULTS ===
for modelo in modelos:
    archivo_respuestas = f"results/prompt/{modelo}/BIOLOGÍA_{modelo}.json"
    if not os.path.exists(archivo_respuestas):
        print(f"\n⚠️ File not found for {modelo}")
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

    print(f"\n📊 Results for model {modelo.upper()}")
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
