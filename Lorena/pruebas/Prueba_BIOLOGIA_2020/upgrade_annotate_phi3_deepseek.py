import json
import requests
import re
import os
from collections import OrderedDict

# Modelos a probar
modelos = {
    "phi3": "phi3:instruct",
    "deepseek": "deepseek-llm"
}

PROMPT_INICIAL = (
    "Eres un experto en biología. Debes responder a la siguiente pregunta de opción múltiple, "
    "eligiendo únicamente la opción correcta (del 1 al 4). "
    "Responde de forma breve, clara y justificada. "
    "Si no estás seguro, responde exactamente: 'No lo sé'.\n\n"
)

archivo_json = "BIOLOGÍA.json"

with open(archivo_json, "r", encoding="utf-8") as f:
    base_data = json.load(f)

# === EJECUCIÓN MODELOS ===
for alias, modelo_real in modelos.items():
    print(f"\n🚀 Procesando con modelo: {modelo_real}")
    data = {"preguntas": []}
    carpeta_salida = f"results/prompt/{alias}"
    os.makedirs(carpeta_salida, exist_ok=True)

    for i, pregunta in enumerate(base_data["preguntas"], 1):
        prompt = PROMPT_INICIAL + pregunta["enunciado"] + "\n\n"
        for idx, opcion in enumerate(pregunta["opciones"], 1):
            prompt += f"{idx}. {opcion}\n"

        print(f"\n📤 [{i}] Enviando pregunta a {modelo_real}...")

        payload = {
            "model": modelo_real,
            "prompt": prompt,
            "stream": False
        }

        try:
            response = requests.post("http://localhost:11434/api/generate", json=payload, timeout=180)
            data_model = response.json()
            texto = data_model.get("response", "").strip()

            print("🧠 Respuesta del modelo:")
            print(texto)

            # Detectar número del 1 al 4 o "no lo sé"
            match = re.search(r'\b([1-4])\b', texto)
            seleccion = int(match.group(1)) if match else None
            if "no lo sé" in texto.lower():
                seleccion = None

            nueva_pregunta = OrderedDict()
            for clave in pregunta:
                if clave not in (alias, f"{alias}_texto"):
                    nueva_pregunta[clave] = pregunta[clave]
            nueva_pregunta[alias] = seleccion
            nueva_pregunta[f"{alias}_texto"] = texto

            data["preguntas"].append(nueva_pregunta)

        except requests.exceptions.Timeout:
            print("❌ Timeout del modelo.")
        except Exception as e:
            print(f"❌ Error en pregunta {i}: {e}")

    salida = os.path.join(carpeta_salida, f"BIOLOGÍA_{alias}.json")
    with open(salida, "w", encoding="utf-8") as f_out:
        json.dump(data, f_out, ensure_ascii=False, indent=2)

    print(f"\n✅ Guardado: {salida}")

# === ANÁLISIS DE RESULTADOS ===
for alias in modelos.keys():
    archivo_respuestas = f"results/prompt/{alias}/BIOLOGÍA_{alias}.json"
    if not os.path.exists(archivo_respuestas):
        print(f"\n⚠️ No se encontró el archivo de {alias}")
        continue

    with open(archivo_respuestas, "r", encoding="utf-8") as f:
        data = json.load(f)

    preguntas = data.get("preguntas", [])
    aciertos = 0
    errores = 0
    sin_respuesta = 0
    errores_detalle = []

    for pregunta in preguntas:
        pred = pregunta.get(alias)
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

    print(f"\n📊 Resultados del modelo {alias.upper()}")
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
