import json
import os

# Ruta al archivo JSON anotado con llama3
archivo_json = "results/2_models/llama3/BIOLOGÍA_llama3.json"  

# Cargar archivo
with open(archivo_json, "r", encoding="utf-8") as f:
    data = json.load(f)

preguntas = data.get("preguntas", [])

# Inicializar contadores
aciertos = 0
errores = 0
sin_respuesta = 0
errores_detalle = []

# Recorrer preguntas
for pregunta in preguntas:
    pred = pregunta.get("llama3")
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

# Estadísticas finales
total = len(preguntas)
respondidas = total - sin_respuesta
acierto_pct = (aciertos / respondidas * 100) if respondidas > 0 else 0

print(f"📊 Resultados del modelo LLaMA 3 sobre {archivo_json}")
print("-" * 50)
print(f"Total de preguntas     : {total}")
print(f"Respondidas por el modelo : {respondidas}")
print(f"Aciertos               : {aciertos}")
print(f"Errores                : {errores}")
print(f"No respondió (None)    : {sin_respuesta}")
print(f"📈 Porcentaje de acierto: {acierto_pct:.2f}%")

# Mostrar los primeros errores (si quieres verlos)
print("\n🔍 Ejemplos de errores:")
for err in errores_detalle[:5]:  # cambia 5 por más si quieres
    print(f"  ➤ Pregunta {err['número']}: predijo {err['predicha']}, correcta {err['correcta']}")
    print(f"    {err['enunciado']}")
