import os
import json

# ==============================
# CONFIGURACIÓN
# ==============================
carpeta_base = "results/1_data_preparation/6_json_final/prueba"
carpeta_rag = "results/2_models/rag_v3"

# dataset base (el que tiene las respuestas_correctas)
archivo_base = "MEDICINA.json"
ruta_base = os.path.join(carpeta_base, archivo_base)

# modelos a comparar
modelos = ["llama3", "mistral", "gemma"]

# cargar dataset base
with open(ruta_base, "r", encoding="utf-8") as f:
    base_data = json.load(f)
base_dict = {p["numero"]: p["respuesta_correcta"] for p in base_data["preguntas"]}

# inicializar métricas
resultados = {m: {"aciertos": 0, "errores": 0, "total": 0, "sin_pred": 0} for m in modelos}

# ==============================
# EVALUACIÓN POR MODELO
# ==============================
for modelo in modelos:
    ruta_rag = os.path.join(carpeta_rag, f"MEDICINA_{modelo}_RAGv3.json")
    if not os.path.exists(ruta_rag):
        print(f"⚠️ No se encontró el archivo de {modelo}.")
        continue

    with open(ruta_rag, "r", encoding="utf-8") as f:
        pred_data = json.load(f)["preguntas"]

    for preg in pred_data:
        num = preg["numero"]
        pred = preg.get(modelo)
        real = base_dict.get(num)

        if pred is None:
            resultados[modelo]["sin_pred"] += 1
        elif real is not None:
            resultados[modelo]["total"] += 1
            if pred == real:
                resultados[modelo]["aciertos"] += 1
            else:
                resultados[modelo]["errores"] += 1

# ==============================
# MOSTRAR RESULTADOS
# ==============================
print("\n📊 RESULTADOS RAG v3 (comparando con respuestas base)\n")
for modelo, met in resultados.items():
    total = met["total"] or 1
    acc = (met["aciertos"] / total) * 100
    print(f"🧠 {modelo.upper()}")
    print(f"   ✅ Aciertos: {met['aciertos']}")
    print(f"   ❌ Errores: {met['errores']}")
    print(f"   ⚠️ Sin predicción: {met['sin_pred']}")
    print(f"   🎯 Accuracy: {acc:.2f}%\n")
