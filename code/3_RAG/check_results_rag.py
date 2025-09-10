import os
import json
from collections import defaultdict

# ==============================
# CONFIG
# ==============================

carpeta_base = "results/1_data_preparation/6_json_final"   # datos con respuestas correctas
carpeta_rag = "results/2_models/rag"                      # resultados RAG
carpeta_salida = "results/3_analysis"
os.makedirs(carpeta_salida, exist_ok=True)

# Modelos usados
modelos = ["llama3", "mistral", "gemma"]

# ==============================
# Cargar archivos base
# ==============================
base_examenes = {}
for archivo in os.listdir(carpeta_base):
    if archivo.endswith(".json"):
        with open(os.path.join(carpeta_base, archivo), "r", encoding="utf-8") as f:
            base_examenes[archivo.replace(".json", "")] = json.load(f)

# ==============================
# Inicializar acumuladores
# ==============================
resumen_por_titulacion = defaultdict(lambda: {m: {"aciertos": 0, "errores": 0, "sin_respuesta": 0, "total": 0} for m in modelos})
resumen_global = {m: {"aciertos": 0, "errores": 0, "sin_respuesta": 0, "total": 0} for m in modelos}

# ==============================
# Procesar resultados RAG
# ==============================
for archivo in os.listdir(carpeta_rag):
    if not archivo.endswith("_RAG.json"):
        continue

    # Ejemplo: MEDICINA_llama3_RAG.json
    partes = archivo.split("_")
    titulacion = partes[0]
    modelo = partes[1]

    ruta_rag = os.path.join(carpeta_rag, archivo)
    with open(ruta_rag, "r", encoding="utf-8") as f:
        predicciones = json.load(f)["preguntas"]

    # Buscar archivo base correspondiente a esa titulación
    for examen, data_base in base_examenes.items():
        if examen.startswith(titulacion):  # Ejemplo: MEDICINA_2020
            preguntas_base = {p["numero"]: p for p in data_base["preguntas"]}

            for preg_pred in predicciones:
                num = preg_pred["numero"]
                if num not in preguntas_base:
                    continue

                correcta = preguntas_base[num]["respuesta_correcta"]
                pred = preg_pred.get(modelo)

                # Actualizar métricas
                resumen_por_titulacion[titulacion][modelo]["total"] += 1
                resumen_global[modelo]["total"] += 1

                if pred is None:
                    resumen_por_titulacion[titulacion][modelo]["sin_respuesta"] += 1
                    resumen_global[modelo]["sin_respuesta"] += 1
                elif pred == correcta:
                    resumen_por_titulacion[titulacion][modelo]["aciertos"] += 1
                    resumen_global[modelo]["aciertos"] += 1
                else:
                    resumen_por_titulacion[titulacion][modelo]["errores"] += 1
                    resumen_global[modelo]["errores"] += 1

# ==============================
# Guardar resultados
# ==============================
salida = {
    "por_titulacion": resumen_por_titulacion,
    "global": resumen_global
}

with open(os.path.join(carpeta_salida, "resumen_accuracy_RAG.json"), "w", encoding="utf-8") as f_out:
    json.dump(salida, f_out, indent=2, ensure_ascii=False)

print("\n✅ Guardado: results/3_analysis/resumen_accuracy_RAG.json")

# Mostrar resultados por pantalla
print("\n📊 Accuracy por titulación y modelo:")
for titulacion, modelos_res in resumen_por_titulacion.items():
    print(f"\n🎓 {titulacion}")
    for modelo, met in modelos_res.items():
        total = met["total"] or 1
        acc = (met["aciertos"] / (total - met["sin_respuesta"])) * 100 if (total - met["sin_respuesta"]) > 0 else 0
        print(f"   🔹 {modelo}: {acc:.2f}% (Aciertos: {met['aciertos']}, Errores: {met['errores']}, Sin respuesta: {met['sin_respuesta']}, Total: {met['total']})")

print("\n📊 Accuracy global por modelo:")
for modelo, met in resumen_global.items():
    total = met["total"] or 1
    acc = (met["aciertos"] / (total - met["sin_respuesta"])) * 100 if (total - met["sin_respuesta"]) > 0 else 0
    print(f"   🧠 {modelo}: {acc:.2f}% (Aciertos: {met['aciertos']}, Errores: {met['errores']}, Sin respuesta: {met['sin_respuesta']}, Total: {met['total']})")
