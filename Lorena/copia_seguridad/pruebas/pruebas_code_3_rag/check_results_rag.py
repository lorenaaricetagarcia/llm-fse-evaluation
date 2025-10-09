import os
import json
from collections import defaultdict

# ==============================
# CONFIG
# ==============================
carpeta_base = "/home/xs1/Desktop/Lorena/results/1_data_preparation/6_json_final"   # respuestas correctas
carpeta_rag = "/home/xs1/Desktop/Lorena/results/2_models/rag_pubmed_live"           # resultados RAG
carpeta_salida = "/home/xs1/Desktop/Lorena/results/3_analysis"
os.makedirs(carpeta_salida, exist_ok=True)

# Modelos a evaluar
modelos = ["llama3", "mistral", "gemma"]

# ==============================
# CARGAR ARCHIVOS BASE
# ==============================
base_examenes = {}
for archivo in os.listdir(carpeta_base):
    if archivo.endswith(".json"):
        with open(os.path.join(carpeta_base, archivo), "r", encoding="utf-8") as f:
            base_examenes[archivo.replace(".json", "")] = json.load(f)

# ==============================
# INICIALIZAR ACUMULADORES
# ==============================
resumen_por_titulacion = defaultdict(lambda: {
    m: {"aciertos": 0, "errores": 0, "sin_respuesta": 0, "total": 0} for m in modelos
})
resumen_global = {
    m: {"aciertos": 0, "errores": 0, "sin_respuesta": 0, "total": 0} for m in modelos
}

# ==============================
# PROCESAR RESULTADOS RAG
# ==============================
for archivo in os.listdir(carpeta_rag):
    # Aceptar nombres tipo MEDICINA_llama3_RAG_PubMed_live.json
    if "_RAG" not in archivo or not archivo.endswith(".json"):
        continue

    partes = archivo.split("_")
    titulacion = partes[0]
    modelo = partes[1]

    ruta_rag = os.path.join(carpeta_rag, archivo)
    with open(ruta_rag, "r", encoding="utf-8") as f:
        predicciones = json.load(f)["preguntas"]

    # Buscar archivo base correspondiente
    for examen, data_base in base_examenes.items():
        if examen.startswith(titulacion):
            preguntas_base = {}
            for p in data_base["preguntas"]:
                try:
                    if p.get("respuesta_correcta") is not None:
                        preguntas_base[int(p["numero"])] = int(p["respuesta_correcta"])
                except Exception:
                    continue

            # Comparar respuestas
            for preg_pred in predicciones:
                try:
                    num = int(preg_pred["numero"])
                    if num not in preguntas_base:
                        continue

                    correcta = preguntas_base[num]
                    pred = preg_pred.get(modelo)

                    resumen_por_titulacion[titulacion][modelo]["total"] += 1
                    resumen_global[modelo]["total"] += 1

                    if pred is None:
                        resumen_por_titulacion[titulacion][modelo]["sin_respuesta"] += 1
                        resumen_global[modelo]["sin_respuesta"] += 1
                    else:
                        pred = int(pred)
                        if pred == correcta or pred + 1 == correcta:
                            resumen_por_titulacion[titulacion][modelo]["aciertos"] += 1
                            resumen_global[modelo]["aciertos"] += 1
                        else:
                            resumen_por_titulacion[titulacion][modelo]["errores"] += 1
                            resumen_global[modelo]["errores"] += 1
                except Exception:
                    continue

# ==============================
# GUARDAR RESULTADOS
# ==============================
salida = {
    "por_titulacion": resumen_por_titulacion,
    "global": resumen_global
}
ruta_salida = os.path.join(carpeta_salida, "resumen_accuracy_RAG_pubmed.json")
with open(ruta_salida, "w", encoding="utf-8") as f_out:
    json.dump(salida, f_out, indent=2, ensure_ascii=False)

print(f"\n✅ Guardado: {ruta_salida}")

# ==============================
# MOSTRAR RESULTADOS
# ==============================
print("\n📊 Accuracy por titulación y modelo:")
for titulacion, modelos_res in resumen_por_titulacion.items():
    print(f"\n🎓 {titulacion}")
    for modelo, met in modelos_res.items():
        total = met["total"]
        if total - met["sin_respuesta"] > 0:
            acc = (met["aciertos"] / (total - met["sin_respuesta"])) * 100
        else:
            acc = 0
        print(f"   🔹 {modelo}: {acc:.2f}% "
              f"(Aciertos: {met['aciertos']}, Errores: {met['errores']}, "
              f"Sin respuesta: {met['sin_respuesta']}, Total: {met['total']})")

print("\n📊 Accuracy global por modelo:")
for modelo, met in resumen_global.items():
    total = met["total"]
    if total - met["sin_respuesta"] > 0:
        acc = (met["aciertos"] / (total - met["sin_respuesta"])) * 100
    else:
        acc = 0
    print(f"   🧠 {modelo}: {acc:.2f}% "
          f"(Aciertos: {met['aciertos']}, Errores: {met['errores']}, "
          f"Sin respuesta: {met['sin_respuesta']}, Total: {met['total']})")
