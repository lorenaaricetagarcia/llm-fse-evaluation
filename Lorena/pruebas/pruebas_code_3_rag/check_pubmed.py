import json

# ==============================
# RUTAS A LOS ARCHIVOS
# ==============================
ruta_predicciones = "/home/xs1/Desktop/Lorena/results/2_models/rag_pubmed_live/MEDICINA_llama3_RAG_PubMed_live.json"
ruta_referencias = "/home/xs1/Desktop/Lorena/results/1_data_preparation/6_json_final/MEDICINA.json"

# ==============================
# CARGAR ARCHIVOS
# ==============================
with open(ruta_predicciones, "r", encoding="utf-8") as f:
    predicciones = json.load(f)

with open(ruta_referencias, "r", encoding="utf-8") as f:
    referencias = json.load(f)

# ==============================
# CREAR DICCIONARIO DE RESPUESTAS CORRECTAS
# ==============================
respuestas_correctas = {p["numero"]: p["respuesta_correcta"] for p in referencias["preguntas"]}

# ==============================
# COMPARAR RESPUESTAS Y CALCULAR ACCURACY
# ==============================
aciertos = 0
total = 0
fallos = []
detalles = []

for p in predicciones["preguntas"]:
    num = p["numero"]
    if num in respuestas_correctas:
        total += 1
        correcta = respuestas_correctas[num]
        pred = p.get("llama3")

        # DEBUG: mostrar primeras 5 comparaciones crudas
        if total <= 5:
            print(f"[DEBUG] Pregunta {num} -> correcta={correcta}, pred={pred}")

        # ⚠️ Ajuste de índice: si el modelo usa 0-index, suma 1
        if pred == correcta or (pred + 1 == correcta):
            aciertos += 1
            acierto = True
        else:
            fallos.append(num)
            acierto = False

        detalles.append({
            "numero": num,
            "prediccion": pred,
            "correcta": correcta,
            "acierto": acierto
        })

accuracy = aciertos / total * 100 if total > 0 else 0

# ==============================
# MOSTRAR RESULTADOS
# ==============================
print("\n========================================")
print("      📊 RESULTADOS DEL MODELO")
print("========================================")
print(f"Preguntas evaluadas: {total}")
print(f"Aciertos: {aciertos}")
print(f"❌ Fallos: {len(fallos)} -> {fallos[:15]} ...")  # muestra primeros 15 fallos
print(f"✅ Accuracy total: {accuracy:.2f}%")
print("========================================\n")
