import json
import requests
import re
import os
import sys
import csv
import pandas as pd
from collections import OrderedDict, defaultdict, Counter

# ================================================================
# CONFIGURACIÓN Y SALIDA
# ================================================================

carpeta_salida = "/home/xs1/Desktop/Lorena/results/2_models/1_prompt/1_no_prompt"
os.makedirs(carpeta_salida, exist_ok=True)

# 🔄 Redirección de salida: consola + archivo
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

# Redirige todo lo que se imprime a pantalla también a un archivo
sys.stdout = DualOutput(os.path.join(carpeta_salida, "log_no_prompt.txt"))

# ================================================================
# CONFIGURACIÓN DE MODELOS Y ARCHIVOS
# ================================================================

modelos = ["llama3", "mistral", "gemma", "deepseek-coder", "phi3"]
carpeta_entrada = "/home/xs1/Desktop/Lorena/results/1_data_preparation/6_json_final"
archivos_json = [f for f in os.listdir(carpeta_entrada) if f.endswith(".json")]

# Estructura para acumular resultados globales
resumen_global = {
    modelo: {
        "aciertos": 0,
        "errores": 0,
        "sin_respuesta": 0,
        "total": 0,
        "errores_detalle": []
    } for modelo in modelos
}

# ================================================================
# LOOP PRINCIPAL POR ARCHIVO Y MODELO
# ================================================================

for archivo_json in archivos_json:
    nombre_base = os.path.splitext(archivo_json)[0]
    ruta_json = os.path.join(carpeta_entrada, archivo_json)

    with open(ruta_json, "r", encoding="utf-8") as f:
        base_data = json.load(f)

    # Aviso si hay duplicados en número
    numeros = [p.get("numero") for p in base_data["preguntas"]]
    dup_count = sum(1 for _, c in Counter(numeros).items() if c > 1)
    if dup_count > 0:
        print(f"⚠️ {archivo_json}: detectados {dup_count} valores duplicados en 'numero' (se comparará por posición)\n")

    for modelo in modelos:
        print(f"\n🚀 Procesando con modelo: {modelo} en archivo: {archivo_json}")
        data = {"preguntas": []}
        carpeta_modelo = f"{carpeta_salida}{modelo}"
        os.makedirs(carpeta_modelo, exist_ok=True)

        # ============================================================
        # GENERACIÓN Y GUARDADO DE RESPUESTAS
        # ============================================================
        for i, pregunta in enumerate(base_data["preguntas"], 1):
            if archivo_json in ["ENFERMERÍA.json", "MEDICINA.json"] and pregunta.get("tipo") != "texto":
                continue

            prompt = f"{pregunta['enunciado']}\n\n"
            for idx, opcion in enumerate(pregunta["opciones"], 1):
                prompt += f"{idx}. {opcion}\n"

            print(f"\n📤 [{i}] Enviando pregunta a {modelo}...")

            payload = {"model": modelo, "prompt": prompt, "stream": False}

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

        # Guardado del JSON del modelo
        salida_json = os.path.join(carpeta_modelo, f"{nombre_base}_{modelo}.json")
        with open(salida_json, "w", encoding="utf-8") as f_out:
            json.dump(data, f_out, ensure_ascii=False, indent=2)
        print(f"\n✅ Guardado: {salida_json}")

        # ============================================================
        # MÉTRICAS – CORREGIDAS (POR POSICIÓN)
        # ============================================================
        print(f"\n📊 Analizando resultados del modelo {modelo.upper()} en {nombre_base}")
        preguntas = data.get("preguntas", [])
        total = len(preguntas)
        aciertos = errores = sin_respuesta = 0
        errores_detalle = []

        for i, pregunta in enumerate(preguntas):
            pred = pregunta.get(modelo)
            correcta = base_data["preguntas"][i].get("respuesta_correcta") if i < len(base_data["preguntas"]) else None

            if pred is None:
                sin_respuesta += 1
            elif correcta is None:
                continue
            elif pred == correcta:
                aciertos += 1
            else:
                errores += 1
                errores_detalle.append({
                    "índice": i + 1,
                    "predicha": pred,
                    "correcta": correcta,
                    "enunciado": pregunta["enunciado"]
                })

        respondidas = total - sin_respuesta
        acierto_pct = (aciertos / respondidas * 100) if respondidas > 0 else 0

        resumen_global[modelo]["aciertos"] += aciertos
        resumen_global[modelo]["errores"] += errores
        resumen_global[modelo]["sin_respuesta"] += sin_respuesta
        resumen_global[modelo]["total"] += total
        resumen_global[modelo]["errores_detalle"].extend(errores_detalle[:3])

        print("-" * 60)
        print(f"Total de preguntas        : {total}")
        print(f"Respondidas por el modelo : {respondidas}")
        print(f"Aciertos                  : {aciertos}")
        print(f"Errores                   : {errores}")
        print(f"No respondió (None)       : {sin_respuesta}")
        print(f"📈 Porcentaje de acierto  : {acierto_pct:.2f}%")

        print("\n🔍 Ejemplos de errores:")
        for err in errores_detalle[:5]:
            print(f"  ➤ Pregunta {err['índice']}: predijo {err['predicha']}, correcta {err['correcta']}")
            print(f"    {err['enunciado']}")

# ================================================================
# RESUMEN GLOBAL FINAL + EXPORTACIÓN CSV Y EXCEL
# ================================================================

print("\n📊📊📊 RESUMEN GLOBAL POR MODELO 📊📊📊")

csv_path = os.path.join(carpeta_salida, "no_prompt_metrics.csv")
excel_path = os.path.join(carpeta_salida, "no_prompt_metrics.xlsx")

rows = []

for modelo in modelos:
    total = resumen_global[modelo]["total"]
    aciertos = resumen_global[modelo]["aciertos"]
    errores = resumen_global[modelo]["errores"]
    sin_respuesta = resumen_global[modelo]["sin_respuesta"]
    respondidas = total - sin_respuesta
    acierto_pct = (aciertos / respondidas * 100) if respondidas > 0 else 0

    print(f"\n🧠 Modelo: {modelo.upper()}")
    print("-" * 50)
    print(f"Total de preguntas         : {total}")
    print(f"Respondidas                : {respondidas}")
    print(f"Aciertos                   : {aciertos}")
    print(f"Errores                    : {errores}")
    print(f"No respondió (None)        : {sin_respuesta}")
    print(f"📈 Porcentaje de acierto   : {acierto_pct:.2f}%")

    print("🔍 Ejemplos de errores:")
    for err in resumen_global[modelo]["errores_detalle"][:5]:
        print(f"  ➤ Pregunta {err['índice']}: predijo {err['predicha']}, correcta {err['correcta']}")
        print(f"    {err['enunciado']}")

    rows.append({
        "Modelo": modelo,
        "Total": total,
        "Respondidas": respondidas,
        "Aciertos": aciertos,
        "Errores": errores,
        "Sin respuesta": sin_respuesta,
        "Accuracy (%)": round(acierto_pct, 2)
    })

# Guardar CSV
with open(csv_path, "w", newline="", encoding="utf-8") as f_csv:
    writer = csv.DictWriter(f_csv, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)

# Guardar Excel
df = pd.DataFrame(rows)
df.to_excel(excel_path, index=False)

print(f"\n✅ Resultados globales guardados en:")
print(f"   • CSV  : {csv_path}")
print(f"   • Excel: {excel_path}")
print("\n✅ Pipeline completado correctamente.")