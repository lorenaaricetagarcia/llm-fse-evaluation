import os
import json
import requests
import re
import wikipediaapi
import pandas as pd
from keybert import KeyBERT
from sentence_transformers import SentenceTransformer
from collections import Counter

# ================================================================
# ⚙️ CONFIGURACIÓN
# ================================================================

# Forzar KeyBERT a CPU
sentence_model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
kw_model = KeyBERT(model=sentence_model)
wiki_wiki = wikipediaapi.Wikipedia('es')

# Modelos a probar
modelos = ["llama3", "mistral", "gemma"]

# Carpetas
carpeta_examenes = "results/1_data_preparation/6_json_final/prueba"
carpeta_salida_modelos = "/home/xs1/Desktop/Lorena/results/2_models/2_rag/1_wikipedia"
carpeta_metricas = carpeta_salida_modelos
os.makedirs(carpeta_examenes, exist_ok=True)
os.makedirs(carpeta_salida_modelos, exist_ok=True)
os.makedirs(carpeta_metricas, exist_ok=True)

# ================================================================
# 🔑 FUNCIÓN PARA EXTRAER KEYWORDS
# ================================================================
def get_keywords(texto):
    keywords = kw_model.extract_keywords(texto, top_n=1)
    if keywords:
        return keywords[0][0]
    return None

# ================================================================
# 🚀 LOOP PRINCIPAL
# ================================================================

resultados_titulacion = {modelo: {} for modelo in modelos}
metricas_globales = []

archivos_json = [f for f in os.listdir(carpeta_examenes) if f.endswith(".json")]

for archivo_json in archivos_json:
    nombre_examen = os.path.splitext(archivo_json)[0]
    ruta_json = os.path.join(carpeta_examenes, archivo_json)

    partes = nombre_examen.split("_")
    titulacion = partes[0] if len(partes) > 0 else "DESCONOCIDO"
    anio = partes[1] if len(partes) > 1 else "SIN_AÑO"

    print(f"\n📘 Procesando titulación: {titulacion} | Año: {anio}")

    with open(ruta_json, "r", encoding="utf-8") as f:
        base_data = json.load(f)

    # ⚠️ Comprobar duplicados en número
    numeros = [p.get("numero") for p in base_data["preguntas"]]
    dup_count = sum(1 for _, c in Counter(numeros).items() if c > 1)
    if dup_count > 0:
        print(f"⚠️ {archivo_json}: detectados {dup_count} duplicados en 'numero' → se comparará por posición.\n")

    for modelo in modelos:
        if titulacion not in resultados_titulacion[modelo]:
            resultados_titulacion[modelo][titulacion] = []

        print(f"   🔹 Modelo: {modelo}")
        preguntas_resultado = []

        for i, pregunta in enumerate(base_data["preguntas"], 1):
            # Solo tipo texto
            if pregunta.get("tipo") != "texto":
                continue

            enunciado = pregunta["enunciado"]
            keyword = get_keywords(enunciado)
            if not keyword:
                print(f"   ❌ No se encontró keyword en la pregunta {i}")
                continue

            # Descargar contexto de Wikipedia
            page = wiki_wiki.page(keyword)
            if not page.exists():
                print(f"   ❌ No hay artículo de Wikipedia para: {keyword}")
                continue

            contexto = page.summary[:1500]  # limitar longitud del contexto

            # Construir prompt RAG
            opciones = "\n".join([f"{idx+1}. {op}" for idx, op in enumerate(pregunta["opciones"])])
            prompt = f"""Usa el siguiente contexto para responder:

{contexto}

Pregunta:
{enunciado}

Opciones:
{opciones}

Responde con el formato: 'La respuesta correcta es la número X.' seguido de una breve explicación.
Si no estás seguro, responde únicamente: 'No estoy seguro.'
"""

            try:
                payload = {"model": modelo, "prompt": prompt, "stream": False}
                response = requests.post("http://localhost:11434/api/generate", json=payload, timeout=180)
                data_model = response.json()
                texto = data_model.get("response", "").strip()
            except Exception as e:
                texto = f"❌ Error en pregunta {i}: {e}"

            print(f"      🧠 Pregunta {i}: {texto[:80]}...")

            match = re.search(r'\b([1-4])\b', texto)
            seleccion = int(match.group(1)) if match else None

            nueva_pregunta = {
                "año": anio,
                "numero": pregunta.get("numero"),
                "enunciado": enunciado,
                "opciones": pregunta.get("opciones"),
                modelo: seleccion,
                f"{modelo}_texto": texto
            }
            preguntas_resultado.append(nueva_pregunta)

        # Guardar resultados por modelo y titulación
        resultados_titulacion[modelo][titulacion].extend(preguntas_resultado)

        # ============================================================
        # 📊 CÁLCULO DE MÉTRICAS (por posición)
        # ============================================================
        total = len(preguntas_resultado)
        aciertos = errores = sin_respuesta = 0

        for i, pregunta in enumerate(preguntas_resultado):
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

        respondidas = total - sin_respuesta
        acierto_pct = (aciertos / respondidas * 100) if respondidas > 0 else 0

        metricas_globales.append({
            "Modelo": modelo,
            "Titulación": titulacion,
            "Año": anio,
            "Total preguntas": total,
            "Respondidas": respondidas,
            "Aciertos": aciertos,
            "Errores": errores,
            "Sin respuesta": sin_respuesta,
            "Accuracy (%)": round(acierto_pct, 2)
        })

        print(f"      ✅ Accuracy {modelo.upper()} ({titulacion} {anio}): {acierto_pct:.2f}%")

# ================================================================
# 💾 GUARDAR RESULTADOS FINALES
# ================================================================

# Guardar JSON por modelo y titulación
for modelo in modelos:
    for titulacion, preguntas in resultados_titulacion[modelo].items():
        salida_json = os.path.join(carpeta_salida_modelos, f"{titulacion}_{modelo}_rag_wikipedia_v1.json")
        with open(salida_json, "w", encoding="utf-8") as f_out:
            json.dump({"preguntas": preguntas}, f_out, ensure_ascii=False, indent=2)
        print(f"\n✅ Guardado JSON: {salida_json}")

# Guardar métricas globales
df_metricas = pd.DataFrame(metricas_globales)
csv_path = os.path.join(carpeta_metricas, "rag_wikipedia_v1_metrics.csv")
excel_path = os.path.join(carpeta_metricas, "rag_wikipedia_v1_metrics.xlsx")

df_metricas.to_csv(csv_path, index=False, encoding="utf-8-sig")
df_metricas.to_excel(excel_path, index=False)

print(f"\n✅ Métricas guardadas en:")
print(f"   • CSV  : {csv_path}")
print(f"   • Excel: {excel_path}")
print("\n🏁 Pipeline RAG-Wikipedia completado correctamente.")
