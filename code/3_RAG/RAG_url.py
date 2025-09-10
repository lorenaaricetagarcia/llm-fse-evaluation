import os
import json
import requests
import re
import wikipediaapi
from keybert import KeyBERT
from sentence_transformers import SentenceTransformer

# ==============================
# ⚙️ CONFIG
# ==============================

# Forzar KeyBERT a CPU
sentence_model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
kw_model = KeyBERT(model=sentence_model)

wiki_wiki = wikipediaapi.Wikipedia('es')

# Modelos a probar
modelos = ["llama3", "mistral", "gemma"]

# Carpeta de entrada con los exámenes
carpeta_examenes = "results/1_data_preparation/6_json_final"
archivos_json = [f for f in os.listdir(carpeta_examenes) if f.endswith(".json")]

# Carpeta de salida
carpeta_salida = "results/2_models/rag"
os.makedirs(carpeta_salida, exist_ok=True)

# ==============================
# 🔑 Función para extraer keywords
# ==============================
def get_keywords(texto):
    keywords = kw_model.extract_keywords(texto, top_n=1)
    if keywords:
        return keywords[0][0]  # Devolver solo la palabra clave
    return None

# ==============================
# 🚀 LOOP sobre exámenes y modelos
# ==============================

# Diccionario acumulador por titulación
resultados_titulacion = {modelo: {} for modelo in modelos}

for archivo_json in archivos_json:
    nombre_examen = os.path.splitext(archivo_json)[0]  # Ejemplo: MEDICINA_2020
    ruta_json = os.path.join(carpeta_examenes, archivo_json)

    # Detectar titulación y año
    partes = nombre_examen.split("_")
    titulacion = partes[0] if len(partes) > 0 else "DESCONOCIDO"
    anio = partes[1] if len(partes) > 1 else "SIN_AÑO"

    print(f"\n📘 Procesando titulación: {titulacion} | Año: {anio}")

    with open(ruta_json, "r", encoding="utf-8") as f:
        base_data = json.load(f)

    for modelo in modelos:
        if titulacion not in resultados_titulacion[modelo]:
            resultados_titulacion[modelo][titulacion] = []

        print(f"   🔹 Modelo: {modelo}")

        for i, pregunta in enumerate(base_data["preguntas"], 1):
            enunciado = pregunta["enunciado"]

            # 1️⃣ Extraer palabra clave de la pregunta
            keyword = get_keywords(enunciado)
            if not keyword:
                print(f"   ❌ No se encontró keyword en pregunta {i}")
                continue

            # 2️⃣ Descargar contexto de Wikipedia
            page = wiki_wiki.page(keyword)
            if not page.exists():
                print(f"   ❌ No hay artículo de Wikipedia para: {keyword}")
                continue

            contexto = page.summary[:1500]  # limitar contexto

            # 3️⃣ Construir prompt con RAG
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

            # 4️⃣ Ejecutar el modelo con Ollama
            try:
                payload = {"model": modelo, "prompt": prompt, "stream": False}
                response = requests.post("http://localhost:11434/api/generate", json=payload, timeout=180)
                data_model = response.json()
                texto = data_model.get("response", "").strip()
            except Exception as e:
                texto = f"❌ Error en pregunta {i}: {e}"

            print(f"      🧠 Pregunta {i}: {texto[:80]}...")  # Mostrar primeras palabras

            # 5️⃣ Detectar número de respuesta
            match = re.search(r'\b([1-4])\b', texto)
            seleccion = int(match.group(1)) if match else None

            # Guardar resultado
            nueva_pregunta = {
                "año": anio,
                "numero": pregunta.get("numero"),
                "enunciado": enunciado,
                "opciones": pregunta.get("opciones"),
                modelo: seleccion,
                f"{modelo}_texto": texto
            }
            resultados_titulacion[modelo][titulacion].append(nueva_pregunta)

# ==============================
# 💾 Guardar resultados finales
# ==============================
for modelo in modelos:
    for titulacion, preguntas in resultados_titulacion[modelo].items():
        salida_json = os.path.join(carpeta_salida, f"{titulacion}_{modelo}_RAG.json")
        with open(salida_json, "w", encoding="utf-8") as f_out:
            json.dump({"preguntas": preguntas}, f_out, ensure_ascii=False, indent=2)
        print(f"\n✅ Guardado JSON: {salida_json}")


