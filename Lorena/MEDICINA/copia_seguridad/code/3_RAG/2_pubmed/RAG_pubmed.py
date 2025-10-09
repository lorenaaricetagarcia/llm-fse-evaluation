# ================================================================
# RAG con abstracts de PubMed (consulta en tiempo real) – versión estable corregida
# ================================================================

import os
import json
import requests
import re
import time
import spacy
import xml.etree.ElementTree as ET
from keybert import KeyBERT
from sentence_transformers import SentenceTransformer
from datasets import load_dataset
import csv
from collections import Counter

# ================================================================
# CONFIGURACIÓN
# ================================================================

print("⏳ Cargando modelos de embeddings y NLP...")

# Keyword extraction y embeddings
sentence_model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
kw_model = KeyBERT(model=sentence_model)
nlp = spacy.load("es_core_news_sm")

# Modelos a evaluar (Ollama)
modelos = ["llama3"]

# Directorios locales
carpeta_examenes = "/home/xs1/Desktop/Lorena/results/1_data_preparation/6_json_final/prueba"
carpeta_salida = "results/2_models/RAG/Pubmed/rag_pubmed_v1_09102025"
os.makedirs(carpeta_salida, exist_ok=True)

archivos_json = [f for f in os.listdir(carpeta_examenes) if f.endswith(".json")]

# ================================================================
# FUNCIONES DE KEYWORDS (KeyBERT + spaCy)
# ================================================================

def get_keywords_keybert(texto, top_n=3):
    try:
        return [kw[0] for kw in kw_model.extract_keywords(texto, top_n=top_n)]
    except Exception:
        return []

def get_keywords_spacy(texto):
    doc = nlp(texto)
    return [token.text.lower() for token in doc if token.pos_ in ["NOUN", "PROPN"]]

# ================================================================
# FUNCIÓN DE BÚSQUEDA EN PUBMED
# ================================================================

def retrieve_pubmed_context(query, top_k=3):
    """Busca abstracts relevantes en PubMed usando la API oficial"""
    try:
        keywords = get_keywords_keybert(query, top_n=3)
        if not keywords:
            keywords = get_keywords_spacy(query)[:3]
        if not keywords:
            return "No se encontraron palabras clave relevantes."

        search_term = "+".join(keywords)
        print(f"🔎 Buscando en PubMed: {search_term}")

        url_search = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?db=pubmed&term={search_term}&retmax={top_k}"
        search_resp = requests.get(url_search)
        tree = ET.fromstring(search_resp.text)
        ids = [id_elem.text for id_elem in tree.findall(".//Id")]

        if not ids:
            return f"No se encontraron resultados en PubMed para: {search_term}"

        id_list = ",".join(ids)
        url_fetch = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=pubmed&id={id_list}&retmode=text&rettype=abstract"
        fetch_resp = requests.get(url_fetch)
        abstracts = fetch_resp.text.strip()

        context = "\n\n".join(abstracts.split("\n\n")[:3])[:2000]
        if not context:
            return "No se pudieron recuperar abstracts."

        time.sleep(0.4)
        return context

    except Exception as e:
        return f"⚠️ Error en búsqueda PubMed: {e}"

# ================================================================
# LOOP PRINCIPAL
# ================================================================

resultados_titulacion = {modelo: {} for modelo in modelos}
keywords_log = []
sin_keywords_keybert = 0
sin_keywords_spacy = 0
coincidencias = 0

for archivo_json in archivos_json:
    nombre_examen = os.path.splitext(archivo_json)[0]
    partes = nombre_examen.split("_")
    titulacion = partes[0] if len(partes) > 0 else "DESCONOCIDO"
    ruta_json = os.path.join(carpeta_examenes, archivo_json)

    with open(ruta_json, "r", encoding="utf-8") as f:
        base_data = json.load(f)

    total_preguntas = len(base_data["preguntas"])
    print(f"\n🔄 Procesando {titulacion} ({total_preguntas} preguntas)...")

    # Aviso de duplicados en número
    numeros = [p.get("numero") for p in base_data["preguntas"]]
    dup_count = sum(1 for _, c in Counter(numeros).items() if c > 1)
    if dup_count > 0:
        print(f"⚠️ Aviso: {dup_count} valores duplicados en 'numero' en {titulacion} (normal si hay varios años).")

    for modelo in modelos:
        print(f"⚙️ Modelo: {modelo}")
        resultados_titulacion[modelo][titulacion] = []

        for i, pregunta in enumerate(base_data["preguntas"], 1):
            enunciado = pregunta["enunciado"]

            keywords_keybert = get_keywords_keybert(enunciado)
            keywords_spacy = get_keywords_spacy(enunciado)
            if not keywords_keybert: sin_keywords_keybert += 1
            if not keywords_spacy: sin_keywords_spacy += 1
            coincidencia_actual = len(set(map(str.lower, keywords_keybert)) & set(keywords_spacy))
            if coincidencia_actual > 0: coincidencias += 1

            keywords_log.append({
                "pregunta": enunciado,
                "keybert": keywords_keybert,
                "spacy": keywords_spacy,
                "coinciden": coincidencia_actual
            })

            contexto = retrieve_pubmed_context(enunciado)
            opciones = "\n".join([f"{idx+1}. {op}" for idx, op in enumerate(pregunta["opciones"])])

            PROMPT_RAG = (
                "Eres un profesional médico altamente capacitado que debe responder una pregunta tipo examen MIR.\n"
                "Analiza la información de la literatura biomédica (PubMed abstracts) y responde de forma concisa.\n"
                "Formato: 'La respuesta correcta es la número X.' (X entre 1 y 4) + una breve justificación.\n"
                "No inventes ni proporciones varias opciones.\n\n"
            )

            prompt = (
                f"{PROMPT_RAG}"
                f"📘 CONTEXTO (PubMed):\n{contexto}\n\n"
                f"❓ PREGUNTA:\n{enunciado}\n\n"
                f"🔢 OPCIONES:\n{opciones}\n"
            )

            try:
                payload = {"model": modelo, "prompt": prompt, "stream": False}
                response = requests.post("http://localhost:11434/api/generate", json=payload, timeout=180)
                data_model = response.json()
                texto = data_model.get("response", "").strip()
            except Exception as e:
                texto = f"❌ Error en pregunta {i}: {e}"

            match = re.search(r"\b([1-4])\b", texto)
            seleccion = int(match.group(1)) if match else None

            nueva_pregunta = {
                "numero": pregunta.get("numero"),
                "enunciado": enunciado,
                "opciones": pregunta.get("opciones"),
                modelo: seleccion,
                f"{modelo}_texto": texto
            }
            resultados_titulacion[modelo][titulacion].append(nueva_pregunta)

# ================================================================
# GUARDAR RESULTADOS Y LOG DE KEYWORDS
# ================================================================

for modelo in modelos:
    for titulacion, preguntas in resultados_titulacion[modelo].items():
        salida_json = os.path.join(carpeta_salida, f"{titulacion}_{modelo}_RAG_PubMed_live.json")
        with open(salida_json, "w", encoding="utf-8") as f_out:
            json.dump({"preguntas": preguntas}, f_out, ensure_ascii=False, indent=2)

ruta_keywords = os.path.join(carpeta_salida, "keywords_resumen_pubmed_live.txt")
with open(ruta_keywords, "w", encoding="utf-8") as f_kw:
    for i, item in enumerate(keywords_log, 1):
        f_kw.write(f"Pregunta {i}:\n")
        f_kw.write(f"  Enunciado: {item['pregunta']}\n")
        f_kw.write(f"  KeyBERT: {item['keybert']}\n")
        f_kw.write(f"  spaCy: {item['spacy']}\n")
        f_kw.write(f"  Coincidencias: {item['coinciden']}\n\n")
    f_kw.write("=== Estadísticas ===\n")
    f_kw.write(f"Preguntas sin keyword (KeyBERT): {sin_keywords_keybert}\n")
    f_kw.write(f"Preguntas sin keyword (spaCy): {sin_keywords_spacy}\n")
    f_kw.write(f"Preguntas con coincidencias entre métodos: {coincidencias}\n")
    f_kw.write(f"Total preguntas procesadas: {len(keywords_log)}\n")

# ================================================================
# MÉTRICAS DE EVALUACIÓN (por posición — corregido)
# ================================================================

carpeta_correctas = "/home/xs1/Desktop/Lorena/results/1_data_preparation/6_json_final"
print("\n📊 Calculando métricas finales comparando con respuestas correctas (por posición)...")

csv_path = os.path.join(carpeta_salida, "resumen_accuracy_rag_pubmed_v1.csv")
with open(csv_path, "w", newline="", encoding="utf-8") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["Modelo", "Titulacion", "Total_Preguntas", "Aciertos", "Errores", "Sin_Contestar", "Accuracy_%"])

    for modelo in modelos:
        total_global = aciertos_global = errores_global = sin_contestar_global = 0
        print(f"\n===============================")
        print(f"📈 Resultados para modelo: {modelo}")
        print("===============================")

        for titulacion, preguntas_modelo in resultados_titulacion[modelo].items():
            ruta_correctas = os.path.join(carpeta_correctas, f"{titulacion}.json")
            if not os.path.exists(ruta_correctas):
                print(f"⚠️ No se encontró archivo de respuestas correctas para {titulacion}")
                continue

            with open(ruta_correctas, "r", encoding="utf-8") as f_corr:
                data_corr = json.load(f_corr)

            total = min(len(preguntas_modelo), len(data_corr["preguntas"]))
            aciertos = errores = sin_contestar = 0

            for i in range(total):
                correcta = data_corr["preguntas"][i].get("respuesta_correcta")
                seleccion = preguntas_modelo[i].get(modelo)
                if seleccion is None:
                    sin_contestar += 1
                elif seleccion == correcta:
                    aciertos += 1
                else:
                    errores += 1

            accuracy = (aciertos / total) * 100 if total > 0 else 0

            print(f"\n🔹 Titulación: {titulacion}")
            print(f"   🧩 Total preguntas: {total}")
            print(f"   ✅ Aciertos: {aciertos}")
            print(f"   ❌ Errores: {errores}")
            print(f"   ⚪ Sin contestar: {sin_contestar}")
            print(f"   🎯 Accuracy: {accuracy:.2f}%")

            writer.writerow([modelo, titulacion, total, aciertos, errores, sin_contestar, f"{accuracy:.2f}"])

            total_global += total
            aciertos_global += aciertos
            errores_global += errores
            sin_contestar_global += sin_contestar

        acc_global = (aciertos_global / total_global) * 100 if total_global > 0 else 0
        print("\n===============================")
        print(f"📊 RESUMEN GLOBAL – {modelo}")
        print(f"🔢 Total preguntas: {total_global}")
        print(f"✅ Aciertos: {aciertos_global}")
        print(f"❌ Errores: {errores_global}")
        print(f"⚪ Sin contestar: {sin_contestar_global}")
        print(f"🎯 Accuracy global: {acc_global:.2f}%")
        print("===============================")

print(f"\n✅ Resultados guardados en CSV: {csv_path}")
print("✅ Pipeline completado correctamente. Resultados guardados en:", carpeta_salida)
