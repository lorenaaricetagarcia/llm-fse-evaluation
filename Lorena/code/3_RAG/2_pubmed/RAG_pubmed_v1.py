# ================================================================
# RAG con abstracts de PubMed (consulta en tiempo real) – versión estable unificada
# ================================================================

import os
import json
import requests
import re
import time
import spacy
import xml.etree.ElementTree as ET
import pandas as pd
from keybert import KeyBERT
from sentence_transformers import SentenceTransformer
from collections import Counter

# ================================================================
# ⚙️ CONFIGURACIÓN
# ================================================================

print("⏳ Cargando modelos y configuraciones...")

# Extracción de keywords y embeddings
sentence_model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
kw_model = KeyBERT(model=sentence_model)
nlp = spacy.load("es_core_news_sm")

# Modelos Ollama a evaluar
modelos = ["llama3"]

# Directorios
carpeta_examenes = "/home/xs1/Desktop/Lorena/results/1_data_preparation/6_json_final"
carpeta_correctas = carpeta_examenes
carpeta_salida = "/home/xs1/Desktop/Lorena/results/2_models/2_rag/2_pubmed"
carpeta_metricas = carpeta_salida
os.makedirs(carpeta_salida, exist_ok=True)
os.makedirs(carpeta_metricas, exist_ok=True)

archivos_json = [f for f in os.listdir(carpeta_examenes) if f.endswith(".json")]

# ================================================================
# 🔍 FUNCIONES DE KEYWORDS Y BÚSQUEDA PUBMED
# ================================================================

def get_keywords_keybert(texto, top_n=3):
    try:
        return [kw[0] for kw in kw_model.extract_keywords(texto, top_n=top_n)]
    except Exception:
        return []

def get_keywords_spacy(texto):
    doc = nlp(texto)
    return [token.text.lower() for token in doc if token.pos_ in ["NOUN", "PROPN"]]

def retrieve_pubmed_context(query, top_k=3):
    """Busca abstracts relevantes en PubMed usando la API oficial (E-utilities)"""
    try:
        keywords = get_keywords_keybert(query, top_n=3)
        if not keywords:
            keywords = get_keywords_spacy(query)[:3]
        if not keywords:
            return "No se encontraron palabras clave relevantes."

        search_term = "+".join(keywords)
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
# 🚀 LOOP PRINCIPAL
# ================================================================

resultados_titulacion = {modelo: {} for modelo in modelos}
keywords_log = []
metricas_globales = []
sin_keywords_keybert = sin_keywords_spacy = coincidencias = 0

for archivo_json in archivos_json:
    nombre_examen = os.path.splitext(archivo_json)[0]
    partes = nombre_examen.split("_")
    titulacion = partes[0] if len(partes) > 0 else "DESCONOCIDO"
    ruta_json = os.path.join(carpeta_examenes, archivo_json)

    with open(ruta_json, "r", encoding="utf-8") as f:
        base_data = json.load(f)

    total_preguntas = len(base_data["preguntas"])
    print(f"\n📘 Procesando {titulacion} ({total_preguntas} preguntas)...")

    numeros = [p.get("numero") for p in base_data["preguntas"]]
    dup_count = sum(1 for _, c in Counter(numeros).items() if c > 1)
    if dup_count > 0:
        print(f"⚠️ Aviso: {dup_count} duplicados en 'numero' → se comparará por posición.")

    for modelo in modelos:
        print(f"\n⚙️ Modelo: {modelo}")
        resultados_titulacion[modelo][titulacion] = []

        for i, pregunta in enumerate(base_data["preguntas"], 1):
            if pregunta.get("tipo") != "texto":
                continue

            enunciado = pregunta["enunciado"]

            # === Extracción de keywords ===
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

            # === Recuperar contexto PubMed ===
            contexto = retrieve_pubmed_context(enunciado)
            opciones = "\n".join([f"{idx+1}. {op}" for idx, op in enumerate(pregunta["opciones"])])

            PROMPT_RAG = (
                "Eres un profesional médico que debe responder una pregunta tipo examen MIR.\n"
                "Utiliza los abstracts de PubMed como fuente de evidencia biomédica.\n"
                "Tu respuesta debe seguir este formato: 'La respuesta correcta es la número X.' (X entre 1 y 4), "
                "seguido de una breve justificación.\n"
                "No inventes ni proporciones varias opciones.\n\n"
            )

            prompt = (
                f"{PROMPT_RAG}"
                f"📚 CONTEXTO (PubMed):\n{contexto}\n\n"
                f"❓ PREGUNTA:\n{enunciado}\n\n"
                f"🔢 OPCIONES:\n{opciones}\n"
            )

            # === Generación con Ollama ===
            try:
                payload = {"model": modelo, "prompt": prompt, "stream": False}
                response = requests.post("http://localhost:11434/api/generate", json=payload, timeout=180)
                data_model = response.json()
                texto = data_model.get("response", "").strip()
            except Exception as e:
                texto = f"❌ Error en pregunta {i}: {e}"

            match = re.search(r"\b([1-4])\b", texto)
            seleccion = int(match.group(1)) if match else None

            resultados_titulacion[modelo][titulacion].append({
                "numero": pregunta.get("numero"),
                "enunciado": enunciado,
                "opciones": pregunta.get("opciones"),
                modelo: seleccion,
                f"{modelo}_texto": texto
            })

# ================================================================
# 💾 GUARDAR RESULTADOS Y LOGS
# ================================================================

for modelo in modelos:
    for titulacion, preguntas in resultados_titulacion[modelo].items():
        salida_json = os.path.join(carpeta_salida, f"{titulacion}_{modelo}_rag_pubmed_v1.json")
        with open(salida_json, "w", encoding="utf-8") as f_out:
            json.dump({"preguntas": preguntas}, f_out, ensure_ascii=False, indent=2)
        print(f"💾 Guardado: {salida_json}")

ruta_keywords = os.path.join(carpeta_salida, "keywords_rag_pubmed_v1.txt")
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
# 📊 MÉTRICAS DE EVALUACIÓN (por posición)
# ================================================================

print("\n📊 Calculando métricas comparando con respuestas correctas (por posición)...")

metricas = []
for modelo in modelos:
    total_global = aciertos_global = errores_global = sin_contestar_global = 0

    for titulacion, preguntas_modelo in resultados_titulacion[modelo].items():
        ruta_correctas = os.path.join(carpeta_correctas, f"{titulacion}.json")
        if not os.path.exists(ruta_correctas):
            continue

        with open(ruta_correctas, "r", encoding="utf-8") as f_corr:
            data_corr = json.load(f_corr)

        total = min(len(preguntas_modelo), len(data_corr["preguntas"]))
        aciertos = errores = sin_contestar = 0

        for i in range(total):
            correcta = data_corr["preguntas"][i].get("respuesta_correcta")
            pred = preguntas_modelo[i].get(modelo)
            if pred is None:
                sin_contestar += 1
            elif pred == correcta:
                aciertos += 1
            else:
                errores += 1

        respondidas = total - sin_contestar
        accuracy = (aciertos / respondidas * 100) if respondidas > 0 else 0

        metricas.append({
            "Modelo": modelo,
            "Titulación": titulacion,
            "Total preguntas": total,
            "Respondidas": respondidas,
            "Aciertos": aciertos,
            "Errores": errores,
            "Sin respuesta": sin_contestar,
            "Accuracy (%)": round(accuracy, 2)
        })

        total_global += total
        aciertos_global += aciertos
        errores_global += errores
        sin_contestar_global += sin_contestar

    acc_global = (aciertos_global / total_global * 100) if total_global > 0 else 0
    print(f"\n🎯 {modelo.upper()} → Accuracy global: {acc_global:.2f}%")

# ================================================================
# 💾 GUARDAR MÉTRICAS (CSV + EXCEL)
# ================================================================

df_metricas = pd.DataFrame(metricas)
csv_path = os.path.join(carpeta_metricas, "rag_pubmed_v1_metrics.csv")
excel_path = os.path.join(carpeta_metricas, "rag_pubmed_v1_metrics.xlsx")

df_metricas.to_csv(csv_path, index=False, encoding="utf-8-sig")
df_metricas.to_excel(excel_path, index=False)

print(f"\n✅ Métricas guardadas en:")
print(f"   • CSV  : {csv_path}")
print(f"   • Excel: {excel_path}")
print("\n🏁 Pipeline RAG-PubMed (en tiempo real) completado correctamente.")

