# ================================================================
# RAG con abstracts de PubMed (consulta en tiempo real) — versión estable v2.2
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
from transformers import pipeline
import torch
from datetime import datetime
from collections import Counter

# ================================================================
# ⚙️ CONFIGURACIÓN INICIAL
# ================================================================

print("⏳ Cargando modelos de embeddings, NLP y traductor...")

# NLP y embeddings
sentence_model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
kw_model = KeyBERT(model=sentence_model)
nlp = spacy.load("es_core_news_sm")

# GPU / CPU
if not torch.cuda.is_available():
    device = -1
    print("⚙️ No se detecta GPU → usando CPU.")
else:
    try:
        torch.cuda.get_device_name(0)
        device = 0
        print(f"✅ GPU detectada: {torch.cuda.get_device_name(0)}")
    except Exception:
        device = -1
        print("⚠️ GPU no compatible → uso forzado de CPU.")

# Traductor ES→EN
try:
    translator = pipeline("translation", model="Helsinki-NLP/opus-mt-es-en", device=device)
    print("✅ Traductor cargado: Helsinki-NLP/opus-mt-es-en")
except Exception as e:
    print(f"⚠️ No se pudo cargar el traductor ({e}). Se usará texto en español.")
    translator = None

# Modelos y rutas
modelos = ["llama3"]
carpeta_examenes = "/home/xs1/Desktop/Lorena/results/1_data_preparation/6_json_final/prueba"
carpeta_correctas = carpeta_examenes
carpeta_salida = "/home/xs1/Desktop/Lorena/results/2_models/2_rag/2_pubmed"
carpeta_metricas = carpeta_salida
os.makedirs(carpeta_salida, exist_ok=True)
os.makedirs(carpeta_metricas, exist_ok=True)
archivos_json = [f for f in os.listdir(carpeta_examenes) if f.endswith(".json")]

# ================================================================
# 🔍 FUNCIONES AUXILIARES
# ================================================================

def get_keywords_keybert(texto, top_n=3):
    try:
        return [kw[0] for kw in kw_model.extract_keywords(texto, top_n=top_n)]
    except Exception:
        return []

def get_keywords_spacy(texto, top_n=5):
    doc = nlp(texto)
    kws = [token.text.lower() for token in doc if token.pos_ in ["NOUN", "PROPN"]]
    seen, out = set(), []
    for k in kws:
        if k not in seen:
            seen.add(k)
            out.append(k)
    return out[:top_n]

def ensure_3_keywords(base_kw):
    kws = [k for k in base_kw if k and isinstance(k, str)]
    extras = ["medicina", "diagnóstico", "tratamiento"]
    for e in extras:
        if len(kws) >= 3:
            break
        if e not in kws:
            kws.append(e)
    return kws[:3]

def translate_keywords_es_en(kws_es):
    if translator is None:
        return kws_es
    try:
        phrase = ", ".join(kws_es)
        translated = translator(phrase)[0]["translation_text"]
        parts = [p.strip().lower() for p in re.split(r"[,;/]", translated) if p.strip()]
        if len(parts) < 3:
            parts = []
            for k in kws_es:
                try:
                    t = translator(k)[0]["translation_text"].strip().lower()
                    parts.append(t)
                except Exception:
                    parts.append(k)
        parts = [re.sub(r"[^a-z0-9 \-()]", "", p) for p in parts]
        return ensure_3_keywords(parts)
    except Exception:
        return kws_es

def _pubmed_esearch(term, retmax):
    url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?db=pubmed&term={term}&retmax={retmax}"
    r = requests.get(url, timeout=30)
    tree = ET.fromstring(r.text)
    return [id_elem.text for id_elem in tree.findall(".//Id")]

def _pubmed_efetch(ids):
    id_list = ",".join(ids)
    url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=pubmed&id={id_list}&retmode=text&rettype=abstract"
    r = requests.get(url, timeout=30)
    return r.text.strip()

def _search_tiered(keywords_list, top_k, lang_tag="[EN]"):
    for n in (3, 2, 1):
        kws = keywords_list[:n]
        term = "+".join(kws)
        print(f"    🔎 PubMed query {lang_tag} ({n} kw): {term}")
        ids = _pubmed_esearch(term, top_k)
        if ids:
            print(f"    📚 PMIDs: {', '.join(ids)}")
            return ids, kws, term
        print("    ⚠️ Sin resultados, probando con menos keywords...")
    return [], [], ""

def retrieve_pubmed_context(query, top_k=3):
    try:
        kw_kb = get_keywords_keybert(query, top_n=3)
        kw_sp = get_keywords_spacy(query, top_n=5)
        base, seen = [], set()
        for k in kw_kb + kw_sp:
            k = k.strip().lower()
            if k and k not in seen:
                seen.add(k)
                base.append(k)
        kws_es = ensure_3_keywords(base[:3])
        kws_en = translate_keywords_es_en(kws_es)

        print(f"    🔍 Keywords ES: {', '.join(kws_es)}")
        print(f"    🌎 Keywords EN: {', '.join(kws_en)}")

        ids, used_kws, term_used = _search_tiered(kws_en, top_k, "[EN]")
        if not ids:
            ids, used_kws, term_used = _search_tiered(kws_es, top_k, "[ES]")

        if not ids:
            return "No se encontraron resultados en PubMed.", [], kws_es, kws_en, term_used

        raw_text = _pubmed_efetch(ids)
        context = "\n\n".join(raw_text.split("\n\n")[:3])[:2000]
        time.sleep(0.4)
        return context, ids, kws_es, kws_en, term_used
    except Exception as e:
        return f"⚠️ Error en búsqueda PubMed: {e}", [], [], [], ""

# ================================================================
# 🚀 LOOP PRINCIPAL
# ================================================================

resultados_titulacion = {modelo: {} for modelo in modelos}
metricas_globales = []

for archivo_json in archivos_json:
    nombre_examen = os.path.splitext(archivo_json)[0]
    partes = nombre_examen.split("_")
    titulacion = partes[0] if len(partes) > 0 else "DESCONOCIDO"
    ruta_json = os.path.join(carpeta_examenes, archivo_json)

    with open(ruta_json, "r", encoding="utf-8") as f:
        base_data = json.load(f)

    total_preguntas = len(base_data["preguntas"])
    print(f"\n🔄 Procesando {titulacion} ({total_preguntas} preguntas)...")

    for modelo in modelos:
        resultados_titulacion[modelo][titulacion] = []
        start_time = datetime.now()

        for i, pregunta in enumerate(base_data["preguntas"], 1):
            if pregunta.get("tipo") != "texto":
                continue

            enunciado = pregunta["enunciado"]
            print(f"\n🧠 Pregunta {i}/{total_preguntas}: {enunciado[:80]}...")
            contexto, pmids, kws_es, kws_en, term_usado = retrieve_pubmed_context(enunciado)
            opciones = "\n".join([f"{idx+1}. {op}" for idx, op in enumerate(pregunta["opciones"])])

            PROMPT_RAG = (
                "Eres un profesional médico que debe responder una pregunta tipo examen MIR.\n"
                "Lee cuidadosamente el CONTEXTO recuperado y luego la PREGUNTA.\n"
                "Responde con el formato: 'La respuesta correcta es la número X' (X entre 1 y 4), "
                "seguido de una breve justificación.\n"
                "No proporciones varias opciones ni respondas 'No estoy seguro'.\n\n"
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
                "keywords_es": kws_es,
                "keywords_en": kws_en,
                "pubmed_term": term_usado,
                "pmids_usados": pmids,
                modelo: seleccion,
                f"{modelo}_texto": texto
            }

            resultados_titulacion[modelo][titulacion].append(nueva_pregunta)

            # Guardado temporal cada 50
            if i % 50 == 0:
                salida_temp = os.path.join(carpeta_salida, f"{titulacion}_{modelo}_rag_pubmed_v3.json")
                with open(salida_temp, "w", encoding="utf-8") as f_temp:
                    json.dump({"preguntas": resultados_titulacion[modelo][titulacion]}, f_temp, ensure_ascii=False, indent=2)
                print(f"💾 Guardado temporal ({i}/{total_preguntas}) en {salida_temp}")

        # Guardado final
        salida_final = os.path.join(carpeta_salida, f"{titulacion}_{modelo}_rag_pubmed_v3.json")
        with open(salida_final, "w", encoding="utf-8") as f_out:
            json.dump({"preguntas": resultados_titulacion[modelo][titulacion]}, f_out, ensure_ascii=False, indent=2)

# ================================================================
# 📊 MÉTRICAS DE EVALUACIÓN (por posición)
# ================================================================

print("\n📈 Calculando métricas comparando con respuestas correctas...\n")

for modelo in modelos:
    for titulacion, preguntas_modelo in resultados_titulacion[modelo].items():
        ruta_correctas = os.path.join(carpeta_correctas, f"{titulacion}.json")
        if not os.path.exists(ruta_correctas):
            print(f"⚠️ No se encontró archivo de respuestas correctas para {titulacion}")
            continue

        with open(ruta_correctas, "r", encoding="utf-8") as f_corr:
            data_corr = json.load(f_corr)

        total = min(len(preguntas_modelo), len(data_corr["preguntas"]))
        aciertos = errores = sin_respuesta = 0

        for i in range(total):
            correcta = data_corr["preguntas"][i].get("respuesta_correcta")
            pred = preguntas_modelo[i].get(modelo)
            if pred is None:
                sin_respuesta += 1
            elif pred == correcta:
                aciertos += 1
            else:
                errores += 1

        respondidas = total - sin_respuesta
        accuracy = (aciertos / respondidas * 100) if respondidas > 0 else 0

        metricas_globales.append({
            "Modelo": modelo,
            "Titulación": titulacion,
            "Total preguntas": total,
            "Respondidas": respondidas,
            "Aciertos": aciertos,
            "Errores": errores,
            "Sin respuesta": sin_respuesta,
            "Accuracy (%)": round(accuracy, 2)
        })

        print(f"📘 {titulacion} – {modelo}: {aciertos}/{total} correctas ({accuracy:.2f}%)")

# ================================================================
# 💾 GUARDAR MÉTRICAS (CSV + EXCEL)
# ================================================================

df_metricas = pd.DataFrame(metricas_globales)
csv_path = os.path.join(carpeta_metricas, "rag_pubmed_v3_metrics.csv")
excel_path = os.path.join(carpeta_metricas, "rag_pubmed_v3_metrics.xlsx")

df_metricas.to_csv(csv_path, index=False, encoding="utf-8-sig")
df_metricas.to_excel(excel_path, index=False)

print(f"\n✅ Métricas guardadas en:")
print(f"   • CSV  : {csv_path}")
print(f"   • Excel: {excel_path}")
print("\n🏁 Pipeline RAG-PubMed v3 completado correctamente.")
