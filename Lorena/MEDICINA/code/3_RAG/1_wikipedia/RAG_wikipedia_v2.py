import os
import json
import requests
import re
import time
import spacy
import wikipediaapi
import pandas as pd
from keybert import KeyBERT
from sentence_transformers import SentenceTransformer
from collections import Counter

# ================================================================
# ⚙️ CONFIGURACIÓN
# ================================================================

sentence_model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
kw_model = KeyBERT(model=sentence_model)
nlp = spacy.load("es_core_news_sm")
wiki_api = wikipediaapi.Wikipedia(language='es')

modelos = ["llama3", "mistral", "gemma"]

carpeta_examenes = "/home/xs1/Desktop/Lorena/results/1_data_preparation/6_json_final/prueba"
carpeta_salida = "/home/xs1/Desktop/Lorena/results/2_models/2_rag/1_wikipedia"
carpeta_metricas = carpeta_salida
os.makedirs(carpeta_salida, exist_ok=True)
os.makedirs(carpeta_metricas, exist_ok=True)

archivos_json = [f for f in os.listdir(carpeta_examenes) if f.endswith(".json")]

# ================================================================
# 🔑 FUNCIONES AUXILIARES
# ================================================================

def get_keywords_keybert(texto, top_n=3):
    keywords = kw_model.extract_keywords(texto, top_n=top_n)
    return [kw[0] for kw in keywords]

def get_keywords_spacy(texto):
    doc = nlp(texto)
    return [token.text.lower() for token in doc if token.pos_ in ["NOUN", "PROPN"]]

def buscar_con_sugerencia(keyword):
    page = wiki_api.page(keyword)
    if page.exists():
        return page.text, False, None

    variantes = [keyword.lower(), keyword.capitalize(), keyword.title()]
    for sugerida in variantes:
        page_alt = wiki_api.page(sugerida)
        if page_alt.exists():
            return page_alt.text, True, sugerida

    return None, False, None

# ================================================================
# 🚀 LOOP PRINCIPAL
# ================================================================

resultados_titulacion = {modelo: {} for modelo in modelos}
metricas_globales = []
keywords_log = []
sugerencias_log = []
sin_keywords_keybert = 0
sin_keywords_spacy = 0
coincidencias = 0
sugerencias_usadas = 0

for archivo_json in archivos_json:
    nombre_examen = os.path.splitext(archivo_json)[0]
    partes = nombre_examen.split("_")
    titulacion = partes[0] if len(partes) > 0 else "DESCONOCIDO"
    ruta_json = os.path.join(carpeta_examenes, archivo_json)

    with open(ruta_json, "r", encoding="utf-8") as f:
        base_data = json.load(f)

    total_preguntas = len(base_data["preguntas"])
    print(f"\n📘 Procesando {titulacion} ({total_preguntas} preguntas)")

    # Comprobar duplicados en número
    numeros = [p.get("numero") for p in base_data["preguntas"]]
    dup_count = sum(1 for _, c in Counter(numeros).items() if c > 1)
    if dup_count > 0:
        print(f"⚠️ Detectados {dup_count} duplicados en 'numero' → se comparará por posición.\n")

    for modelo in modelos:
        print(f"\n🔄 Modelo: {modelo}")
        if titulacion not in resultados_titulacion[modelo]:
            resultados_titulacion[modelo][titulacion] = []

        preguntas_resultado = []

        for i, pregunta in enumerate(base_data["preguntas"], 1):
            if pregunta.get("tipo") != "texto":
                continue

            print(f"   🧪 Pregunta {i}/{total_preguntas}")
            enunciado = pregunta["enunciado"]

            # === KEYWORDS ===
            keywords_keybert = get_keywords_keybert(enunciado)
            keywords_spacy = get_keywords_spacy(enunciado)

            if not keywords_keybert:
                sin_keywords_keybert += 1
            if not keywords_spacy:
                sin_keywords_spacy += 1

            coincidencia_actual = len(set(map(str.lower, keywords_keybert)) & set(keywords_spacy))
            if coincidencia_actual > 0:
                coincidencias += 1

            keywords_log.append({
                "pregunta": enunciado,
                "keybert": keywords_keybert,
                "spacy": keywords_spacy,
                "coinciden": coincidencia_actual
            })

            # === CONTEXTO MULTIPLE DESDE WIKIPEDIA ===
            contextos = []
            for kw in keywords_keybert:
                try:
                    contenido, usada_sugerencia, sugerida = buscar_con_sugerencia(kw)
                    if contenido:
                        contextos.append(contenido)
                        if usada_sugerencia:
                            sugerencias_usadas += 1
                            sugerencias_log.append({"original": kw, "sugerida": sugerida})
                    time.sleep(0.3)
                except Exception as e:
                    print(f"⚠️ Error al buscar '{kw}': {e}")
                    continue

            if not contextos:
                continue

            contexto_completo = "\n\n".join(contextos[:3])[:2000]

            # === PROMPT RAG ===
            opciones = "\n".join([f"{idx+1}. {op}" for idx, op in enumerate(pregunta["opciones"])])
            PROMPT_RAG = (
                "Eres un profesional médico que debe responder una pregunta tipo examen clínico (MIR).\n"
                "Lee la información recuperada de Wikipedia y responde utilizando razonamiento clínico.\n"
                "Tu respuesta debe tener el formato: 'La respuesta correcta es la número X.' (X entre 1 y 4), "
                "seguido de una breve justificación.\n"
                "Si no estás seguro, responde solo: 'No estoy seguro.'\n\n"
            )

            prompt = (
                f"{PROMPT_RAG}"
                f"📚 CONTEXTO (Wikipedia):\n{contexto_completo}\n\n"
                f"❓ PREGUNTA:\n{enunciado}\n\n"
                f"🔢 OPCIONES:\n{opciones}\n"
            )

            # === LLAMADA AL MODELO ===
            try:
                payload = {"model": modelo, "prompt": prompt, "stream": False}
                response = requests.post("http://localhost:11434/api/generate", json=payload, timeout=180)
                data_model = response.json()
                texto = data_model.get("response", "").strip()
            except Exception as e:
                texto = f"❌ Error en pregunta {i}: {e}"

            match = re.search(r'\b([1-4])\b', texto)
            seleccion = int(match.group(1)) if match else None

            nueva_pregunta = {
                "numero": pregunta.get("numero"),
                "enunciado": enunciado,
                "opciones": pregunta.get("opciones"),
                modelo: seleccion,
                f"{modelo}_texto": texto
            }
            preguntas_resultado.append(nueva_pregunta)

        resultados_titulacion[modelo][titulacion].extend(preguntas_resultado)

        # ============================================================
        # 📊 MÉTRICAS POR POSICIÓN
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

        print(f"✅ Accuracy {modelo.upper()} ({titulacion}): {accuracy:.2f}%")

# ================================================================
# 💾 GUARDAR RESULTADOS
# ================================================================

# Guardar resultados en JSON
for modelo in modelos:
    for titulacion, preguntas in resultados_titulacion[modelo].items():
        salida_json = os.path.join(carpeta_salida, f"{titulacion}_{modelo}_rag_wikipedia_v2.json")
        with open(salida_json, "w", encoding="utf-8") as f_out:
            json.dump({"preguntas": preguntas}, f_out, ensure_ascii=False, indent=2)
        print(f"💾 Guardado JSON: {salida_json}")

# Guardar métricas globales
df_metricas = pd.DataFrame(metricas_globales)
csv_path = os.path.join(carpeta_metricas, "rag_wikipedia_v2_metrics.csv")
excel_path = os.path.join(carpeta_metricas, "rag_wikipedia_v2_metrics.xlsx")

df_metricas.to_csv(csv_path, index=False, encoding="utf-8-sig")
df_metricas.to_excel(excel_path, index=False)

print(f"\n✅ Métricas guardadas en:")
print(f"   • CSV  : {csv_path}")
print(f"   • Excel: {excel_path}")

# Guardar logs de keywords y sugerencias
ruta_keywords = os.path.join(carpeta_salida, "keywords_resumen_v2.txt")
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
    f_kw.write(f"Preguntas con coincidencias: {coincidencias}\n")
    f_kw.write(f"Total procesadas: {len(keywords_log)}\n")

ruta_sugerencias = os.path.join(carpeta_salida, "sugerencias_usadas_v2.txt")
with open(ruta_sugerencias, "w", encoding="utf-8") as f_sug:
    f_sug.write("=== Sugerencias de Wikipedia usadas ===\n\n")
    for item in sugerencias_log:
        f_sug.write(f"Original: {item['original']} → Sugerida: {item['sugerida']}\n")
    f_sug.write(f"\nTotal sugerencias usadas: {sugerencias_usadas}\n")

print("\n🏁 Pipeline RAG-Wikipedia con sugerencias completado correctamente.")
