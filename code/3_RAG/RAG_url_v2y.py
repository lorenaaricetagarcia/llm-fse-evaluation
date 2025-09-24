import os
import json
import requests
import re
import wikipediaapi
import wikipedia
import spacy
from keybert import KeyBERT
from sentence_transformers import SentenceTransformer

# ==============================
# CONFIGURACIÓN
# ==============================

sentence_model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
kw_model = KeyBERT(model=sentence_model)
nlp = spacy.load("es_core_news_sm")
wiki_api = wikipediaapi.Wikipedia(language='es')

wikipedia.set_lang("es")

modelos = ["llama3", "mistral", "gemma"]

carpeta_examenes = "/home/xs1/Desktop/Lorena/results/1_data_preparation/6_json_final/prueba"
carpeta_salida = "results/2_models/rag"
os.makedirs(carpeta_salida, exist_ok=True)
archivos_json = [f for f in os.listdir(carpeta_examenes) if f.endswith(".json")]

# ==============================
# FUNCIONES DE KEYWORDS Y WIKI
# ==============================

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

    sugerencias = wikipedia.search(keyword)
    for sugerida in sugerencias:
        page_alt = wiki_api.page(sugerida)
        if page_alt.exists():
            return page_alt.text, True, sugerida

    return None, False, None

# ==============================
# LOOP PRINCIPAL
# ==============================

resultados_titulacion = {modelo: {} for modelo in modelos}
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

    for modelo in modelos:
        print(f"\n🔄 Procesando modelo: {modelo} para titulación: {titulacion} ({total_preguntas} preguntas)")
        if titulacion not in resultados_titulacion[modelo]:
            resultados_titulacion[modelo][titulacion] = []

        for i, pregunta in enumerate(base_data["preguntas"], 1):
            print(f"   🧪 Pregunta {i}/{total_preguntas} en modelo {modelo}")
            enunciado = pregunta["enunciado"]

            # KEYWORDS
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

            # CONTEXTO MULTIPLE
            contextos = []
            for kw in keywords_keybert:
                contenido, usada_sugerencia, sugerida = buscar_con_sugerencia(kw)
                if contenido:
                    contextos.append(contenido)
                    if usada_sugerencia:
                        sugerencias_usadas += 1
                        sugerencias_log.append({
                            "original": kw,
                            "sugerida": sugerida
                        })

            if not contextos:
                continue

            contexto_completo = "\n\n".join(contextos)

            # PROMPT
            opciones = "\n".join([f"{idx+1}. {op}" for idx, op in enumerate(pregunta["opciones"])])
            PROMPT_RAG = (
                "Eres un profesional médico altamente capacitado que se está preparando para un examen clínico.\n"
                "Responde la siguiente pregunta de opción múltiple utilizando primero la información recuperada que se proporciona.\n"
                "Si la respuesta está claramente respaldada por el contenido recuperado, úsalo para justificar tu elección.\n"
                "Si el contenido recuperado no contiene suficiente información, entonces recurre a tu conocimiento médico interno para responder.\n"
                "Tu respuesta debe seguir este formato: 'La respuesta correcta es la número X.' (donde X es un número del 1 al 4), "
                "seguido de una explicación detallada utilizando razonamiento clínico y terminología médica.\n"
                "No adivines ni proporciones múltiples opciones. Si no estás seguro, responde únicamente: 'No estoy seguro.'\n"
                "Sé preciso, conciso y clínicamente relevante.\n\n"
            )

            prompt = (
                PROMPT_RAG +
                f"Retrieved content:\n{contexto_completo}\n\n"
                f"Question:\n{enunciado}\n\n"
                f"Options:\n{opciones}\n"
            )

            # LLAMADA AL MODELO
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
            resultados_titulacion[modelo][titulacion].append(nueva_pregunta)

# ==============================
# GUARDAR RESULTADOS JSON
# ==============================

for modelo in modelos:
    for titulacion, preguntas in resultados_titulacion[modelo].items():
        salida_json = os.path.join(carpeta_salida, f"{titulacion}_{modelo}_RAG.json")
        with open(salida_json, "w", encoding="utf-8") as f_out:
            json.dump({"preguntas": preguntas}, f_out, ensure_ascii=False, indent=2)

# ==============================
# GUARDAR KEYWORDS Y ESTADÍSTICAS
# ==============================

ruta_keywords = os.path.join(carpeta_salida, "keywords_resumen.txt")
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

# ==============================
# GUARDAR SUGERENCIAS USADAS
# ==============================

ruta_sugerencias = os.path.join(carpeta_salida, "sugerencias_usadas.txt")
with open(ruta_sugerencias, "w", encoding="utf-8") as f_sug:
    f_sug.write("=== Sugerencias de Wikipedia usadas ===\n\n")
    for item in sugerencias_log:
        f_sug.write(f"Original: {item['original']} → Sugerida: {item['sugerida']}\n")
    f_sug.write("\n=== Estadísticas ===\n")
    f_sug.write(f"Sugerencias usadas: {sugerencias_usadas}\n")
    f_sug.write