import re
import json

# Ruta al archivo de texto previamente extraído
ruta_txt = "mir_extraido_completo.txt"

# Leer el contenido del archivo
with open(ruta_txt, "r", encoding="utf-8") as file:
    texto = file.read()

# Expresión regular para capturar preguntas y sus opciones
patron = re.compile(
    r"\n?(\d+)\.\s*(.*?)\s*"
    r"1\.\s*(.*?)\s*"
    r"2\.\s*(.*?)\s*"
    r"3\.\s*(.*?)\s*"
    r"4\.\s*(.*?)(?=\n\d+\.|\Z)",  # detecta el final o la siguiente pregunta
    re.DOTALL
)

# Extraer coincidencias y construir estructura
preguntas = []
for match in patron.finditer(texto):
    numero = int(match.group(1))
    enunciado = match.group(2).strip().replace('\n', ' ')
    opciones = [match.group(i).strip().replace('\n', ' ') for i in range(3, 7)]

    preguntas.append({
        "numero": numero,
        "enunciado": enunciado,
        "opciones": opciones
    })

# Construir estructura JSON
estructura_json = {
    "archivo": "mir.pdf",
    "preguntas": preguntas
}

# Guardar como archivo JSON
with open("mir_preguntas.json", "w", encoding="utf-8") as salida:
    json.dump(estructura_json, salida, ensure_ascii=False, indent=2)

print("✅ Archivo JSON generado: mir_preguntas.json")
