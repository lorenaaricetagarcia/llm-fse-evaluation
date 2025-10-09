import re
from collections import defaultdict, Counter

ruta_resumen = "results/3_analysis/txt_resumen_examen/resumen_completo.txt"

# Diccionarios para agrupar
fallos_por_pregunta = defaultdict(list)
aciertos_por_pregunta = defaultdict(int)
sin_respuesta_por_pregunta = defaultdict(int)

with open(ruta_resumen, "r", encoding="utf-8") as f:
    contenido = f.read()

# Buscar bloques de errores con regex
errores = re.findall(r"➤ Pregunta (\d+): predijo (\d+|None), correcta (\d+)[^\n]*\n\s+(.*)", contenido)

for numero, predicha, correcta, enunciado in errores:
    clave = f"{numero} - {enunciado.strip()[:100]}..."  # limitar texto
    if predicha == "None":
        sin_respuesta_por_pregunta[clave] += 1
    elif predicha == correcta:
        aciertos_por_pregunta[clave] += 1
    else:
        fallos_por_pregunta[clave].append((predicha, correcta))

# Mostrar patrones más comunes
print("\n📌 Preguntas con más errores:")
for pregunta, errores in Counter(fallos_por_pregunta).most_common(5):
    print(f"\n❌ {pregunta}")
    print(f"  ↪ N. fallos: {len(errores)}")
    for p, c in errores:
        print(f"    - Predijo: {p}, Correcta: {c}")

print("\n✅ Preguntas más acertadas:")
for pregunta, aciertos in aciertos_por_pregunta.items():
    if aciertos >= 3:
        print(f"\n✔️ {pregunta}")
        print(f"  ↪ N. aciertos: {aciertos}")

print("\n🕳️ Preguntas sin respuesta más frecuentes:")
for pregunta, n in sin_respuesta_por_pregunta.items():
    if n >= 2:
        print(f"\n🚫 {pregunta}")
        print(f"  ↪ N. veces sin respuesta: {n}")
