import json

with open("all_mir_questions.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# Mostrar claves del JSON principal
print("🔍 Claves principales del JSON:", list(data.keys()) if isinstance(data, dict) else type(data))

# Si es un diccionario con listas dentro, inspeccionamos una de ellas
if isinstance(data, dict):
    for key, value in data.items():
        print(f"\n🔹 Mostrando primeras entradas de '{key}':")
        if isinstance(value, list):
            for i, item in enumerate(value[:3], start=1):
                print(f"\n--- {key} {i} ---")
                print(json.dumps(item, indent=2, ensure_ascii=False))
        else:
            print(f"{key} no es una lista. Tipo: {type(value)}")
        break  # solo la primera clave por ahora
elif isinstance(data, list):
    for i, pregunta in enumerate(data[:3], start=1):
        print(f"\n--- Pregunta {i} ---")
        print(json.dumps(pregunta, indent=2, ensure_ascii=False))
else:
    print("❌ Formato JSON no reconocido.")
