import os
import json

input_dir = "json_con_respuesta"
output_dir = "json_con_tipo"

os.makedirs(output_dir, exist_ok=True)

for filename in os.listdir(input_dir):
    if filename.endswith(".json"):
        with open(os.path.join(input_dir, filename), "r", encoding="utf-8") as f:
            data = json.load(f)

        for pregunta in data.get("preguntas", []):
            enunciado = pregunta.get("enunciado", "").lower()
            if "pregunta asociada a la imagen" in enunciado:
                pregunta["tipo"] = "imagen"
            else:
                pregunta["tipo"] = "texto"

        output_path = os.path.join(output_dir, filename)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        print(f"✅ Tipo añadido en: {filename}")
