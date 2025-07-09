import os
import json

input_dir = "results/4_json_corregido"
output_dir = "results/5_json_type"

os.makedirs(output_dir, exist_ok=True)

titulaciones_con_imagenes = {"MEDICINA", "ENFERMERÍA"}

# Contadores globales
total_global = 0
texto_global = 0
imagen_global = 0

for filename in os.listdir(input_dir):
    if filename.endswith(".json"):
        with open(os.path.join(input_dir, filename), "r", encoding="utf-8") as f:
            data = json.load(f)

        titulacion = data.get("titulacion", "").upper()

        total = 0
        texto = 0
        imagen = 0

        for pregunta in data.get("preguntas", []):
            total += 1
            enunciado = pregunta.get("enunciado", "").lower()

            if titulacion in titulaciones_con_imagenes and "pregunta asociada a la imagen" in enunciado:
                pregunta["tipo"] = "imagen"
                imagen += 1
            else:
                pregunta["tipo"] = "texto"
                texto += 1

        # Actualizar contadores globales
        total_global += total
        texto_global += texto
        imagen_global += imagen

        # Guardar JSON actualizado
        output_path = os.path.join(output_dir, filename)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        print(f"✅ {filename} procesado:")
        print(f"   📊 Total preguntas: {total}")
        print(f"   🖼️ Imagen: {imagen} | 📝 Texto: {texto}")

# Mostrar resumen global
print("\n📦 RESUMEN GLOBAL:")
print(f"🔢 Total preguntas procesadas: {total_global}")
print(f"🖼️ Total tipo imagen: {imagen_global}")
print(f"📝 Total tipo texto: {texto_global}")

if total_global > 0:
    p_imagen = round((imagen_global / total_global) * 100, 2)
    p_texto = round((texto_global / total_global) * 100, 2)
    print(f"\n📊 Porcentajes globales:")
    print(f"   🖼️ Imagen: {p_imagen}%")
    print(f"   📝 Texto: {p_texto}%")
else:
    print("⚠️ No se procesaron preguntas.")