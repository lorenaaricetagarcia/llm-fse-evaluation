import fitz  # PyMuPDF
import re
import os
from PIL import Image
import pytesseract

# Parámetros
pdf_path = "Cuaderno_2023_FARMACIA_0_C.pdf"  # Ruta local al PDF
output_dir = "imagenes_farmacia_detectadas"
os.makedirs(output_dir, exist_ok=True)

# Abrir PDF
doc = fitz.open(pdf_path)

# Resultado
preguntas_con_imagen = []

# Recorrer páginas
for i, page in enumerate(doc):
    images = page.get_images(full=True)
    if not images:
        continue  # No hay imágenes

    # Extraer imagen de página
    pix = page.get_pixmap(dpi=300)
    image_path = os.path.join(output_dir, f"pagina_{i+1}.png")
    pix.save(image_path)

    # OCR para detectar número de pregunta (sin lang="spa")
    text = pytesseract.image_to_string(Image.open(image_path))
    match = re.search(r"\b(\d{1,3})\.", text)
    numero_pregunta = int(match.group(1)) if match else None

    preguntas_con_imagen.append({
        "pagina": i + 1,
        "pregunta_detectada": numero_pregunta,
        "imagen_extraida": image_path
    })

# Mostrar resumen
for info in preguntas_con_imagen:
    print(f"📄 Página {info['pagina']}: Pregunta {info['pregunta_detectada']} -> Imagen guardada en {info['imagen_extraida']}")
