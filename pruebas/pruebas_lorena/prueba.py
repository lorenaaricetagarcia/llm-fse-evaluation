import pdfplumber
import json

def pdf_to_json(pdf_path, json_path):
    """
    Convierte un archivo PDF a formato JSON.

    Args:
        pdf_path: La ruta al archivo PDF de entrada.
        json_path: La ruta donde se guardará el archivo JSON de salida.
    """
    try:
        with pdfplumber.open(pdf_path) as pdf:
            all_text = ""
            for page in pdf.pages:
                all_text += page.extract_text() + "//n"

        data = {"content": all_text}

        with open(json_path, 'w', encoding='utf-8') as json_file:
            json.dump(data, json_file, indent=4, ensure_ascii=False)

        print(f"PDF convertido a JSON exitosamente. Archivo guardado en: {json_path}")

    except Exception as e:
        print(f"Error durante la conversión: {e}")

# Ejemplo de uso
pdf_file = "C://Users//loren//OneDrive//Escritorio//TFM_UCM//examenes_mir//BIOLOGÍA//cuaderno_texto//Cuaderno_2020_BIOLOGÍA_0_C.pdf"
json_file = "C://Users//loren//OneDrive//Escritorio//TFM_UCM//examenes_mir//BIOLOGÍA//cuaderno_texto"
pdf_to_json(pdf_file, json_file)