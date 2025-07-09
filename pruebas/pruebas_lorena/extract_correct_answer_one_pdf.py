from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait, Select
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from bs4 import BeautifulSoup
import time

def obtener_respuestas_desde_busqueda(titulacion, convocatoria, version=0):
    options = webdriver.ChromeOptions()
    options.add_argument("--disable-gpu")
    options.add_argument("--no-sandbox")

    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
    wait = WebDriverWait(driver, 10)

    try:
        print("🌐 Abriendo página...")
        driver.get("https://fse.mscbs.gob.es/fseweb/view/public/datosanteriores/cuadernosExamen/busquedaConvocatoria.xhtml")

        # Titulación
        select_titulacion = Select(wait.until(EC.presence_of_element_located((By.ID, "mainForm:titulacionSelect"))))
        select_titulacion.select_by_visible_text(titulacion)
        time.sleep(0.5)

        # Convocatoria
        select_convocatoria = Select(wait.until(EC.presence_of_element_located((By.ID, "mainForm:anyosSelect"))))
        select_convocatoria.select_by_visible_text(str(convocatoria))
        time.sleep(0.5)

        # Versión
        select_version = Select(wait.until(EC.presence_of_element_located((By.ID, "mainForm:versionSelect"))))
        select_version.select_by_visible_text(str(version))
        time.sleep(0.5)

        # Seleccionar "Hoja de Respuestas" usando ID directo
        radio_respuestas = wait.until(EC.element_to_be_clickable((By.ID, "mainForm:j_idt87:1")))
        radio_respuestas.click()
        time.sleep(0.5)

        # Pulsar botón Ver
        boton_ver = wait.until(EC.element_to_be_clickable((By.ID, "mainForm:j_idt91")))
        boton_ver.click()
        time.sleep(2)

        # Extraer HTML y parsear con BeautifulSoup
        soup = BeautifulSoup(driver.page_source, "html.parser")
        tabla = soup.find("table")

        respuestas = {}
        if tabla:
            celdas = tabla.find_all("td")
            for i in range(0, len(celdas), 2):
                try:
                    numero = int(celdas[i].text.strip())
                    correcta = int(celdas[i + 1].text.strip())
                    respuestas[numero] = correcta
                except:
                    continue
        else:
            print("⚠️ No se encontró la tabla de respuestas.")

        driver.quit()
        return respuestas

    except Exception as e:
        print(f"❌ Error durante la extracción: {e}")
        driver.quit()
        return {}

# --- EJEMPLO DE USO ---
if __name__ == "__main__":
    respuestas = obtener_respuestas_desde_busqueda("BIOLOGÍA", 2023, 0)
    print("📋 Respuestas correctas:")
    print(respuestas)
