from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait, Select
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from bs4 import BeautifulSoup
import os
import json
import time

# Carpeta de salida
os.makedirs("2_respuestas_json", exist_ok=True)

# Configuración del navegador (headless para no abrir ventana)
options = webdriver.ChromeOptions()
options.add_argument("--headless=new")
options.add_argument("--disable-gpu")
options.add_argument("--no-sandbox")

# URL base
URL = "https://fse.mscbs.gob.es/fseweb/view/public/datosanteriores/cuadernosExamen/busquedaConvocatoria.xhtml"

def extraer_respuestas(titulacion, convocatoria, driver, wait):
    try:
        # Cargar página base
        driver.get(URL)

        # Seleccionar titulación
        select_titulacion = Select(wait.until(EC.presence_of_element_located((By.ID, "mainForm:titulacionSelect"))))
        select_titulacion.select_by_visible_text(titulacion)
        time.sleep(1)

        # Seleccionar convocatoria
        select_convocatoria = Select(wait.until(EC.presence_of_element_located((By.ID, "mainForm:anyosSelect"))))
        select_convocatoria.select_by_visible_text(convocatoria)
        time.sleep(1)

        # Seleccionar versión 0
        select_version = Select(wait.until(EC.presence_of_element_located((By.ID, "mainForm:versionSelect"))))
        versiones = [opt.text.strip() for opt in select_version.options if opt.get_attribute("value")]

        if "0" not in versiones:
            print("    ⚠️ No hay versión 0, se omite.")
            return

        select_version.select_by_visible_text("0")
        time.sleep(1)

        # Seleccionar radio "Hoja de Respuestas"
        radios = driver.find_elements(By.XPATH, '//input[@type="radio" and contains(@id,"mainForm:j_idt87:")]')
        for radio in radios:
            label = driver.find_element(By.CSS_SELECTOR, f'label[for="{radio.get_attribute("id")}"]')
            if "Hoja de Respuestas" in label.text:
                radio.click()
                break
        else:
            print("    ⚠️ No se encontró radio de Hoja de Respuestas.")
            return

        # Pulsar botón "Ver"
        ver_btn = wait.until(EC.element_to_be_clickable((By.ID, "mainForm:j_idt91")))
        ver_btn.click()
        time.sleep(2)

        # Parsear tabla con BeautifulSoup
        soup = BeautifulSoup(driver.page_source, "html.parser")
        tabla = soup.find("table")

        if not tabla:
            print("    ⚠️ No se encontró tabla de respuestas.")
            return

        respuestas = {}
        celdas = tabla.find_all("td")
        for i in range(0, len(celdas), 2):
            try:
                num = int(celdas[i].text.strip())
                val = int(celdas[i + 1].text.strip())
                respuestas[num] = val
            except:
                continue

        # Guardar archivo
        nombre_archivo = f"{titulacion}_{convocatoria}.json".replace(" ", "_")
        ruta_salida = os.path.join("2_respuestas_json", nombre_archivo)

        with open(ruta_salida, "w", encoding="utf-8") as f:
            json.dump(respuestas, f, indent=2, ensure_ascii=False)

        print(f"    ✅ Respuestas guardadas: {nombre_archivo}")

    except Exception as e:
        print(f"    ❌ Error procesando {titulacion} {convocatoria}: {e}")

# Iniciar navegador
driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
wait = WebDriverWait(driver, 10)

try:
    driver.get(URL)
    select_titulacion = Select(wait.until(EC.presence_of_element_located((By.ID, "mainForm:titulacionSelect"))))
    titulaciones = [opt.text.strip() for opt in select_titulacion.options if opt.get_attribute("value")]

    for titulacion in titulaciones:
        print(f"\n🧪 Titulación: {titulacion}")

        # Recargar y volver a seleccionar titulación para convocatorias
        driver.get(URL)
        select_titulacion = Select(wait.until(EC.presence_of_element_located((By.ID, "mainForm:titulacionSelect"))))
        select_titulacion.select_by_visible_text(titulacion)
        time.sleep(1)

        select_convocatoria = Select(wait.until(EC.presence_of_element_located((By.ID, "mainForm:anyosSelect"))))
        convocatorias = [opt.text.strip() for opt in select_convocatoria.options if opt.get_attribute("value")]

        for convocatoria in convocatorias:
            print(f"  📅 Convocatoria: {convocatoria}")
            extraer_respuestas(titulacion, convocatoria, driver, wait)

finally:
    driver.quit()
    print("\n🏁 Finalizado.")
