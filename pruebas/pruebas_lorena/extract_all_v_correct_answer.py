from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait, Select
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from bs4 import BeautifulSoup   # Parsear HTML
import os
import json
import time

# Carpeta de salida
os.makedirs("2_respuestas_json", exist_ok=True)

# Configuración del navegador
options = webdriver.ChromeOptions()
options.add_argument("--headless=new")
options.add_argument("--disable-gpu")
options.add_argument("--no-sandbox")

# Cargar página base
URL = "https://fse.mscbs.gob.es/fseweb/view/public/datosanteriores/cuadernosExamen/busquedaConvocatoria.xhtml"

def extraer_respuestas(titulacion, convocatoria, version, driver, wait):
    try:
        # Reabrir la página cada vez para evitar errores de estado
        driver.get(URL)

        # Seleccionar titulación
        select_titulacion = Select(wait.until(EC.presence_of_element_located((By.ID, "mainForm:titulacionSelect"))))
        select_titulacion.select_by_visible_text(titulacion)
        time.sleep(1)

        # Seleccionar convocatoria
        select_convocatoria = Select(wait.until(EC.presence_of_element_located((By.ID, "mainForm:anyosSelect"))))
        select_convocatoria.select_by_visible_text(convocatoria)
        time.sleep(1)

        # Seleccionar versión
        select_version = Select(wait.until(EC.presence_of_element_located((By.ID, "mainForm:versionSelect"))))
        select_version.select_by_visible_text(version)
        time.sleep(1)

        # Seleccionar "Hoja de Respuestas"
        radios = driver.find_elements(By.XPATH, '//input[@type="radio" and contains(@id,"mainForm:j_idt87:")]')
        for radio in radios:
            label = driver.find_element(By.CSS_SELECTOR, f'label[for="{radio.get_attribute("id")}"]')
            if "Hoja de Respuestas" in label.text:
                radio.click()
                break
        else:
            print("      ⚠️ No se encontró radio de Hoja de Respuestas")
            return

        # Clic en "Ver"
        ver_btn = wait.until(EC.element_to_be_clickable((By.ID, "mainForm:j_idt91")))
        ver_btn.click()
        time.sleep(2)

        # Extraer tabla
        soup = BeautifulSoup(driver.page_source, "html.parser") # Usa BeautifulSoup para buscar elementos
        tabla = soup.find("table")

        if not tabla:
            print("      ⚠️ No se encontró tabla de respuestas.")
            return

        respuestas = {} # Inicializa diccionario respuestas
        celdas = tabla.find_all("td")   # Obtiene todas las celdas de la tabla

        for i in range(0, len(celdas), 2):  # Recorre las celdas en pares (número de pregunta y opción correcta)
            try:
                num = int(celdas[i].text.strip())
                val = int(celdas[i + 1].text.strip())
                respuestas[num] = val   # Guarda la relación pregunta respuesta
            except:
                continue

        nombre_archivo = f"{titulacion}_{convocatoria}_v{version}".replace(" ", "_") + ".json"
        with open(os.path.join("2_respuestas_json", nombre_archivo), "w", encoding="utf-8") as f:
            json.dump(respuestas, f, indent=2, ensure_ascii=False)

        print(f"      ✅ Respuestas guardadas: {nombre_archivo}")

    except Exception as e:
        print(f"      ❌ Error procesando versión {version}: {e}")

# Lanzar navegador
driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
wait = WebDriverWait(driver, 10)

try:
    driver.get(URL)
    select_titulacion = Select(wait.until(EC.presence_of_element_located((By.ID, "mainForm:titulacionSelect"))))
    titulaciones = [opt.text.strip() for opt in select_titulacion.options if opt.get_attribute("value")]

    for titulacion in titulaciones:
        print(f"\n🧪 Titulación: {titulacion}")

        # Cargar página para obtener convocatorias
        driver.get(URL)
        select_titulacion = Select(wait.until(EC.presence_of_element_located((By.ID, "mainForm:titulacionSelect"))))
        select_titulacion.select_by_visible_text(titulacion)
        time.sleep(1)

        select_convocatoria = Select(wait.until(EC.presence_of_element_located((By.ID, "mainForm:anyosSelect"))))
        convocatorias = [opt.text.strip() for opt in select_convocatoria.options if opt.get_attribute("value")]

        for convocatoria in convocatorias:
            print(f"  📅 Convocatoria: {convocatoria}")

            driver.get(URL)
            select_titulacion = Select(wait.until(EC.presence_of_element_located((By.ID, "mainForm:titulacionSelect"))))
            select_titulacion.select_by_visible_text(titulacion)
            time.sleep(1)

            select_convocatoria = Select(wait.until(EC.presence_of_element_located((By.ID, "mainForm:anyosSelect"))))
            select_convocatoria.select_by_visible_text(convocatoria)
            time.sleep(1)

            try:
                select_version = Select(wait.until(EC.presence_of_element_located((By.ID, "mainForm:versionSelect"))))
                versiones = [opt.text.strip() for opt in select_version.options if opt.get_attribute("value")]
            except:
                print("    ⚠️ Sin versiones disponibles.")
                continue

            for version in versiones:
                print(f"    🔢 Versión: {version}")
                extraer_respuestas(titulacion, convocatoria, version, driver, wait)

finally:
    driver.quit()
    print("\n🏁 Finalizado.")