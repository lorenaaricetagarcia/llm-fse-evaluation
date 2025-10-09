import os
import json
import time
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait, Select
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from bs4 import BeautifulSoup

# Carpeta de salida
os.makedirs("results/1_data_preparation/2_respuestas_json", exist_ok=True)

# Configuración del navegador (headless para evitar abrir ventanas)
options = webdriver.ChromeOptions()
options.add_argument("--disable-gpu")
options.add_argument("--no-sandbox")

# URL base
URL_ANTERIORES = "https://fse.mscbs.gob.es/fseweb/view/public/datosanteriores/cuadernosExamen/busquedaConvocatoria.xhtml"

def extraer_respuestas(titulacion, convocatoria, driver, wait):
    try:
        if convocatoria != "2024":
            driver.get(URL_ANTERIORES)
            print("    🧭 Página cargada.")

            Select(wait.until(EC.presence_of_element_located((By.ID, "mainForm:titulacionSelect")))).select_by_visible_text(titulacion)
            time.sleep(1)

            Select(wait.until(EC.presence_of_element_located((By.ID, "mainForm:anyosSelect")))).select_by_visible_text(convocatoria)
            time.sleep(1)

            select_version = Select(wait.until(EC.presence_of_element_located((By.ID, "mainForm:versionSelect"))))
            versiones = [opt.text.strip() for opt in select_version.options if opt.get_attribute("value")]
            if "0" not in versiones:
                print("    ⚠️ No hay versión 0, se omite.")
                return
            select_version.select_by_visible_text("0")
            time.sleep(1)

            radios = driver.find_elements(By.XPATH, '//input[@type="radio" and contains(@id,"mainForm:j_idt87:")]')
            for radio in radios:
                label = driver.find_element(By.CSS_SELECTOR, f'label[for="{radio.get_attribute("id")}"]')
                if "Hoja de Respuestas" in label.text:
                    radio.click()
                    break
            else:
                print("    ⚠️ No se encontró radio de Hoja de Respuestas.")
                return

            wait.until(EC.element_to_be_clickable((By.ID, "mainForm:j_idt91"))).click()
            time.sleep(2)

        else:

            print("    ↪ Entrando en modo especial para 2024...")

            driver.get(URL_ANTERIORES)
            print("    🧭 Página cargada.")

            Select(wait.until(EC.presence_of_element_located((By.ID, "mainForm:titulacionSelect")))).select_by_visible_text(titulacion)
            time.sleep(1)

            Select(wait.until(EC.presence_of_element_located((By.ID, "mainForm:anyosSelect")))).select_by_visible_text(convocatoria)
            time.sleep(1)

            select_version = Select(wait.until(EC.presence_of_element_located((By.ID, "mainForm:versionSelect"))))
            versiones = [opt.text.strip() for opt in select_version.options if opt.get_attribute("value")]
            if "0" not in versiones:
                print("    ⚠️ No hay versión 0, se omite.")
                return
            select_version.select_by_visible_text("0")
            time.sleep(1)

            radios = driver.find_elements(By.XPATH, '//input[@type="radio" and contains(@id,"mainForm:j_idt87:")]')
            for radio in radios:
                label = driver.find_element(By.CSS_SELECTOR, f'label[for="{radio.get_attribute("id")}"]')
                if "Hoja de Respuestas" in label.text:
                    radio.click()
                    break
            else:
                print("    ⚠️ No se encontró radio de Hoja de Respuestas.")
                return

            wait.until(EC.element_to_be_clickable((By.ID, "mainForm:j_idt91"))).click()
            time.sleep(2)

            try:
                acceso = wait.until(EC.element_to_be_clickable((By.ID, "mainForm:j_idt93")))
                acceso.click()
                print("    🔓 Click en 'Para consultar las respuestas correctas de la convocatoria actual, pulse aqui.'")
            except Exception as e:
                print("    ❌ Fallo en 'Para consultar las respuestas correctas de la convocatoria actual, pulse aqui.'", e)
                return

            try:
                acceso = wait.until(EC.element_to_be_clickable((By.ID, "mainForm:accAbierto")))
                acceso.click()
                print("    🔓 Click en 'Acceso sin identificación'")
            except Exception as e:
                print("    ❌ Fallo en acceso sin identificación:", e)
                return


            time.sleep(2)

            try:
                sel_titulacion = Select(wait.until(EC.presence_of_element_located((By.ID, "mainForm:titulaciones"))))
                sel_titulacion.select_by_visible_text(titulacion)
                print(f"    ✅ Titulación seleccionada: {titulacion}")
            except Exception as e:
                print("    ❌ Fallo al seleccionar titulación:", e)
                return

            time.sleep(1)

            try:
                # Esperar y seleccionar el desplegable por name
                sel_version = Select(wait.until(EC.presence_of_element_located(
                    (By.XPATH, '//*[@id="mainForm:versiones"]/select')
                )))

                versiones = [opt.text.strip() for opt in sel_version.options if opt.get_attribute("value")]
                if "0" not in versiones:
                    sel_version.select_by_visible_text("1")
                    time.sleep(1)
                else:
                    sel_version.select_by_visible_text("0")
                    time.sleep(1)

            except Exception as e:
                print("    ❌ Fallo al seleccionar versión:", e)
                return

            time.sleep(1)

            try:
                buscar = wait.until(EC.element_to_be_clickable((By.XPATH, '//*[@id="mainForm:j_idt107"]')))


                time.sleep(1)

                buscar.click()
                print("    🔍 Click en botón Buscar.")
            except Exception as e:
                print("    ❌ Fallo al hacer click en buscar:", e)
                return

            time.sleep(2)


        # Extraer respuestas
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
        ruta_salida = os.path.join("results/1_data_preparation/2_respuestas_json", nombre_archivo)

        with open(ruta_salida, "w", encoding="utf-8") as f:
            json.dump(respuestas, f, indent=2, ensure_ascii=False)

        print(f"    ✅ Respuestas guardadas: {nombre_archivo}")

    except Exception as e:
        print(f"    ❌ Error procesando {titulacion} {convocatoria}: {e}")

# Inicialización de navegador
driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
wait = WebDriverWait(driver, 10)

try:
    driver.get(URL_ANTERIORES)
    select_titulacion = Select(wait.until(EC.presence_of_element_located((By.ID, "mainForm:titulacionSelect"))))
    titulaciones = [opt.text.strip() for opt in select_titulacion.options if opt.get_attribute("value")]

    for titulacion in titulaciones:
        print(f"\n🧪 Titulación: {titulacion}")

        driver.get(URL_ANTERIORES)
        Select(wait.until(EC.presence_of_element_located((By.ID, "mainForm:titulacionSelect")))).select_by_visible_text(titulacion)
        time.sleep(1)

        select_convocatoria = Select(wait.until(EC.presence_of_element_located((By.ID, "mainForm:anyosSelect"))))
        convocatorias = [opt.text.strip() for opt in select_convocatoria.options if opt.get_attribute("value")]

        for convocatoria in convocatorias:
            print(f"  📅 Convocatoria: {convocatoria}")
            extraer_respuestas(titulacion, convocatoria, driver, wait)

finally:
    driver.quit()
    print("\n🏁 Finalizado.")