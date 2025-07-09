from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import Select, WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.chrome.service import Service
import time
import os

# Crear carpeta raíz
main_output_dir = os.path.abspath("examenes_mir")
os.makedirs(main_output_dir, exist_ok=True)

# Configurar navegador
options = webdriver.ChromeOptions()
prefs = {
    "download.prompt_for_download": False,
    "plugins.always_open_pdf_externally": True
}
options.add_experimental_option("prefs", prefs)

# Iniciar navegador
driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
wait = WebDriverWait(driver, 10)

# Abrir la página
driver.get("https://fse.mscbs.gob.es/fseweb/view/public/datosanteriores/cuadernosExamen/busquedaConvocatoria.xhtml")

# Obtener todas las titulaciones
select_titulacion = Select(wait.until(EC.presence_of_element_located((By.ID, "mainForm:titulacionSelect"))))
titulaciones = [option.get_attribute("value") for option in select_titulacion.options if option.get_attribute("value")] # Automaticamente detecta todas las titulaciones

# Mapeo del texto visible en la web a nombres de carpeta
tipo_mapeo = { # Identifica el tipo de PDF por su etiqueta (texto o imágenes) y descarga ambos si están disponibles
    "Cuaderno de Texto": "cuaderno_texto",
    "Cuaderno de Imágenes": "cuaderno_imagenes"
}

for titulacion in titulaciones:
    print(f"Procesando titulación: {titulacion}")

    # Re-seleccionar titulación
    select_titulacion = Select(wait.until(EC.presence_of_element_located((By.ID, "mainForm:titulacionSelect")))) # Vuelve a capturar los elementos select después de cada cambio
    select_titulacion.select_by_value(titulacion)
    time.sleep(2)

    titulacion_dir = os.path.join(main_output_dir, titulacion)
    os.makedirs(titulacion_dir, exist_ok=True)

    # Obtener convocatorias
    select_convocatoria = Select(wait.until(EC.presence_of_element_located((By.ID, "mainForm:anyosSelect"))))
    convocatorias = [option.text.strip() for option in select_convocatoria.options if option.text.strip()]

    for convocatoria in convocatorias:
        print(f"Convocatoria: {convocatoria}")
        select_convocatoria = Select(wait.until(EC.presence_of_element_located((By.ID, "mainForm:anyosSelect"))))
        select_convocatoria.select_by_visible_text(convocatoria)
        time.sleep(1.5)

        # Obtener versiones
        select_version_element = wait.until(EC.presence_of_element_located((By.ID, "mainForm:versionSelect")))
        versiones = [option.text.strip() for option in Select(select_version_element).options if option.text.strip()]

        for version in versiones:
            print(f"Versión: {version}")

            select_version_element = wait.until(EC.presence_of_element_located((By.ID, "mainForm:versionSelect")))
            Select(select_version_element).select_by_visible_text(version)
            time.sleep(1)

            # Detectar radio buttons disponibles
            radios = driver.find_elements(By.XPATH, '//input[@type="radio" and contains(@id,"mainForm:j_idt87:")]')

            for radio in radios:
                radio_id = radio.get_attribute("id")
                label = driver.find_element(By.CSS_SELECTOR, f'label[for="{radio_id}"]')
                tipo_visible = label.text.strip()

                # Solo descargar si es texto o imágenes
                if tipo_visible in tipo_mapeo:
                    nombre_carpeta = tipo_mapeo[tipo_visible]
                    print(f"Descargando tipo: {nombre_carpeta}")

                    subcarpeta = os.path.join(titulacion_dir, nombre_carpeta) # Usa descargas por subcarpetas específicas (cuaderno_texto, cuaderno_imagenes) para mantener organizados los archivos por tipo y titulación
                    os.makedirs(subcarpeta, exist_ok=True)

                    # Cambiar carpeta de descarga
                    driver.execute_cdp_cmd("Page.setDownloadBehavior", { # Cambia dinámicamente la carpeta de descarga durante la ejecución
                        "behavior": "allow",
                        "downloadPath": subcarpeta
                    })

                    radio.click()
                    time.sleep(0.5)

                    ver_btn = driver.find_element(By.ID, "mainForm:j_idt91")
                    ver_btn.click()
                    time.sleep(3)

driver.quit()
print("Finalizado correctamente.")
