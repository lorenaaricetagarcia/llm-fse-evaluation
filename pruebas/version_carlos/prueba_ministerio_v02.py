from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import Select, WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
import time
import os

# Crear carpeta para guardar los PDFs
output_dir = os.path.abspath("examenes_mir")
os.makedirs(output_dir, exist_ok=True)

# Configuración del navegador
options = webdriver.ChromeOptions()
options.add_experimental_option("prefs", {
    "download.default_directory": output_dir,    # Carpeta de descarga
    "download.prompt_for_download": False,       # Sin preguntar
    "plugins.always_open_pdf_externally": True   # Descargar PDF en vez de abrirlo
})
from selenium.webdriver.chrome.service import Service
driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)

# Abrir la página
driver.get("https://fse.mscbs.gob.es/fseweb/view/public/datosanteriores/cuadernosExamen/busquedaConvocatoria.xhtml")

# Esperar que cargue el selector
wait = WebDriverWait(driver, 10)

titulaciones = ["MEDICINA", "FARMACIA"]

for titulacion in titulaciones:
    print(f"Procesando titulación: {titulacion}")
    # Seleccionar titulación: Medicina
    select_titulacion = Select(wait.until(EC.presence_of_element_located((By.ID, "mainForm:titulacionSelect"))))
    select_titulacion.select_by_value(titulacion)
    time.sleep(2)  # esperar que se carguen convocatorias

    # Obtener todas las convocatorias disponibles
    select_convocatoria = Select(driver.find_element(By.ID, "mainForm:anyosSelect"))
    convocatorias = [option.text for option in select_convocatoria.options if option.text.strip() != '']

    for convocatoria in convocatorias:
        print(f"Procesando convocatoria: {convocatoria}")
        select_convocatoria.select_by_visible_text(convocatoria)
        time.sleep(1.5)
        
        select_version = Select(driver.find_element(By.ID, "mainForm:versionSelect"))
        versiones = [option.text for option in select_version.options if option.text.strip() != '']

        for version in versiones:
            print(f"Procesando version: {version}")
            select_version.select_by_visible_text(version)
            time.sleep(1.5)
                            
            label = driver.find_element(By.ID, "mainForm:j_idt87:0")
            label.click()
            time.sleep(1.5)
            
            # Click en el botón "Ver"
            ver_btn = driver.find_element(By.ID, "mainForm:j_idt91")
            ver_btn.click()
            time.sleep(1.5) 
            
            if titulacion == "MEDICINA":
                label = driver.find_element(By.ID, "mainForm:j_idt87:1")
                label.click()
                time.sleep(1.5)
                ver_btn.click()

        time.sleep(2)

driver.quit()