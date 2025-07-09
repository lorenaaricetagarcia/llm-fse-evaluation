from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import Select, WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
import time
import os

# Configuración del navegador
options = webdriver.ChromeOptions()
from selenium.webdriver.chrome.service import Service
driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)

# Crear carpeta para guardar los PDFs
output_dir = "examenes_mir"
os.makedirs(output_dir, exist_ok=True)

# Abrir la página
driver.get("https://fse.mscbs.gob.es/fseweb/view/public/datosanteriores/cuadernosExamen/busquedaConvocatoria.xhtml")

# Esperar que cargue el selector
wait = WebDriverWait(driver, 10)

# Seleccionar titulación: Medicina
select_titulacion = Select(wait.until(EC.presence_of_element_located((By.ID, "formCuadernos:titulos"))))
select_titulacion.select_by_visible_text("Medicina")
time.sleep(2)  # esperar que se carguen convocatorias

# Obtener todas las convocatorias disponibles
select_convocatoria = Select(driver.find_element(By.ID, "formCuadernos:convocatorias"))
convocatorias = [option.text for option in select_convocatoria.options if option.text.strip() != '']

for convocatoria in convocatorias:
    print(f"Procesando convocatoria: {convocatoria}")
    select_convocatoria.select_by_visible_text(convocatoria)
    time.sleep(1.5)

    # Click en el botón "Ver"
    ver_btn = driver.find_element(By.ID, "formCuadernos:botonBuscar")
    ver_btn.click()

    # Esperar resultados
    wait.until(EC.presence_of_element_located((By.CLASS_NAME, "ui-datatable-data")))

    # Extraer los enlaces de descarga
    enlaces = driver.find_elements(By.XPATH, "//a[contains(@href, '.pdf')]")
    for enlace in enlaces:
        href = enlace.get_attribute("href")
        nombre = href.split("/")[-1]
        ruta = os.path.join(output_dir, f"{convocatoria}_{nombre}")
        try:
            import requests
            r = requests.get(href)
            with open(ruta, "wb") as f:
                f.write(r.content)
            print(f"Descargado: {ruta}")
        except Exception as e:
            print(f"Error al descargar {href}: {e}")

    # Volver atrás
    driver.back()
    time.sleep(2)

driver.quit()