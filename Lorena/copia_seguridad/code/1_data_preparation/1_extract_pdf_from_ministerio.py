import os
import time
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import Select, WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.chrome.service import Service

# 📁 Crear carpeta raíz
main_output_dir = os.path.abspath("examenes_mir_v_0")
os.makedirs(main_output_dir, exist_ok=True)

# 🌐 Configuración del navegador
options = webdriver.ChromeOptions()
prefs = {
    "download.prompt_for_download": False,
    "plugins.always_open_pdf_externally": True
}
options.add_experimental_option("prefs", prefs)

# 🚀 Iniciar navegador
driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
wait = WebDriverWait(driver, 10)

# Abrir página inicial
driver.get("https://fse.mscbs.gob.es/fseweb/view/public/datosanteriores/cuadernosExamen/busquedaConvocatoria.xhtml")

# Obtener titulaciones disponibles
select_titulacion = Select(wait.until(EC.presence_of_element_located((By.ID, "mainForm:titulacionSelect"))))
titulaciones = [option.get_attribute("value") for option in select_titulacion.options if option.get_attribute("value")]

# Etiquetas de tipos de PDF
tipo_mapeo = {
    "Cuaderno de Texto": "cuaderno_texto",
    "Cuaderno de Imágenes": "cuaderno_imagenes"
}

# Iterar por titulaciones
for titulacion in titulaciones:
    print(f"📘 Procesando titulación: {titulacion}")

    # Volver a capturar el select por seguridad
    select_titulacion = Select(wait.until(EC.presence_of_element_located((By.ID, "mainForm:titulacionSelect"))))
    select_titulacion.select_by_value(titulacion)
    time.sleep(2)

    titulacion_dir = os.path.join(main_output_dir, titulacion)
    os.makedirs(titulacion_dir, exist_ok=True)

    # Obtener convocatorias
    select_convocatoria = Select(wait.until(EC.presence_of_element_located((By.ID, "mainForm:anyosSelect"))))
    convocatorias = [option.text.strip() for option in select_convocatoria.options if option.text.strip()]

    for convocatoria in convocatorias:
        print(f"  📅 Convocatoria: {convocatoria}")
        select_convocatoria = Select(wait.until(EC.presence_of_element_located((By.ID, "mainForm:anyosSelect"))))
        select_convocatoria.select_by_visible_text(convocatoria)
        time.sleep(1.5)

        try:
            # Intentar seleccionar solo la versión 0
            select_version = Select(wait.until(EC.presence_of_element_located((By.ID, "mainForm:versionSelect"))))
            version_0_found = any(opt.text.strip() == "0" for opt in select_version.options)

            if not version_0_found:
                print("    ⚠️ No hay versión 0, se omite.")
                continue

            select_version.select_by_visible_text("0")
            time.sleep(1)

            # Buscar tipo de PDF (texto o imagen)
            radios = driver.find_elements(By.XPATH, '//input[@type="radio" and contains(@id,"mainForm:j_idt87:")]')

            for radio in radios:
                radio_id = radio.get_attribute("id")
                label = driver.find_element(By.CSS_SELECTOR, f'label[for="{radio_id}"]')
                tipo_visible = label.text.strip()

                if tipo_visible in tipo_mapeo:
                    nombre_carpeta = tipo_mapeo[tipo_visible]
                    print(f"    📥 Descargando tipo: {nombre_carpeta}")

                    subcarpeta = os.path.join(titulacion_dir, nombre_carpeta)
                    os.makedirs(subcarpeta, exist_ok=True)

                    # Cambiar carpeta de descarga dinámica
                    driver.execute_cdp_cmd("Page.setDownloadBehavior", {
                        "behavior": "allow",
                        "downloadPath": subcarpeta
                    })

                    radio.click()
                    time.sleep(0.5)

                    # Clic en "Ver"
                    ver_btn = driver.find_element(By.ID, "mainForm:j_idt91")
                    ver_btn.click()
                    time.sleep(3)

        except Exception as e:
            print(f"    ❌ Error con {titulacion} - {convocatoria} v0: {e}")

driver.quit()
print("\n✅ Finalizado correctamente.")
