import os
import requests
import time
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from webdriver_manager.chrome import ChromeDriverManager

def download_all_pdfs(destination_folder="mir_pdfs"):
    # Configuración navegador
    options = webdriver.ChromeOptions()
    options.add_argument('--no-sandbox')
    options.add_argument('--disable-dev-shm-usage')

    # Lanzar navegador
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)

    # URL de la página
    url = "https://mirial.es/examen-mir/24-examen-mir/174-descarga-todos-los-examen-mir-en-pdf"
    driver.get(url)
    time.sleep(5)  # esperar a que cargue

    # Crear carpeta
    os.makedirs(destination_folder, exist_ok=True)

    # Enlaces a PDFs
    links = driver.find_elements(By.TAG_NAME, "a")
    pdf_urls = [link.get_attribute("href") for link in links if link.get_attribute("href") and link.get_attribute("href").endswith(".pdf")]

    for pdf_url in pdf_urls:
        filename = pdf_url.split("/")[-1]
        filepath = os.path.join(destination_folder, filename)
        try:
            response = requests.get(pdf_url)
            with open(filepath, "wb") as f:
                f.write(response.content)
            print(f"Descargado: {filename}")
        except Exception as e:
            print(f"Error al descargar {pdf_url}: {e}")

    driver.quit()

download_all_pdfs()