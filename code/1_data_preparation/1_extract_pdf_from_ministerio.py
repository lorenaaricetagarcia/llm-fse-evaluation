"""
Script Title: Automated Download of FSE Examination Booklets
Author: Lorena Ariceta Garcia
TFM: AI-Based Diagnosis: Ally or Risk?
     An Analysis of Language Models

Description
-----------
This script automates the retrieval and download of FSE examination booklets
from the public website of the Spanish Ministry of Health (MSCBS).
For each available medical specialization and examination year,
only the documents corresponding to Version 0 are downloaded.

Two types of examination booklets are considered:
    1) Text Booklet   (Text-based examination content)
    2) Image Booklet  (Image-based examination content)

The downloaded files are automatically organized into the following
directory structure:

    FSE_exams_v0/
        └── <specialization>/
            ├── text_booklet/
            └── image_booklet/

Requirements
------------
- Python 3.x
- selenium
- webdriver-manager
- Google Chrome browser

Methodological Notes
--------------------
- Selenium is used to automate browser interaction and form submission.
- Explicit waits (WebDriverWait) are applied to ensure page elements
  are fully loaded before interaction.
- Chrome DevTools Protocol (CDP) is used to dynamically modify
  the download directory during execution.
"""

import os
import time

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import Select, WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.chrome.service import Service


# ---------------------------------------------------------------------
# 1. Output directory configuration
# ---------------------------------------------------------------------
MAIN_OUTPUT_DIRECTORY = os.path.abspath("FSE_exams_v0")
os.makedirs(MAIN_OUTPUT_DIRECTORY, exist_ok=True)


# ---------------------------------------------------------------------
# 2. Browser configuration
# ---------------------------------------------------------------------
chrome_options = webdriver.ChromeOptions()

# Disable download confirmation dialogs and force PDF files to be downloaded
download_preferences = {
    "download.prompt_for_download": False,
    "plugins.always_open_pdf_externally": True,
}
chrome_options.add_experimental_option("prefs", download_preferences)

# Initialize Chrome WebDriver using webdriver-manager
driver = webdriver.Chrome(
    service=Service(ChromeDriverManager().install()),
    options=chrome_options
)

wait = WebDriverWait(driver, 10)


# ---------------------------------------------------------------------
# 3. Access target website
# ---------------------------------------------------------------------
TARGET_URL = "https://fse.sanidad.gob.es/fseweb/#/principal/datosAnteriores/cuadernosExamen"
driver.get(TARGET_URL)


# ---------------------------------------------------------------------
# 4. Retrieve available medical specializations
# ---------------------------------------------------------------------
specialization_selector = Select(
    wait.until(
        EC.presence_of_element_located((By.ID, "mainForm:titulacionSelect"))
    )
)

specializations = [
    option.get_attribute("value")
    for option in specialization_selector.options
    if option.get_attribute("value")
]


# Mapping between visible document labels and internal folder names
DOCUMENT_TYPE_MAPPING = {
    "Cuaderno de Texto": "text_booklet",
    "Cuaderno de Imágenes": "image_booklet",
}


# ---------------------------------------------------------------------
# 5. Iteration over specializations and examination years
# ---------------------------------------------------------------------
for specialization in specializations:
    print(f"Processing specialization: {specialization}")

    # Re-locate selector to avoid stale element references
    specialization_selector = Select(
        wait.until(
            EC.presence_of_element_located((By.ID, "mainForm:titulacionSelect"))
        )
    )
    specialization_selector.select_by_value(specialization)
    time.sleep(2)

    specialization_directory = os.path.join(
        MAIN_OUTPUT_DIRECTORY, specialization
    )
    os.makedirs(specialization_directory, exist_ok=True)

    # Retrieve available examination years
    year_selector = Select(
        wait.until(
            EC.presence_of_element_located((By.ID, "mainForm:anyosSelect"))
        )
    )

    examination_years = [
        option.text.strip()
        for option in year_selector.options
        if option.text.strip()
    ]

    for year in examination_years:
        print(f"  Processing examination year: {year}")

        year_selector = Select(
            wait.until(
                EC.presence_of_element_located((By.ID, "mainForm:anyosSelect"))
            )
        )
        year_selector.select_by_visible_text(year)
        time.sleep(1.5)

        try:
            # ---------------------------------------------------------
            # 5.1 Select Version 0 only
            # ---------------------------------------------------------
            version_selector = Select(
                wait.until(
                    EC.presence_of_element_located((By.ID, "mainForm:versionSelect"))
                )
            )

            version_zero_available = any(
                option.text.strip() == "0"
                for option in version_selector.options
            )

            if not version_zero_available:
                print("    Version 0 not available. Skipping.")
                continue

            version_selector.select_by_visible_text("0")
            time.sleep(1)

            # ---------------------------------------------------------
            # 5.2 Identify available document types
            # ---------------------------------------------------------
            radio_buttons = driver.find_elements(
                By.XPATH,
                '//input[@type="radio" and contains(@id,"mainForm:j_idt87:")]'
            )

            for radio_button in radio_buttons:
                radio_id = radio_button.get_attribute("id")
                label_element = driver.find_element(
                    By.CSS_SELECTOR, f'label[for="{radio_id}"]'
                )
                visible_label = label_element.text.strip()

                if visible_label in DOCUMENT_TYPE_MAPPING:
                    folder_name = DOCUMENT_TYPE_MAPPING[visible_label]
                    print(f"    Downloading document type: {folder_name}")

                    target_directory = os.path.join(
                        specialization_directory, folder_name
                    )
                    os.makedirs(target_directory, exist_ok=True)

                    # Dynamically set Chrome download directory via CDP
                    driver.execute_cdp_cmd(
                        "Page.setDownloadBehavior",
                        {
                            "behavior": "allow",
                            "downloadPath": target_directory,
                        },
                    )

                    # Select document type and trigger download
                    radio_button.click()
                    time.sleep(0.5)

                    view_button = driver.find_element(
                        By.ID, "mainForm:j_idt91"
                    )
                    view_button.click()
                    time.sleep(3)

        except Exception as exception:
            print(
                f"    Error while processing {specialization} "
                f"- {year} (Version 0): {exception}"
            )


# ---------------------------------------------------------------------
# 6. Cleanup and termination
# ---------------------------------------------------------------------
driver.quit()
print("\nProcess completed successfully.")
