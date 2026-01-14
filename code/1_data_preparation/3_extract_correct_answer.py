"""
Script Title: Automated Extraction of FSE Official Answer Sheets (Version 0)
Author: Lorena Ariceta Garcia
TFM: AI-Based Diagnosis: Ally or Risk?
     An Analysis of Language Models

Description
-----------
This script automates the extraction of official FSE answer sheets from the
public website of the Spanish Ministry of Health (MSCBS).

For each available medical specialization and examination year, the script
attempts to retrieve the answers associated with Version 0. The extracted
answers are parsed from the resulting HTML table and stored as JSON files.

A dedicated workflow is implemented for the 2024 examination year due to an
additional access pathway required to consult the official correct answers.

Output
------
The script creates the following output directory:

    results/1_data_preparation/2_respuestas_json/

Each JSON file is named as:

    <specialization>_<year>.json

and contains a dictionary mapping question numbers to the official correct
option number.

Requirements
------------
- Python 3.x
- selenium
- webdriver-manager
- beautifulsoup4 (bs4)
- Google Chrome browser

Methodological Notes
--------------------
- Selenium is used to automate form interactions and navigation.
- Explicit waits (WebDriverWait) are applied to ensure that dynamic elements
  are fully loaded before interaction.
- BeautifulSoup is used to parse the final HTML response table containing
  question numbers and correct answers.
"""

import os
import json
import time

from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait, Select
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager


# ---------------------------------------------------------------------
# 1. Output directory configuration
# ---------------------------------------------------------------------
OUTPUT_DIRECTORY = "results/1_data_preparation/2_respuestas_json"
os.makedirs(OUTPUT_DIRECTORY, exist_ok=True)


# ---------------------------------------------------------------------
# 2. Browser configuration
# ---------------------------------------------------------------------
chrome_options = webdriver.ChromeOptions()
chrome_options.add_argument("--disable-gpu")
chrome_options.add_argument("--no-sandbox")

# Base URL for previous examination materials
BASE_URL = (
    "https://fse.mscbs.gob.es/fseweb/view/public/datosanteriores/"
    "cuadernosExamen/busquedaConvocatoria.xhtml"
)


# ---------------------------------------------------------------------
# 3. Core extraction function
# ---------------------------------------------------------------------
def extract_answers(specialization: str, year: str, driver, wait) -> None:
    """
    Extract the official answer sheet for a given specialization and year,
    restricted to Version 0 when available.

    The extracted answers are saved as a JSON file in OUTPUT_DIRECTORY.
    """
    try:
        driver.get(BASE_URL)
        print("    Page loaded.")

        # Select specialization
        Select(
            wait.until(
                EC.presence_of_element_located((By.ID, "mainForm:titulacionSelect"))
            )
        ).select_by_visible_text(specialization)
        time.sleep(1)

        # Select year
        Select(
            wait.until(
                EC.presence_of_element_located((By.ID, "mainForm:anyosSelect"))
            )
        ).select_by_visible_text(year)
        time.sleep(1)

        # Select version (Version 0 only)
        version_selector = Select(
            wait.until(
                EC.presence_of_element_located((By.ID, "mainForm:versionSelect"))
            )
        )
        available_versions = [
            opt.text.strip()
            for opt in version_selector.options
            if opt.get_attribute("value")
        ]

        if "0" not in available_versions:
            print("    Version 0 not available. Skipping.")
            return

        version_selector.select_by_visible_text("0")
        time.sleep(1)

        # Select "Answer Sheet" radio option
        radio_buttons = driver.find_elements(
            By.XPATH,
            '//input[@type="radio" and contains(@id,"mainForm:j_idt87:")]'
        )

        for radio_button in radio_buttons:
            label = driver.find_element(
                By.CSS_SELECTOR,
                f'label[for="{radio_button.get_attribute("id")}"]'
            )
            if "Hoja de Respuestas" in label.text:
                radio_button.click()
                break
        else:
            print("    Answer sheet radio option not found.")
            return

        # Click "View"
        wait.until(EC.element_to_be_clickable((By.ID, "mainForm:j_idt91"))).click()
        time.sleep(2)

        # -----------------------------------------------------------------
        # Special workflow for 2024
        # -----------------------------------------------------------------
        if year == "2024":
            print("    Entering special workflow for 2024...")

            try:
                access_link = wait.until(
                    EC.element_to_be_clickable((By.ID, "mainForm:j_idt93"))
                )
                access_link.click()
                print("    Clicked link to consult current-year correct answers.")
            except Exception as exc:
                print("    Failed to open 2024 access link:", exc)
                return

            try:
                anonymous_access = wait.until(
                    EC.element_to_be_clickable((By.ID, "mainForm:accAbierto"))
                )
                anonymous_access.click()
                print("    Clicked 'Access without identification'.")
            except Exception as exc:
                print("    Failed to access without identification:", exc)
                return

            time.sleep(2)

            # Select specialization (inside the 2024 answers access page)
            try:
                specialization_selector = Select(
                    wait.until(
                        EC.presence_of_element_located((By.ID, "mainForm:titulaciones"))
                    )
                )
                specialization_selector.select_by_visible_text(specialization)
                print(f"    Specialization selected: {specialization}")
            except Exception as exc:
                print("    Failed to select specialization:", exc)
                return

            time.sleep(1)

            # Select version on the new page
            try:
                version_selector_2024 = Select(
                    wait.until(
                        EC.presence_of_element_located(
                            (By.XPATH, '//*[@id="mainForm:versiones"]/select')
                        )
                    )
                )
                versions_2024 = [
                    opt.text.strip()
                    for opt in version_selector_2024.options
                    if opt.get_attribute("value")
                ]

                if "0" in versions_2024:
                    version_selector_2024.select_by_visible_text("0")
                else:
                    # Fallback behavior kept consistent with the original script
                    version_selector_2024.select_by_visible_text("1")

                time.sleep(1)

            except Exception as exc:
                print("    Failed to select version (2024):", exc)
                return

            # Click search
            try:
                search_button = wait.until(
                    EC.element_to_be_clickable((By.XPATH, '//*[@id="mainForm:j_idt107"]'))
                )
                time.sleep(1)
                search_button.click()
                print("    Clicked 'Search'.")
            except Exception as exc:
                print("    Failed to click search (2024):", exc)
                return

            time.sleep(2)

        # -----------------------------------------------------------------
        # Parse answers table
        # -----------------------------------------------------------------
        soup = BeautifulSoup(driver.page_source, "html.parser")
        table = soup.find("table")

        if not table:
            print("    No answers table found.")
            return

        answers = {}
        cells = table.find_all("td")

        for i in range(0, len(cells), 2):
            try:
                question_number = int(cells[i].text.strip())
                correct_option = int(cells[i + 1].text.strip())
                answers[question_number] = correct_option
            except Exception:
                continue

        # Save JSON file
        filename = f"{specialization}_{year}.json".replace(" ", "_")
        output_path = os.path.join(OUTPUT_DIRECTORY, filename)

        with open(output_path, "w", encoding="utf-8") as file:
            json.dump(answers, file, indent=2, ensure_ascii=False)

        print(f"    Answers saved: {filename}")

    except Exception as exc:
        print(f"    Error while processing {specialization} {year}: {exc}")


# ---------------------------------------------------------------------
# 4. Main execution block
# ---------------------------------------------------------------------
driver = webdriver.Chrome(
    service=Service(ChromeDriverManager().install()),
    options=chrome_options
)
wait = WebDriverWait(driver, 10)

try:
    driver.get(BASE_URL)

    specialization_selector = Select(
        wait.until(
            EC.presence_of_element_located((By.ID, "mainForm:titulacionSelect"))
        )
    )
    specializations = [
        opt.text.strip()
        for opt in specialization_selector.options
        if opt.get_attribute("value")
    ]

    for specialization in specializations:
        print(f"\nProcessing specialization: {specialization}")

        driver.get(BASE_URL)
        Select(
            wait.until(
                EC.presence_of_element_located((By.ID, "mainForm:titulacionSelect"))
            )
        ).select_by_visible_text(specialization)
        time.sleep(1)

        year_selector = Select(
            wait.until(
                EC.presence_of_element_located((By.ID, "mainForm:anyosSelect"))
            )
        )
        years = [
            opt.text.strip()
            for opt in year_selector.options
            if opt.get_attribute("value")
        ]

        for year in years:
            print(f"  Processing examination year: {year}")
            extract_answers(specialization, year, driver, wait)

finally:
    driver.quit()
    print("\nProcess completed.")
