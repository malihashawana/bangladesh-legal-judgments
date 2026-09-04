import os
import time
import requests
import pandas as pd
from bs4 import BeautifulSoup
from urllib.parse import urljoin

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException

from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.chrome.service import Service

BASE_URL = "https://supremecourt.gov.bd/web/judgments.php"

PDF_DIR = "data/pdfs"
METADATA_CSV = "data/metadata.csv"

MAX_PAGES = 5

os.makedirs(PDF_DIR, exist_ok=True)


def make_driver():
    options = Options()
    # comment this while debugging if you want to see the browser
    options.add_argument("--headless=new")

    options.add_argument("--disable-gpu")
    options.add_argument("--window-size=1920,1080")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")

    service = Service(ChromeDriverManager().install())

    driver = webdriver.Chrome(service=service, options=options)

    return driver


def scrape_current_page(driver, seen, rows, counter):

    soup = BeautifulSoup(driver.page_source, "html.parser")

    links = soup.find_all("a")

    print(f"Found {len(links)} links")

    for a in links:

        href = a.get("href")
        text = a.get_text(strip=True)

        if not href:
            continue

        if "/resources/documents/" not in href:
            continue

        if not href.lower().endswith(".pdf"):
            continue

        pdf_url = urljoin("https://supremecourt.gov.bd/web/", href)

        if pdf_url in seen:
            continue

        seen.add(pdf_url)

        filename = f"J{counter:05}.pdf"

        filepath = os.path.join(PDF_DIR, filename)

        print(f"[{counter}] {text}")

        try:

            r = requests.get(pdf_url, timeout=60)

            r.raise_for_status()

            with open(filepath, "wb") as f:
                f.write(r.content)

            rows.append({
                "id": f"J{counter:05}",
                "case_name": text,
                "pdf_url": pdf_url,
                "saved_file": filename,
                "original_filename": pdf_url.split("/")[-1]
            })

            counter += 1

            time.sleep(0.5)

        except Exception as e:

            print("Download failed:", e)

    return counter


def next_page(driver):

    try:

        next_btn = driver.find_element(By.PARTIAL_LINK_TEXT, "Next")

        driver.execute_script("arguments[0].scrollIntoView();", next_btn)

        old = driver.page_source

        next_btn.click()

        WebDriverWait(driver, 10).until(
            lambda d: d.page_source != old
        )

        time.sleep(2)

        return True

    except Exception:

        return False


def main():

    driver = make_driver()

    driver.get(BASE_URL)

    time.sleep(3)

    rows = []

    seen = set()

    counter = 1

    page = 1

    while page <= MAX_PAGES:

        print("=" * 60)
        print("PAGE", page)
        print("=" * 60)

        counter = scrape_current_page(driver, seen, rows, counter)

        if not next_page(driver):
            print("No more pages.")
            break

        page += 1

    driver.quit()

    df = pd.DataFrame(rows)

    df.to_csv(METADATA_CSV, index=False, encoding="utf-8-sig")

    print(df.head())

    print()

    print("Downloaded:", len(df))


if __name__ == "__main__":
    main()