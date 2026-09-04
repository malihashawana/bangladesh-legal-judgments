import os
import re
import requests
import pandas as pd
from bs4 import BeautifulSoup
from urllib.parse import urljoin

BASE = "https://supremecourt.gov.bd/web/judgments.php"

html = requests.get(BASE).text
soup = BeautifulSoup(html, "html.parser")

os.makedirs("data/pdfs", exist_ok=True)

rows = []

count = 1

for a in soup.find_all("a"):

    href = a.get("href")
    text = a.get_text(strip=True)

    if (
    href
    and "/resources/documents/" in href
    and href.lower().endswith(".pdf")
):

        pdf_url = urljoin(BASE, href)

        filename = f"J{count:05}.pdf"

        print(f"Downloading {filename}")

        pdf = requests.get(pdf_url)

        with open(f"data/pdfs/{filename}", "wb") as f:
            f.write(pdf.content)

        rows.append({
    "id": f"J{count:05}",
    "case_name": text,
    "original_filename": pdf_url.split("/")[-1],
    "pdf_url": pdf_url,
    "saved_file": filename
})

        count += 1

df = pd.DataFrame(rows)

df.to_csv(
    "data/metadata.csv",
    index=False,
    encoding="utf-8-sig"
)

print(df.head())

print(f"\nDownloaded {len(df)} judgments.")