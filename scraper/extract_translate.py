import os
import fitz
from tqdm import tqdm
from deep_translator import GoogleTranslator

PDF_FOLDER = "data/pdfs"
EN_FOLDER = "data/texts_en"
BN_FOLDER = "data/texts_bn"

os.makedirs(EN_FOLDER, exist_ok=True)
os.makedirs(BN_FOLDER, exist_ok=True)


def extract_text(pdf_path):
    doc = fitz.open(pdf_path)

    text = ""

    for page in doc:
        text += page.get_text()

    doc.close()

    return text


translator = GoogleTranslator(source="en", target="bn")

for pdf in tqdm(os.listdir(PDF_FOLDER)):

    if not pdf.endswith(".pdf"):
        continue

    pdf_path = os.path.join(PDF_FOLDER, pdf)

    english = extract_text(pdf_path)

    if len(english.strip()) < 100:
        print(pdf, "No readable text.")
        continue

    with open(
        os.path.join(
            EN_FOLDER,
            pdf.replace(".pdf", ".txt")
        ),
        "w",
        encoding="utf-8",
    ) as f:

        f.write(english)

    chunks = []

    STEP = 4000

    for i in range(0, len(english), STEP):

        part = english[i:i + STEP]

        try:
            bn = translator.translate(part)
        except Exception:
            bn = ""

        chunks.append(bn)

    bangla = "\n".join(chunks)

    with open(
        os.path.join(
            BN_FOLDER,
            pdf.replace(".pdf", ".txt")
        ),
        "w",
        encoding="utf-8",
    ) as f:

        f.write(bangla)

print("Finished.")