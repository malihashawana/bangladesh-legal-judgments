"""
STEP: Classify each judgment as either:
    - "full_judgment"  -> has a real Facts/Arguments/Order narrative,
                           suitable for hierarchical summarization
    - "short_order"    -> a brief procedural ruling with no real
                           "story" to summarize section-by-section

Why we need this:
    We found that some judgments (e.g. short Company Matter orders)
    have NO facts section not because our pattern-matching failed,
    but because that kind of document genuinely doesn't narrate facts.
    Your framework's hierarchical summarization should be evaluated
    on FULL judgments. Short orders are a different kind of document
    and shouldn't be forced into the same structure.

How classification works:
    1. Word count of the cleaned English text (main signal)
    2. Case type extracted from the case name in metadata.csv
       (e.g. "Company Matter 341/2026" -> case type = "Company Matter")
    3. Cross-checked against whether section_split.py found a
       "facts" section for that judgment

    A judgment is classified as "short_order" if its word count is
    below WORD_COUNT_THRESHOLD. Everything else is "full_judgment".
    The case-type and facts-found columns are shown alongside so you
    can visually confirm the classification makes sense, not just
    trust the number blindly.

Input:
    data/texts_en_clean/*.txt
    data/structured_judgments.json   (from section_split.py)
    data/metadata.csv                (from your scraper)

Output:
    data/document_classification.json
    -> a summary table printed to the terminal, grouped by case type

Run:
    python processing/classify_documents.py
"""

import os
import re
import json
import csv

TEXTS_DIR = "data/texts_en_clean"
STRUCTURED_PATH = "data/structured_judgments.json"
METADATA_PATH = "data/metadata.csv"
OUTPUT_PATH = "data/document_classification.json"

# Judgments with fewer words than this are treated as short
# procedural orders rather than full narrative judgments.
# This is a starting point -- adjust after reviewing the printed
# table if the cutoff doesn't match what you see.
WORD_COUNT_THRESHOLD = 300


def extract_case_type(case_name):
    """
    Turns "Company Matter 341/2026" into "Company Matter",
    "Writ Petition 5568/2019(...)" into "Writ Petition", etc.
    Strips everything from the first digit onward.
    """
    if not case_name:
        return "Unknown"
    match = re.match(r"^([A-Za-z\.\(\)\s]+?)\s*\d", case_name.strip())
    if match:
        return match.group(1).strip()
    return case_name.strip()


def load_metadata():
    """Returns a dict: {id -> case_type}"""
    case_types = {}
    if not os.path.isfile(METADATA_PATH):
        print(f"Warning: {METADATA_PATH} not found, case types will be 'Unknown'")
        return case_types

    with open(METADATA_PATH, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            judgment_id = row.get("id", "").strip()
            case_name = row.get("case_name", "").strip()
            case_types[judgment_id] = extract_case_type(case_name)
    return case_types


def load_structured():
    """Returns a dict: {id -> sections_found list}"""
    if not os.path.isfile(STRUCTURED_PATH):
        print(f"Warning: {STRUCTURED_PATH} not found, run section_split.py first")
        return {}

    with open(STRUCTURED_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    return {entry["id"]: entry["sections_found"] for entry in data}


def word_count(text):
    return len(text.split())


def main():
    if not os.path.isdir(TEXTS_DIR):
        print(f"Could not find folder: {TEXTS_DIR}")
        return

    case_types = load_metadata()
    sections_found_map = load_structured()

    files = sorted(f for f in os.listdir(TEXTS_DIR) if f.lower().endswith(".txt"))
    print(f"Found {len(files)} files in {TEXTS_DIR}\n")

    results = []
    for fname in files:
        judgment_id = os.path.splitext(fname)[0]
        path = os.path.join(TEXTS_DIR, fname)
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()

        wc = word_count(text)
        doc_type = "short_order" if wc < WORD_COUNT_THRESHOLD else "full_judgment"
        case_type = case_types.get(judgment_id, "Unknown")
        facts_found = "facts" in sections_found_map.get(judgment_id, [])

        results.append({
            "id": judgment_id,
            "case_type": case_type,
            "word_count": wc,
            "doc_type": doc_type,
            "facts_found": facts_found,
        })

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    # ---- Print per-judgment table ----
    print(f"{'ID':10s} {'case type':22s} {'words':>6s}  {'doc_type':13s} facts_found")
    for r in results:
        print(f"{r['id']:10s} {r['case_type']:22s} {r['word_count']:6d}  "
              f"{r['doc_type']:13s} {r['facts_found']}")

    # ---- Print summary grouped by case type ----
    print("\n=== Summary by case type ===")
    by_type = {}
    for r in results:
        by_type.setdefault(r["case_type"], []).append(r)

    for case_type, rows in sorted(by_type.items()):
        avg_words = sum(r["word_count"] for r in rows) / len(rows)
        full_count = sum(1 for r in rows if r["doc_type"] == "full_judgment")
        facts_ok = sum(1 for r in rows if r["facts_found"])
        print(f"{case_type:22s} n={len(rows):3d}  avg_words={avg_words:7.0f}  "
              f"full_judgment={full_count:3d}  facts_found={facts_ok:3d}")

    full_total = sum(1 for r in results if r["doc_type"] == "full_judgment")
    short_total = len(results) - full_total
    print(f"\nOverall: {full_total} full_judgment, {short_total} short_order "
          f"(threshold = {WORD_COUNT_THRESHOLD} words)")
    print(f"\nSaved classification to {OUTPUT_PATH}")
    print("\nNext: review the table above. If the word-count threshold looks")
    print("wrong for a case type (e.g. some 'full' ones are really short")
    print("orders, or vice versa), tell me and we'll adjust the threshold")
    print("or add case-type-specific rules.")


if __name__ == "__main__":
    main()