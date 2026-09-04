"""
STEP: Clean raw English judgment text before any summarization.

Problems this fixes:
    1. Repeated headers/footers (e.g. court name, page numbers) that
       got extracted on every single page of the PDF
    2. Words broken across line-wraps, e.g. "judg-\nment" -> "judgment"
    3. Extra/inconsistent blank lines and spacing
    4. Common junk lines (page numbers like "Page 3 of 12", stray
       "www.supremecourt.gov.bd" footer text, etc.)

What this does NOT do:
    - Does not touch your original files in data/texts_en/ at all
    - Does not change wording/content, only removes noise/formatting junk

Input:
    data/texts_en/*.txt

Output:
    data/texts_en_clean/*.txt   (same filenames, cleaned content)

Run:
    python processing/clean_texts_en.py
"""

import os
import re
from collections import Counter

INPUT_DIR = "data/texts_en"
OUTPUT_DIR = "data/texts_en_clean"

# Lines matching these patterns are almost always junk, not real
# judgment content. Add more here if you spot other junk patterns.
JUNK_LINE_PATTERNS = [
    r"^\s*page\s+\d+\s*(of\s*\d+)?\s*$",         # "Page 3", "Page 3 of 12"
    r"^\s*-?\s*\d+\s*-?\s*$",                     # a lone page number like "12"
    r"^\s*www\.\S+\s*$",                          # website footer
    r"^\s*supreme\s*court\s*of\s*bangladesh\s*$", # repeated court header
    r"^\s*high\s*court\s*division\s*$",           # repeated header
]

JUNK_REGEXES = [re.compile(p, re.IGNORECASE) for p in JUNK_LINE_PATTERNS]


def is_junk_line(line):
    stripped = line.strip()
    if not stripped:
        return False  # blank lines are handled separately, not junk here
    for pattern in JUNK_REGEXES:
        if pattern.match(stripped):
            return True
    return False


def remove_repeated_lines(lines, min_repeats=4):
    """
    If the exact same line appears many times within one document
    (e.g. a header repeated on every page), it's very likely a
    header/footer, not real content. Remove lines that repeat at
    least `min_repeats` times.
    """
    counts = Counter(line.strip() for line in lines if line.strip())
    repeated = {line for line, count in counts.items() if count >= min_repeats}

    cleaned = []
    for line in lines:
        if line.strip() in repeated:
            continue
        cleaned.append(line)
    return cleaned, repeated


def fix_line_wrapped_words(text):
    """
    Fixes words that got split across a line break with a hyphen,
    e.g.:
        "The respon-\ndent filed"  ->  "The respondent filed"
    """
    return re.sub(r"(\w)-\n(\w)", r"\1\2", text)


def collapse_whitespace(text):
    """
    Collapses 3+ blank lines into a single blank line, and trims
    trailing spaces on each line.
    """
    lines = [line.rstrip() for line in text.split("\n")]
    text = "\n".join(lines)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def clean_text(raw_text):
    lines = raw_text.split("\n")

    # Step 1: fix hyphenated line-wraps (do this on the joined text)
    joined = "\n".join(lines)
    joined = fix_line_wrapped_words(joined)
    lines = joined.split("\n")

    # Step 2: drop obvious junk lines (page numbers, footers)
    lines = [line for line in lines if not is_junk_line(line)]

    # Step 3: drop lines repeated many times in this document (headers)
    lines, repeated = remove_repeated_lines(lines)

    # Step 4: collapse extra whitespace
    text = collapse_whitespace("\n".join(lines))

    return text, repeated


def main():
    if not os.path.isdir(INPUT_DIR):
        print(f"Could not find folder: {INPUT_DIR}")
        return

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    files = sorted(f for f in os.listdir(INPUT_DIR) if f.lower().endswith(".txt"))
    print(f"Found {len(files)} files in {INPUT_DIR}\n")

    for fname in files:
        in_path = os.path.join(INPUT_DIR, fname)
        out_path = os.path.join(OUTPUT_DIR, fname)

        with open(in_path, "r", encoding="utf-8", errors="ignore") as f:
            raw_text = f.read()

        cleaned, repeated_lines = clean_text(raw_text)

        with open(out_path, "w", encoding="utf-8") as f:
            f.write(cleaned)

        before_len = len(raw_text)
        after_len = len(cleaned)
        removed_pct = 100 * (1 - after_len / before_len) if before_len else 0

        print(f"{fname:30s} {before_len:6d} -> {after_len:6d} chars "
              f"({removed_pct:4.1f}% removed), "
              f"{len(repeated_lines)} repeated header/footer lines stripped")

    print(f"\nDone. Cleaned files saved to {OUTPUT_DIR}")
    print("Your original files in", INPUT_DIR, "were not modified.")


if __name__ == "__main__":
    main()