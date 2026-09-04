"""
STEP: Clean Bangla-translated judgment text before summarization.

This does everything the English cleaner does (repeated headers,
broken line-wraps, extra whitespace, junk lines), PLUS a few fixes
specific to Bangla / machine-translated text:

    1. Unicode normalization (NFC)
       Bangla has characters that can be typed/stored in more than
       one way but LOOK identical on screen (e.g. a vowel sign
       combined differently). If we don't normalize this, the model
       can treat two visually-identical words as different words.
       This is a very common, easy-to-miss bug in Bangla NLP.

    2. Punctuation fix
       Bangla should end sentences with '।' (the Bangla full stop),
       but Google Translate output often leaves regular English '.'
       instead. We convert sentence-ending '.' to '।' where safe.

    3. Stray leftover English fragments
       Google Translate sometimes leaves case citations, section
       numbers, or English words untranslated inside Bangla
       sentences (e.g. "179 CR of 2018"). We don't delete these
       (they can be legally important, like a case number!) — we
       just flag how many such lines exist so you can spot-check them.

Input:
    data/texts_bn/*.txt

Output:
    data/texts_bn_clean/*.txt

Run:
    python processing/clean_texts_bn.py
"""

import os
import re
import unicodedata
from collections import Counter

INPUT_DIR = "data/texts_bn"
OUTPUT_DIR = "data/texts_bn_clean"

JUNK_LINE_PATTERNS = [
    r"^\s*page\s+\d+\s*(of\s*\d+)?\s*$",
    r"^\s*পৃষ্ঠা\s*\d+\s*$",          # Bangla for "page N"
    r"^\s*-?\s*\d+\s*-?\s*$",
    r"^\s*www\.\S+\s*$",
    r"^\s*সুপ্রিম\s*কোর্ট.*$",         # repeated "Supreme Court..." header
]

JUNK_REGEXES = [re.compile(p, re.IGNORECASE) for p in JUNK_LINE_PATTERNS]


def is_junk_line(line):
    stripped = line.strip()
    if not stripped:
        return False
    for pattern in JUNK_REGEXES:
        if pattern.match(stripped):
            return True
    return False


def remove_repeated_lines(lines, min_repeats=4):
    counts = Counter(line.strip() for line in lines if line.strip())
    repeated = {line for line, count in counts.items() if count >= min_repeats}
    cleaned = [line for line in lines if line.strip() not in repeated]
    return cleaned, repeated


def fix_line_wrapped_words(text):
    return re.sub(r"(\w)-\n(\w)", r"\1\2", text)


def collapse_whitespace(text):
    lines = [line.rstrip() for line in text.split("\n")]
    text = "\n".join(lines)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def normalize_unicode(text):
    """
    Standardizes Bangla character encoding so visually-identical
    characters are always stored the same way underneath.
    """
    return unicodedata.normalize("NFC", text)


def fix_sentence_ending_punctuation(text):
    """
    Converts a '.' that ends a Bangla sentence into '।' (Bangla
    full stop), but leaves '.' alone inside things like abbreviations,
    decimal numbers, or case citations (e.g. "C.R. No. 4266/15"),
    since blindly replacing those would corrupt legal references.
    """
    # Only replace '.' when followed by whitespace + a Bangla letter
    # (i.e. clearly starting a new Bangla sentence), not mid-abbreviation.
    bangla_char_range = r"\u0980-\u09FF"
    pattern = re.compile(rf"\.(\s+)(?=[{bangla_char_range}])")
    return pattern.sub(r"।\1", text)


def count_stray_english_lines(text):
    """
    Counts lines that are mostly English (likely untranslated
    fragments like citations). Doesn't remove anything -- just
    reports it so you can manually check if needed.
    """
    count = 0
    for line in text.split("\n"):
        stripped = line.strip()
        if not stripped:
            continue
        letters = re.findall(r"[a-zA-Z\u0980-\u09FF]", stripped)
        if not letters:
            continue
        english_letters = re.findall(r"[a-zA-Z]", stripped)
        if len(english_letters) / len(letters) > 0.7:
            count += 1
    return count


def clean_text(raw_text):
    text = normalize_unicode(raw_text)

    lines = text.split("\n")
    joined = "\n".join(lines)
    joined = fix_line_wrapped_words(joined)
    lines = joined.split("\n")

    lines = [line for line in lines if not is_junk_line(line)]
    lines, repeated = remove_repeated_lines(lines)

    text = collapse_whitespace("\n".join(lines))
    text = fix_sentence_ending_punctuation(text)

    stray_count = count_stray_english_lines(text)

    return text, repeated, stray_count


def main():
    if not os.path.isdir(INPUT_DIR):
        print(f"Could not find folder: {INPUT_DIR}")
        return

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    files = sorted(f for f in os.listdir(INPUT_DIR) if f.lower().endswith(".txt"))
    print(f"Found {len(files)} files in {INPUT_DIR}\n")

    total_stray = 0

    for fname in files:
        in_path = os.path.join(INPUT_DIR, fname)
        out_path = os.path.join(OUTPUT_DIR, fname)

        with open(in_path, "r", encoding="utf-8", errors="ignore") as f:
            raw_text = f.read()

        cleaned, repeated_lines, stray_count = clean_text(raw_text)
        total_stray += stray_count

        with open(out_path, "w", encoding="utf-8") as f:
            f.write(cleaned)

        before_len = len(raw_text)
        after_len = len(cleaned)
        removed_pct = 100 * (1 - after_len / before_len) if before_len else 0

        print(f"{fname:30s} {before_len:6d} -> {after_len:6d} chars "
              f"({removed_pct:4.1f}% removed), "
              f"{len(repeated_lines)} repeated lines stripped, "
              f"{stray_count} mostly-English lines flagged")

    print(f"\nDone. Cleaned files saved to {OUTPUT_DIR}")
    print("Your original files in", INPUT_DIR, "were not modified.")
    print(f"\nTotal mostly-English lines flagged across all files: {total_stray}")
    print("These are likely untranslated legal citations/terms — worth a")
    print("quick spot-check, but usually fine to leave as-is since case")
    print("numbers and Act names often stay in English intentionally.")


if __name__ == "__main__":
    main()