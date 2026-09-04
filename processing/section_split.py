"""
STEP: Split each judgment's English text into 3 parts:
    - facts       (what happened / background of the case)
    - arguments   (what each side submitted / argued)
    - order       (the court's decision / judgment / order)

Why this step exists:
    Your framework summarizes each section SEPARATELY (hierarchical
    summarization). Before we can summarize sections, we first need
    to know where each section starts and ends in the raw text.

How it works:
    Bangladeshi High Court judgments usually contain recognizable
    heading words, even if formatting isn't perfectly consistent
    (all-caps, numbered, underlined, etc). We search for the FIRST
    line that matches each group of keywords, and treat everything
    between one heading and the next as that section's text.

Input:
    data/texts_en/*.txt   (plain English text extracted from each judgment PDF)

Output:
    data/structured_judgments.json
    -> one entry per judgment with the 3 sections + a confidence flag

Run:
    python processing/section_split.py
"""

import os
import re
import json

TEXTS_DIR = "data/texts_en_clean"
OUTPUT_PATH = "data/structured_judgments.json"

# Heading keywords that usually mark the START of each section.
# Written as regex patterns, case-insensitive, matched against a whole line.
# Add more variations here as you discover them in your own judgments.
SECTION_PATTERNS = {
    "facts": [
        r"^\s*facts?\b",
        r"^\s*statement of facts",
        r"^\s*background\b",
        r"^\s*brief facts",
        r"^\s*facts? of the case",
        r"^\s*the (short|brief) facts",
        r"^\s*facts? leading to",
        r"^\s*this (rule|appeal|revision|petition|application) (was issued|is directed against|arises)",
        r"^\s*by this (application|petition|rule)",
        r"the (prosecution|plaintiff|petitioner|appellant)('s)?\s*case,?\s*in brief,?\s*(is|was)\s*that",
        r"the fact(s)? of the case,?\s*in brief,?\s*(is|are|was|were)\s*that",
    ],
    "arguments": [
        r"^\s*submissions?\b",
        r"^\s*arguments?\b",
        r"^\s*submissions? of the parties",
        r"^\s*contentions?\b",
        r"^\s*heard the learned",
        r"^\s*mr\.?\s+\w+.*(learned advocate|learned counsel)",
        r"^\s*ms\.?\s+\w+.*(learned advocate|learned counsel)",
        r"learned (advocate|counsel) (for|appearing for) the (petitioner|appellant|respondent|opposite party|state)",
        r"^\s*having heard the learned",
        r"^\s*learned counsel for (both|the) (sides|parties)",
        r"^\s*per contra,?\b",
        r"^\s*mr\.?\s+\w+.*(contended|submitted|argued) that",
        r"^\s*ms\.?\s+\w+.*(contended|submitted|argued) that",
    ],
    "order": [
        r"^\s*judgment\b",
        r"^\s*order\b",
        r"^\s*decision\b",
        r"^\s*findings? and order",
        r"^\s*discussion and findings?",
        r"^\s*conclusion\b",
        r"^\s*in the result",
        r"^\s*accordingly,?\b",
        r"^\s*in view of the above",
        r"^\s*for the reasons stated above",
        r"^\s*the rule is (made absolute|discharged)",
        r"^\s*it is ordered that",
        r"^\s*this (rule|appeal|revision|petition) is (allowed|dismissed|disposed)",
    ],
}


def find_section_starts(lines):
    """
    Scan each line of the judgment and record the line number where
    each section FIRST appears to start.
    Returns a dict like {"facts": 12, "arguments": 40, "order": 88}
    (some keys may be missing if that heading wasn't found)
    """
    found = {}
    for i, line in enumerate(lines):
        clean_line = line.strip()
        if not clean_line:
            continue

        for section_name, patterns in SECTION_PATTERNS.items():
            if section_name in found:
                continue  # already found the first occurrence, skip
            for pattern in patterns:
                if re.match(pattern, clean_line, re.IGNORECASE):
                    found[section_name] = i
                    break
    return found


def split_into_sections(text):
    """
    Given the full judgment text, return:
        sections: dict with facts / arguments / order text
        confidence: "high", "partial", or "low" depending on how many
                    of the 3 sections were successfully located
        facts_inferred: True if 'facts' wasn't found by a heading match
                        and was instead filled in using the fallback
                        rule below
    """
    lines = text.split("\n")
    starts = find_section_starts(lines)

    # Sort found sections by where they appear in the document
    ordered = sorted(starts.items(), key=lambda kv: kv[1])

    sections = {"facts": "", "arguments": "", "order": ""}

    for idx, (name, start_line) in enumerate(ordered):
        end_line = ordered[idx + 1][1] if idx + 1 < len(ordered) else len(lines)
        section_text = "\n".join(lines[start_line:end_line]).strip()
        sections[name] = section_text

    facts_inferred = False

    # FALLBACK RULE: if no "facts" heading was matched, but at least one
    # other section WAS found (arguments or order), treat everything
    # from the top of the document up to that first found section as
    # the facts text. This is reasonable because facts/background
    # almost always come first in a judgment, even when the judge
    # doesn't label it with a heading word.
    if "facts" not in starts and ordered:
        first_other_start_line = ordered[0][1]
        inferred_text = "\n".join(lines[:first_other_start_line]).strip()
        if inferred_text:
            sections["facts"] = inferred_text
            facts_inferred = True

    found_count = len(starts)
    if found_count == 3:
        confidence = "high"
    elif found_count >= 1:
        confidence = "partial"
    else:
        confidence = "low"

    return sections, confidence, list(starts.keys()), facts_inferred


def main():
    if not os.path.isdir(TEXTS_DIR):
        print(f"Could not find folder: {TEXTS_DIR}")
        print("Update TEXTS_DIR at the top of this script to match your setup.")
        return

    results = []
    files = sorted(f for f in os.listdir(TEXTS_DIR) if f.lower().endswith(".txt"))

    print(f"Found {len(files)} text files in {TEXTS_DIR}\n")

    confidence_counts = {"high": 0, "partial": 0, "low": 0}

    for fname in files:
        path = os.path.join(TEXTS_DIR, fname)
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()

        sections, confidence, found_sections, facts_inferred = split_into_sections(text)
        confidence_counts[confidence] += 1

        judgment_id = os.path.splitext(fname)[0]
        results.append({
            "id": judgment_id,
            "source_file": fname,
            "confidence": confidence,
            "sections_found": found_sections,
            "facts_inferred": facts_inferred,
            "facts": sections["facts"],
            "arguments": sections["arguments"],
            "order": sections["order"],
        })

        flag = " (facts inferred by position)" if facts_inferred else ""
        print(f"{judgment_id:15s} confidence={confidence:8s} found={found_sections}{flag}")

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\nSaved {len(results)} structured judgments to {OUTPUT_PATH}")
    print("Confidence breakdown:", confidence_counts)

    inferred_count = sum(1 for r in results if r["facts_inferred"])
    print(f"\n{inferred_count} judgments had their 'facts' text filled in by "
          f"position (no exact heading match, but text before the first "
          f"found section was used instead).")

    # Diagnostic: which specific section (facts / arguments / order) is
    # most often NOT found? This tells us exactly what to fix next,
    # instead of guessing.
    missing_counts = {"facts": 0, "arguments": 0, "order": 0}
    missing_examples = {"facts": [], "arguments": [], "order": []}
    for r in results:
        for section_name in missing_counts:
            if section_name not in r["sections_found"]:
                missing_counts[section_name] += 1
                if len(missing_examples[section_name]) < 3:
                    missing_examples[section_name].append(r["id"])

    print("\nMissing-section counts (out of", len(results), "judgments):")
    for name, count in missing_counts.items():
        examples = ", ".join(missing_examples[name])
        print(f"  {name:10s} missing in {count:3d} judgments  e.g. {examples}")

    print("\nTip: open one of the example judgment .txt files listed above")
    print("and look at the actual heading wording it uses for that missing")
    print("section — send it to me and I'll add the exact pattern.")


if __name__ == "__main__":
    main()