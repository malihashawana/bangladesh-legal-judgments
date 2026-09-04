"""
STEP 1: Evaluation setup — build reference summaries + eval config.

Install first (run once in your venv):
    pip install rouge-score bert-score sacrebleu nltk pandas matplotlib seaborn

What this does:
    Your baseline summaries (data/summaries_baseline_mt5.json) have no
    human-written gold summary to score against. ROUGE/BLEU/BERTScore
    all need *some* reference text.

    Fix used here: extract the "Order"/operative portion (the final
    judgment/decision, usually the last ~150-250 words of each
    judgment) from data/texts_bn_clean/*.txt as a proxy reference
    summary. This is the closest thing to a "gold summary" you have
    without manually writing 51 reference summaries.

    NOTE: this is a proxy, not a true gold standard — flag this in
    your paper/report. If you want real gold summaries later, this
    script also drops a template (data/reference_summaries_TEMPLATE.json)
    you can hand-edit for a subset (even 10-15 judgments) for a
    cleaner human-reference evaluation.

Input:
    data/texts_bn_clean/*.txt
    data/document_classification.json

Output:
    data/reference_summaries.json   (proxy references, auto-extracted)
    data/reference_summaries_TEMPLATE.json  (empty slots for manual gold summaries)

Run:
    python processing/build_reference_summaries.py
"""

import os
import json
import re

TEXTS_BN_DIR = "data/texts_bn_clean"
CLASSIFICATION_PATH = "data/document_classification.json"
OUTPUT_PROXY = "data/reference_summaries.json"
OUTPUT_TEMPLATE = "data/reference_summaries_TEMPLATE.json"

# Bangla markers that often precede the operative/order portion of a judgment
ORDER_MARKERS = [
    "অর্ডার", "আদেশ", "রায়ের শেষাংশ", "ফলে", "ফলস্বরূপ",
    "তদনুসারে", "সুতরাং", "অতএব", "বিধি", "আপিল"
]


def extract_order_section(text, min_words=40, max_words=250):
    """
    Heuristic extraction of the concluding operative portion of a
    judgment, to use as a proxy reference summary.
    """
    words = text.split()
    if len(words) <= max_words:
        return text.strip()

    # Search backwards from the end for the last strong order-marker
    # paragraph boundary, else just fall back to last N words.
    paragraphs = [p for p in text.split("\n") if p.strip()]
    tail_candidate = []
    collected_words = 0

    for para in reversed(paragraphs):
        tail_candidate.insert(0, para)
        collected_words += len(para.split())
        if collected_words >= min_words and any(m in para for m in ORDER_MARKERS):
            break
        if collected_words >= max_words:
            break

    proxy = "\n".join(tail_candidate).strip()
    proxy_words = proxy.split()
    if len(proxy_words) > max_words:
        proxy = " ".join(proxy_words[-max_words:])
    return proxy


def main():
    if not os.path.isdir(TEXTS_BN_DIR):
        print(f"Could not find folder: {TEXTS_BN_DIR}")
        return

    with open(CLASSIFICATION_PATH, "r", encoding="utf-8") as f:
        classification = {d["id"]: d for d in json.load(f)}

    files = sorted(f for f in os.listdir(TEXTS_BN_DIR) if f.lower().endswith(".txt"))
    proxy_refs = []
    template_refs = []
    skipped_binary = []

    for fname in files:
        judgment_id = os.path.splitext(fname)[0]
        path = os.path.join(TEXTS_BN_DIR, fname)

        with open(path, "rb") as f:
            raw = f.read()

        # skip files that are placeholder/binary (not real extracted text)
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError:
            skipped_binary.append(judgment_id)
            continue

        if len(text.strip()) < 200:
            skipped_binary.append(judgment_id)
            continue

        proxy_summary = extract_order_section(text)

        proxy_refs.append({
            "id": judgment_id,
            "case_type": classification.get(judgment_id, {}).get("case_type", "Unknown"),
            "reference_summary_bn": proxy_summary,
            "reference_type": "proxy_order_extraction"
        })

        template_refs.append({
            "id": judgment_id,
            "case_type": classification.get(judgment_id, {}).get("case_type", "Unknown"),
            "reference_summary_bn": ""  # fill manually for a clean gold subset
        })

    os.makedirs("data", exist_ok=True)
    with open(OUTPUT_PROXY, "w", encoding="utf-8") as f:
        json.dump(proxy_refs, f, ensure_ascii=False, indent=2)

    with open(OUTPUT_TEMPLATE, "w", encoding="utf-8") as f:
        json.dump(template_refs, f, ensure_ascii=False, indent=2)

    print(f"Built {len(proxy_refs)} proxy reference summaries -> {OUTPUT_PROXY}")
    print(f"Blank template for manual gold summaries -> {OUTPUT_TEMPLATE}")
    if skipped_binary:
        print(f"\nSkipped {len(skipped_binary)} files (binary/placeholder, no real text): {skipped_binary}")


if __name__ == "__main__":
    main()