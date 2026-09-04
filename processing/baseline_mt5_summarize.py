"""
STEP: Baseline Bangla summarization using mT5 XL-Sum.

What this does:
    For each judgment, take the FULL Bangla-translated text (from
    data/texts_bn/) and generate one short Bangla summary using a
    general-purpose, pre-trained summarization model.

Why we need this:
    This is your COMPARISON BASELINE. It has no idea about Facts vs
    Arguments vs Order, and does no fact-checking. Later, when you
    compare it against your own framework's output, you'll be able to
    show concretely how much better section-aware + verified
    summarization is (fewer hallucinations, better structure, etc).

Model used:
    csebuetnlp/mT5_multilingual_XLSum
    - A well-known, widely-cited model in Bangla NLP research
    - Runs fine on CPU (no GPU required) — just slower than a GPU would be
    - ~580M parameters, so expect maybe 5-20 seconds per judgment on
      a laptop CPU, depending on text length and your machine

Install requirements (run once in your venv):
    pip install transformers torch sentencepiece

Input:
    data/texts_bn/*.txt

Output:
    data/summaries_baseline_mt5.json

Run:
    python processing/baseline_mt5_summarize.py
"""

import os
import json
import time

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

TEXTS_BN_DIR = "data/texts_bn_clean"
OUTPUT_PATH = "data/summaries_baseline_mt5.json"

MODEL_NAME = "csebuetnlp/mT5_multilingual_XLSum"

# mT5 XL-Sum was trained with a max input length of 512 tokens.
# Longer judgments will be truncated — this is a known limitation of
# this baseline model (worth mentioning in your paper as a weakness
# that YOUR hierarchical approach avoids, since you summarize smaller
# sections instead of the whole document at once).
MAX_INPUT_TOKENS = 512
MAX_SUMMARY_TOKENS = 84  # roughly matches XL-Sum's typical summary length


def load_model():
    print(f"Loading model: {MODEL_NAME}")
    print("(first run will download ~2.3GB, this can take a few minutes)")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
    model.eval()
    return tokenizer, model


def summarize(text, tokenizer, model):
    inputs = tokenizer(
        text,
        return_tensors="pt",
        max_length=MAX_INPUT_TOKENS,
        truncation=True,
    )
    output_ids = model.generate(
        **inputs,
        max_length=MAX_SUMMARY_TOKENS,
        num_beams=4,
        no_repeat_ngram_size=3,
    )
    summary = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return summary


def main():
    if not os.path.isdir(TEXTS_BN_DIR):
        print(f"Could not find folder: {TEXTS_BN_DIR}")
        print("Update TEXTS_BN_DIR at the top of this script if needed.")
        return

    tokenizer, model = load_model()

    files = sorted(f for f in os.listdir(TEXTS_BN_DIR) if f.lower().endswith(".txt"))
    print(f"\nFound {len(files)} Bangla text files. Starting summarization...\n")

    results = []
    for i, fname in enumerate(files, start=1):
        path = os.path.join(TEXTS_BN_DIR, fname)
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read().strip()

        if not text:
            print(f"[{i}/{len(files)}] {fname}: empty file, skipping")
            continue

        start = time.time()
        summary = summarize(text, tokenizer, model)
        elapsed = time.time() - start

        judgment_id = os.path.splitext(fname)[0]
        results.append({
            "id": judgment_id,
            "source_file": fname,
            "model": "mT5_multilingual_XLSum",
            "summary_bn": summary,
        })

        print(f"[{i}/{len(files)}] {judgment_id} done in {elapsed:.1f}s")
        print(f"    -> {summary}\n")

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\nSaved {len(results)} baseline summaries to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()