"""
STEP 3: Compute BERTScore (semantic similarity) and factual
precision/recall (does the summary's content actually appear in the
source judgment?) for the baseline mT5 summaries.

Install first (run once in your venv):
    pip install bert-score

Metrics computed:
    1. BERTScore (P/R/F1) - semantic similarity between summary and
       reference, using a multilingual transformer. Better than
       ROUGE for catching paraphrases/synonyms, which matters a lot
       for Bangla where ROUGE's exact-word-overlap is too strict.

    2. Factual precision - of the "content words" (nouns/numbers/
       key terms) in the generated summary, what fraction actually
       appear somewhere in the SOURCE judgment (not just the
       reference)? Low factual precision = hallucination.

    3. Factual recall - of the important terms in the source
       judgment (using its own reference/order section as a stand-in
       for "what matters"), how many did the summary capture?

Input:
    data/summaries_baseline_mt5.json
    data/reference_summaries.json
    data/texts_bn_clean/*.txt   (full source text, for factual check)

Output:
    data/eval_bertscore_factual_baseline.csv
    data/eval_bertscore_factual_baseline.json

Run:
    python processing/eval_bertscore_factual.py --model baseline_mt5
"""

import os
import re
import json
import argparse
import pandas as pd
from bert_score import score as bertscore_score

SUMMARIES_PATH = "data/summaries_baseline_mt5.json"
REFERENCES_PATH = "data/reference_summaries.json"
TEXTS_BN_DIR = "data/texts_bn_clean"


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_source_text(judgment_id):
    path = os.path.join(TEXTS_BN_DIR, f"{judgment_id}.txt")
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
    except UnicodeDecodeError:
        return None
    if len(text.strip()) < 200:
        return None
    return text


# Bangla numerals + digits, useful as "hard facts" that must not be hallucinated
NUMERIC_PATTERN = re.compile(r"[０-９0-9০-৯]+(?:[.,][０-９0-9০-৯]+)*")

# Words shorter than this are usually function words / not meaningful "facts"
MIN_TOKEN_LEN = 3


def extract_content_tokens(text):
    """
    Rough extraction of 'content-bearing' tokens (proxy for facts):
    numbers/dates and longer words. Not a proper NER system, but
    good enough as a hallucination/coverage signal without needing
    a Bangla NER model.
    """
    tokens = text.split()
    content_tokens = set()
    for tok in tokens:
        cleaned = tok.strip("।,.()[]{}:;\"'—-")
        if not cleaned:
            continue
        if NUMERIC_PATTERN.fullmatch(cleaned):
            content_tokens.add(cleaned)
        elif len(cleaned) >= MIN_TOKEN_LEN:
            content_tokens.add(cleaned)
    return content_tokens


def factual_precision_recall(summary_text, source_text, reference_text):
    summary_tokens = extract_content_tokens(summary_text)
    source_tokens = extract_content_tokens(source_text)
    reference_tokens = extract_content_tokens(reference_text)

    if not summary_tokens:
        return 0.0, 0.0

    # Precision: of what the summary claims, how much is grounded in the source?
    grounded = summary_tokens & source_tokens
    precision = len(grounded) / len(summary_tokens)

    # Recall: of the important terms in the reference (proxy for "what
    # should be covered"), how many did the summary capture?
    if not reference_tokens:
        recall = 0.0
    else:
        covered = summary_tokens & reference_tokens
        recall = len(covered) / len(reference_tokens)

    return precision, recall


def main(model_name, summaries_path, output_prefix):
    summaries = {d["id"]: d for d in load_json(summaries_path)}
    references = {d["id"]: d for d in load_json(REFERENCES_PATH)}

    rows = []
    hyps_for_bertscore = []
    refs_for_bertscore = []
    ids_for_bertscore = []
    skipped = []

    for judgment_id, ref_entry in references.items():
        ref_text = ref_entry.get("reference_summary_bn", "").strip()
        hyp_entry = summaries.get(judgment_id)

        if not ref_text or hyp_entry is None:
            skipped.append(judgment_id)
            continue

        hyp_text = hyp_entry.get("summary_bn", "").strip()
        if not hyp_text:
            skipped.append(judgment_id)
            continue

        source_text = load_source_text(judgment_id)
        if source_text is None:
            skipped.append(judgment_id)
            continue

        precision, recall = factual_precision_recall(hyp_text, source_text, ref_text)
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

        rows.append({
            "id": judgment_id,
            "case_type": ref_entry.get("case_type", "Unknown"),
            "model": model_name,
            "factual_precision": precision,
            "factual_recall": recall,
            "factual_f1": f1,
        })

        hyps_for_bertscore.append(hyp_text)
        refs_for_bertscore.append(ref_text)
        ids_for_bertscore.append(judgment_id)

    print(f"Running BERTScore on {len(hyps_for_bertscore)} documents "
          f"(this downloads a multilingual model on first run, may take a few minutes)...")

    # bert-base-multilingual-cased covers Bangla reasonably; lang="bn" not
    # directly supported by default model list, so we pass model_type explicitly.
    P, R, F1 = bertscore_score(
        hyps_for_bertscore,
        refs_for_bertscore,
        model_type="bert-base-multilingual-cased",
        verbose=True
    )

    bertscore_map = {
        doc_id: {"bertscore_p": p.item(), "bertscore_r": r.item(), "bertscore_f1": f1.item()}
        for doc_id, p, r, f1 in zip(ids_for_bertscore, P, R, F1)
    }

    for row in rows:
        row.update(bertscore_map.get(row["id"], {"bertscore_p": None, "bertscore_r": None, "bertscore_f1": None}))

    df = pd.DataFrame(rows)

    os.makedirs("data", exist_ok=True)
    df.to_csv(f"{output_prefix}.csv", index=False, encoding="utf-8-sig")
    df.to_json(f"{output_prefix}.json", orient="records", force_ascii=False, indent=2)

    print(f"\nScored {len(df)} documents for model='{model_name}'")
    if skipped:
        print(f"Skipped {len(skipped)} (missing text/hyp/ref): {skipped}")

    print("\n=== AVERAGE SCORES ===")
    numeric_cols = [
        "factual_precision", "factual_recall", "factual_f1",
        "bertscore_p", "bertscore_r", "bertscore_f1"
    ]
    print(df[numeric_cols].mean().round(4).to_string())

    print(f"\nSaved -> {output_prefix}.csv / .json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="baseline_mt5", help="Model name label")
    parser.add_argument("--summaries", default=SUMMARIES_PATH, help="Path to summaries JSON")
    parser.add_argument("--out", default="data/eval_bertscore_factual_baseline", help="Output prefix (no extension)")
    args = parser.parse_args()
    main(args.model, args.summaries, args.out)