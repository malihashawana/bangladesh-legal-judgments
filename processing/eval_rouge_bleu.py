"""
STEP 2: Compute ROUGE-1/2/L and BLEU for the mT5 baseline summaries
against the proxy reference summaries built in Step 1.

Install first (run once in your venv):
    pip install rouge-score sacrebleu pandas

Input:
    data/summaries_baseline_mt5.json   (from your existing baseline script)
    data/reference_summaries.json      (from Step 1)

Output:
    data/eval_rouge_bleu_baseline.csv
    data/eval_rouge_bleu_baseline.json (with per-doc + averaged scores)

Run:
    python processing/eval_rouge_bleu.py --model baseline_mt5
"""

import os
import json
import argparse
import pandas as pd
from rouge_score import rouge_scorer
import sacrebleu

SUMMARIES_PATH = "data/summaries_baseline_mt5.json"
REFERENCES_PATH = "data/reference_summaries.json"
OUTPUT_CSV = "data/eval_rouge_bleu_baseline.csv"
OUTPUT_JSON = "data/eval_rouge_bleu_baseline.json"


class BanglaWhitespaceTokenizer:
    """
    Simple whitespace tokenizer to replace rouge_score's default
    tokenizer, which only recognizes ASCII word characters and
    therefore strips all Bangla text (causing silent 0.0 scores).
    """
    def tokenize(self, text):
        return text.split()


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def compute_rouge(scorer, hyp, ref):
    scores = scorer.score(ref, hyp)
    return {
        "rouge1_f": scores["rouge1"].fmeasure,
        "rouge1_p": scores["rouge1"].precision,
        "rouge1_r": scores["rouge1"].recall,
        "rouge2_f": scores["rouge2"].fmeasure,
        "rougeL_f": scores["rougeL"].fmeasure,
    }


def compute_bleu(hyp, ref):
    # sacrebleu handles Bangla fine since it doesn't restrict to ASCII
    result = sacrebleu.sentence_bleu(hyp, [ref])
    return result.score


def main(model_name, summaries_path, output_prefix):
    summaries = {d["id"]: d for d in load_json(summaries_path)}
    references = {d["id"]: d for d in load_json(REFERENCES_PATH)}

    scorer = rouge_scorer.RougeScorer(
        ["rouge1", "rouge2", "rougeL"],
        use_stemmer=False,
        tokenizer=BanglaWhitespaceTokenizer()
    )

    rows = []
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

        rouge_scores = compute_rouge(scorer, hyp_text, ref_text)
        bleu_score = compute_bleu(hyp_text, ref_text)

        rows.append({
            "id": judgment_id,
            "case_type": ref_entry.get("case_type", "Unknown"),
            "model": model_name,
            "hyp_len_words": len(hyp_text.split()),
            "ref_len_words": len(ref_text.split()),
            "bleu": bleu_score,
            **rouge_scores,
        })

    df = pd.DataFrame(rows)

    os.makedirs("data", exist_ok=True)
    df.to_csv(f"{output_prefix}.csv", index=False, encoding="utf-8-sig")
    df.to_json(f"{output_prefix}.json", orient="records", force_ascii=False, indent=2)

    print(f"Scored {len(df)} documents for model='{model_name}'")
    if skipped:
        print(f"Skipped {len(skipped)} (missing hypothesis or reference): {skipped}")

    print("\n=== AVERAGE SCORES ===")
    numeric_cols = ["bleu", "rouge1_f", "rouge1_p", "rouge1_r", "rouge2_f", "rougeL_f"]
    print(df[numeric_cols].mean().round(4).to_string())

    print(f"\nSaved -> {output_prefix}.csv / .json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="baseline_mt5", help="Model name label")
    parser.add_argument("--summaries", default=SUMMARIES_PATH, help="Path to summaries JSON")
    parser.add_argument("--out", default="data/eval_rouge_bleu_baseline", help="Output prefix (no extension)")
    args = parser.parse_args()
    main(args.model, args.summaries, args.out)