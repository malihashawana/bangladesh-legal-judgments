import json
import re
from pathlib import Path

STRUCTURED_FILE = Path("data/structured_judgments.json")
EN_FOLDER = Path("data/texts_en_clean")
BN_FOLDER = Path("data/texts_bn_clean")
OUTPUT_FILE = Path("data/final_decisions.json")


DECISION_PATTERNS = [
    r"\brule is made absolute\b",
r"\brule made absolute\b",
r"\brule is made absolute in part\b",
r"\brule is made absolute in part and discharged in part\b",
r"\brule is discharged\b",
r"\brule is disposed of\b",
r"\brule stands discharged\b",
r"\brule stands absolute\b",
r"\bfind merit in\b",
    r"\ballowed\b",
    r"\bdismissed\b",
    r"\bdischarged\b",
    r"\brejected\b",
    r"\baffirmed\b",
    r"\bset[\s-]*aside\b",
    r"\bmodified\b",
    r"\bcommuted\b",
    r"\bacquitted\b",
    r"\bconviction\b",
    r"\bsentence\b",
    r"\bdisposed of\b",
    r"\bstands recalled\b",
    r"\bstands vacated\b",
    r"\bis hereby directed\b",
    r"\bwe direct\b",
    r"\bno order as to costs\b",
    r"খারিজ",
    r"মঞ্জুর",
    r"আদেশ",
    r"নির্দেশ"
]


REASONING_PATTERNS = [
    r"in view of",
    r"from the above discussion",
    r"we are of the view",
    r"this court is.*satisfied",
    r"we find",
    r"we hold",
    r"therefore",
    r"thus",
    r"consequently",
    r"for the reasons",
    r"accordingly"
]


def normalize(text):
    if not text:
        return ""

    text = text.replace("\r\n", "\n")
    text = text.replace("\r", "\n")

    # Preserve paragraph boundaries first
    text = re.sub(r"\n\s*\n+", "<PARA>", text)

    # Join PDF line-wrapped text into continuous sentences
    text = re.sub(r"\n+", " ", text)

    text = re.sub(r"[ \t]+", " ", text)

    # Restore paragraph boundaries
    text = text.replace("<PARA>", "\n\n")

    # Fix spaces before punctuation
    text = re.sub(r"\s+([,.;:!?])", r"\1", text)

    return text.strip()


def split_sentences(text):
    text = normalize(text)

    parts = re.split(
        r'(?<=[.!?।])\s+|\n+',
        text
    )

    return [
        x.strip()
        for x in parts
        if len(x.strip()) >= 25
    ]


def matches(sentence, patterns):
    return any(
        re.search(
            pattern,
            sentence,
            flags=re.IGNORECASE
        )
        for pattern in patterns
    )


def get_full_text(judgment_id):

    en_file = EN_FOLDER / f"{judgment_id}.txt"

    if en_file.exists():
        return (
            en_file.read_text(
                encoding="utf-8",
                errors="ignore"
            ),
            "en_clean"
        )

    bn_file = BN_FOLDER / f"{judgment_id}.txt"

    if bn_file.exists():
        return (
            bn_file.read_text(
                encoding="utf-8",
                errors="ignore"
            ),
            "bn_clean"
        )

    return "", "missing"


def extract_tail_evidence(text):

    sents = split_sentences(text)

    if not sents:
        return [], [], 0

    # Final 25% of judgment, with minimum/maximum bounds.
    tail_size = max(
        20,
        int(len(sents) * 0.25)
    )

    tail_size = min(
        tail_size,
        120
    )

    tail = sents[-tail_size:]

    decision = [
        sent
        for sent in tail
        if matches(
            sent,
            DECISION_PATTERNS
        )
    ]

    reasoning = [
        sent
        for sent in tail
        if matches(
            sent,
            REASONING_PATTERNS
        )
    ]

    # Keep the latest decision statements because
    # final disposition normally occurs near the end.
    decision = decision[-10:]

    # Reasoning immediately before the final disposition
    # is generally most informative.
    reasoning = reasoning[-8:]

    return decision, reasoning, tail_size


def process(item):

    judgment_id = item["id"]

    text, source = get_full_text(
        judgment_id
    )

    decision, reasoning, tail_size = (
        extract_tail_evidence(text)
    )

    return {
        "id": judgment_id,
        "source_file": item["source_file"],
        "text_source": source,
        "tail_sentences_examined": tail_size,
        "final_reasoning_evidence": reasoning,
        "final_decision_evidence": decision,
        "decision_found": bool(decision)
    }


def main():

    with open(
        STRUCTURED_FILE,
        "r",
        encoding="utf-8"
    ) as f:
        judgments = json.load(f)

    results = [
        process(item)
        for item in judgments
    ]

    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(
            results,
            f,
            ensure_ascii=False,
            indent=2
        )

    found = sum(
        x["decision_found"]
        for x in results
    )

    missing = [
        x["id"]
        for x in results
        if not x["decision_found"]
    ]

    print(
        f"Processed judgments: {len(results)}"
    )

    print(
        f"Decision evidence found: "
        f"{found}/{len(results)}"
    )

    print(
        "Missing decision evidence:",
        missing
    )

    print(
        f"Saved: {OUTPUT_FILE}"
    )


if __name__ == "__main__":
    main()