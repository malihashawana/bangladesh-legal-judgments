import json
import re
from pathlib import Path


STRUCTURED_FILE = Path("data/structured_judgments.json")
TEXT_FOLDER = Path("data/texts_en_clean")
FINAL_DECISION_FILE = Path("data/final_decisions.json")

OUTPUT_FILE = Path("data/reasoning_evidence.json")


# ============================================================
# TEXT CLEANING
# ============================================================

def normalize(text):
    if not text:
        return ""

    text = text.replace("\r\n", "\n")
    text = text.replace("\r", "\n")

    # preserve paragraph boundaries
    text = re.sub(r"\n\s*\n+", "<PARA>", text)

    # join PDF line wraps
    text = re.sub(r"\n+", " ", text)

    text = re.sub(r"[ \t]+", " ", text)

    text = text.replace("<PARA>", "\n\n")

    text = re.sub(
        r"\s+([,.;:!?])",
        r"\1",
        text
    )

    return text.strip()


def sentences(text):
    text = normalize(text)

    if not text:
        return []

    # Protect common legal abbreviations
    replacements = {
        "No.": "No<prd>",
        "Nos.": "Nos<prd>",
        "Mr.": "Mr<prd>",
        "Mrs.": "Mrs<prd>",
        "Dr.": "Dr<prd>",
        "J.": "J<prd>",
        "Ltd.": "Ltd<prd>",
    }

    for old, new in replacements.items():
        text = text.replace(old, new)

    parts = re.split(
        r'(?<=[.!?])\s+|\n\n+',
        text
    )

    output = []

    for part in parts:

        part = part.replace(
            "<prd>",
            "."
        ).strip()

        if len(part) >= 35:
            output.append(part)

    return output


def unique(items):
    result = []

    for item in items:
        item = item.strip()

        if item and item not in result:
            result.append(item)

    return result


def contains_any(text, terms):
    lower = text.lower()

    return any(
        term.lower() in lower
        for term in terms
    )


# ============================================================
# LOAD FULL JUDGMENT
# ============================================================

def load_full_text(judgment_id):

    path = (
        TEXT_FOLDER /
        f"{judgment_id}.txt"
    )

    if not path.exists():
        return ""

    return path.read_text(
        encoding="utf-8",
        errors="ignore"
    )


# ============================================================
# FACT EXTRACTION
# ============================================================

def extract_facts(text):

    sents = sentences(text)

    if not sents:
        return []

    fact_markers = [
        "relevant facts",
        "facts for disposal",
        "facts leading",
        "brief facts",
        "short facts",
        "admitted facts",
        "undisputed facts",
        "facts emerging",
        "facts of the case",
        "case of the petitioner",
        "case of the prosecution",
        "factual background"
    ]

    exclude = [
        "supreme court of bangladesh",
        "high court division",
        "present:",
        "heard on",
        "judgment on",
        "for the petitioner",
        "for the respondent",
        "learned advocate submits",
        "learned advocate contends"
    ]

    found = []

    # First preference:
    # sentences immediately following explicit fact headings
    for i, sent in enumerate(sents):

        if contains_any(
            sent,
            fact_markers
        ):

            # include marker sentence + following narrative
            for candidate in sents[
                i:min(i + 6, len(sents))
            ]:

                if not contains_any(
                    candidate,
                    exclude
                ):
                    found.append(
                        candidate
                    )

            if found:
                break

    # Second preference:
    # strong factual language
    if len(found) < 3:

        fact_signals = [
            "it is undisputed",
            "it is admitted",
            "admitted position",
            "it appears from the record",
            "the petitioner joined",
            "the petitioner retired",
            "was arrested",
            "was convicted",
            "was filed",
            "filed the instant",
            "was registered",
            "occurred on",
            "the company failed",
            "the materials on record disclose",
            "respondents refrained",
            "pension benefits",
            "audit objection",
            "share transfer"
        ]

        for sent in sents:

            if contains_any(
                sent,
                exclude
            ):
                continue

            if contains_any(
                sent,
                fact_signals
            ):
                found.append(
                    sent
                )

            if len(
                unique(found)
            ) >= 6:
                break

    return unique(found)[:6]


# ============================================================
# LEGAL ISSUE EXTRACTION
# ============================================================

def extract_issues(text):

    sents = sentences(text)

    if not sents:
        return []

    strong_markers = [
        "issues arise for determination",
        "issue arises for determination",
        "issue for determination",
        "issues for determination",
        "question for determination",
        "questions for determination",
        "question requiring determination",
        "point for determination",
        "required to examine",
        "requires determination",
        "the question is whether",
        "the issue is whether"
    ]

    found = []

    # Capture explicit issue section
    for i, sent in enumerate(sents):

        if contains_any(
            sent,
            strong_markers
        ):

            found.append(sent)

            # Following sentences often contain
            # First issue / Second issue / Whether...
            for candidate in sents[
                i + 1:min(i + 5, len(sents))
            ]:

                lower = candidate.lower()

                if (
                    "whether" in lower
                    or "issue no" in lower
                    or "firstly" in lower
                    or "secondly" in lower
                ):
                    found.append(
                        candidate
                    )

            break

    # Search explicit Issue No. headings
    for sent in sents:

        lower = sent.lower()

        if (
            "issue no." in lower
            or "issue no " in lower
        ):

            found.append(
                sent
            )

    # fallback: carefully selected whether sentences
    if not found:

        for sent in sents:

            lower = sent.lower()

            if (
                "whether" in lower
                and len(sent) < 700
            ):

                # avoid lawyer submissions
                if not contains_any(
                    sent,
                    [
                        "learned advocate",
                        "he submits",
                        "she submits",
                        "counsel submits"
                    ]
                ):
                    found.append(
                        sent
                    )

            if len(found) >= 3:
                break

    return unique(found)[:4]


# ============================================================
# LEGAL BASIS EXTRACTION
# ============================================================

def is_real_legal_basis(sentence):

    s = sentence.lower()

    patterns = [

        # Section 51 / sections 81(2)
        r"\bsections?\s+\d",

        # Article 102
        r"\barticles?\s+\d",

        # Order XX Rule ...
        r"\border\s+[ivxlcdm0-9]+\s+rule",

        # Rule 4
        r"\brule\s+\d",

        # Named acts
        r"\b[a-z][a-z\s’'\-]+act,\s*\d{4}",

        # Codes
        r"\bcode of\b",

        # Constitution
        r"\bconstitution\b",

        # Ordinance
        r"\bordinance\b",

        # Named regulatory order
        r"\bpension simplification order,\s*\d{4}",

        # Service Rules 2018 etc.
        r"\bservice rules?\s+(?:of\s+)?\d{4}",
    ]

    return any(
        re.search(
            pattern,
            s,
            flags=re.IGNORECASE
        )
        for pattern in patterns
    )


def extract_legal_basis(text):

    sents = sentences(text)

    found = []

    exclude = [
        "rule is made absolute",
        "rule is discharged",
        "rule stands discharged",
        "rule is disposed",
        "no order as to costs",
        "communicate the judgment",
        "learned advocate prays"
    ]

    for sent in sents:

        if contains_any(
            sent,
            exclude
        ):
            continue

        if is_real_legal_basis(
            sent
        ):
            found.append(
                sent
            )

    return unique(found)[:8]


# ============================================================
# COURT REASONING EXTRACTION
# ============================================================

def extract_reasoning(text):

    sents = sentences(text)

    court_signals = [
        "we are of the view",
        "we find that",
        "we find",
        "this court finds",
        "this court is satisfied",
        "court is satisfied",
        "we hold",
        "we observe",
        "we are unable to accept",
        "it is obvious",
        "it is evident",
        "it appears that",
        "it transpires",
        "therefore",
        "thus",
        "consequently",
        "in view of the findings",
        "in view of the discussion",
        "from the above discussion",
        "cannot be relied upon",
        "without any lawful authority"
    ]

    counsel_signals = [
        "learned advocate submits",
        "learned advocate contends",
        "learned advocate argued",
        "learned senior advocate",
        "he submits",
        "she submits",
        "he argued",
        "she argued",
        "he contends",
        "she contends",
        "prays for",
        "prayer is"
    ]

    found = []

    # Court reasoning generally becomes stronger
    # in the latter part of judgments.
    start = int(
        len(sents) * 0.35
    )

    for sent in sents[start:]:

        if contains_any(
            sent,
            counsel_signals
        ):
            continue

        if contains_any(
            sent,
            court_signals
        ):
            found.append(
                sent
            )

    return unique(found)[-8:]


# ============================================================
# PROCESS
# ============================================================

def process(
    item,
    decision_map
):

    jid = item["id"]

    full_text = load_full_text(
        jid
    )

    facts = extract_facts(
        full_text
    )

    issues = extract_issues(
        full_text
    )

    legal_basis = (
        extract_legal_basis(
            full_text
        )
    )

    reasoning = extract_reasoning(
        full_text
    )

    decision_record = (
        decision_map.get(
            jid,
            {}
        )
    )

    decisions = (
        decision_record.get(
            "final_decision_evidence",
            []
        )
    )

    components = {
        "issue": bool(issues),
        "facts": bool(facts),
        "legal_basis":
            bool(legal_basis),
        "reasoning":
            bool(reasoning),
        "decision":
            bool(decisions)
    }

    return {

        "id": jid,

        "source_file":
            item.get(
                "source_file",
                f"{jid}.txt"
            ),

        "facts_inferred":
            item.get(
                "facts_inferred",
                False
            ),

        "legal_issue_evidence":
            issues,

        "relevant_fact_evidence":
            facts,

        "legal_basis_evidence":
            legal_basis,

        "court_reasoning_evidence":
            reasoning,

        "decision_evidence":
            decisions,

        "components_found":
            components
    }


# ============================================================
# MAIN
# ============================================================

def main():

    with open(
        STRUCTURED_FILE,
        "r",
        encoding="utf-8"
    ) as f:

        judgments = json.load(f)

    with open(
        FINAL_DECISION_FILE,
        "r",
        encoding="utf-8"
    ) as f:

        decisions = json.load(f)

    decision_map = {
        x["id"]: x
        for x in decisions
    }

    output = [

        process(
            item,
            decision_map
        )

        for item in judgments
    ]

    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            output,
            f,
            ensure_ascii=False,
            indent=2
        )

    print(
        "Processed judgments:",
        len(output)
    )

    print(
        "\nComponents found:"
    )

    for component in [
        "issue",
        "facts",
        "legal_basis",
        "reasoning",
        "decision"
    ]:

        count = sum(
            x["components_found"][
                component
            ]
            for x in output
        )

        print(
            f"  {component}: "
            f"{count}/{len(output)}"
        )

    print(
        "\nSaved:",
        OUTPUT_FILE
    )


if __name__ == "__main__":
    main()