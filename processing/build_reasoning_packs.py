import json
from pathlib import Path


STRUCTURED_FILE = Path("data/structured_judgments.json")
EVIDENCE_FILE = Path("data/reasoning_evidence.json")
DECISION_FILE = Path("data/final_decisions.json")

OUTPUT_FILE = Path("data/reasoning_packs.json")


def load_json(path):
    with open(
        path,
        "r",
        encoding="utf-8"
    ) as f:
        return json.load(f)


def main():

    structured = load_json(
        STRUCTURED_FILE
    )

    reasoning_evidence = load_json(
        EVIDENCE_FILE
    )

    decisions = load_json(
        DECISION_FILE
    )

    evidence_map = {
        x["id"]: x
        for x in reasoning_evidence
    }

    decision_map = {
        x["id"]: x
        for x in decisions
    }

    results = []

    for judgment in structured:

        jid = judgment["id"]

        ev = evidence_map.get(
            jid,
            {}
        )

        dec = decision_map.get(
            jid,
            {}
        )

        pack = {
            "id": jid,

            "source_file":
                judgment.get(
                    "source_file"
                ),

            "facts_inferred":
                judgment.get(
                    "facts_inferred",
                    False
                ),

            "legal_issue_evidence":
                ev.get(
                    "legal_issue_evidence",
                    []
                ),

            "fact_evidence":
                ev.get(
                    "relevant_fact_evidence",
                    []
                ),

            "legal_basis_evidence":
                ev.get(
                    "legal_basis_evidence",
                    []
                ),

            "court_reasoning_evidence":
                dec.get(
                    "final_reasoning_evidence",
                    []
                ),

            "court_decision_evidence":
                dec.get(
                    "final_decision_evidence",
                    []
                )
        }

        results.append(pack)

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

    print(
        f"Reasoning packs created: "
        f"{len(results)}"
    )

    print(
        f"Saved: {OUTPUT_FILE}"
    )


if __name__ == "__main__":
    main()