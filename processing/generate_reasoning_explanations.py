import json
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch


INPUT_FILE = Path("data/reasoning_packs.json")
OUTPUT_FILE = Path("data/reasoning_explanations_generated.json")

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"


def join_evidence(items):
    if not items:
        return "Evidence unavailable."

    return "\n".join(
        f"- {x}"
        for x in items
    )


def build_prompt(item):

    return f"""
You are a legal explanation assistant.

Your task is to explain the court judgment in clear, simple Bangla.

IMPORTANT RULES:
1. Use ONLY the evidence provided below.
2. Do NOT invent facts, laws, arguments, reasons, or decisions.
3. If evidence for a component is insufficient, write:
   "পর্যাপ্ত প্রমাণ পাওয়া যায়নি।"
4. Preserve the meaning of the court's decision exactly.
5. Do not give legal advice.
6. Keep each section concise.
7. Legal Act names, section numbers and case-specific identifiers may remain in English.

CASE ID:
{item["id"]}

FACT EVIDENCE:
{join_evidence(item.get("fact_evidence", []))}

LEGAL ISSUE EVIDENCE:
{join_evidence(item.get("legal_issue_evidence", []))}

LEGAL BASIS EVIDENCE:
{join_evidence(item.get("legal_basis_evidence", []))}

COURT REASONING EVIDENCE:
{join_evidence(item.get("court_reasoning_evidence", []))}

COURT DECISION EVIDENCE:
{join_evidence(item.get("court_decision_evidence", []))}

Return ONLY valid JSON in exactly this format:

{{
  "facts": "সহজ বাংলায় মামলার গুরুত্বপূর্ণ ঘটনা",
  "legal_issue": "আদালতের সামনে মূল আইনি প্রশ্ন",
  "legal_basis": "প্রযোজ্য আইন, ধারা বা বিধান",
  "court_reasoning": "আদালত কেন এই সিদ্ধান্তে পৌঁছেছে",
  "decision": "আদালতের চূড়ান্ত সিদ্ধান্ত",
  "plain_bangla_explanation": "২-৪ বাক্যে পুরো মামলার সহজ বাংলা ব্যাখ্যা"
}}
"""


def clean_json_output(text):

    text = text.strip()

    if "```json" in text:
        text = text.split("```json", 1)[1]
        text = text.split("```", 1)[0]

    elif "```" in text:
        text = text.split("```", 1)[1]
        text = text.split("```", 1)[0]

    text = text.strip()

    start = text.find("{")
    end = text.rfind("}")

    if start != -1 and end != -1:
        text = text[start:end + 1]

    return text


def main():

    print("Loading model:", MODEL_NAME)

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME
    )

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float32,
        device_map="cpu"
    )

    model.eval()

    with open(
        INPUT_FILE,
        "r",
        encoding="utf-8"
    ) as f:
        items = json.load(f)

    outputs = []

    for index, item in enumerate(items[:3], 1):

        print(
            f"[{index}/{len(items)}] "
            f"Generating {item['id']}..."
        )

        prompt = build_prompt(item)

        messages = [
            {
                "role": "user",
                "content": prompt
            }
        ]

        formatted = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        inputs = tokenizer(
            formatted,
            return_tensors="pt"
        )

        with torch.no_grad():

            generated = model.generate(
                **inputs,
                max_new_tokens=450,
                do_sample=False,
                temperature=None,
                top_p=None
            )

        output_tokens = generated[
            0,
            inputs["input_ids"].shape[1]:
        ]

        response = tokenizer.decode(
            output_tokens,
            skip_special_tokens=True
        )

        cleaned = clean_json_output(
            response
        )

        try:

            parsed = json.loads(
                cleaned
            )

            parsed["id"] = item["id"]

            outputs.append(
                parsed
            )

        except Exception as e:

            print(
                "JSON parse failed:",
                item["id"],
                e
            )

            outputs.append({
                "id": item["id"],
                "facts": "",
                "legal_issue": "",
                "legal_basis": "",
                "court_reasoning": "",
                "decision": "",
                "plain_bangla_explanation": "",
                "raw_output": response,
                "parse_error": str(e)
            })

        # Save after every judgment
        with open(
            OUTPUT_FILE,
            "w",
            encoding="utf-8"
        ) as f:

            json.dump(
                outputs,
                f,
                ensure_ascii=False,
                indent=2
            )

    print("\nCompleted:", len(outputs))

    print(
        "Saved:",
        OUTPUT_FILE
    )


if __name__ == "__main__":
    main()