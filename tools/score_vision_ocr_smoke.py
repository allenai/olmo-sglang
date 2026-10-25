"""Run official olmOCR checks on the capped integration-smoke generations.

Requires olmocr==0.4.27. These diagnostic counts are not benchmark accuracy:
only eight frozen pages are present and their responses may be truncated.
"""

import argparse
import json
from collections import Counter
from pathlib import Path

from olmocr.bench.tests import load_single_test


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inference_json")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    inference = json.loads(Path(args.inference_json).read_text())
    rows = []
    counts = Counter()
    for record in inference["records"]:
        if not record["name"].startswith("ocr-"):
            continue
        page = record["page"]
        tests = [{"category": page["category"], "test": row} for row in page["tests"]]
        if not any(row["test"]["type"] == "baseline" for row in tests):
            tests.append(
                {
                    "category": "baseline",
                    "test": {
                        "pdf": page["tests"][0]["pdf"],
                        "page": 1,
                        "id": record["name"] + "-baseline",
                        "type": "baseline",
                    },
                }
            )
        for entry in tests:
            test = entry["test"]
            # Do not silently turn missing scorer dependencies into model errors.
            passed, explanation = load_single_test(dict(test)).run(
                record["result"]["text"]
            )
            rows.append(
                {
                    "request": record["name"],
                    "id": test["id"],
                    "type": test["type"],
                    "category": entry["category"],
                    "passed": bool(passed),
                    "explanation": explanation,
                    "finish_reason": record["result"]["meta_info"]["finish_reason"],
                }
            )
            counts["total"] += 1
            counts["passed"] += bool(passed)
    if not rows:
        raise ValueError("No OCR records to score")
    result = {
        "scope": "Capped integration smoke; not benchmark accuracy",
        "counts": dict(counts),
        "tests": rows,
    }
    Path(args.output).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["counts"]))


if __name__ == "__main__":
    main()
