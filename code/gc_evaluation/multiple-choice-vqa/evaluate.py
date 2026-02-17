"""
DALPHIN Multiple-Choice VQA Evaluation Script

Evaluates multiple-choice predictions against ground truth.
Computes average accuracy per data subset (full, reader, semi) and per category (full set only).

Expected input structure:
    /opt/app/resources/
        ground_truth.xlsx (ground truth labels)
    /input/
        *.csv (predicted labels)

Output:
    /output/metrics.json
"""

import json
import pandas as pd
from pathlib import Path
from pprint import pprint
import numpy as np
from collections import defaultdict
import re

INPUT_DIRECTORY = Path("/input")
OUTPUT_DIRECTORY = Path("/output")

READER_SUBSETS = {"reader-no-semi-expert", "reader-semi-expert"}
SEMI_SUBSET = "reader-semi-expert"


class DalphinMultipleChoiceEvaluation:
    def __init__(self, input_dir: Path):
        self.gt_path = Path("/opt/app/resources/ground_truth.xlsx")
        self.pred_dir = input_dir

    # ---------- Validation ----------
    def validate_submission(self):
        errors = []

        gt_df = pd.read_excel(self.gt_path)

        gt_ids = {
            qid for qid in gt_df["question-id"].astype(str)
            if qid.endswith("mc")
        }

        csv_files = list(self.pred_dir.glob("*.csv"))
        if len(csv_files) != 1:
            errors.append("Exactly one CSV file is required")

        else:
            pred_df = pd.read_csv(csv_files[0])
            if "question-id" not in pred_df.columns:
                errors.append("Prediction CSV missing 'question-id' column")

            if "response" not in pred_df.columns:
                errors.append("Prediction CSV missing 'response' column")

            if "question-id" in pred_df.columns:
                pred_ids = {
                    qid for qid in pred_df["question-id"].astype(str)
                    if qid.endswith("mc")
                }

                if gt_ids != pred_ids:
                    errors.append(
                        "Question-ID mismatches:\n"
                        f"  Only in ground truth: {sorted(gt_ids - pred_ids)}\n"
                        f"  Only in predictions: {sorted(pred_ids - gt_ids)}"
                    )

            if {"question-id", "response"}.issubset(pred_df.columns):
                invalid_answers = []
                for _, row in pred_df.iterrows():
                    qid = str(row["question-id"])
                    if not qid.endswith("mc"):
                        continue
                    answer = str(row.get("response", "")).strip().rstrip('.,').lower()
                    if not re.fullmatch(r"[a-f]", answer):
                        invalid_answers.append((qid, row.get("response")))

                if invalid_answers:
                    for qid, ans in invalid_answers:
                        print(
                            f"Warning: prediction for question-id {qid} is invalid ('{ans}'), "
                            "responses must be a single letter from 'a' to 'f'. "
                            "If the invalid prediction contains letters, the first letter will be used; "
                            "otherwise the prediction will be treated as incorrect."
                        )

        if errors:
            raise ValueError("Submission validation failed:\n" + "\n".join(errors))

    # ---------- Evaluation ----------
    def evaluate(self) -> dict:
        gt_df = pd.read_excel(self.gt_path)

        gt_df = gt_df[gt_df["question-id"].str.endswith("mc")]

        pred_csv = next(self.pred_dir.glob("*.csv"))
        pred_df = pd.read_csv(pred_csv)
        pred_lookup = dict(zip(pred_df["question-id"], pred_df["response"]))

        category_scores = defaultdict(list)
        subset_scores = {"full": [], "reader": [], "semi": []}

        for _, row in gt_df.iterrows():
            qid = row["question-id"]
            category = str(row["category"])
            subset_raw = str(row["subset"])

            expected = str(row["expected answer"]).strip().lower()
            expected_items = [e.strip() for e in expected.split(";") if e.strip()]

            response = pred_lookup.get(qid, "")

            generated = str(response).strip().rstrip(".,").lower() if isinstance(response, str) else ""
            curated = generated[0] if re.fullmatch(r"[a-z]", generated) else ""

            score = float(curated in expected_items)

            category_scores[category].append(score)
            subset_scores["full"].append(score)

            if subset_raw in READER_SUBSETS:
                subset_scores["reader"].append(score)
            if subset_raw == SEMI_SUBSET:
                subset_scores["semi"].append(score)

        return {
            "ByCategory": {
                k: round(float(np.mean(v)), 3) if v else 0.0
                for k, v in category_scores.items()
            },
            "BySubset": {
                k: round(float(np.mean(v)), 3) if v else 0.0
                for k, v in subset_scores.items()
            }
        }

# --- Main Execution ---
def main():
    evaluator = DalphinMultipleChoiceEvaluation(INPUT_DIRECTORY)
    
    evaluator.validate_submission()

    metrics = evaluator.evaluate()
    pprint(metrics)

    write_metrics(metrics=metrics)
    return 0

def write_metrics(*, metrics):
    write_json_file(location=OUTPUT_DIRECTORY / "metrics.json", content=metrics)

def write_json_file(*, location, content):
    with open(location, "w") as f:
        f.write(json.dumps(content, indent=4))

if __name__ == "__main__":
    raise SystemExit(main())