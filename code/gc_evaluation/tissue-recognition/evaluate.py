"""
DALPHIN Tissue Recognition Evaluation Script

Evaluates organ/tissue predictions against ground truth.
Computes average hierarchical organ recognition scores per data subset (full, reader, semi) and per subspecialty (full set only).

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
import re
from collections import defaultdict
from organ_scoring import load_taxonomy, compute_organ_score

INPUT_DIRECTORY = Path("/input")
OUTPUT_DIRECTORY = Path("/output")

READER_SUBSETS = {"reader-no-semi-expert", "reader-semi-expert"}
SEMI_SUBSET = "reader-semi-expert"

INCLUDED_SUBSPECIALTIES = {
    "genitourinary",
    "gastrointestinal",
    "soft tissue",
    "breast",
    "thoracic",
    "nephropathology",
    "hepatopancreatobiliary",
    "dermatopathology",
}


class DalphinTissueEvaluation:
    def __init__(self, input_dir: Path):
        self.gt_path = Path("/opt/app/resources/ground_truth.xlsx")
        self.taxonomy_path = Path("/opt/app/resources/taxonomy.yaml")
        self.pred_dir = input_dir
        
        self.nodes, self.lookup, self.graph = load_taxonomy(self.taxonomy_path)
        self.synonym_patterns = self._load_synonym_patterns()

    # ---------- Validation ----------
    def validate_submission(self):
        errors = []

        gt_df = pd.read_excel(self.gt_path)

        gt_ids = {
            qid for qid in gt_df["question-id"].astype(str)
            if qid.endswith("tissue")
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
                    if qid.endswith("tissue")
                }

                if gt_ids != pred_ids:
                    errors.append(
                        "Question-ID mismatches:\n"
                        f"  Only in ground truth: {sorted(gt_ids - pred_ids)}\n"
                        f"  Only in predictions: {sorted(pred_ids - gt_ids)}"
                    )

        if errors:
            raise ValueError("Submission validation failed:\n" + "\n".join(errors))

    # ---------- Evaluation ----------
    def evaluate(self) -> dict:
        gt_df = pd.read_excel(self.gt_path)

        gt_df = gt_df[gt_df["question-id"].str.endswith("tissue")]

        pred_csv = next(self.pred_dir.glob("*.csv"))
        pred_df = pd.read_csv(pred_csv)
        pred_lookup = dict(zip(pred_df["question-id"], pred_df["response"]))

        subspecialty_scores = defaultdict(list)
        subset_scores = {"full": [], "reader": [], "semi": []}

        for _, row in gt_df.iterrows():
            qid = row["question-id"]
            expected = str(row["expected answer"])
            subspecialty = str(row["subspecialty"])
            subset_raw = str(row["subset"])

            response = pred_lookup.get(qid, "")

            if not isinstance(response, str) or not response.strip():
                print(f"Warning: empty prediction for question-id {qid}")

            curated = self._extract_response(response)

            score = self._max_score_against_expected(curated, expected)

            if subspecialty in INCLUDED_SUBSPECIALTIES:
                subspecialty_scores[subspecialty].append(score)

            subset_scores["full"].append(score)
            if subset_raw in READER_SUBSETS:
                subset_scores["reader"].append(score)
            if subset_raw == SEMI_SUBSET:
                subset_scores["semi"].append(score)

        return {
            "BySubspecialty": {
                k: round(float(np.mean(v)), 3) if v else 0.0
                for k, v in subspecialty_scores.items()
            },
            "BySubset": {
                k: round(float(np.mean(v)), 3) if v else 0.0
                for k, v in subset_scores.items()
            }
        }

    # ---------- Helpers ----------
    def _max_score_against_expected(self, response, expected):
        expected_items = [e.strip() for e in expected.split(";") if e.strip()]
        return max(
            compute_organ_score(response, e, self.lookup, self.graph)
            for e in expected_items
        ) if expected_items else 0.0

    def _load_synonym_patterns(self):
        patterns = []
        for syn in self.lookup.keys():
            patterns.append((re.compile(rf"\b{re.escape(syn)}\b"), syn))
        return patterns
    
    def _find_best_match(self, answer_cleaned):
        matches = []

        for pattern, _ in self.synonym_patterns:
            for m in pattern.finditer(answer_cleaned):
                start, end = m.start(), m.end()
                matched_text = answer_cleaned[start:end]
                length = end - start
                matches.append((length, start, matched_text))

        if not matches:
            return None

        # Sort by:
        # 1) longest match first
        # 2) earliest occurrence
        matches.sort(key=lambda x: (-x[0], x[1]))

        return matches[0][2]

    def _extract_response(self, answer: str) -> str:
        if not isinstance(answer, str) or not answer.strip():
            return "Not found: No answer provided"

        cleaned = re.sub(r"\([^)]*\)", "", answer.lower())
        cleaned = re.sub(r"\s+", " ", cleaned).strip()

        best_match = self._find_best_match(cleaned)
        if best_match:
            return best_match
        
        return answer

# --- Main Execution ---
def main():
    evaluator = DalphinTissueEvaluation(INPUT_DIRECTORY)
    
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