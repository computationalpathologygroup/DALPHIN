"""
DALPHIN Neoplastic Status Evaluation Script

Evaluates neoplastic status predictions against ground truth.
Computes average F1, MCC, precision, recall, and specificity scores per data subset (full, reader, semi).

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
from sklearn.metrics import matthews_corrcoef, f1_score, precision_score

INPUT_DIRECTORY = Path("/input")
OUTPUT_DIRECTORY = Path("/output")

READER_SUBSETS = {"reader-no-semi-expert", "reader-semi-expert"}
SEMI_SUBSET = "reader-semi-expert"


class DalphinNeoplasticStatusEvaluation:
    def __init__(self, input_dir: Path):
        self.gt_path = Path("/opt/app/resources/ground_truth.xlsx")
        self.pred_dir = input_dir

    # ---------- Validation ----------
    def validate_submission(self):
        errors = []

        gt_df = pd.read_excel(self.gt_path)

        gt_ids = {
            qid for qid in gt_df["question-id"].astype(str)
            if qid.endswith("neoplasm")
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
                    if qid.endswith("neoplasm")
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
                    if not qid.endswith("neoplasm"):
                        continue
                    answer = str(row.get("response", "")).strip().rstrip('.,').lower()
                    if answer not in {"yes", "no"} and answer != "":
                        invalid_answers.append((qid, row.get("response")))

                if invalid_answers:
                    for qid, ans in invalid_answers:
                        errors.append(
                        f"Invalid response for question-id {qid}: '{ans}'. "
                        "Responses must be one of {'yes', 'no'}."
                    )

        if errors:
            raise ValueError("Submission validation failed:\n" + "\n".join(errors))

    # ---------- Evaluation ----------
    def evaluate(self) -> dict:
        gt_df = pd.read_excel(self.gt_path)

        gt_df = gt_df[gt_df["question-id"].str.endswith("neoplasm")]

        pred_csv = next(self.pred_dir.glob("*.csv"))
        pred_df = pd.read_csv(pred_csv)
        pred_lookup = dict(zip(pred_df["question-id"], pred_df["response"]))

        records = []
        for _, row in gt_df.iterrows():
            qid = row["question-id"]
            expected = str(row["expected answer"]).strip().lower()
            subset_raw = str(row["subset"])

            response = pred_lookup.get(qid, "")

            generated = str(response).strip().rstrip('.,').lower()

            records.append({
                "question-id": qid,
                "expected_yes": 1 if expected == "yes" else 0,
                "generated_yes": 1 if generated == "yes" else 0,
                "subset_raw": subset_raw
            })

        df_metrics = pd.DataFrame(records)

        subset_masks = {
            "full": np.ones(len(df_metrics), dtype=bool),
            "reader": df_metrics["subset_raw"].isin(READER_SUBSETS),
            "semi": df_metrics["subset_raw"] == SEMI_SUBSET
        }

        metrics_by_subset = {}

        for subset_name, mask in subset_masks.items():
            df_sub = df_metrics[mask]
            if df_sub.empty:
                metrics_by_subset[subset_name] = {
                    "sensitivity": None,
                    "specificity": None,
                    "F1": None,
                    "MCC": None,
                    "precision": None
                }
                continue

            y_true = df_sub["expected_yes"].astype(int)
            y_pred = df_sub["generated_yes"].astype(int)

            subset_metrics = {}

            if np.any(y_true == 1):
                subset_metrics["sensitivity"] = float(np.mean(y_pred[y_true == 1]))
            else:
                subset_metrics["sensitivity"] = None

            if np.any(y_true == 0):
                subset_metrics["specificity"] = float(1 - np.mean(y_pred[y_true == 0]))
            else:
                subset_metrics["specificity"] = None

            if len(np.unique(y_true)) > 1:
                subset_metrics["F1"] = f1_score(y_true, y_pred, zero_division=0)
                subset_metrics["MCC"] = matthews_corrcoef(y_true, y_pred)
                subset_metrics["precision"] = precision_score(y_true, y_pred, zero_division=0)
            else:
                subset_metrics["F1"] = None
                subset_metrics["MCC"] = None
                subset_metrics["precision"] = None

            metrics_by_subset[subset_name] = subset_metrics

        return metrics_by_subset

# --- Main Execution ---
def main():
    evaluator = DalphinNeoplasticStatusEvaluation(INPUT_DIRECTORY)
    
    evaluator.validate_submission()

    metrics = evaluator.evaluate()
    metrics = round_floats(metrics, decimals=3)
    pprint(metrics)

    write_metrics(metrics=metrics)
    return 0

def write_metrics(*, metrics):
    write_json_file(location=OUTPUT_DIRECTORY / "metrics.json", content=metrics)

def write_json_file(*, location, content):
    with open(location, "w") as f:
        f.write(json.dumps(content, indent=4))

def round_floats(obj, decimals=3):
    if isinstance(obj, float):
        return round(obj, decimals)
    elif isinstance(obj, dict):
        return {k: round_floats(v, decimals) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [round_floats(v, decimals) for v in obj]
    else:
        return obj

if __name__ == "__main__":
    raise SystemExit(main())