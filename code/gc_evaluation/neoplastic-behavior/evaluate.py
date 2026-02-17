"""
DALPHIN Neoplastic Behavior Evaluation Script

Evaluates neoplastic behavior predictions against ground truth.
Computes multiclass Cohen's kappa and MCC, and per-class F1 and MCC (benign, malignant), per data subset (full and reader).

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
from sklearn.metrics import matthews_corrcoef, f1_score, cohen_kappa_score

INPUT_DIRECTORY = Path("/input")
OUTPUT_DIRECTORY = Path("/output")

READER_SUBSETS = {"reader-no-semi-expert", "reader-semi-expert"}


class DalphinNeoplasticBehaviorEvaluation:
    def __init__(self, input_dir: Path):
        self.gt_path = Path("/opt/app/resources/ground_truth.xlsx")
        self.pred_dir = input_dir

    # ---------- Validation ----------
    def validate_submission(self):
        errors = []

        gt_df = pd.read_excel(self.gt_path)

        gt_ids = {
            qid for qid in gt_df["question-id"].astype(str)
            if qid.endswith("behavior")
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
                    if qid.endswith("behavior")
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
                    if not qid.endswith("behavior"):
                        continue
                    answer = str(row.get("response", "")).strip().rstrip('.,').lower()
                    if answer not in {"benign", "malignant", "uncertain", "in situ"} and answer != "":
                        invalid_answers.append((qid, row.get("response")))

                if invalid_answers:
                    for qid, ans in invalid_answers:
                        errors.append(
                        f"Invalid response for question-id {qid}: '{ans}'. "
                        "Responses must be one of {'benign', 'malignant', 'uncertain', 'in situ'}."
                    )

        if errors:
            raise ValueError("Submission validation failed:\n" + "\n".join(errors))

    # ---------- Evaluation ----------
    def evaluate(self) -> dict:
        gt_df = pd.read_excel(self.gt_path)

        gt_df = gt_df[(gt_df["question-id"].str.endswith("behavior")) & (gt_df["expected answer"] != "NOT APPLICABLE")]

        pred_csv = next(self.pred_dir.glob("*.csv"))
        pred_df = pd.read_csv(pred_csv)
        pred_lookup = dict(zip(pred_df["question-id"], pred_df["response"]))

        class_names = ['benign', 'uncertain', 'in situ', 'malignant']
        records = []

        for _, row in gt_df.iterrows():
            qid = row["question-id"]
            expected = str(row["expected answer"]).strip().lower()
            subset_raw = str(row["subset"])

            response = pred_lookup.get(qid, "")
            
            generated = str(response).strip().rstrip('.,').lower()

            generated_onehot = {cls: int(cls == generated) for cls in class_names}
            expected_onehot = {cls: int(cls == expected) for cls in class_names}

            records.append({
                "question-id": qid,
                "subset_raw": subset_raw,
                **{f"expected_{cls}": val for cls, val in expected_onehot.items()},
                **{f"generated_{cls}": val for cls, val in generated_onehot.items()}
            })

        df_metrics = pd.DataFrame(records)

        subset_masks = {
            "full": np.ones(len(df_metrics), dtype=bool),
            "reader": df_metrics["subset_raw"].isin(READER_SUBSETS)
        }

        metrics_by_subset = {}

        for subset_name, mask in subset_masks.items():
            df_sub = df_metrics[mask]
            if df_sub.empty:
                metrics_by_subset[subset_name] = {key: None for key in (
                    'MCC', 'cohen_kappa', 'benign_F1', 'benign_MCC', 'malignant_F1', 'malignant_MCC',
                    )}
                continue

            subset_metrics = {}

            # Overall metrics
            y_true_labels = np.argmax(df_sub[[f'expected_{cls}' for cls in class_names]].values, axis=1)
            y_pred_labels = np.argmax(df_sub[[f'generated_{cls}' for cls in class_names]].values, axis=1)
            
            subset_metrics['MCC'] = matthews_corrcoef(y_true_labels, y_pred_labels)
            subset_metrics['cohen_kappa'] = cohen_kappa_score(y_true_labels, y_pred_labels)

            # Per-class metrics for benign and malignant
            for cls in ['benign', 'malignant']:
                y_true_bin = df_sub[f'expected_{cls}'].astype(int)
                y_pred_bin = df_sub[f'generated_{cls}'].astype(int)

                subset_metrics[f'{cls}_F1'] = f1_score(y_true_bin, y_pred_bin, zero_division=0)
                subset_metrics[f'{cls}_MCC'] = matthews_corrcoef(y_true_bin, y_pred_bin)

            metrics_by_subset[subset_name] = subset_metrics

        return metrics_by_subset

# --- Main Execution ---
def main():
    evaluator = DalphinNeoplasticBehaviorEvaluation(INPUT_DIRECTORY)
    
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