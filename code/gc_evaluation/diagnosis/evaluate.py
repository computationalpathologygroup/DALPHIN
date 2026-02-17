"""
DALPHIN Diagnosis Evaluation Script

Evaluates diagnosis predictions against ground truth.
Computes average similarity scores per data subset (full, reader, semi) and per difficulty, cancer incidence, neoplastic status, and subspecialty (full set only).

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
from bert_enhanced_metrics import compute_enhanced_bert_metrics
from metrics import cosine_similarity_biobert, machine_translation_metrics, bert_score_metric
import sys
import os

INPUT_DIRECTORY = Path("/input")
OUTPUT_DIRECTORY = Path("/output")

READER_SUBSETS = {"reader-no-semi-expert", "reader-semi-expert"}
SEMI_SUBSET = "reader-semi-expert"

METRIC_COLUMNS = [
    "similarity_biobert",
    "bert_precision",
    "bert_recall",
    "bert_f1",
    "bleu_1",
    "rouge_l",
    "meteor",
    "cider",
]

INCLUDED_INCIDENCES = {
    "common",
    "rare",
}

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


class DalphinDiagnosisEvaluation:
    def __init__(self, input_dir: Path):
        self.gt_path = Path("/opt/app/resources/ground_truth.xlsx")
        self.pred_dir = input_dir

    # ---------- Validation ----------
    def validate_submission(self):
        errors = []

        gt_df = pd.read_excel(self.gt_path)

        gt_ids = {
            qid for qid in gt_df["question-id"].astype(str)
            if qid.endswith("diagnosis")
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

            if {"question-id", "response"}.issubset(pred_df.columns):
                pred_ids = {
                    qid for qid in pred_df["question-id"].astype(str)
                    if qid.endswith("diagnosis")
                }

                if gt_ids != pred_ids:
                    errors.append(
                        "Question-ID mismatches:\n"
                        f"  Only in ground truth: {sorted(gt_ids - pred_ids)}\n"
                        f"  Only in predictions: {sorted(pred_ids - gt_ids)}"
                    )

                for _, row in pred_df.iterrows():
                    qid = str(row["question-id"])
                    if not qid.endswith("diagnosis"):
                        continue

                    answer = row.get("response", "")

                    if not isinstance(answer, str) or not answer.strip():
                        print(f"Warning: empty prediction for question-id {qid}")

        if errors:
            raise ValueError("Submission validation failed:\n" + "\n".join(errors))

    # ---------- Evaluation ----------
    def evaluate(self) -> dict:
        gt_df = pd.read_excel(self.gt_path)

        gt_df["question-id"] = gt_df["question-id"].astype(str)
        gt_df = gt_df[gt_df["question-id"].str.endswith("diagnosis")]

        pred_csv = next(self.pred_dir.glob("*.csv"))
        pred_df = pd.read_csv(pred_csv)
        pred_lookup = dict(zip(pred_df["question-id"], pred_df["response"]))

        difficulty_scores = defaultdict(lambda: defaultdict(list))
        incidence_scores = defaultdict(lambda: defaultdict(list))
        neoplastic_scores = defaultdict(lambda: defaultdict(list))
        subspecialty_scores = defaultdict(lambda: defaultdict(list))
        subset_scores = {
            "full": defaultdict(list),
            "reader": defaultdict(list),
            "semi": defaultdict(list),
        }

        combined_df = gt_df.copy()
        combined_df["response"] = combined_df["question-id"].map(pred_lookup).fillna("")

        self._evaluate_similarity(
            df=combined_df,
            expected_col="expected answer",
            response_col="response"
        )

        for _, row in combined_df.iterrows():
            difficulty = str(row["difficulty"])
            incidence = str(row["cancer incidence"])
            neoplastic = str(row["neoplastic"])
            subspecialty = str(row["subspecialty"])
            subset_raw = str(row["subset"])

            for metric in METRIC_COLUMNS:
                value = row.get(metric)

                if pd.isna(value):
                    continue

                value = float(value)

                difficulty_scores[difficulty][metric].append(value)

                if incidence in INCLUDED_INCIDENCES:
                    incidence_scores[incidence][metric].append(value)

                neoplastic_scores[neoplastic][metric].append(value)

                if subspecialty in INCLUDED_SUBSPECIALTIES:
                    subspecialty_scores[subspecialty][metric].append(value)

                subset_scores["full"][metric].append(value)

                if subset_raw in READER_SUBSETS:
                    subset_scores["reader"][metric].append(value)

                if subset_raw == SEMI_SUBSET:
                    subset_scores["semi"][metric].append(value)

        metrics = {
            "ByDifficulty": {
                difficulty: {
                    metric: round(float(np.mean(values)), 3) if values else None
                    for metric, values in metric_dict.items()
                }
                for difficulty, metric_dict in difficulty_scores.items()
            },
            "ByCancerIncidence": {
                incidence: {
                    metric: round(float(np.mean(values)), 3) if values else None
                    for metric, values in metric_dict.items()
                }
                for incidence, metric_dict in incidence_scores.items()
            },
            "ByNeoplasticStatus": {
                neoplastic: {
                    metric: round(float(np.mean(values)), 3) if values else None
                    for metric, values in metric_dict.items()
                }
                for neoplastic, metric_dict in neoplastic_scores.items()
            },
            "BySubspecialty": {
                subspecialty: {
                    metric: round(float(np.mean(values)), 3) if values else None
                    for metric, values in metric_dict.items()
                }
                for subspecialty, metric_dict in subspecialty_scores.items()
            },
            "BySubset": {
                subset: {
                    metric: round(float(np.mean(values)), 3) if values else None
                    for metric, values in metric_dict.items()
                }
                for subset, metric_dict in subset_scores.items()
            }
        }

        return metrics
    
    # ---------- Helpers ----------
    def _evaluate_similarity(self, df, expected_col, response_col):
        if df.empty:
            return {
                "avg_similarity_biobert": None,
                "avg_bertscore_f1": None,
                "avg_bertscore_precision": None,
                "avg_bertscore_recall": None,
                "avg_bleu_1": None,
                "avg_cider": None,
                "avg_rouge_l": None,
                "avg_meteor": None
            }, pd.DataFrame()

        # Prepare lists for all (pred, true_item) pairs
        pred_texts = []
        true_items = []
        row_indices = []
        for idx, row in df.iterrows():
            pred = str(row[response_col]).strip().lower()
            true_split = [item.strip().lower() for item in str(row[expected_col]).split(';') if item.strip()]
            for true_item in true_split:
                pred_texts.append(pred)
                true_items.append(true_item)
                row_indices.append(idx)

        # Compute metrics for all pairs in batch
        # Enhanced BERT metrics
        try:
            bert_metrics = compute_enhanced_bert_metrics(true_items, pred_texts)
            biobert_sim = bert_metrics.get('biobert_similarity', [0.0] * len(true_items))
            bert_precision = bert_metrics.get('bert_precision', [0.0] * len(true_items))
            bert_recall = bert_metrics.get('bert_recall', [0.0] * len(true_items))
            bert_f1 = bert_metrics.get('bert_f1', [0.0] * len(true_items))
        except Exception as e:
            biobert_sim = cosine_similarity_biobert(true_items, pred_texts)
            bert_precision, bert_recall, bert_f1 = bert_score_metric(true_items, pred_texts)

        # Machine translation metrics
        with self._suppress_stdout():
            mt_scores = machine_translation_metrics(true_items, pred_texts)
        bleu_1 = mt_scores["BLEU-1"]
        rouge_l = mt_scores["ROUGE-L"]
        meteor = mt_scores["METEOR"]
        cider_individual = mt_scores["CIDEr_individual"]

        # Aggregate max per row
        import collections
        agg = collections.defaultdict(list)
        for i, idx in enumerate(row_indices):
            agg[idx].append({
                "biobert_sim": biobert_sim[i],
                "bert_precision": bert_precision[i],
                "bert_recall": bert_recall[i],
                "bert_f1": bert_f1[i],
                "bleu_1": bleu_1[i],
                "rouge_l": rouge_l[i],
                "meteor": meteor[i],
                "cider": cider_individual[i],
            })

        # For each row, take max for each metric
        metrics_per_row = {
            "similarity_biobert": [],
            "bert_precision": [],
            "bert_recall": [],
            "bert_f1": [],
            "bleu_1": [],
            "rouge_l": [],
            "meteor": [],
            "cider": [],
        }
        for idx in df.index:
            items = agg.get(idx, [])
            if items:
                metrics_per_row["similarity_biobert"].append(max(x["biobert_sim"] for x in items))
                metrics_per_row["bert_precision"].append(max(x["bert_precision"] for x in items))
                metrics_per_row["bert_recall"].append(max(x["bert_recall"] for x in items))
                metrics_per_row["bert_f1"].append(max(x["bert_f1"] for x in items))
                metrics_per_row["bleu_1"].append(max(x["bleu_1"] for x in items))
                metrics_per_row["rouge_l"].append(max(x["rouge_l"] for x in items))
                metrics_per_row["meteor"].append(max(x["meteor"] for x in items))
                metrics_per_row["cider"].append(max(x["cider"] for x in items))
            else:
                # If no items, fill with 0.0
                for k in metrics_per_row:
                    metrics_per_row[k].append(0.0)

        # Assign to df
        for k in METRIC_COLUMNS:
            df[k] = metrics_per_row[k]
    
    from contextlib import contextmanager

    @contextmanager
    def _suppress_stdout(self):
        old_stdout = sys.stdout
        sys.stdout = open(os.devnull, "w")
        try:
            yield
        finally:
            sys.stdout.close()
            sys.stdout = old_stdout

# --- Main Execution ---
def main():
    evaluator = DalphinDiagnosisEvaluation(INPUT_DIRECTORY)
    
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