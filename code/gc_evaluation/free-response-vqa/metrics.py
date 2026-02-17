import torch
from bert_score import BERTScorer
from sentence_transformers import SentenceTransformer, util
import os
import locale


def mean_pooling(model_output, attention_mask):
    """
    Mean pooling to obtain a fixed-size sentence embedding.
    """
    token_embeddings = (
        model_output.last_hidden_state
    )  # (batch_size, seq_len, hidden_size)
    input_mask_expanded = (
        attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    )
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, dim=1)
    sum_mask = input_mask_expanded.sum(dim=1)
    return sum_embeddings / sum_mask


def get_biobert_model():
    if not hasattr(get_biobert_model, "model"):
        get_biobert_model.model = SentenceTransformer("pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb", 
                                                      cache_folder="/opt/models/biobert")
    return get_biobert_model.model


def cosine_similarity_biobert(reports_true, reports_pred, model=None):
    if model is None:
        model = get_biobert_model()
    embeddings_true = model.encode(reports_true, convert_to_tensor=True)
    embeddings_pred = model.encode(reports_pred, convert_to_tensor=True)
    cosine_matrix = util.pytorch_cos_sim(embeddings_true, embeddings_pred)
    cosine_similarities = [cosine_matrix[i, i].item() for i in range(len(reports_true))]
    return cosine_similarities


def machine_translation_metrics(reports_true, reports_pred):
    """
    Calculate machine translation metrics (BLEU-1, BLEU-2, BLEU-3, BLEU-4, CIDEr, ROUGE, METEOR).

    Args:
        reports_true (list of str): List of reference texts.
        reports_pred (list of str): List of hypothesis texts.

    Returns:
        Dictionary containing per-report scores and averaged scores for BLEU, CIDEr, ROUGE, and METEOR.
    """
    # Initialize scorers and tracking variables
    scorers = _initialize_scorers()
    scores = _initialize_scores_dict()
    meteor_state = {"should_skip": False}

    # Compute per-sentence metrics
    _compute_per_sentence_metrics(
        reports_true, reports_pred, scorers, scores, meteor_state
    )

    # Compute corpus-level CIDEr scores
    _compute_cider_scores(reports_true, reports_pred, scores)

    return scores


def _initialize_scorers():
    """Initialize all metric scorers with proper configuration."""
    from pycocoevalcap.bleu.bleu import Bleu
    from pycocoevalcap.rouge.rouge import Rouge
    from pycocoevalcap.meteor.meteor import Meteor

    scorers = {"bleu": Bleu(), "rouge": Rouge(), "meteor": None}

    # Initialize METEOR with locale handling
    _setup_meteor_locale()

    try:
        scorers["meteor"] = Meteor()
    except Exception as e:
        print(f"⚠️  Warning: Could not initialize METEOR: {e}")

    return scorers


def _setup_meteor_locale():
    """Set up proper locale for METEOR Java backend."""
    try:
        # Force English locale for consistent decimal separators in METEOR
        os.environ["LC_NUMERIC"] = "C"
        locale.setlocale(locale.LC_NUMERIC, "C")
    except:
        pass  # Ignore locale setting errors


def _initialize_scores_dict():
    """Initialize the scores dictionary structure."""
    return {
        "BLEU-1": [],
        "BLEU-2": [],
        "BLEU-3": [],
        "BLEU-4": [],
        "ROUGE-L": [],
        "METEOR": [],
    }


def _compute_per_sentence_metrics(
    reports_true, reports_pred, scorers, scores, meteor_state
):
    """Compute BLEU, ROUGE, and METEOR metrics for each sentence pair."""
    for text_true, text_pred in zip(reports_true, reports_pred):
        # Prepare texts for evaluation
        reference, candidate = _prepare_texts_for_scoring(text_true, text_pred)

        # Compute BLEU scores
        bleu_scores = _compute_bleu(reference, candidate, scorers["bleu"])

        # Compute METEOR score
        meteor_score = _compute_meteor(
            reference, candidate, scorers["meteor"], meteor_state
        )

        # Compute ROUGE score
        rouge_score = _compute_rouge(reference, candidate, scorers["rouge"])

        # Store scores
        _store_sentence_scores(scores, bleu_scores, rouge_score, meteor_score)


def _prepare_texts_for_scoring(text_true, text_pred):
    """Clean and prepare texts for scoring."""
    text_true = _clean_text_for_meteor(str(text_true).strip())
    text_pred = _clean_text_for_meteor(str(text_pred).strip())

    # Handle empty texts after cleaning
    if not text_true:
        text_true = "empty"
    if not text_pred:
        text_pred = "empty"

    reference = {"id1": [text_true]}
    candidate = {"id1": [text_pred]}

    return reference, candidate


def _clean_text_for_meteor(text):
    """Clean text to prevent METEOR Java backend issues."""
    # Replace newlines with spaces (common cause of METEOR failures)
    text = text.replace("\n", " ").replace("\r", " ")
    # Replace multiple spaces with single spaces
    text = " ".join(text.split())
    # Remove or replace problematic characters that interfere with METEOR's pipe format
    text = text.replace("|||", " | | | ")  # Replace triple pipes
    # Remove other control characters that might cause issues
    text = "".join(c for c in text if c.isprintable() or c.isspace())
    return text.strip()


def _compute_bleu(reference, candidate, bleu_scorer):
    """Compute BLEU scores (1-4 grams)."""
    score_b, _ = bleu_scorer.compute_score(reference, candidate)
    return score_b


def _compute_rouge(reference, candidate, rouge_scorer):
    """Compute ROUGE-L score."""
    score_r, _ = rouge_scorer.compute_score(reference, candidate)
    return score_r


def _compute_meteor(reference, candidate, meteor_scorer, meteor_state):
    """Compute METEOR score with error handling."""

    if meteor_state["should_skip"]:
        return -1.0  # Indicate METEOR was skipped
    else:
        score_r, _ = meteor_scorer.compute_score(reference, candidate)
        return score_r


def _store_sentence_scores(scores, bleu_scores, rouge_score, meteor_score):
    """Store individual sentence scores in the scores dictionary."""
    scores["BLEU-1"].append(bleu_scores[0])
    scores["BLEU-2"].append(bleu_scores[1])
    scores["BLEU-3"].append(bleu_scores[2])
    scores["BLEU-4"].append(bleu_scores[3])
    scores["ROUGE-L"].append(rouge_score)
    scores["METEOR"].append(meteor_score)


def _compute_cider_scores(reports_true, reports_pred, scores):
    """Compute CIDEr scores at the dataset level."""
    from pycocoevalcap.cider.cider import Cider

    try:
        cider_scorer = Cider()

        # Clean texts for CIDEr computation
        cleaned_true = [
            str(ref).strip() if str(ref).strip() else "empty" for ref in reports_true
        ]
        cleaned_pred = [
            str(hyp).strip() if str(hyp).strip() else "empty" for hyp in reports_pred
        ]

        reference_corpus = {i: [ref] for i, ref in enumerate(cleaned_true)}
        candidate_corpus = {i: [hyp] for i, hyp in enumerate(cleaned_pred)}

        cider_score, cider_individual_scores = cider_scorer.compute_score(
            reference_corpus, candidate_corpus
        )
        scores["CIDEr"] = cider_score
        scores["CIDEr_individual"] = cider_individual_scores

    except Exception as e:
        print(f"⚠️  Warning: CIDEr computation failed, using 0.0: {e}")
        scores["CIDEr"] = 0.0
        scores["CIDEr_individual"] = [0.0] * len(reports_true)


def get_bertscorer():
    if not hasattr(get_bertscorer, "scorer"):
        get_bertscorer.scorer = BERTScorer(model_type="microsoft/deberta-xlarge-mnli", 
                                            num_layers=40, device="cpu")
    return get_bertscorer.scorer


def bert_score_metric(reports_true, reports_pred, scorer=None):
    if scorer is None:
        scorer = get_bertscorer()
    P, R, F1 = scorer.score(reports_true, reports_pred)
    # Convert tensors to lists
    p_list = P.cpu().numpy().tolist()
    r_list = R.cpu().numpy().tolist()
    f1_list = F1.cpu().numpy().tolist()
    return p_list, r_list, f1_list
