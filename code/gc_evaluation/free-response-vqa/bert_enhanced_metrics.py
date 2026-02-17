import torch
from bert_score import BERTScorer
from sentence_transformers import SentenceTransformer, util

def get_biobert_model():
    if not hasattr(get_biobert_model, "model"):
        get_biobert_model.model = SentenceTransformer("pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb", 
                                                      cache_folder="/opt/models/biobert")
    return get_biobert_model.model

def get_bertscorer():
    if not hasattr(get_bertscorer, "scorer"):
        get_bertscorer.scorer = BERTScorer(model_type="microsoft/deberta-xlarge-mnli", 
                                            num_layers=40, device="cpu")
    return get_bertscorer.scorer

def mean_pooling(model_output, attention_mask):
    """
    Mean pooling to obtain a fixed-size sentence embedding.
    """
    token_embeddings = model_output.last_hidden_state  # (batch_size, seq_len, hidden_size)
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, dim=1)
    sum_mask = input_mask_expanded.sum(dim=1)
    return sum_embeddings / sum_mask

def cosine_similarity_biobert(reports_true, reports_pred):
    """
    Computes cosine similarity between two lists of texts using BioBERT model.
    Lazily instantiates BioBERT model only once per process.
    """
    model = get_biobert_model()
    embeddings_true = model.encode(reports_true, convert_to_tensor=True)
    embeddings_pred = model.encode(reports_pred, convert_to_tensor=True)
    # Compute cosine similarities in one go (this returns a matrix)
    cosine_matrix = util.pytorch_cos_sim(embeddings_true, embeddings_pred)
    # Extract the diagonal (each pair's similarity)
    cosine_similarities = [cosine_matrix[i, i].item() for i in range(len(reports_true))]
    return cosine_similarities

def bert_score_metric(reports_true, reports_pred):
    """
    Compute BERTScore metrics (Precision, Recall, F1).
    """
    scorer = get_bertscorer()
    P, R, F1 = scorer.score(reports_pred, reports_true)
    # Convert tensors to lists
    p_list = P.cpu().numpy().tolist()
    r_list = R.cpu().numpy().tolist()
    f1_list = F1.cpu().numpy().tolist()
    return p_list, r_list, f1_list

def compute_enhanced_bert_metrics(true_texts, pred_texts):
    """
    Compute all BERT-based metrics in one function call.
    Returns dictionary with all BERT metric scores per pair.
    """
    if not true_texts or not pred_texts:
        return {}
    
    try:
        # BERTScore
        bert_p, bert_r, bert_f1 = bert_score_metric(true_texts, pred_texts)
    except Exception as e:
        print(f"Warning: BERTScore computation failed: {e}")
        bert_p = bert_r = bert_f1 = [0.0] * len(true_texts)
    
    try:
        # BioBERT similarity
        biobert_sim = cosine_similarity_biobert(true_texts, pred_texts)
    except Exception as e:
        print(f"Warning: BioBERT similarity computation failed: {e}")
        biobert_sim = [0.0] * len(true_texts)
    
    return {
        'bert_precision': bert_p,
        'bert_recall': bert_r,
        'bert_f1': bert_f1,
        'biobert_similarity': biobert_sim
    }