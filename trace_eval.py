import argparse
import os
import numpy as np
from scipy import stats
from sklearn.metrics import average_precision_score
from sentence_transformers import SentenceTransformer, util
import torch
from tqdm import tqdm
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", type=str, default="bert-base-uncased", help="base model")
    parser.add_argument("--pretrain_model", type=str, default="./output/groovy/groovy", help="path of the pretrained model to finetune")
    parser.add_argument("--dataset_root", type=str, default="./datasets", help="root directory of the dataset")
    parser.add_argument("--project", type=str, default="groovy", help="project name")
    return parser.parse_args()

def get_paths(args):
    req_dir = os.path.join(args.dataset_root, args.project, "req")
    src_dir = os.path.join(args.dataset_root, args.project, "src")
    rtm_file = os.path.join(args.dataset_root, args.project, "RTM_CLASS.txt")
    return req_dir, src_dir, rtm_file

def load_data(req_dir, src_dir, rtm_file):
    requirements = {}
    source_code = {}
    ground_truth = {}

    for filename in os.listdir(req_dir):
        with open(os.path.join(req_dir, filename), 'r') as file:
            requirements[os.path.splitext(filename)[0]] = file.read()

    for filename in os.listdir(src_dir):
        with open(os.path.join(src_dir, filename), 'r') as file:
            source_code[os.path.splitext(filename)[0]] = file.read()

    with open(rtm_file, 'r') as file:
        for line in file:
            req, src, score = line.strip().split()
            ground_truth[(req, src)] = float(score)

    return requirements, source_code, ground_truth

def calculate_similarity_matrix(model, requirements, source_code):
    def preprocess_text(text):
        return str(text).strip()[:512] if text else ""

    req_texts = [preprocess_text(text) for text in requirements.values() if text]
    src_texts = [preprocess_text(text) for text in source_code.values() if text]

    if not req_texts or not src_texts:
        raise ValueError("No valid input texts found after preprocessing")

    logger.info(f"Number of requirements: {len(req_texts)}")
    logger.info(f"Number of source code files: {len(src_texts)}")

    batch_size = 32
    req_embeddings = torch.cat([model.encode(req_texts[i:i+batch_size], convert_to_tensor=True) 
                                for i in tqdm(range(0, len(req_texts), batch_size), desc="Encoding requirements")])
    src_embeddings = torch.cat([model.encode(src_texts[i:i+batch_size], convert_to_tensor=True) 
                                for i in tqdm(range(0, len(src_texts), batch_size), desc="Encoding source code")])
    
    similarity_matrix = util.pytorch_cos_sim(req_embeddings, src_embeddings)
    
    return similarity_matrix, list(requirements.keys()), list(source_code.keys())

def calculate_ap(y_true, y_pred):
    """
    Calculate Average Precision (AP) for a single query.
    
    :param y_true: Binary relevance labels (1 for relevant, 0 for irrelevant)
    :param y_pred: Predicted scores or similarities
    :return: AP value
    """
    sorted_indices = np.argsort(y_pred)[::-1]
    y_true = y_true[sorted_indices]
    
    relevant_docs = np.sum(y_true)
    if relevant_docs == 0:
        return 0.0
    
    precisions = np.cumsum(y_true) / np.arange(1, len(y_true) + 1)
    ap = np.sum(precisions * y_true) / relevant_docs
    
    return ap

def calculate_metrics(similarity_matrix, ground_truth, req_files, src_files):
    ap_scores = []
    precisions_at_10 = []
    recalls_at_10 = []

    for i, req in enumerate(req_files):
        y_true = np.array([ground_truth.get((req, src), 0) for src in src_files])
        y_pred = similarity_matrix[i].cpu().numpy()
        
        ap = calculate_ap(y_true, y_pred)
        ap_scores.append(ap)

        top_10_indices = np.argsort(y_pred)[-10:]
        relevant = set(src for src in src_files if ground_truth.get((req, src), 0) > 0)
        retrieved = set(np.array(src_files)[top_10_indices])
        
        true_positives = len(relevant.intersection(retrieved))
        precisions_at_10.append(true_positives / 10 if relevant else 0)
        recalls_at_10.append(true_positives / len(relevant) if relevant else 0)

    map_score = np.mean(ap_scores)
    precision_at_10 = np.mean(precisions_at_10)
    recall_at_10 = np.mean(recalls_at_10)

    return ap_scores, map_score, precision_at_10, recall_at_10

def calculate_p_value_and_cohens_d(group1, group2):
    t_statistic, p_value = stats.ttest_ind(group1, group2)
    n1, n2 = len(group1), len(group2)
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    cohens_d = (np.mean(group1) - np.mean(group2)) / pooled_std
    return p_value, cohens_d

def evaluate_traceability(model, req_dir, src_dir, rtm_file):
    try:
        requirements, source_code, ground_truth = load_data(req_dir, src_dir, rtm_file)
        similarity_matrix, req_files, src_files = calculate_similarity_matrix(model, requirements, source_code)
        ap_scores, map_score, precision_at_10, recall_at_10 = calculate_metrics(similarity_matrix, ground_truth, req_files, src_files)
        
        logger.info(f"MAP: {map_score:.4f}")
        logger.info(f"Precision@10: {precision_at_10:.4f}")
        logger.info(f"Recall@10: {recall_at_10:.4f}")
        
        # Log individual AP scores
        for i, ap in enumerate(ap_scores):
            logger.info(f"AP for query {i+1}: {ap:.4f}")
        
        return ap_scores, map_score, precision_at_10, recall_at_10
    except Exception as e:
        logger.error(f"An error occurred during traceability evaluation: {str(e)}")
        return None, None, None, None

def evaluate(args):
    try:
        model = SentenceTransformer(args.pretrain_model)
        req_dir, src_dir, rtm_file = get_paths(args)
        ap_scores, map_score, precision_at_10, recall_at_10 = evaluate_traceability(model, req_dir, src_dir, rtm_file)
        
        if ap_scores is not None:
            logger.info(f"Evaluation completed successfully.")
            logger.info(f"MAP: {map_score:.4f}")
            logger.info(f"Precision@10: {precision_at_10:.4f}")
            logger.info(f"Recall@10: {recall_at_10:.4f}")
            
            # Display AP statistics
            logger.info(f"AP Min: {min(ap_scores):.4f}")
            logger.info(f"AP Max: {max(ap_scores):.4f}")
            logger.info(f"AP Median: {np.median(ap_scores):.4f}")
            
            # Here you would typically compare with another model's results
            # For demonstration, we'll just use a random sample
            other_ap_scores = np.random.rand(len(ap_scores))
            p_value, cohens_d = calculate_p_value_and_cohens_d(ap_scores, other_ap_scores)
            
            logger.info(f"p-value: {p_value:.4f}")
            logger.info(f"Cohen's d: {cohens_d:.4f}")
        else:
            logger.error("Evaluation failed.")
    except Exception as e:
        logger.error(f"An error occurred during evaluation: {str(e)}")

if __name__ == '__main__':
    args = parse_args()
    
    evaluate(args)