import argparse 
from utils import load_json, save_json
import numpy as np
import re 
from numpy.typing import NDArray 
from sklearn.metrics import mean_squared_error, cohen_kappa_score, classification_report

def int_or_str(value: str):
    try:
        return int(value)
    except ValueError:
        return value

def parse_args():
    parser = argparse.ArgumentParser(description="Compute classification performance metrics for verdict verification experiment.")
    parser.add_argument("--results_path", type=str, help="Path to the file where verdict verification experiment results are stored.")
    parser.add_argument("--attr_methods", type=str, nargs="+", choices=["context_cite_32", "context_cite_64", "context_cite_128", "context_cite_256", "semantic_similarity", "leave_one_out", "nli_post_hoc_naive", "nli_post_hoc_sliding_window_3", "nli_post_hoc_sliding_window_5", "nli_post_hoc_greedy_sampling", "llm_post_hoc", "longcite_llm_direct"], help="Which attribution method to use.")
    parser.add_argument("--ks", type=int_or_str, nargs="+", help="Numbers k of how many source sentences to give the model for verification.")
    return parser.parse_args()

LABELS = ["Supported", "Conflicting Evidence/Cherrypicking", "Refuted", "Not Enough Evidence"]
ABSTENTION_THRESH = 0.7

def ranked_probability_score(y_true: NDArray, y_pred_probs: NDArray):
    """
    Calculate the mean Ranked Probability Score (RPS) metric for multiple predictions.
    
    Args:
        y_true (NDArray[int]):
            Ground-truth outcomes. 
            Shape: (n_predictions,)
        y_pred_probs (NDArray[float]):
            Predicted probability distributions.
            Shape: (n_predictions, n_classes)

    Returns:
        rps (float):
            Mean Ranked Probability Score over all predictions.
    """

    n, C = y_pred_probs.shape

    # one-hot encode the ground-truth
    y_true_probs = np.zeros((n, C)) 
    y_true_probs[np.arange(n), y_true] = 1

    # calculate cumulative probability distributions
    true_cdf = np.cumsum(y_true_probs, axis=1)
    pred_cdf = np.cumsum(y_pred_probs, axis=1) 

    # calculate mean RPS 
    rps_scores = 1/(C-1) * np.sum((true_cdf[:,:-1] - pred_cdf[:,:-1])**2, axis=1)
    rps_mean = np.mean(rps_scores).item() 
    return rps_mean, rps_scores

def calc_metrics(results, k, attr_method):

    # filter invalid results, e.g., for BadRequestError in DeepSeek API
    invalid_mask = [res["pred_distributions"][attr_method][f"k={k}"]["verdict_dist_4_classes"] is None for res in results["results"]]
    filtered_results = [res for is_invalid, res in zip(invalid_mask, results["results"]) if not is_invalid]

    # predicted Not Enough Evidence (i.e. abstained)
    pred_nee_mask = np.array([res["pred_distributions"][attr_method][f"k={k}"]["verdict_dist_4_classes"][3] >= ABSTENTION_THRESH for res in filtered_results])

    # ground truth Not Enough Evidence
    gt_nee_mask = np.array([res["label"] == "Not Enough Evidence" for res in filtered_results])

    # include instances with "not enough evidence" for confusion matrix-based metrics
    y_pred = np.array([np.argmax(res["pred_distributions"][attr_method][f"k={k}"]["verdict_dist_3_classes"]) for res in filtered_results])
    y_pred[pred_nee_mask] = 3
    y_true = np.array([LABELS.index(res["label"]) for res in filtered_results])

    cm_metrics = classification_report(
        y_true,
        y_pred,
        target_names=LABELS,
        zero_division=0.0,
        output_dict=True,
    )

    # exclude instances with either prediction or ground truth "not enough evidence" for ordinal metrics
    y_pred = np.array([np.argmax(res["pred_distributions"][attr_method][f"k={k}"]["verdict_dist_3_classes"]) for res in filtered_results])[~(pred_nee_mask | gt_nee_mask)]
    y_pred_probs = np.array([res["pred_distributions"][attr_method][f"k={k}"]["verdict_dist_3_classes"] for res in filtered_results])[~(pred_nee_mask | gt_nee_mask)]
    y_true = np.array([LABELS.index(res["label"]) for res in filtered_results])[~(pred_nee_mask | gt_nee_mask)]

    mse = mean_squared_error(y_true, y_pred)
    kappa = cohen_kappa_score(y_true, y_pred, weights="quadratic").item()
    mean_rps, rps_scores = ranked_probability_score(y_true, y_pred_probs)

    abstention_info = {
        "n_abstentions": np.sum(pred_nee_mask).item(),
        "n_gt_nee": np.sum(gt_nee_mask).item(),
        "n_abstained_or_gt_nee": np.sum((pred_nee_mask | gt_nee_mask)).item(),
        "abstained_or_gt_nee_mask": (pred_nee_mask | gt_nee_mask).tolist(),
        "invalid_instances_mask": invalid_mask,
    }

    return cm_metrics, mse, kappa, mean_rps, rps_scores.tolist(), abstention_info

def main(config=None):

    if config is None:
        config = parse_args()

    results = load_json(config.results_path)

    # compute metrics
    results.setdefault("metrics", {})
    for attr_method in config.attr_methods:
        results["metrics"].setdefault(attr_method, {})
        for k in config.ks:
            cm_metrics, mse, kappa, mean_rps, rps_scores, abstention_info = calc_metrics(results, k, attr_method)
            results["metrics"][attr_method][f"k={k}"] = {
                "abstention_info": abstention_info,
                "confusion_matrix_metrics": cm_metrics,
                "mean_squared_error": mse,
                "cohens_kappa": kappa,
                "mean_ranked_probability_score": mean_rps, 
                "ranked_probability_scores": rps_scores,
            }

    save_json(config.results_path, results)
    print(f"Saved results to: {config.results_path}")

if __name__ == "__main__":
    main()