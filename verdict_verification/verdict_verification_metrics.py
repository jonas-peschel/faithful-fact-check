import argparse 
from utils import load_json, save_json
import numpy as np
from numpy.typing import NDArray 
from sklearn.metrics import mean_squared_error, cohen_kappa_score, classification_report

def parse_args():
    parser = argparse.ArgumentParser(description="Compute classification performance metrics for verdict verification experiment.")
    parser.add_argument("--results_path", type=str, help="Path to the file where verdict verification experiment results are stored.")

    return parser.parse_args()

LABELS = ["Supported", "Conflicting Evidence/Cherrypicking", "Refuted", "Not Enough Evidence"]

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

    # include instances with "not enough evidence" for confusion matrix-based metrics
    y_pred = np.array([np.argmax(res["class_distributions"][attr_method][f"k={k}"]) if res["label"] != "Not Enough Evidence" else 3 for res in results["results"]])
    y_true = np.array([np.argmax(res["class_distributions"]["ground_truth"]) if res["label"] != "Not Enough Evidence" else 3 for res in results["results"]])

    cm_metrics = classification_report(
        y_true,
        y_pred,
        target_names=LABELS,
        zero_division=np.nan,
        output_dict=True,
    )

    # exclude instances with "not enough evidence" for ordinal metrics
    y_pred = np.array([np.argmax(res["class_distributions"][attr_method][f"k={k}"]) for res in results["results"] if res["label"] != "Not Enough Evidence"])
    y_pred_probs = np.array([res["class_distributions"][attr_method][f"k={k}"] for res in results["results"] if res["label"] != "Not Enough Evidence"])
    y_true = np.array([np.argmax(res["class_distributions"]["ground_truth"]) for res in results["results"] if res["label"] != "Not Enough Evidence"])

    mse = mean_squared_error(y_true, y_pred)
    kappa = cohen_kappa_score(y_true, y_pred, weights="quadratic").item()
    rps, rps_scores = ranked_probability_score(y_true, y_pred_probs)

    return cm_metrics, mse, kappa, rps

def main(config=None):

    if config is None:
        config = parse_args()

    results = load_json(config.results_path)
    attr_methods = list(results["results"][0]["class_distributions"].keys())
    attr_methods.remove("ground_truth")

    # TODO: add ks as cli argument later
    ks = ["all", "cite"]

    # compute metrics
    results["metrics"] = {}
    for attr_method in attr_methods:
        results["metrics"][attr_method] = {}
        for k in ks:
            cm_metrics, mse, kappa, rps = calc_metrics(results, k, attr_method)
            results["metrics"][attr_method][f"k={k}"] = {
                "confusion_matrix_metrics": cm_metrics,
                "mean_squared_error": mse,
                "cohens_kappa": kappa,
                "ranked_probability_score": rps,
            }

    save_json(config.results_path, results)
    print(f"Saved results to: {config.results_path}")

if __name__ == "__main__":
    main()