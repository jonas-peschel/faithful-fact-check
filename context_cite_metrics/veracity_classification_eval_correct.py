import argparse 
import numpy as np
from sklearn.metrics import classification_report
from utils import load_json, save_json

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluation of answer correctness via veracity classification performance for fact-checking datasets.")
    parser.add_argument("--results_path", type=str, default=None, help="Path to the json file with results formatted using format_results.py with predicted labels.")
   
    return parser.parse_args()

def get_label_names(dataset_name):

    if dataset_name == "averitec":
        return ['Supported', 'Refuted', 'Conflicting Evidence/Cherrypicking', 'Not Enough Evidence']

def main(config=None):

    if config is None:
        config = parse_args()

    results = load_json(config.results_path)
    dataset_name = results[0]["dataset"]
    LABELS = get_label_names(dataset_name)
    label_to_idx = {l: i for i, l in enumerate(LABELS)}

    y_true = [label_to_idx[res["label"]] for res in results]
    y_pred = [label_to_idx[res["pred_label"]] for res in results]   

    # classification metrics
    results_dict = classification_report(
        y_true,
        y_pred,
        target_names=LABELS,
        zero_division=np.nan,
        output_dict=True,
    )

    # append results dict to the input file and save it
    results.append({
        "classification_performance": results_dict,
    })
    save_json(config.results_path, results)

    # print results
    print(classification_report(
        y_true,
        y_pred,
        target_names=LABELS,
        zero_division=np.nan,
    ))

if __name__ == "__main__":
    main() 