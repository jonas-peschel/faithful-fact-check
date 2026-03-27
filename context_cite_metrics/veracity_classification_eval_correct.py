import argparse 
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from utils import load_json, save_json

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluation of answer correctness via veracity classification performance for fact-checking datasets.")
    parser.add_argument("--results_path", type=str, default=None, help="Path to the json file with results formatted using format_results.py with predicted labels.")
   
    return parser.parse_args()

def get_label_names(dataset_name):

    if (dataset_name == "averitec" or dataset_name == "averitec_short_ans" 
        or dataset_name == "averitec_web_evidence"  or dataset_name == "averitec_web_evidence_short_ans"):
        return ['Supported', 'Refuted', 'Conflicting Evidence/Cherrypicking', 'Not Enough Evidence']

def get_label_names_pretty(dataset_name):

    if (dataset_name == "averitec" or dataset_name == "averitec_short_ans" 
        or dataset_name == "averitec_web_evidence"  or dataset_name == "averitec_web_evidence_short_ans"):
        return ['Supported', 'Refuted', 'Conflicting Evidence', 'Not Enough Evidence']

def main(config=None):

    if config is None:
        config = parse_args()

    results = load_json(config.results_path)
    if results[-1].get("classification_performance"):
        results = results[:-1]

    dataset_name = results[0]["dataset"]
    LABELS = get_label_names(dataset_name)
    LABELS_PRETTY = get_label_names_pretty(dataset_name)
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

    #--- confusion matrix ---#
    cm = confusion_matrix(y_true, y_pred)
    cm_normalized = cm / np.sum(cm, axis=1, keepdims=True)

    # annotations with raw count + percentage
    annot = []
    for i in range(cm.shape[0]):
        annot_row = []
        for j in range(cm.shape[1]):
            count = cm[i,j]
            ratio = cm_normalized[i,j]
            annot_row.append(f"{count}\n({100*ratio:.1f}%)")
        annot.append(annot_row)
    annot = np.array(annot)

    # plot
    fig, ax = plt.subplots() 
    sns.heatmap(
        cm_normalized,
        annot=annot, 
        fmt="", 
        cmap="Blues",
        vmin=0,
        vmax=1,
        xticklabels=LABELS_PRETTY,
        yticklabels=LABELS_PRETTY,
        ax=ax,
    )
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=9)

    # save plot
    plot_save_path = f"plots/confusion_matrix_{dataset_name}.png"
    fig.savefig(plot_save_path, bbox_inches="tight")
    print(f"Saved confusion matrix to: {plot_save_path}")

if __name__ == "__main__":
    main() 