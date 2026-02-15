import argparse 
import numpy as np 
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import json
from pathlib import Path
import re

def parse_args():

    parser = argparse.ArgumentParser(description="Plot results for ContextCite top-k log-prob drop metric.")
    parser.add_argument("results_path", type=str, help="Path to the file where attribution scores and experiment results (metrics) are stored.")
    parser.add_argument("plots_savepath", type=str, help="Path where to save the generated plots.")
    parser.add_argument("--plot_title", type=str, default="Top-k Log-Probability Drop Metric", help="Title for the plot.")
    parser.add_argument("--ks", type=int, nargs="+", choices=range(1,10), default=None, help="For which k's to plot the results.")

    return parser.parse_args()

def load_json(filepath):

    with open(filepath) as f:
        data = json.load(f)

    return data

def aggregate_log_prob_drops(results, ks):
    """
    Calculate mean log-prob drop over all sentences in the data for
    each answer attribution method and number of ablations k.

    Args:
        results (Dict):
            Contains the results from metrics computations.
        ks (List[str]):
            Numbers of context ablations for which the metric has been computed.

    Returns:
        mean_drops (NDArray[float]):
            Mean log-prob drop values. Shape: (n_methods, n_ks)
    """

    methods = list(results["results"][0]["methods"].keys())

    drops = []  # (n_methods, n_Ks, n_sentences)
    for method in methods:
        drops_method = []   # (n_Ks, n_sentences)
        for data_point_result in results["results"]:
            top_k_drop_dict = data_point_result["methods"][method]["metrics"]["top_k_drop"]
            top_k_drops = []
            for k in ks:
                top_k_drops.append(top_k_drop_dict[f"top_{k}_drop"])    
            top_k_drops = np.array(top_k_drops) # top_k_drops: (n_Ks, n_sentences in this data point)
            drops_method.append(top_k_drops)

        # concatenate for all data_points for the method in drops_method
        drops_method = np.concat(drops_method, axis=1)  # (n_Ks, n_sentences)
        drops.append(drops_method)

    drops = np.array(drops)
    mean_drops = drops.mean(axis=2) # (n_methods, n_Ks)

    return mean_drops

#--- Plotting Functions ---#
def plot_top_k_log_prob_drop(mean_drops, labels, ks, title="Top-k Log-Probability Drop Metric"):
    """
    Plot bar plot of top-k log-prob drop metric.

    Args:
        mean_drops (NDArray[float]): 
            Top-k log-probability drop for different attribution methods and different k.
            Mean values over data points and answer sentences.
            Shape: (n_methods, n_ks)
        labels (List[str]):
            Names of the different attribution methods.
        ks (List[int]):
            Numbers of context ablations for which the metric has been computed.
        title (str, optional):
            Plot title.
    """

    fig, ax = plt.subplots()

    x = np.arange(mean_drops.shape[1])
    bar_width = 1 / (mean_drops.shape[0] + 1)

    multiplier = 0
    for drops, label in zip(mean_drops, labels):
        offset = bar_width * multiplier 
        rects = ax.bar(x+offset, drops, bar_width, label=label, edgecolor="white", linewidth=0.5)
        multiplier += 1

    ax.set_title(title)
    ax.set_ylabel("Log-prob drop")
    ax.set_xticks(x + bar_width*(mean_drops.shape[0]-1)/2, [f"k={k}" for k in ks])
    ax.legend()
    ax.set_axisbelow(True)
    ax.grid(axis="y", linestyle="--", alpha=0.7)

    return fig
#--- Plotting Functions ---#

def main(config=None):

    if config is None:
        config = parse_args()

    # load metrics results
    results_path = Path(config.results_path)
    results = load_json(results_path)

    # use all ks from the data if not provided
    if config.ks is None:
        # good luck trying to read this
        config.ks = [int(re.compile(r"top_(\d+)_drop").match(key).group(1)) for key in list(list(results["results"][0]["methods"].values())[0]["metrics"]["top_k_drop"].keys())]    

    # aggregate mean over the data
    mean_drops = aggregate_log_prob_drops(results, config.ks)

    # plot and save
    fig = plot_top_k_log_prob_drop(mean_drops, labels=list(results["results"][0]["methods"].keys()), ks=config.ks, title=config.plot_title)

    plots_savepath = Path(config.plots_savepath)
    plots_savepath.parent.mkdir(exist_ok=True, parents=True)
    fig.savefig(plots_savepath)


if __name__ == "__main__":
    main()