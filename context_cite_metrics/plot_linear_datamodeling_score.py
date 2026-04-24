import argparse 
import numpy as np 
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from utils import load_json, order_results, METH2COL, METH2LABEL, DATASET2LABEL

def parse_args():

    parser = argparse.ArgumentParser(description="Plot results for ContextCite linear datamodeling score metric.")
    parser.add_argument("results_path", type=str, help="Path to the file where attribution scores and experiment results (metrics) are stored.")
    parser.add_argument("plots_savepath", type=str, help="Path where to save the generated plots.")
    parser.add_argument("--plot_title", type=str, default="Linear Datamodeling Score (LDS)", help="Title for the plot.")
    parser.add_argument("--is_error_bars", action="store_true", help="Whether to plot the error bars with the standard error of the mean.")

    return parser.parse_args()

def aggregate_lds(results, attr_methods):
    """
    Calculate mean linear datamodeling score over all sentences in the data 
    for each answer attribution method.

    Args:
        results (Dict):
            Contains the results from metrics computations.
        attr_methods (List[str]):
            Names of attribution methods for which to calculate LDS.

    Returns:
        mean_lds (NDArray[float]):
            Mean linear datamodeling scores. Shape: (n_methods,)
        sem_lds (NDArray[float]):
            Standard error of the mean of the linear datamodeling scores.
            Shape: (n_methods,)
    """

    lds = []
    for method in attr_methods:
        lds_method = []
        for data_point_result in results["results"]:
            if "metrics" in data_point_result["methods"][method]:
                lds_method.extend(data_point_result["methods"][method]["metrics"]["LDS"])
            else:  # skip missing data
                continue

        lds.append(lds_method)

    lds = np.array(lds, dtype=float)    # using dtype=float to convert None values back to np.nan
    # delete any columns (i.e. answer sentences) for which at least one entry is invalid (NaN)
    mask = np.all(~np.isnan(lds), axis=0)
    print(f"Dropped {lds.shape[1]-(mask.sum())}/{lds.shape[1]} sentences with NaN values.")
    lds = lds[:,mask]

    mean_lds = np.mean(lds, axis=1, keepdims=True)
    sem_lds = np.std(lds, axis=1, keepdims=True) / np.sqrt(lds.shape[1])

    return mean_lds, sem_lds

#--- Plotting Function ---#
def plot_linear_datamodeling_score(mean_lds, sem_lds, labels, dataset_name, is_error_bars=False, title=None):
    """
    Plot bar plot of linear datamodeling score metric.

    Args:
        mean_lds (NDArray[float]): 
            Linear datamodeling score for different attribution methods.
            Mean values over data points and answer sentences.
            Shape: (n_methods,)
        sem_lds (NDArray[float]): 
            Standard error of the mean for different attribution methods
            over data points and answer sentences.
            Shape: (n_methods,)
        labels (List[str]):
            Names of the different attribution methods.
        dataset_name (str):
            Name of dataset.
        title (str, optional):
            Plot title.
    """

    fig, ax = plt.subplots()

    x = np.arange(mean_lds.shape[1])
    bar_width = 1 / (mean_lds.shape[0] + 1)

    mean_lds, sem_lds, labels = order_results(mean_lds, sem_lds, labels)  # ordering for the plot

    multiplier = 0
    for mean, sem, label in zip(mean_lds, sem_lds, labels):
        offset = bar_width * multiplier 
        if is_error_bars:
            rects = ax.bar(x+offset, mean, bar_width, label=METH2LABEL[label], 
                        edgecolor="white", linewidth=0.5, color=METH2COL[label], yerr=sem, capsize=2, error_kw={"ecolor": "black", "lw": 1.0})
        else:
            rects = ax.bar(x+offset, mean, bar_width, label=label, edgecolor="white", linewidth=0.5, color=METH2COL[label])
        multiplier += 1

    ax.set_title(title or DATASET2LABEL[dataset_name])
    ax.set_ylabel("Linear datamodeling score")
    ax.set_xticks([], [])
    ax.legend(bbox_to_anchor=(1.04, 0), loc="lower left", borderaxespad=0)  # place legend outside of plot
    ax.set_axisbelow(True)
    ax.grid(axis="y", linestyle="--", alpha=0.7)

    return fig
#--- Plotting Function ---#



def main(config=None):

    if config is None:
        config = parse_args()

    # load metrics results
    results_path = Path(config.results_path)
    results = load_json(results_path)

    # get attribution methods and remove methods for which no LDS can be calculated
    attr_methods = list(results["results"][0]["methods"].keys())
    if "llm_post_hoc" in attr_methods:
        attr_methods.remove("llm_post_hoc")
    if "nli_post_hoc_greedy_sampling" in attr_methods:
        attr_methods.remove("nli_post_hoc_greedy_sampling")
    if "longcite_llm_direct" in attr_methods:
        attr_methods.remove("longcite_llm_direct")

    # aggregate mean and standard error of the mean over the data
    mean_lds, sem_lds = aggregate_lds(results, attr_methods)

    # plot and save
    fig = plot_linear_datamodeling_score(mean_lds, sem_lds, labels=attr_methods, 
                                         dataset_name=results["metadata"]["dataset"], is_error_bars=config.is_error_bars, title=config.plot_title)

    plots_savepath = Path(config.plots_savepath)
    plots_savepath.parent.mkdir(exist_ok=True, parents=True)
    fig.savefig(plots_savepath, bbox_inches="tight")
    print(f"Saved results to: {plots_savepath}")

if __name__ == "__main__":
    main()